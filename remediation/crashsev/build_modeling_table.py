"""Build the de-identified local modelling table from the licensed raw extract (Route A).

Writes ``_local_data/modeling_table_09_12.parquet`` (git-ignored) — a fast-loading, privacy-
reduced subset of the raw workbook that the governed CLI consumes. It:

* validates the raw file against the Gate-0 contract;
* keeps the **target, temporal, pre-event, at-event** columns (the modelling features), the
  **post-outcome** columns (kept ONLY so the leakage factorial can toggle them on), plus the
  ``Crash Number`` key and ``DateTime``;
* **drops** point coordinates, free-text street/intersection, and officer/report/segment ids
  (privacy) and single-valued constants.

Values are left untouched (numeric-sentinel cleaning happens in the CLI, deterministically), so
the parquet passes the same contract as the raw file and produces an identical study.

It also writes a committable **aggregate** audit (`data/modeling_table_audit.md`) with counts,
per-year class balance, and per-feature null rates — no per-crash rows.

Run: ``python -m crashsev.build_modeling_table --data "<path to Crash Level 09-12 (1).xlsx>"``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import contracts as K

PKG_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = PKG_ROOT / "_local_data"
DATA_DIR = PKG_ROOT / "data"

# Identifier-tier columns that are privacy-sensitive and NOT needed by the CLI (dropped).
DROP_IDENTIFIERS = {
    "Latitude", "Longitude", "Street", "Intersecting Street", "Officer ID", "Report ID",
    "CDS Number", "Route", "Milepoint", "Maintenance Station", "Officer Agency",
    "Reporting Agency", "Detachment", "Date",
}


def build(data_path: str) -> Path:
    schema = K.load_schema(); mapping = K.load_target_mapping(); ledger = K.load_ledger()
    raw = pd.read_excel(data_path, engine="openpyxl") if str(data_path).lower().endswith((".xlsx", ".xls")) \
        else pd.read_csv(data_path, low_memory=False)
    K.assert_valid(raw, schema, mapping)

    tier = ledger.tier()
    keep = [c for c in raw.columns
            if c in {"Crash Number", "DateTime"}
            or (tier.get(c) in {"target", "post_outcome", "temporal", "pre_event", "at_event"}
                and c not in DROP_IDENTIFIERS)]
    table = raw[keep].copy()

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    # CSV (not parquet) to avoid a heavy pyarrow dependency in the minimal lock (AA2-009);
    # this file is git-ignored and local, so format portability trumps size.
    out = LOCAL_DIR / "modeling_table_09_12.csv"
    table.to_csv(out, index=False)

    _write_audit(table, mapping, schema, ledger)
    print(f"wrote {out}  shape={table.shape}  (dropped {len(raw.columns) - len(keep)} privacy/constant cols)")
    return out


def _write_audit(table: pd.DataFrame, mapping, schema, ledger) -> None:
    from . import target as T
    _, y, audit = T.map_severity_from_mapping(table[mapping.target_column], mapping)
    yy = y.dropna().astype(int)
    tier = ledger.tier()
    allowed = [c for c in ledger.allowed() if c in table.columns]

    # per-year class balance (aggregate only)
    yr = pd.to_numeric(table["Year"], errors="coerce")
    by_year = []
    for year in sorted(yr.dropna().unique()):
        m = (yr == year) & y.notna()
        vc = y[m].astype("Int64").value_counts().sort_index()
        by_year.append((int(year), int(m.sum()), {int(k): int(v) for k, v in vc.items()}))

    # per-feature null rate AFTER sentinel cleaning (numeric) — aggregate
    null_lines = []
    for c in allowed:
        s = table[c]
        rule = schema.numeric_sentinels.get(c)
        if rule and pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors="coerce")
            bad = v.isin(rule.get("null_values", []))
            if "valid_min" in rule: bad = bad | (v < rule["valid_min"])
            if "valid_max" in rule: bad = bad | (v > rule["valid_max"])
            nr = float(bad.mean())
        else:
            nr = float((s.astype(str).str.strip().str.lower().isin(
                [t for t in schema.string_missing_tokens if t]) | s.isna()).mean())
        null_lines.append((c, tier.get(c, "?"), round(nr, 3)))
    null_lines.sort(key=lambda t: -t[2])

    lines = [
        "# Modelling table audit (aggregate; no per-crash rows)\n",
        f"Source documented columns kept: **{table.shape[1]}**  ·  rows: **{len(table):,}**",
        f"Usable after fail-closed target mapping: **{int(yy.notna().sum()):,}**  ·  "
        f"quarantined: **{audit.n_quarantined:,}**  ·  blank→{audit.blank_maps_to}\n",
        "## Class balance by year (development = 2009-2011, final test = 2012)\n",
        "| year | usable crashes | class 0 | class 1 | class 2 |",
        "|---|---|---|---|---|",
    ]
    for year, n, vc in by_year:
        lines.append(f"| {year} | {n:,} | {vc.get(0,0):,} | {vc.get(1,0):,} | {vc.get(2,0):,} |")
    lines += [
        "\n## Feature missingness after sentinel cleaning (top 15)\n",
        "| feature | tier | missing rate |", "|---|---|---|",
    ]
    for c, t, nr in null_lines[:15]:
        lines.append(f"| {c} | {t} | {nr:.3f} |")
    lines.append("\n*Generated by `crashsev/build_modeling_table.py`; no per-crash records are shown.*")
    (DATA_DIR / "modeling_table_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the de-identified local modelling table.")
    ap.add_argument("--data", required=True, help="Path to the licensed raw crash extract.")
    build(ap.parse_args(argv).data)


if __name__ == "__main__":
    main()
