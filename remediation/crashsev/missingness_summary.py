"""crashsev.missingness_summary — missingness structure of the primary feature tier (MISS-002).

Development years ONLY (no final-year outcome is read). For every strict-tier input field this
reports missingness overall, by development year, and by (development) target class — the three
axes along which informative missingness could bias the study — and states the prespecified
high-missingness rule that feeds the low-missingness sensitivity benchmark:

    RULE (prespecified): strict-tier fields with > 50% missingness on the development rows are
    removed in `configs/route_r_09_12_lowmiss_sensitivity.yml`.

Usage:
    python -m crashsev.missingness_summary --data _local_data/modeling_table_09_12.csv \
        --config configs/route_r_09_12.yml --out experiment
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cli as CLI
from . import contracts as K
from . import target as T

PKG_ROOT = Path(__file__).resolve().parents[1]
RULE_THRESHOLD = 0.50


def _missing_mask(s: pd.Series, tokens) -> pd.Series:
    toks = {str(t).strip().lower() for t in tokens if str(t).strip()}
    if pd.api.types.is_numeric_dtype(s):
        return s.isna()
    sv = s.astype(str).str.strip()
    return s.isna() | (sv == "") | sv.str.lower().isin(toks | {"nan"})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Strict-tier missingness structure (MISS-002).")
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(PKG_ROOT / "experiment"))
    args = ap.parse_args(argv)
    cfg = CLI.load_config(args.config)

    prep = CLI.prepare_development(args.data, cfg)   # structural isolation: dev rows only
    schema = prep.schema
    X = prep.X_dev
    years = pd.Series(prep.years_dev).astype(int)
    y = prep.y_dev
    uy = sorted(years.unique().tolist())

    rows = []
    for c in X.columns:
        m = _missing_mask(X[c], schema.string_missing_tokens)
        overall = float(m.mean())
        by_year = {yr: float(m[years.to_numpy() == yr].mean()) for yr in uy}
        by_cls = {k: float(m[(y == k).to_numpy()].mean()) for k in (0, 1, 2)}
        rows.append((c, overall, by_year, by_cls))
    rows.sort(key=lambda t: -t[1])

    flagged = [c for c, o, *_ in rows if o > RULE_THRESHOLD]
    lines = [
        "# Strict-tier missingness structure — development years only (MISS-002, v4.1)\n",
        f"Fields: {len(rows)} (the v4 primary strict tier after invariant drop). "
        f"Missing = NaN / blank / documented missing token.\n",
        f"**Prespecified high-missingness rule:** fields with > {RULE_THRESHOLD:.0%} development "
        f"missingness are removed in the low-missingness sensitivity benchmark "
        f"(`configs/route_r_09_12_lowmiss_sensitivity.yml`). Flagged: "
        + (", ".join(f"`{c}`" for c in flagged) if flagged else "none") + ".\n",
        "| field | overall | " + " | ".join(str(yr) for yr in uy) + " | class 0 | class 1 | class 2 |",
        "|---|---|" + "---|" * (len(uy) + 3),
    ]
    for c, o, by, bc in rows[:25]:
        yr_cells = " | ".join(f"{by[yr]:.3f}" for yr in uy)
        lines.append(f"| {c} | {o:.3f} | {yr_cells} | {bc[0]:.3f} | {bc[1]:.3f} | {bc[2]:.3f} |")
    if len(rows) > 25:
        lines.append(f"| *(… {len(rows) - 25} further fields below 25th rank, all lower missingness)* "
                     + "| " * (len(uy) + 4) + "|")
    lines += [
        "",
        "**Reading.** Class-conditional missingness differences flag potentially informative",
        "missingness (a value's absence correlating with the outcome or with reporting practice);",
        "year-conditional differences flag reporting drift. The low-missingness sensitivity",
        "(paper §7.8) tests whether the primary contrast survives removing the flagged fields.",
        "",
        f"*Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by "
        f"`crashsev/missingness_summary.py`; development rows only; aggregates only.*",
    ]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "missingness_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[missingness] {len(rows)} fields; flagged >{RULE_THRESHOLD:.0%}: {flagged}")
    print(f"[missingness] wrote {out / 'missingness_summary.md'}")


if __name__ == "__main__":
    main()
