"""Codebook & analytical-unit verification for the raw 2009-2012 crash-level extract.

This is a *read-only forensic audit*, not part of the modeling pipeline. It exists to
justify (or refute) the single most consequential inferential decision in the Route A
study: that a **blank ("NaN") Crash Severity in this extract denotes a property-damage-only
(KABCO 'O') crash**, and that the KABCO->ordinal collapse reproduces the reported class
balance. Every claim the Route A paper makes about the target rests on this file's evidence.

It writes three artifacts to ``remediation/data/`` (all safe to commit — no coordinates,
no per-crash rows, only aggregate counts and column metadata):

* ``codebook_evidence_09_12.json`` — machine-readable evidence bundle.
* ``column_inventory_09_12.csv``   — all 100 raw columns: dtype, null rate, n_unique, sample.
* ``codebook_verification_09_12.md`` — human-readable audit note.

Run:  ``python data_audit/verify_codebook_09_12.py``
The raw workbook path is read from ``$CRASH_XLSX_09_12`` or defaults to ``data/raw/`` in the repo
(a reviewer supplies their own lawful copy -- see ``research/DATA_AUTHORITY_AND_ACCESS.md``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
DEFAULT_XLSX = str(DATA_DIR / "raw" / "Crash Level 09-12 (1).xlsx")  # repo-relative; supply a lawful copy
XLSX = Path(os.environ.get("CRASH_XLSX_09_12", DEFAULT_XLSX))

SEVERITY_COL = "Crash Severity"
YEAR_COL = "Year"
KEY_COL = "Crash Number"

# Old-style ABC injury labels observed in this extract -> KABCO letter.
LABEL_TO_KABCO = {
    "Fatal": "K",
    "Incapacitating": "A",
    "Non-Incapacitating": "B",
    "Possible": "C",
    # blank / NaN is hypothesised to be 'O' (property-damage-only) — TESTED below, not assumed.
}
# Genuinely unusable severity states -> quarantine (fail-closed, never coerced to a class).
QUARANTINE_LABELS = {"Unknown", "Not Reported", "Null value"}

# KABCO -> 3-level ordinal collapse used throughout the study.
KABCO_TO_ORDINAL = {"O": 0, "C": 1, "B": 1, "A": 2, "K": 2}

# Outcome / post-crash columns that describe the injury result (leakage w.r.t. severity).
OUTCOME_COLS = [
    "Number of Fatalities",
    "Number of Injuries with Fatalities",
    "Number of Injuries without Fatailites",  # (sic — misspelled in source)
    "Unit 1 Person 1 Injury",
]

# Reported class balance from the original paper (3-level), for cross-check only.
PAPER_DIST = {0: 0.677, 1: 0.287, 2: 0.035}


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def main() -> None:
    if not XLSX.exists():
        raise SystemExit(f"raw workbook not found: {XLSX}\nset $CRASH_XLSX_09_12 to its path.")
    df = pd.read_excel(XLSX, engine="openpyxl")
    n = len(df)
    evidence: dict = {"source_file": XLSX.name, "n_rows": int(n), "n_cols": int(df.shape[1])}

    # ---- 1. column inventory ------------------------------------------------
    # Redact the sample value for privacy-sensitive columns (a single real coordinate /
    # street / officer id / timestamp / case number must not be committed).
    redact = {
        "Latitude", "Longitude", "Street", "Intersecting Street", "Officer ID", "Report ID",
        "CDS Number", "Route", "Milepoint", "DateTime", "Date", "Crash Number",
        "Maintenance Station", "Election District", "Officer Agency",
    }
    inv_rows = []
    for c in df.columns:
        s = df[c]
        nonnull = s.dropna()
        sample = nonnull.iloc[0] if len(nonnull) else ""
        inv_rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "n_null": int(s.isna().sum()),
                "null_rate": round(float(s.isna().mean()), 4),
                "n_unique": int(s.nunique(dropna=True)),
                "sample_value": "[redacted]" if c in redact else str(sample)[:60],
            }
        )
    pd.DataFrame(inv_rows).to_csv(DATA_DIR / "column_inventory_09_12.csv", index=False)

    # ---- 2. severity label census ------------------------------------------
    sev = df[SEVERITY_COL]
    blank = sev.isna() | (sev.astype(str).str.strip() == "")
    census = {str(k): int(v) for k, v in sev.value_counts(dropna=False).items()}
    evidence["severity_census"] = census
    evidence["n_blank_severity"] = int(blank.sum())

    # ---- 3. THE decisive test: are blank-severity crashes injury-free? ------
    # If NaN == 'O' (property-damage-only), these rows must show 0 fatalities and no injuries.
    def outcome_profile(mask: pd.Series) -> dict:
        prof = {"n": int(mask.sum())}
        for oc in OUTCOME_COLS:
            if oc not in df.columns:
                prof[oc] = "ABSENT"
                continue
            vals = _num(df.loc[mask, oc])
            if vals.notna().any():
                prof[oc] = {
                    "n_nonnull": int(vals.notna().sum()),
                    "n_gt0": int((vals > 0).sum()),
                    "max": float(np.nanmax(vals.values)) if vals.notna().any() else None,
                    "mean": round(float(np.nanmean(vals.values)), 4) if vals.notna().any() else None,
                }
            else:
                # non-numeric outcome (e.g. injury text); report value distribution head
                txt = df.loc[mask, oc].astype(str).str.strip()
                prof[oc] = {"top_values": {k: int(v) for k, v in txt.value_counts().head(6).items()}}
        return prof

    evidence["blank_severity_outcome_profile"] = outcome_profile(blank)
    evidence["fatal_severity_outcome_profile"] = outcome_profile(sev.astype(str).str.strip() == "Fatal")
    evidence["incap_severity_outcome_profile"] = outcome_profile(sev.astype(str).str.strip() == "Incapacitating")

    # Fraction of blank-severity rows with ANY positive fatality/injury signal:
    fatal_num = _num(df["Number of Fatalities"]) if "Number of Fatalities" in df else pd.Series(np.nan, index=df.index)
    inj_wf = _num(df["Number of Injuries with Fatalities"]) if "Number of Injuries with Fatalities" in df else pd.Series(np.nan, index=df.index)
    inj_nf = _num(df["Number of Injuries without Fatailites"]) if "Number of Injuries without Fatailites" in df else pd.Series(np.nan, index=df.index)
    any_injury_signal = (fatal_num.fillna(0) > 0) | (inj_wf.fillna(0) > 0) | (inj_nf.fillna(0) > 0)
    evidence["blank_and_injury_signal"] = {
        "n_blank": int(blank.sum()),
        "n_blank_with_injury_or_fatality": int((blank & any_injury_signal).sum()),
        "pct_blank_contaminated": round(float((blank & any_injury_signal).mean() / blank.mean()) * 100, 3)
        if blank.mean() else None,
    }

    # ---- 4. 3-level distribution under the hypothesised mapping -------------
    def to_ordinal(label, is_blank) -> int | float:
        if is_blank:
            return 0  # O
        lab = str(label).strip()
        if lab in QUARANTINE_LABELS:
            return np.nan
        kabco = LABEL_TO_KABCO.get(lab)
        if kabco is None:
            return np.nan
        return KABCO_TO_ORDINAL[kabco]

    y = pd.Series([to_ordinal(l, b) for l, b in zip(sev.values, blank.values)], index=df.index)
    usable = y.dropna()
    dist = {int(k): round(float(v), 4) for k, v in usable.value_counts(normalize=True).sort_index().items()}
    evidence["n_quarantined"] = int(y.isna().sum())
    evidence["n_usable"] = int(usable.notna().sum())
    evidence["ordinal_distribution"] = dist
    evidence["paper_distribution"] = PAPER_DIST
    evidence["max_abs_dist_gap_vs_paper"] = round(
        max(abs(dist.get(k, 0) - PAPER_DIST[k]) for k in PAPER_DIST), 4
    )

    # ---- 5. year & key sanity ----------------------------------------------
    evidence["year_counts"] = {int(k): int(v) for k, v in df[YEAR_COL].value_counts().sort_index().items()}
    evidence["key_is_unique_per_row"] = bool(df[KEY_COL].nunique() == n)
    evidence["usable_by_year"] = {
        int(yr): int(((df[YEAR_COL] == yr) & y.notna()).sum()) for yr in sorted(df[YEAR_COL].dropna().unique())
    }
    evidence["severe_by_year"] = {
        int(yr): int(((df[YEAR_COL] == yr) & (y == 2)).sum()) for yr in sorted(df[YEAR_COL].dropna().unique())
    }

    (DATA_DIR / "codebook_evidence_09_12.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    # ---- 6. human-readable note --------------------------------------------
    bc = evidence["blank_and_injury_signal"]
    verdict = (
        "SUPPORTED" if (bc["pct_blank_contaminated"] or 0) < 1.0 else "REFUTED / needs review"
    )
    md = f"""# Codebook verification — raw 2009-2012 crash-level extract

**Source:** `{XLSX.name}`  ·  rows = {n:,}  ·  cols = {df.shape[1]}
**Analytical unit:** crash (`Crash Number` unique per row = {evidence['key_is_unique_per_row']})

## The decisive question: does blank `Crash Severity` mean property-damage-only ('O')?

- Blank-severity rows: **{bc['n_blank']:,}** ({bc['n_blank']/n:.1%} of file)
- Of those, rows carrying *any* fatality/injury count > 0: **{bc['n_blank_with_injury_or_fatality']:,}**
  (**{bc['pct_blank_contaminated']}%** of blank rows)
- **Verdict: {verdict}.** If ~0% of blank-severity crashes carry an injury/fatality signal,
  treating blank as KABCO 'O' (no-injury / PDO) is empirically justified for this extract.

## Fail-closed quarantine (never coerced to a class)

Labels `{sorted(QUARANTINE_LABELS)}` -> quarantined: **{evidence['n_quarantined']:,}** rows.

## 3-level ordinal distribution vs. the original paper

| class | this extract | paper |
|---|---|---|
| 0 (O)         | {dist.get(0,0):.3f} | {PAPER_DIST[0]:.3f} |
| 1 (C,B)       | {dist.get(1,0):.3f} | {PAPER_DIST[1]:.3f} |
| 2 (A,K)       | {dist.get(2,0):.3f} | {PAPER_DIST[2]:.3f} |

Max abs. gap vs paper: **{evidence['max_abs_dist_gap_vs_paper']:.3f}** (usable N = {evidence['n_usable']:,}).

## Temporal & minority support for a chronological final test

Usable crashes by year: {evidence['usable_by_year']}
Severe (class 2) by year: {evidence['severe_by_year']}

*This note is generated by `data_audit/verify_codebook_09_12.py`; do not edit by hand.*
"""
    (DATA_DIR / "codebook_verification_09_12.md").write_text(md, encoding="utf-8")

    print(json.dumps(
        {
            "n_rows": n,
            "blank_severity": evidence["n_blank_severity"],
            "pct_blank_contaminated": bc["pct_blank_contaminated"],
            "ordinal_distribution": dist,
            "max_gap_vs_paper": evidence["max_abs_dist_gap_vs_paper"],
            "n_usable": evidence["n_usable"],
            "n_quarantined": evidence["n_quarantined"],
            "severe_by_year": evidence["severe_by_year"],
            "verdict": verdict,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
