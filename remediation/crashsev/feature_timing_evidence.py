"""crashsev.feature_timing_evidence — per-field timing-evidence ledger for the paper
(FEAT-TIME-001, v4.1 revision).

POST HOC documentation generator; modifies no frozen artifact. Renders
``paper/FEATURE_TIMING_EVIDENCE.md`` from the authoritative contract ledger
(``data/feature_availability_ledger.csv``), the frozen run's retained-column list, and the
development-rows missingness structure. Every timing judgement in this study is the author's,
made from field names, the Alaska crash-report form inventory, and reporting-practice
reasoning — no field has custodian-verified recording-time evidence, so NO field is graded
``VERIFIED_EARLY``. The study uses a conservative retrospective feature tier; it has NOT
validated an operational deployment timestamp at which all retained fields are simultaneously
available and unrevised.

Status vocabulary (per revision directive, plus one extension):
  VERIFIED_EARLY / PLAUSIBLY_EARLY / TIMING_UNCERTAIN / OUTCOME_PROXIMAL / OUTCOME_DERIVED /
  EXCLUDED_IDENTIFIER / SPLIT_ONLY, plus EXCLUDED_CONSTANT (invariant columns; extension noted).

Usage (from remediation/):
    python -m crashsev.feature_timing_evidence --data _local_data/modeling_table_09_12.csv \
        --config configs/route_r_09_12.yml --out paper
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from . import cli as CLI
from .missingness_summary import _missing_mask

PKG_ROOT = Path(__file__).resolve().parents[1]
CONTRAST_FLAG = 0.05   # |missing(class0) - missing(class2)| above this => reporting-thoroughness risk

# fields the in-code schema.py grades as damage/outcome-proximal although the contract ledger
# admits them to the broad tier — surfaced explicitly (LEDGER-TENSION-001)
LEDGER_TENSIONS = {"Non Vehicle Damage", "Unit 1 Person 1 Ejected"}


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=PKG_ROOT, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _status(row, retained: set, strict: set, dropped_invariant: set, year_col: str):
    col, avail = row["column"], str(row["availability"]).strip()
    if avail == "target":
        return "TARGET (excluded)", "the outcome itself"
    if avail == "post_outcome":
        return "OUTCOME_DERIVED", "mechanically determined by / recorded because of the outcome"
    if avail == "identifier":
        return "EXCLUDED_IDENTIFIER", "identifier / privacy exclusion"
    if avail == "constant":
        return "EXCLUDED_CONSTANT", "constant column (no information)"
    if col in dropped_invariant:
        return "EXCLUDED_CONSTANT", "invariant on development rows (auto-dropped, MISS-001)"
    if col == year_col:
        return "SPLIT_ONLY", "used only as the chronological split axis"
    if avail == "temporal":
        if col in retained:
            return "PLAUSIBLY_EARLY", "crash-timestamp attribute (known at event; author-judged)"
        return "EXCLUDED_IDENTIFIER", "raw timestamp key (year is the only temporal split input)"
    if avail == "pre_event":
        if col in retained:
            return ("PLAUSIBLY_EARLY",
                    "infrastructure/exposure attribute whose VALUE predates the crash; report-entry "
                    "timing still author-judged")
        return "TIMING_UNCERTAIN", "pre-event value but excluded from the primary tier"
    # at_event
    if col in strict and col in retained:
        return "PLAUSIBLY_EARLY", "scene-observable at-event field (author-judged strict tier)"
    if col in LEDGER_TENSIONS:
        return ("OUTCOME_PROXIMAL",
                "broad-tier-admissible in the contract ledger but damage/response-adjacent "
                "(in-code schema grades it post-outcome) — LEDGER-TENSION-001")
    return ("TIMING_UNCERTAIN",
            "at-event field whose decision-time availability/revision status is not established; "
            "excluded from the primary restricted tier, admitted to the broad sensitivity only")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate paper/FEATURE_TIMING_EVIDENCE.md (post hoc).")
    ap.add_argument("--data", default=str(PKG_ROOT / "_local_data" / "modeling_table_09_12.csv"))
    ap.add_argument("--config", default=str(PKG_ROOT / "configs" / "route_r_09_12.yml"))
    ap.add_argument("--out", default=str(PKG_ROOT / "paper"))
    args = ap.parse_args(argv)
    cfg = CLI.load_config(args.config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    frozen = json.loads((PKG_ROOT / "experiment" / "final_results.json").read_text("utf-8"))
    retained = set(frozen["allowed_feature_columns"])
    dropped_invariant = set(frozen["dropped_invariant_features"])
    led = pd.read_csv(PKG_ROOT / "data" / "feature_availability_ledger.csv")
    strict = set(led.loc[led["strict_scene_tier"] == True, "column"])
    broad = set(led.loc[led["allowed_post_crash_triage"] == True, "column"])

    prep = CLI.prepare_development(args.data, cfg)
    Xd, yd = prep.X_dev, prep.y_dev.to_numpy()
    tokens = prep.schema.string_missing_tokens
    miss = {}
    for c in Xd.columns:
        m = _missing_mask(Xd[c], tokens).to_numpy()
        miss[c] = (float(m.mean()), float(m[yd == 0].mean()) - float(m[yd == 2].mean()))

    rows = []
    for _, r in led.sort_values("column").iterrows():
        col = r["column"]
        status, stage = _status(r, retained, strict, dropped_invariant, cfg["year_col"])
        is_ret = col in retained
        mo, contrast = miss.get(col, (None, None))
        thorough = ("elevated" if (contrast is not None and abs(contrast) >= CONTRAST_FLAG)
                    else ("low" if contrast is not None else "n/a"))
        basis = ("author-judged" if status in
                 ("PLAUSIBLY_EARLY", "TIMING_UNCERTAIN", "OUTCOME_PROXIMAL") else
                 "structural (name/identifier/constant/outcome definition)")
        risk = {"OUTCOME_DERIVED": "n/a (excluded)", "EXCLUDED_IDENTIFIER": "n/a (excluded)",
                "EXCLUDED_CONSTANT": "n/a (excluded)", "TARGET (excluded)": "n/a",
                "SPLIT_ONLY": "n/a"}.get(status)
        if risk is None:
            risk = ("medium (report fields can be revised during investigation)"
                    if status != "PLAUSIBLY_EARLY" or thorough == "elevated"
                    else "low-to-medium (author-judged)")
        rows.append({
            "source_field": col,
            "representation": ("numeric (median-imputed)" if col in frozen["numeric_features"]
                               else ("one-hot categorical (most-frequent-imputed, infrequent "
                                     "bucket)" if is_ret else "—")),
            "retained": "yes (strict tier)" if is_ret else
                        ("broad tier only" if col in broad and str(r["availability"]) not in
                         ("identifier", "constant", "target", "post_outcome") and col not in
                         dropped_invariant and col != cfg["year_col"] else "no"),
            "status": status,
            "stage": stage,
            "basis": basis,
            "dev_missing": ("" if mo is None else f"{mo:.3f}"),
            "reporting_thoroughness_risk": thorough,
            "revision_risk": risk,
            "ledger_rationale": str(r.get("rationale", "")).strip(),
        })

    n_by = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    lines = [
        "# Feature timing evidence — per-field ledger (FEAT-TIME-001; post hoc, v4.1 revision)\n",
        "POST HOC documentation generated from the authoritative contract ledger "
        "(`data/feature_availability_ledger.csv`), the frozen run `final_8af9d5bc23d8`'s retained "
        "columns, and development-rows missingness. **No field in this study has "
        "custodian-verified recording-time evidence, so no field is graded `VERIFIED_EARLY`; "
        "every early/uncertain grade is the author's judgement.** The study uses a conservative "
        "retrospective feature tier and has NOT validated a single operational prediction "
        "timestamp at which all retained fields are simultaneously available and unrevised — "
        "that validation is roadmap Phase 1 work.\n",
        f"Status counts: " + ", ".join(f"`{k}` {v}" for k, v in sorted(n_by.items())) +
        ". (`EXCLUDED_CONSTANT` extends the directive vocabulary for invariant columns.)\n",
        "`reporting_thoroughness_risk` = *elevated* when development-rows missingness differs by "
        f"more than {CONTRAST_FLAG:.0%} between class 0 and class 2 — the field's *presence* is "
        "outcome-correlated, so retained-field missingness patterns can encode reporting "
        "completeness (see `experiment/target_reporting_process_audit.json`, q8/q9).\n",
        "| source field | model representation | retained | timing status | likely recording "
        "stage | evidence basis | dev missing | thoroughness risk | revision risk | ledger rationale |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in
                     ("source_field", "representation", "retained", "status", "stage", "basis",
                      "dev_missing", "reporting_thoroughness_risk", "revision_risk",
                      "ledger_rationale")) + " |")
    lines += [
        "",
        f"*Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by "
        f"`crashsev/feature_timing_evidence.py` (commit {_git_commit()[:8]}); regenerate with "
        "`python -m crashsev.feature_timing_evidence`.*",
    ]
    (out / "FEATURE_TIMING_EVIDENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[feat-timing] wrote {out / 'FEATURE_TIMING_EVIDENCE.md'} ({len(rows)} fields; "
          f"statuses: {n_by})")


if __name__ == "__main__":
    main()
