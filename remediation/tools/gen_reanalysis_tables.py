"""Generate the structured cell artifact for the prior-iteration re-analysis tables.

Reads the transcribed Iteration III confusion matrices
(``data/confusion_matrices_from_paper.json``), recomputes every metric exactly with
``crashsev.metrics.metrics_from_cm``, and emits
``reanalysis/reanalysis_table_cells.json``: one formatted string per
``(table_id, row_key, column_key)`` cell, plus the full-precision value it rounds from.

The manuscript verifier compares the LaTeX tables ``tab:prior-performance`` and
``tab:prior-severe`` cell-by-cell against this artifact (positional, not substring),
and regenerates the artifact in memory to reject a stale committed copy.

Deterministic: no timestamps; provenance is the source-artifact SHA-256. Values are
formatted at 3 dp with Python's round-half-even, matching the manuscript convention.
Metrics follow the manuscript's F1-CONV-001 reporting convention
(``zero_division="zero"``): precision for a never-predicted class is genuinely
undefined and formats as the em-dash placeholder ``---``; recall and F1 for such a
class are measured zeros under the count-based definition F1 = 2TP/(2TP+FP+FN), and
macro-F1 averages the class F1 values including those zeros.

Usage: python tools/gen_reanalysis_tables.py [--check]
  --check: regenerate and compare against the committed artifact; exit 1 on drift.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REM))
from crashsev import metrics as M  # noqa: E402

SOURCE = REM / "data" / "confusion_matrices_from_paper.json"
OUT = REM / "reanalysis" / "reanalysis_table_cells.json"

MODEL_ROWS = [
    ("majority", "Majority baseline"),
    ("decision_tree", "Decision tree"),
    ("xgboost", "XGBoost"),
    ("mlrf_random_forest", "MLRF / random forest"),
    ("ebm", "EBM"),
]

TABLES = {
    "tab:prior-performance": [
        ("Acc.", "accuracy"),
        ("oMAE", "ordinal_mae"),
        ("QWK", "qwk"),
        ("Macro-F1", "macro_f1"),
        ("Bal. acc.", "balanced_accuracy"),
    ],
    "tab:prior-severe": [
        ("Sev. P", "severe_precision"),
        ("Sev. R", "severe_recall"),
        ("Sev. F1", "severe_f1"),
        ("Pred. 0", "predicted_class0_share"),
        ("2-step", "two_step_error_rate"),
    ],
}

UNDEFINED = "---"


def fmt3(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return UNDEFINED
    return f"{round(float(v), 3):.3f}"


def build() -> dict:
    raw = json.loads(SOURCE.read_text("utf-8"))
    mats = {k: np.asarray(raw["reported"][k]["confusion_matrix"], dtype=float)
            for k, _ in MODEL_ROWS if k != "majority"}
    row_marg = np.stack([m.sum(axis=1) for m in mats.values()])
    if not (row_marg == row_marg[0]).all():
        raise SystemExit("transcribed matrices disagree on row marginals; refusing to derive majority row")
    maj = np.zeros((3, 3), dtype=float)
    maj[:, 0] = row_marg[0]
    mats["majority"] = maj

    n_total = int(row_marg[0].sum())
    tables = {}
    for table_id, cols in TABLES.items():
        rows = {}
        for key, row_label in MODEL_ROWS:
            met = M.metrics_from_cm(mats[key], zero_division="zero")  # F1-CONV-001 reporting convention
            cells = {}
            for col_label, metric in cols:
                v = met[metric]
                exact = None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
                cells[col_label] = {"formatted": fmt3(v), "exact": exact, "metric": metric}
            rows[row_label] = cells
        tables[table_id] = {
            "columns": [c for c, _ in cols],
            "row_order": [label for _, label in MODEL_ROWS],
            "rows": rows,
        }
    return {
        "description": (
            "Structured cell values for the prior-iteration re-analysis tables "
            "(manuscript tab:prior-performance / tab:prior-severe), recomputed exactly from "
            "the transcribed Iteration III confusion matrices under the F1-CONV-001 "
            "reporting convention (zero_division='zero': precision undefined ('---') when a "
            "class is never predicted; recall and F1 measured zeros via the count-based "
            "F1 = 2TP/(2TP+FP+FN); macro-F1 averages class F1 values including zeros). "
            "Formatted at 3 dp, round-half-even. The manuscript verifier checks each LaTeX "
            "cell positionally against this artifact."
        ),
        "f1_zero_division_convention": "zero (F1-CONV-001; see experiment/metric_conventions.json)",
        "source_artifact": "data/confusion_matrices_from_paper.json",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "n": n_total,
        "tables": tables,
    }


def main() -> int:
    art = build()
    payload = json.dumps(art, indent=2, sort_keys=False) + "\n"
    if "--check" in sys.argv[1:]:
        if not OUT.exists():
            print(f"[FAIL] committed artifact missing: {OUT}")
            return 1
        if OUT.read_text("utf-8") != payload:
            print(f"[FAIL] committed artifact is stale: regenerate with tools/gen_reanalysis_tables.py")
            return 1
        print(f"[PASS] {OUT.name} is current (regeneration byte-identical)")
        return 0
    OUT.write_text(payload, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
