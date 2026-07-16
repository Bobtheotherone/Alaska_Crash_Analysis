"""Generate the COMPLETE Appendix A result tables from the frozen artifact (r3 pass, L-03).

Section 6.4 declares the hard-label metric suite: accuracy, balanced accuracy, macro-F1,
QWK, class-specific precision and recall, within-one-level accuracy, and two-step error.
The audit-era Appendix A tables were titled "Complete" while omitting macro-F1,
within-one accuracy, two-step error, and the non-severe per-class precision/recall.
This tool makes the titles true: it derives EVERY declared hard-label metric for all
evaluated models from ``experiment/final_results.json`` (recomputing each value from the
stored confusion matrices and asserting exact agreement with the stored fields wherever
those are defined) and emits

  * ``experiment/full_results_table_cells.json`` — structured per-(table,row,column)
    cells, exact + formatted, under the F1-CONV-001 reporting convention; and
  * ``paper/latex/full_results_tables.tex``    — the three Appendix A longtables,
    ``\\input`` by the manuscript and never edited by hand.

Recomputed F1 uses the count-based form 2TP/(2TP+FP+FN); the frozen artifact stored the
algebraically identical ratio form 2PR/(P+R), which can differ by ~1 ulp in floating
point, so F1 fields are compared with a 1e-12 relative tolerance AND required to agree
at the printed 3-dp precision; every non-F1 field must match the stored value exactly.

F1-CONV-001 (see experiment/metric_conventions.json): precision for a never-predicted
class is undefined and prints as '---'; recall and F1 for such a class are measured
zeros (F1 = 2TP/(2TP+FP+FN)); macro-F1 averages class F1 values including those zeros.
The frozen artifact's NaN *storage* fields are mapped through this convention; no frozen
file is modified.

Deterministic output (no timestamps). Usage:
    python tools/gen_full_results_tables.py [--check]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REM))
from crashsev import metrics as M  # noqa: E402

FR = REM / "experiment" / "final_results.json"
OUT_CELLS = REM / "experiment" / "full_results_table_cells.json"
OUT_TEX = REM / "paper" / "latex" / "full_results_tables.tex"

# display order = ascending frozen 2012 oMAE; majority and ordinal_median share one row
ROWS = [
    ("Random forest (unweighted)", "random_forest_unweighted", "Ablation"),
    ("Ordinal RF (unweighted)", "ordinal_random_forest_unweighted", "Ablation"),
    ("Ordinal RF", "ordinal_random_forest", "Primary"),
    ("Majority / ordinal median", "majority", "Baseline"),
    ("Prior probability", "prior_probability", "Baseline (probabilistic)"),
    ("Random forest (weighted)", "random_forest", "Secondary"),
    ("XGBoost", "xgboost", "Secondary"),
    ("EBM", "ebm", "Exploratory (final-only)"),
    ("Proportional-odds logit", "proportional_odds", "Baseline (nonconverged)"),
    ("Frank--Hall logit", "frank_hall_logistic", "Baseline"),
    ("Multinomial logit", "multinomial_logistic", "Baseline"),
    ("Decision tree", "decision_tree", "Secondary"),
    ("Shallow tree", "shallow_tree", "Baseline"),
]
PRIMARY_DISPLAY = "Ordinal RF"
SUPERSCRIPT = {  # display name -> (column marker cell, row-key marker)
    "Ordinal RF (unweighted)": ("sev_prec", "a"),
    "Proportional-odds logit": ("rowkey", "b"),
}
UNDEF = "---"


def fmt3(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return UNDEF
    return f"{round(float(v), 3):.3f}"


def fmt4(v) -> str:
    return f"{round(float(v), 4):.4f}"


def delta_ci(ci) -> str:
    return (f"{round(ci['difference'], 4):+.4f} "
            f"[{round(ci['ci_low'], 4):+.4f},{round(ci['ci_high'], 4):+.4f}]")


def _exact(v):
    return None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)


def build() -> dict:
    fr = json.loads(FR.read_text("utf-8"))
    res = fr["results"]

    # majority and ordinal_median are one display row: assert they are identical
    for f in ("ordinal_mae", "accuracy", "confusion_matrix"):
        assert res["majority"][f] == res["ordinal_median"][f], \
            "majority and ordinal_median diverged; the merged display row is invalid"

    derived = {}
    for display, key, role in ROWS:
        stored = res[key]
        rep = M.metrics_from_cm(stored["confusion_matrix"], zero_division="zero")
        # the recomputation must agree with every DEFINED stored field: exactly for
        # non-F1 metrics; within 1e-12 relative tolerance AND at printed 3-dp precision
        # for F1 fields (count-based vs ratio form differ by ~1 ulp; see module docstring)
        F1_FIELDS = {"macro_f1", "severe_f1"}
        for f in ("accuracy", "ordinal_mae", "qwk", "balanced_accuracy",
                  "within_one_accuracy", "two_step_error_rate", "macro_f1",
                  "severe_precision", "severe_recall", "severe_f1",
                  "predicted_class0_share"):
            s = stored[f]
            if isinstance(s, float) and math.isnan(s):
                continue  # stored NaN under METRIC-001; reporting value defined via F1-CONV-001
            if f in F1_FIELDS:
                assert math.isclose(rep[f], s, rel_tol=1e-12), \
                    f"{key}.{f}: recomputed {rep[f]!r} not within 1e-12 of stored {s!r}"
                assert fmt3(rep[f]) == fmt3(s), \
                    f"{key}.{f}: printed 3-dp values diverge ({fmt3(rep[f])} vs {fmt3(s)})"
            else:
                assert rep[f] == s, f"{key}.{f}: recomputed {rep[f]!r} != stored {s!r}"
        for c in range(3):
            for f in ("precision", "recall"):
                s = stored["per_class"][str(c)][f]
                if isinstance(s, float) and math.isnan(s):
                    assert math.isnan(rep["per_class"][c][f]), (key, c, f)
                else:
                    assert rep["per_class"][c][f] == s, (key, c, f)
        derived[display] = {"stored": stored, "rep": rep, "role": role, "key": key}

    def delta(display):
        key = derived[display]["key"]
        if key == "majority":
            return "Reference"
        return delta_ci(fr["paired_difference_vs_baseline"][key])

    tables = {
        "tab:full-results-a": {
            "columns": ["oMAE", "$\\Delta$ vs majority (CI)", "Acc.", "Bal. acc.", "QWK"],
            "cells": {d: [fmt4(v["rep"]["ordinal_mae"]), delta(d), fmt3(v["rep"]["accuracy"]),
                          fmt3(v["rep"]["balanced_accuracy"]), fmt3(v["rep"]["qwk"])]
                      for d, v in derived.items()},
        },
        "tab:full-results-detail": {
            "columns": ["Macro-F1", "Within-1", "Two-step",
                        "Prec. 0", "Rec. 0", "Prec. 1", "Rec. 1", "Prec. 2", "Rec. 2"],
            "cells": {d: [fmt3(v["rep"]["macro_f1"]), fmt3(v["rep"]["within_one_accuracy"]),
                          fmt3(v["rep"]["two_step_error_rate"])]
                         + [fmt3(v["rep"]["per_class"][c][f])
                            for c in range(3) for f in ("precision", "recall")]
                      for d, v in derived.items()},
        },
        "tab:full-results-b": {
            "columns": ["Sev. rec.", "Sev. prec.", "Sev. F1", "Pred. 0", "Role"],
            "cells": {d: [fmt3(v["rep"]["severe_recall"]), fmt3(v["rep"]["severe_precision"]),
                          fmt3(v["rep"]["severe_f1"]), fmt3(v["rep"]["predicted_class0_share"]),
                          v["role"]]
                      for d, v in derived.items()},
        },
    }
    return {
        "description": (
            "Structured cell values for the COMPLETE Appendix A result tables "
            "(tab:full-results-a / tab:full-results-detail / tab:full-results-b), derived "
            "from the frozen experiment/final_results.json: every value recomputed from the "
            "stored confusion matrices and asserted equal to the stored fields where those "
            "are defined; NaN storage fields mapped through the F1-CONV-001 reporting "
            "convention (precision undefined '---'; recall/F1 measured zeros; macro-F1 "
            "averages including zeros). oMAE at 4 dp, deltas as +/-0.xxxx [lo,hi], all other "
            "metrics at 3 dp, round-half-even. The manuscript verifier checks the generated "
            "LaTeX byte-for-byte and each cell positionally."
        ),
        "source_artifact": "experiment/final_results.json",
        "run_id": fr["run_id"],
        "f1_zero_division_convention": "zero (F1-CONV-001; see experiment/metric_conventions.json)",
        "row_order": [d for d, _, _ in ROWS],
        "roles": {d: r for d, _, r in ROWS},
        "tables": tables,
    }


def _texify(display: str, cells, table_id: str, bold: bool) -> str:
    parts = []
    rowkey = display
    if display in SUPERSCRIPT and SUPERSCRIPT[display][0] == "rowkey" \
            and table_id == "tab:full-results-b":
        rowkey = display + "\\textsuperscript{" + SUPERSCRIPT[display][1] + "}"
    parts.append(rowkey)
    for j, c in enumerate(cells):
        cell = c
        if c != UNDEF and c not in ("Reference",) and not c[0].isalpha():
            if c.startswith(("+", "-")):
                # delta with interval: $-0.0144$ [$-0.0231,-0.0063$]
                v, br = c.split(" [", 1)
                cell = f"${v}$ [${br[:-1]}$]"
        if (display == "Ordinal RF (unweighted)" and table_id == "tab:full-results-b"
                and j == 1 and c == "1.000"):
            cell = "1.000\\textsuperscript{a}"
        parts.append(cell)
    row = " & ".join(parts) + " \\\\"
    if bold:
        segs = row[:-3].split(" & ")
        row = " & ".join(f"\\textbf{{{s}}}" for s in segs) + " \\\\"
    return row


def _longtable(art: dict, table_id: str, caption: str, colspec: str,
               header_cells: str, note: str, tabcolsep: str | None = None) -> str:
    tbl = art["tables"][table_id]
    n = len(tbl["columns"]) + 1
    lines = ([f"\\begingroup\\setlength{{\\tabcolsep}}{{{tabcolsep}}}"] if tabcolsep else []) + [
        f"\\begin{{longtable}}{{{colspec}}}",
        f"\\caption{{{caption}}}\\label{{{table_id}}}\\\\",
        "\\toprule",
        header_cells + " \\\\",
        "\\midrule",
        "\\endfirsthead",
        f"\\multicolumn{{{n}}}{{l}}{{\\footnotesize\\itshape Table \\thetable\\ continued from previous page}}\\\\",
        "\\toprule",
        header_cells + " \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        f"\\multicolumn{{{n}}}{{r}}{{\\footnotesize Continued on next page}}\\\\",
        "\\endfoot",
        "\\bottomrule",
        f"\\multicolumn{{{n}}}{{@{{}}p{{0.97\\textwidth}}@{{}}}}{{\\rule{{0pt}}{{11pt}}\\footnotesize\\itshape {note}}}\\\\",
        "\\endlastfoot",
    ]
    for d in art["row_order"]:
        lines.append(_texify(d, tbl["cells"][d], table_id, bold=(d == PRIMARY_DISPLAY)))
    lines.append("\\end{longtable}")
    if tabcolsep:
        lines.append("\\endgroup")
    return "\n".join(lines)


HEADERS = {
    "tab:full-results-a":
        "\\textbf{Model} & \\textbf{oMAE} & \\textbf{$\\Delta$ vs majority (\\CI{})} & "
        "\\textbf{Acc.} & \\textbf{Bal. acc.} & \\textbf{QWK}",
    "tab:full-results-detail":
        "\\textbf{Model} & \\textbf{Macro-F1} & \\textbf{Within-1} & \\textbf{Two-step} & "
        "\\textbf{Prec.~0} & \\textbf{Rec.~0} & \\textbf{Prec.~1} & \\textbf{Rec.~1} & "
        "\\textbf{Prec.~2} & \\textbf{Rec.~2}",
    "tab:full-results-b":
        "\\textbf{Model} & \\textbf{Sev. rec.} & \\textbf{Sev. prec.} & \\textbf{Sev. F1} & "
        "\\textbf{Pred. 0} & \\textbf{Role}",
}
COLSPECS = {
    "tab:full-results-a": "@{}L{0.25\\textwidth}R{0.09\\textwidth}L{0.26\\textwidth}"
                          "R{0.08\\textwidth}R{0.09\\textwidth}R{0.08\\textwidth}@{}",
    "tab:full-results-detail": "@{}L{0.185\\textwidth}R{0.078\\textwidth}R{0.072\\textwidth}"
                               "R{0.078\\textwidth}R{0.062\\textwidth}R{0.062\\textwidth}"
                               "R{0.062\\textwidth}R{0.062\\textwidth}R{0.062\\textwidth}"
                               "R{0.062\\textwidth}@{}",
    "tab:full-results-b": "@{}L{0.24\\textwidth}R{0.09\\textwidth}R{0.09\\textwidth}"
                          "R{0.09\\textwidth}R{0.09\\textwidth}L{0.24\\textwidth}@{}",
}
CAPTIONS = {
    "tab:full-results-a": "Complete corrected 2012 performance and agreement metrics.",
    "tab:full-results-detail": "Complete corrected 2012 hard-label detail: macro-F1, "
                               "ordinal tolerance, and per-class precision and recall.",
    "tab:full-results-b": "Complete corrected 2012 severe-class behavior and declared model role.",
}
NOTES = {
    "tab:full-results-a":
        "oMAE = ordinal mean absolute error; $\\Delta$ = paired difference versus the majority "
        "baseline; CI = confidence interval (paired percentile case bootstrap, "
        "\\cref{sec:calib}); Acc. = accuracy; Bal. acc. = balanced accuracy; QWK = quadratic "
        "weighted kappa; RF = random forest; EBM = Explainable Boosting Machine.",
    "tab:full-results-detail":
        "Within-1 = within-one-level accuracy; Two-step = fraction of maximum-distance "
        "(two-step) errors; Prec.\\ $k$ / Rec.\\ $k$ = precision and recall of working mapped "
        "class $k$. All F1 values follow the count-based convention of \\cref{sec:metrics}: an "
        "em dash marks precision that is undefined because the class is never predicted; the "
        "corresponding recall and F1 are measured zeros, and macro-F1 averages the class F1 "
        "values including those zeros.",
    "tab:full-results-b":
        "Sev. rec./prec./F1 = severe-class recall, precision, and F1 (count-based convention, "
        "\\cref{sec:metrics}); Pred. 0 = fraction of crashes predicted as class 0. "
        "\\textsuperscript{a} Precision 1.000 rests on approximately one severe prediction and "
        "is not substantively meaningful. \\textsuperscript{b} The optimizer did not converge "
        "on the restricted-tier design.",
}


def render_tex(art: dict) -> str:
    head = ("% GENERATED FILE — do not edit by hand.\n"
            "% Source: tools/gen_full_results_tables.py, derived from the frozen\n"
            "% experiment/final_results.json under the F1-CONV-001 reporting convention.\n")
    # \newpage between the tables: each fits one page, and starting a longtable mid-page
    # would let it split, which trips a pdfTeX 'infinite glue shrinkage' warning under
    # this preamble.
    body = "\n\n\\newpage\n".join(
        _longtable(art, tid, CAPTIONS[tid], COLSPECS[tid], HEADERS[tid], NOTES[tid],
                   tabcolsep="3pt" if tid == "tab:full-results-detail" else None)
        for tid in ("tab:full-results-a", "tab:full-results-detail", "tab:full-results-b"))
    return head + body + "\n"


def main() -> int:
    art = build()
    cells_payload = json.dumps(art, indent=2) + "\n"
    tex_payload = render_tex(art)
    if "--check" in sys.argv[1:]:
        ok = True
        if not OUT_CELLS.exists() or OUT_CELLS.read_text("utf-8") != cells_payload:
            print("[FAIL] full_results_table_cells.json stale or missing; regenerate")
            ok = False
        if not OUT_TEX.exists() or OUT_TEX.read_text("utf-8") != tex_payload:
            print("[FAIL] full_results_tables.tex stale or missing; regenerate")
            ok = False
        if ok:
            print("[PASS] complete-results tables are current (regeneration byte-identical)")
        return 0 if ok else 1
    OUT_CELLS.write_text(cells_payload, encoding="utf-8")
    OUT_TEX.write_text(tex_payload, encoding="utf-8")
    print(f"wrote {OUT_CELLS}\nwrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
