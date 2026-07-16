"""Generate the complete development-results appendix table (r3 pass, L-06 / Phase 3).

Renders every model of the frozen development report — per-fold scores, seed
variability, trial-mean aggregate, eligibility status, and designation outcome — so the
role-based designation can be audited without private explanation. Derived entirely
from ``experiment/development_report.json`` (pinned by FROZEN.lock) plus the registry
role taxonomy in ``crashsev/models.py``; emits

  * ``experiment/dev_results_table_cells.json`` — structured cells (exact + formatted);
  * ``paper/latex/dev_results_table.tex``       — the Appendix longtable, ``\\input`` by
    the manuscript and never edited by hand.

Selection semantics reproduced here and asserted against the artifact:
  * eligibility = registry kind ``candidate`` (ablations, baselines, and the final-only
    EBM are excluded by prespecified role);
  * criterion  = lowest UNWEIGHTED mean development oMAE over fold x seed trials
    (2 rolling-origin folds; seeds 1, 2, 3 for stochastic models; deterministic models
    run once) — folds are equally weighted, never pooled at prediction level;
  * tie-break  = stable minimum in frozen registry order (no candidate tie occurred;
    the three trivial baselines tied exactly and the comparator fell to ``majority``,
    first in the frozen baseline list);
  * the designated candidate is refit on all mapped development rows at the frozen run
    seed before the governed evaluation.

Deterministic output (no timestamps). Usage:
    python tools/gen_dev_results_table.py [--check]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REM))

DR = REM / "experiment" / "development_report.json"
OUT_CELLS = REM / "experiment" / "dev_results_table_cells.json"
OUT_TEX = REM / "paper" / "latex" / "dev_results_table.tex"

DISPLAY = {
    "ordinal_random_forest_unweighted": "Ordinal RF (unweighted)",
    "random_forest_unweighted": "Random forest (unweighted)",
    "ordinal_random_forest": "Ordinal RF (weighted)",
    "majority": "Majority",
    "ordinal_median": "Ordinal median",
    "prior_probability": "Prior probability",
    "random_forest": "Random forest (weighted)",
    "xgboost": "XGBoost",
    "frank_hall_logistic": "Frank--Hall logit",
    "shallow_tree": "Shallow tree",
    "decision_tree": "Decision tree",
    "multinomial_logistic": "Multinomial logit",
    "proportional_odds": "Proportional-odds logit",
    "ebm": "EBM",
}
ROLE = {
    "candidate": "Candidate",
    "ablation": "Ablation",
    "baseline": "Baseline",
    "exploratory": "Exploratory (final-only)",
}
UNDEF = "---"


def fmt4(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return UNDEF
    return f"{round(float(v), 4):.4f}"


def build() -> dict:
    from crashsev import models as Mo

    dr = json.loads(DR.read_text("utf-8"))
    cv = dr["dev_cv"]
    trials = [t for t in cv["per_trial"] if "ordinal_mae" in t]
    folds = sorted({t["fold"] for t in trials})
    assert len(folds) == 2, f"expected the two rolling-origin folds, got {folds}"

    reg = Mo.build_registry(include_optional=True, seed=int(dr["config"]["seed"]))
    kinds = {name: spec.kind for name, spec in reg.items()}

    # reproduce the frozen selection semantics and assert them against the artifact
    agg = cv["aggregate"]
    cand = [(n, agg[n]["mean_ordinal_mae"]) for n in agg if kinds.get(n) == "candidate"]
    designated = min(cand, key=lambda t: t[1])[0]
    assert designated == "ordinal_random_forest", designated
    assert sorted(v for _, v in cand).count(min(v for _, v in cand)) == 1, \
        "a candidate tie appeared; the tie-break narrative must be revised"
    base_scores = [(b, agg[b]["mean_ordinal_mae"]) for b in dr["config"]["baseline_set"] if b in agg]
    comparator = min(base_scores, key=lambda t: t[1])[0]
    assert comparator == "majority", comparator
    tied_baselines = [b for b, v in base_scores if v == min(v for _, v in base_scores)]
    assert tied_baselines == ["majority", "ordinal_median", "prior_probability"], tied_baselines

    rows = {}
    for name in agg:
        per_fold = {}
        for f in folds:
            vals = [t["ordinal_mae"] for t in trials if t["model"] == name and t["fold"] == f]
            per_fold[f] = {
                "seed_values": vals,
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals)) if len(vals) > 1 else None,
            }
        seeds = sorted({t["seed"] for t in trials if t["model"] == name})
        eligible = kinds.get(name) == "candidate"
        outcome = UNDEF
        if name == designated:
            outcome = "Designated primary"
        elif name == comparator:
            outcome = "Designated comparator"
        elif name in tied_baselines:
            outcome = "Tied comparator score"
        rows[name] = {
            "display": DISPLAY[name],
            "role": ROLE[kinds[name]],
            "per_fold": per_fold,
            "seeds": seeds,
            "aggregate_mean": agg[name]["mean_ordinal_mae"],
            "aggregate_sd": agg[name]["std_ordinal_mae"],
            "n_trials": agg[name]["n_trials"],
            "eligible": eligible,
            "outcome": outcome,
        }
    # EBM: absent from development by design
    if "ebm" not in rows:
        rows["ebm"] = {
            "display": DISPLAY["ebm"], "role": ROLE["exploratory"], "per_fold": {},
            "seeds": [], "aggregate_mean": None, "aggregate_sd": None, "n_trials": 0,
            "eligible": False, "outcome": UNDEF,
        }

    order = sorted((n for n in rows if n != "ebm"),
                   key=lambda n: (rows[n]["aggregate_mean"], n != "majority", n)) + ["ebm"]
    return {
        "description": (
            "Structured cells for the complete development-results appendix table "
            "(tab:dev-results): every model of the frozen development report with per-fold "
            "scores, seed variability, trial-mean aggregate, eligibility, and designation "
            "outcome. Derived from experiment/development_report.json; the selection "
            "semantics (candidate-kind eligibility, unweighted trial mean, stable-min "
            "tie-break, majority comparator from the frozen baseline list) are re-executed "
            "and asserted at generation time."
        ),
        "source_artifact": "experiment/development_report.json",
        "folds": folds,
        "cv_seeds": dr["config"]["cv_seeds"],
        "designated_candidate": designated,
        "comparator": comparator,
        "tied_comparator_baselines": tied_baselines,
        "row_order": order,
        "rows": rows,
    }


def _fold_cell(row: dict, fold: str) -> str:
    pf = row["per_fold"].get(fold)
    if not pf:
        return UNDEF
    if pf["sd"] is None:
        return fmt4(pf["mean"])
    return f"{fmt4(pf['mean'])} $\\pm$ {fmt4(pf['sd'])}"


def _agg_cell(row: dict) -> str:
    if row["aggregate_mean"] is None:
        return UNDEF
    return f"{fmt4(row['aggregate_mean'])} $\\pm$ {fmt4(row['aggregate_sd'])}"


def render_tex(art: dict) -> str:
    f10, f11 = art["folds"]
    header = ("\\textbf{Model} & \\textbf{Role} & \\textbf{Val.~2010} & "
              "\\textbf{Val.~2011} & \\textbf{Seeds} & \\textbf{Aggregate oMAE} & "
              "\\textbf{Eligible} & \\textbf{Outcome}")
    note = ("Rolling-origin development validation of the frozen registry "
            "(\\artifact{experiment/development_report.json}). Val.\\ 2010 / Val.\\ 2011 = "
            "ordinal MAE on the fold validating on that year (train on 2009, and on "
            "2009--2010, respectively); stochastic models show the mean $\\pm$ SD over "
            "development seeds 1--3, deterministic models run once (their seed column shows "
            "the configured run seed, which does not influence their fit). Aggregate = unweighted "
            "mean $\\pm$ SD over all fold$\\times$seed trials --- folds are equally weighted, "
            "not pooled at prediction level. Eligible = registered \\emph{candidate} role; "
            "ablations, baselines, and the final-only EBM are excluded from designation by "
            "prespecified role. The three trivial baselines tie exactly; the comparator "
            "designation falls to the majority predictor, first in the frozen baseline list. "
            "No candidate tie occurred; a tie would resolve to the earlier frozen-registry "
            "entry. RF = random forest; EBM = Explainable Boosting Machine.")
    lines = [
        "% GENERATED FILE — do not edit by hand.",
        "% Source: tools/gen_dev_results_table.py, derived from the frozen",
        "% experiment/development_report.json.",
        "% \\newpage: the 14 double-line rows plus caption and note fill most of a page;",
        "% starting mid-page would strand the last row on a lone continuation page.",
        "\\newpage",
        "\\begingroup\\setlength{\\tabcolsep}{3pt}",
        "\\begin{longtable}{@{}L{0.18\\textwidth}L{0.10\\textwidth}R{0.115\\textwidth}"
        "R{0.115\\textwidth}L{0.07\\textwidth}R{0.135\\textwidth}L{0.075\\textwidth}"
        "L{0.115\\textwidth}@{}}",
        "\\caption{Complete development-phase results, eligibility, and designation record.}"
        "\\label{tab:dev-results}\\\\",
        "\\toprule", header + " \\\\", "\\midrule", "\\endfirsthead",
        "\\multicolumn{8}{l}{\\footnotesize\\itshape Table \\thetable\\ continued from previous page}\\\\",
        "\\toprule", header + " \\\\", "\\midrule", "\\endhead",
        "\\midrule",
        "\\multicolumn{8}{r}{\\footnotesize Continued on next page}\\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\multicolumn{8}{@{}p{0.97\\textwidth}@{}}{\\rule{0pt}{11pt}\\footnotesize\\itshape "
        + note + "}\\\\",
        "\\endlastfoot",
    ]
    for name in art["row_order"]:
        r = art["rows"][name]
        seeds = ",".join(str(s) for s in r["seeds"]) if len(r["seeds"]) > 1 else \
            (str(r["seeds"][0]) if r["seeds"] else UNDEF)
        cells = [r["display"], r["role"], _fold_cell(r, f10), _fold_cell(r, f11),
                 seeds, _agg_cell(r), "Yes" if r["eligible"] else "No", r["outcome"]]
        row = " & ".join(cells) + " \\\\"
        if name == art["designated_candidate"]:
            row = " & ".join(f"\\textbf{{{c}}}" for c in cells) + " \\\\"
        lines.append(row)
    lines.append("\\end{longtable}")
    lines.append("\\endgroup")
    return "\n".join(lines) + "\n"


def main() -> int:
    art = build()
    cells_payload = json.dumps(art, indent=2) + "\n"
    tex_payload = render_tex(art)
    if "--check" in sys.argv[1:]:
        ok = True
        if not OUT_CELLS.exists() or OUT_CELLS.read_text("utf-8") != cells_payload:
            print("[FAIL] dev_results_table_cells.json stale or missing; regenerate")
            ok = False
        if not OUT_TEX.exists() or OUT_TEX.read_text("utf-8") != tex_payload:
            print("[FAIL] dev_results_table.tex stale or missing; regenerate")
            ok = False
        if ok:
            print("[PASS] development-results table is current (regeneration byte-identical)")
        return 0 if ok else 1
    OUT_CELLS.write_text(cells_payload, encoding="utf-8")
    OUT_TEX.write_text(tex_payload, encoding="utf-8")
    print(f"wrote {OUT_CELLS}\nwrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
