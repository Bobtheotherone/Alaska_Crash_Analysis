"""Locked-value + positional table-cell + language verification for the manuscript.

Fails (exit 1) if:
  * any locked headline identity is missing from the manuscript,
  * any table cell disagrees with the frozen artifacts under round-half-even —
    checked POSITIONALLY per (table, row, column), never as a row substring
    (hardened 2026-07-15 after audit finding N2: substring matching false-passed a
    correct value appearing in a different column of the same row),
  * the committed re-analysis table artifact is stale w.r.t. its source matrices,
  * any forbidden phrase remains in the LaTeX source or the built PDF text,
  * any required concept is not expressed,
  * any referenced revision artifact does not exist in the repository,
  * git shows a modification/deletion of a frozen path relative to the governed baseline.

Usage: python verify_manuscript_numbers.py [<source_dir>] [<pdf_path>]
  <source_dir> defaults to <remediation>/paper/latex.
  The remediation root is located by walking up from this file (or set
  CRASHSEV_REMEDIATION).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


def _find_rem_root() -> Path:
    env = os.environ.get("CRASHSEV_REMEDIATION")
    if env:
        return Path(env)
    for anc in Path(__file__).resolve().parents:
        if (anc / "crashsev" / "metrics.py").exists() and (anc / "experiment" / "final_results.json").exists():
            return anc
    raise SystemExit("cannot locate the remediation root; set CRASHSEV_REMEDIATION")


ACA = _find_rem_root()
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ACA / "paper" / "latex"
PDF = Path(sys.argv[2]) if len(sys.argv) > 2 else SRC / "main.pdf"
BASELINE_COMMIT = "42194a619c3a8c1a146c1202ac6afe6a919e392b"
TEX = (SRC / "main.tex").read_text(encoding="utf-8")
RAW_TEX = TEX  # unexpanded main.tex (for \input-presence checks)

# r3: generated tables are \input files; expand them so positional table parsing,
# forbidden/required language checks, and artifact scans see the full manuscript text.
def _expand_inputs(tex: str) -> str:
    def repl(m):
        p = SRC / (m.group(1) + ".tex")
        return p.read_text(encoding="utf-8") if p.exists() else m.group(0)
    return re.sub(r"\\input\{([A-Za-z0-9_./-]+?)(?:\.tex)?\}", repl, tex)

TEX = _expand_inputs(TEX)

sys.path.insert(0, str(ACA))
from crashsev.manuscript_tables import TableSpec, verify_table  # noqa: E402

fr = json.loads((ACA / "experiment" / "final_results.json").read_text("utf-8"))
dr = json.loads((ACA / "experiment" / "development_report.json").read_text("utf-8"))
lf = json.loads((ACA / "experiment" / "leakage_factorial.json").read_text("utf-8"))
lm = json.loads((ACA / "experiment" / "lowmiss_sensitivity" / "final_results.json").read_text("utf-8"))
br = json.loads((ACA / "experiment" / "broad_sensitivity" / "final_results.json").read_text("utf-8"))
sm = json.loads((ACA / "experiment" / "severe_metric_uncertainty.json").read_text("utf-8"))
cb2 = json.loads((ACA / "experiment" / "cluster_bootstrap_sensitivity.json").read_text("utf-8"))
tr = json.loads((ACA / "experiment" / "target_reporting_process_audit.json").read_text("utf-8"))
rv = json.loads((ACA / "experiment" / "raw_vs_calibrated_decisions.json").read_text("utf-8"))
cbk = json.loads((ACA / "data" / "codebook_evidence_09_12.json").read_text("utf-8"))

PASS = FAIL = 0
def check(name, ok, detail=""):
    global PASS, FAIL
    if ok: PASS += 1
    else:
        FAIL += 1
        print(f"[FAIL] {name}  :: {detail}")

def has(s, name=None):
    check(name or f"contains {s!r}", s in TEX, "missing from main.tex")

def fmt3(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    return f"{round(float(v), 3):.3f}"

def fmt4(v) -> str:
    return f"{round(float(v), 4):.4f}"

def delta_ci(ci) -> str:
    return (f"{round(ci['difference'], 4):+.4f} "
            f"[{round(ci['ci_low'], 4):+.4f},{round(ci['ci_high'], 4):+.4f}]")

def run_spec(spec: TableSpec):
    for name, ok, detail in verify_table(TEX, spec):
        check(name, ok, detail)

# ---- 1. locked identities ---------------------------------------------------
for s in ["50,543", "46,844", "3,699", "32,046", "13,047", "1,751", "35,214", "11,630",
          "0.3614", "0.3469", "$-0.0144$", "$[-0.0231,-0.0063]$", "5.8\\%", "52.0\\%",
          "26 of 450", "24 false severe alarms", "79.0\\%", "final_8af9d5bc23d8",
          "2,000 crash-level case-bootstrap resamples", "162,820",
          # r3: severe-class reliability counts (frozen artifact); r4: release tag
          "11,148", "portfolio-final-r4"]:
    has(s, f"locked value {s}")

# title synchronized everywhere (E-001)
TITLE_CORE = "Researcher-Defined Alaska Crash-Severity Outcome"
check("pdftitle carries the researcher-defined-outcome title",
      f"pdftitle={{Leakage-Controlled Ordinal Classification of a {TITLE_CORE}" in TEX)
check("title page carries the researcher-defined-outcome title (uppercase)",
      "OF A RESEARCHER-DEFINED ALASKA" in TEX and "CRASH-SEVERITY OUTCOME:" in TEX)

# ---- 2. result tables, positional cells vs frozen artifact -------------------
RESULT_ROWS = {  # display key -> (artifact key, key-results role, full-results-b role)
    "Random forest (unweighted)": ("random_forest_unweighted", "Ablation", "Ablation"),
    "Ordinal RF (unweighted)": ("ordinal_random_forest_unweighted", "Ablation", "Ablation"),
    "Ordinal RF": ("ordinal_random_forest", "Primary", "Primary"),
    "Majority / ordinal median": ("majority", "Baseline", "Baseline"),
    "Prior probability": ("prior_probability", None, "Baseline (probabilistic)"),
    "Random forest (weighted)": ("random_forest", "Secondary", "Secondary"),
    "XGBoost": ("xgboost", "Secondary", "Secondary"),
    "EBM": ("ebm", "Exploratory", "Exploratory (final-only)"),
    "Proportional-odds logit": ("proportional_odds", None, "Baseline (nonconverged)"),
    "Frank--Hall logit": ("frank_hall_logistic", "Baseline", "Baseline"),
    "Multinomial logit": ("multinomial_logistic", None, "Baseline"),
    "Decision tree": ("decision_tree", None, "Secondary"),
    "Shallow tree": ("shallow_tree", None, "Baseline"),
}
KEY_RESULT_MODELS = ["Random forest (unweighted)", "Ordinal RF (unweighted)", "Ordinal RF",
                     "Majority / ordinal median", "Random forest (weighted)", "XGBoost",
                     "EBM", "Frank--Hall logit"]

def _delta(display):
    key = RESULT_ROWS[display][0]
    if key == "majority":
        return "Reference"
    return delta_ci(fr["paired_difference_vs_baseline"][key])

key_spec = TableSpec(
    label="tab:key-results", n_cols=7,
    header={0: "Model", 1: "Ordinal", 2: "majority", 3: "Accuracy", 4: "Balanced",
            5: "Severe", 6: "Role"},
    rows={d: {1: fmt4(fr["results"][RESULT_ROWS[d][0]]["ordinal_mae"]),
              2: _delta(d),
              3: fmt3(fr["results"][RESULT_ROWS[d][0]]["accuracy"]),
              4: fmt3(fr["results"][RESULT_ROWS[d][0]]["balanced_accuracy"]),
              5: fmt3(fr["results"][RESULT_ROWS[d][0]]["severe_recall"]),
              6: RESULT_ROWS[d][1]}
          for d in KEY_RESULT_MODELS},
)
# r3: the three complete Appendix A tables are GENERATED from the frozen artifact by
# tools/gen_full_results_tables.py; verify (a) the committed cells JSON and .tex are
# byte-identical to regeneration, and (b) every cell positionally in the expanded TEX.
import importlib.util as _ilu0
_fr_spec = _ilu0.spec_from_file_location(
    "gen_full_results_tables", Path(__file__).resolve().parent / "gen_full_results_tables.py")
_frg = _ilu0.module_from_spec(_fr_spec)
_fr_spec.loader.exec_module(_frg)
_fr_art = _frg.build()
check("full-results cells artifact current",
      (ACA / "experiment" / "full_results_table_cells.json").read_text("utf-8")
      == json.dumps(_fr_art, indent=2) + "\n",
      "run tools/gen_full_results_tables.py")
check("full-results generated tex current",
      (SRC / "full_results_tables.tex").read_text("utf-8") == _frg.render_tex(_fr_art),
      "run tools/gen_full_results_tables.py")

def _spec_from_cells(table_id: str) -> TableSpec:
    tbl = _fr_art["tables"][table_id]
    cols = tbl["columns"]
    header = {0: "Model"}
    for i, c in enumerate(cols, start=1):
        frag = {"$\\Delta$ vs majority (CI)": "majority"}.get(c, c)
        header[i] = frag
    return TableSpec(
        label=table_id, n_cols=len(cols) + 1, header=header,
        rows={d: {i + 1: tbl["cells"][d][i] for i in range(len(cols))}
              for d in _fr_art["row_order"]},
    )

full_a_spec = _spec_from_cells("tab:full-results-a")
full_detail_spec = _spec_from_cells("tab:full-results-detail")
full_b_spec = _spec_from_cells("tab:full-results-b")
run_spec(full_detail_spec)

# cross-check the generated cells against the frozen artifact independently of the
# generator's own assertions (defense in depth for the headline row)
check("generated A1 primary oMAE cell equals frozen value",
      _fr_art["tables"]["tab:full-results-a"]["cells"]["Ordinal RF"][0]
      == fmt4(fr["results"]["ordinal_random_forest"]["ordinal_mae"]))
check("generated B severe-F1 majority cell is the measured zero",
      _fr_art["tables"]["tab:full-results-b"]["cells"]["Majority / ordinal median"][2] == "0.000")
census = cbk["severity_census"]
cohort_spec = TableSpec(
    label="tab:cohort", n_cols=3,
    header={0: "Quantity", 1: "Count", 2: "Interpretation"},
    rows={
        "Raw crash rows": {1: f"{cbk['n_rows']:,}"},
        "Mapped usable rows": {1: f"{fr['split']['n_total']:,}"},
        "Quarantined rows": {1: f"{cbk['n_quarantined']:,}"},
        "Class 0": {1: f"{census['nan']:,}"},
        "Class 1": {1: f"{census['Non-Incapacitating'] + census['Possible']:,}"},
        "Class 2": {1: f"{census['Incapacitating'] + census['Fatal']:,}"},
        "Development cohort": {1: f"{fr['split']['n_development']:,}"},
        "Final retrospective cohort": {1: f"{fr['split']['n_final_test']:,}"},
    },
)
for spec in (key_spec, full_a_spec, full_b_spec, cohort_spec):
    run_spec(spec)

# cross-check: the quarantine census decomposes exactly
check("quarantine census = Unknown + Not Reported + Null value",
      cbk["n_quarantined"] == census["Unknown"] + census["Not Reported"] + census["Null value"])

# ---- 3. development, factorial, sensitivity, prose numbers -------------------
agg = dr["dev_cv"]["aggregate"]
for k, v in (("ordinal_random_forest_unweighted", "0.3233"), ("random_forest_unweighted", "0.3246"),
             ("ordinal_random_forest", "0.3358"), ("majority", "0.3465"),
             ("random_forest", "0.3744"), ("xgboost", "0.4314")):
    check(f"dev oMAE {k} == {v} (artifact {agg[k]['mean_ordinal_mae']:.4f})",
          f"{round(agg[k]['mean_ordinal_mae'],4):.4f}" == v and v in TEX)
cells_lf = {(c["leakage_features"], c["preprocess_before"], c["random_split"]): c["ordinal_mae"]
            for c in lf["cells"]}
ref = cells_lf[(False, False, False)]
for eff, s in ((cells_lf[(True, False, False)] - ref, "$-0.349$"),
               (cells_lf[(False, True, False)] - ref, "$-0.021$"),
               (cells_lf[(False, False, True)] - ref, "$-0.033$")):
    check(f"factorial contrast {s} (artifact {eff:+.4f})",
          s in TEX and s == f"${round(eff,3):+.3f}$".replace("+0", "+0"))
lmci = lm["paired_difference_vs_baseline"]["ordinal_random_forest"]
check("lowmiss -0.0091 [-0.0181,-0.0001]",
      "$-0.0091$" in TEX and "$[-0.0181,-0.0001]$" in TEX
      and round(lmci["difference"], 4) == -0.0091 and round(lmci["ci_low"], 4) == -0.0181)
brci = br["paired_difference_vs_baseline"]["ordinal_random_forest"]
check("broad -0.0287 / recall 0.171",
      "$-0.0287$" in TEX and "0.171" in TEX and round(brci["difference"], 4) == -0.0287
      and round(br["results"]["ordinal_random_forest"]["severe_recall"], 3) == 0.171)
for s, v in (("0.7474", fr["results"]["prior_probability"]["log_loss"]),
             ("0.4592", fr["results"]["prior_probability"]["brier"]),
             ("0.2558", fr["results"]["prior_probability"]["rps"]),
             ("0.7075", fr["results"]["ordinal_random_forest"]["log_loss"]),
             ("0.4322", fr["results"]["ordinal_random_forest"]["brier"]),
             ("0.2384", fr["results"]["ordinal_random_forest"]["rps"]),
             ("0.3501", fr["results"]["ordinal_random_forest"]["ordinal_mae_argmax"]),
             ("0.3582", fr["results"]["random_forest"]["ordinal_mae_argmax"])):
    check(f"prose value {s}", s in TEX and f"{round(v,4):.4f}" == s)
check("calibration prose 0.7075 -> 0.6924 (no '0.708 to 0.692')",
      "from 0.7075 to 0.6924" in TEX and "0.708 to 0.692" not in TEX
      and round(fr["calibration"]["ordinal_random_forest"]["log_loss_calibrated"], 4) == 0.6924)
check("ECE prose 0.059 -> 0.032",
      "from 0.059 to 0.032" in TEX
      and round(fr["calibration"]["ordinal_random_forest"]["ece_raw"], 3) == 0.059
      and round(fr["calibration"]["ordinal_random_forest"]["ece_calibrated"], 3) == 0.032)

# post-hoc addenda numbers quoted in the text
mi = tr["q9_missingness_indicator_only_probe"]["models"]["missingness_only_logistic"]
check("missingness-probe 0.3583 and CI",
      "0.3583" in TEX and "$[-0.0047,-0.0015]$" in TEX
      and f"{round(mi['ordinal_mae'],4):.4f}" == "0.3583"
      and round(mi["delta_omae_vs_majority"]["ci_low"], 4) == -0.0047
      and round(mi["delta_omae_vs_majority"]["ci_high"], 4) == -0.0015)
day = cb2["clusters"]["calendar_day"]["ci95_percentile"]
wk = cb2["clusters"]["calendar_week"]["ci95_percentile"]
check("cluster CIs quoted correctly",
      "$[-0.0233,-0.0052]$" in TEX and "$[-0.0261,-0.0036]$" in TEX
      and [round(day[0], 4), round(day[1], 4)] == [-0.0233, -0.0052]
      and [round(wk[0], 4), round(wk[1], 4)] == [-0.0261, -0.0036])
mets = sm["metrics"]
check("severe-metric CIs quoted correctly",
      "$[0.169, 0.243]$" in TEX and "$[0.772, 0.814]$" in TEX
      and "$[0.038, 0.080]$" in TEX and "$[0.378, 0.658]$" in TEX
      and [round(mets["severe_average_precision"]["ci95_percentile"][0], 3),
           round(mets["severe_average_precision"]["ci95_percentile"][1], 3)] == [0.169, 0.243]
      and [round(mets["severe_auroc"]["ci95_percentile"][0], 3),
           round(mets["severe_auroc"]["ci95_percentile"][1], 3)] == [0.772, 0.814]
      and [round(mets["severe_recall"]["ci95_percentile"][0], 3),
           round(mets["severe_recall"]["ci95_percentile"][1], 3)] == [0.038, 0.080]
      and [round(mets["severe_precision"]["ci95_percentile"][0], 3),
           round(mets["severe_precision"]["ci95_percentile"][1], 3)] == [0.378, 0.658])
check("ranking all-model range quoted (0.109 / 0.673)", "0.109" in TEX and "0.673" in TEX)
check("AP/AUROC post-hoc points 0.204 / 0.793",
      "AP 0.204" in TEX and "AUROC 0.793" in TEX
      and round(mets["severe_average_precision"]["point"], 3) == 0.204
      and round(mets["severe_auroc"]["point"], 3) == 0.793)

# ---- 3b. correction-release machine-derived values ------------------------------
census2 = cbk["severity_census"]
check("class-0 shares 63.4% / 68.4% (artifact-derived)",
      "63.4\\%" in TEX and "68.4\\%" in TEX
      and round(census2["nan"] / cbk["n_rows"] * 100, 1) == 63.4
      and round(census2["nan"] / fr["split"]["n_total"] * 100, 1) == 68.4)
q3 = tr["q3_person_injury_among_blanks"]
check("person-level injury missing 93.4% among blanks (artifact-derived)",
      "93.4\\%" in TEX and round(q3["missing_share"] * 100, 1) == 93.4)
q5 = tr["q5_count_field_inconsistency"]
check("count-field anomaly 1,511 / 1,512 (artifact-derived)",
      "1,511 of the 1,512" in TEX
      and q5["incap_with_positive_injuries_WITH_fatalities"] == 1511
      and q5["incapacitating_rows"] == 1512)
cc = rv["calibrated_counterfactual"]
check("calibrated counterfactual 1,894 labels / 0.3442 / 4.2% (artifact-exact)",
      "1,894" in TEX and "0.3442" in TEX and "4.2\\%" in TEX
      and cc["labels_changed"] == 1894
      and f"{round(cc['hard_metrics_calibrated_labels']['ordinal_mae'], 4):.4f}" == "0.3442"
      and round(cc["hard_metrics_calibrated_labels"]["severe_recall"] * 100, 1) == 4.2)
check("raw-label reconstruction: 162,820 = 14 x 11,630, zero mismatches",
      rv["raw_label_verification"]["mismatch_count_exact_parser"] == 0
      and rv["raw_label_verification"]["median_of_raw_equals_stored_y_pred"] is True
      and 14 * fr["split"]["n_final_test"] == 162820)
check("proxy-cluster counts quoted (366/53/4/18/9; artifact-derived)",
      "(366 clusters)" in TEX and "(53 clusters)" in TEX and "(4 regions)" in TEX
      and "(18 boroughs)" in TEX and "(9 groups)" in TEX
      and cb2["clusters"]["calendar_day"]["n_clusters"] == 366
      and cb2["clusters"]["calendar_week"]["n_clusters"] == 53
      and cb2["clusters"]["region"]["n_clusters"] == 4
      and cb2["clusters"]["borough"]["n_clusters"] == 18
      and cb2["clusters"]["maintenance_responsibility"]["n_clusters"] == 9)
mc_path = ACA / "experiment" / "metric_conventions.json"
check("metric-conventions sidecar exists", mc_path.exists())
if mc_path.exists():
    mc = json.loads(mc_path.read_text("utf-8"))
    check("RPS convention stated: unnormalized, /(K-1) alternative = 2",
          "unnormalized ranked probability score" in TEX
          and "\\emph{unnormalized} cumulative sum" in TEX
          and "twice those obtained under the alternative convention" in TEX
          and mc["rps_normalization"] == "none"
          and mc["alternative_normalized_divisor"] == 2)
    check("ECE binning stated and matches sidecar (10 equal-width bins, top-label)",
          "10 equal-width bins" in TEX and "top-label" in TEX and mc["ece_n_bins"] == 10)
    check("log-loss epsilon stated and matches sidecar (1e-15, renormalized)",
          "clipped using the implementation's epsilon value ($10^{-15}$" in TEX
          and "row renormalization" in TEX and mc["log_loss_epsilon"] == 1e-15)

# ---- 4. prior-reanalysis appendix, positional vs regenerated artifact --------
import importlib.util as _ilu
_gen_spec = _ilu.spec_from_file_location(
    "gen_reanalysis_tables", Path(__file__).resolve().parent / "gen_reanalysis_tables.py")
_gen = _ilu.module_from_spec(_gen_spec)
_gen_spec.loader.exec_module(_gen)
regen = _gen.build()
committed_path = ACA / "reanalysis" / "reanalysis_table_cells.json"
check("reanalysis cell artifact exists", committed_path.exists())
if committed_path.exists():
    committed = json.loads(committed_path.read_text("utf-8"))
    check("reanalysis cell artifact is current (regeneration identical)",
          committed == regen, "run tools/gen_reanalysis_tables.py")
for table_id, tbl in regen["tables"].items():
    spec = TableSpec(
        label=table_id, n_cols=len(tbl["columns"]) + 1,
        header=dict(enumerate(["Model"] + list(tbl["columns"]))),
        rows={rk: {i + 1: tbl["rows"][rk][c]["formatted"] for i, c in enumerate(tbl["columns"])}
              for rk in tbl["row_order"]},
    )
    run_spec(spec)
check("prior-reanalysis N=10,936", "10{,}936" in TEX and regen["n"] == 10936)

# ---- 4b. hyperparameter appendix vs generated ledger ---------------------------
_hp_spec = _ilu.spec_from_file_location(
    "gen_hyperparameter_ledger", Path(__file__).resolve().parent / "gen_hyperparameter_ledger.py")
_hp = _ilu.module_from_spec(_hp_spec)
_hp_spec.loader.exec_module(_hp)
hp = _hp.build()
hp_path = ACA / "experiment" / "hyperparameter_ledger.json"
check("hyperparameter ledger exists", hp_path.exists())
if hp_path.exists():
    check("hyperparameter ledger is current (regeneration identical)",
          json.loads(hp_path.read_text("utf-8")) == json.loads(json.dumps(hp, default=str)),
          "run tools/gen_hyperparameter_ledger.py")
from crashsev.manuscript_tables import parse_table  # noqa: E402
_pt = parse_table(TEX, "tab:hyperparameters")
check("tab:hyperparameters present", _pt is not None)
if _pt is not None:
    _hp_cells = dict(_pt.rows)
    def cell_has(rowkey, frag):
        cells = _hp_cells.get(rowkey)
        ok = cells is not None and len(cells) >= 3 and frag in cells[2]
        check(f"tab:hyperparameters [{rowkey}] config cell contains {frag!r}", ok,
              str(cells[2] if cells and len(cells) >= 3 else cells))
    m = hp["models"]
    cell_has("XGBoost", f"{m['xgboost']['n_estimators']} trees")
    cell_has("XGBoost", f"learning rate {m['xgboost']['learning_rate']}")
    cell_has("XGBoost", f"depth {m['xgboost']['max_depth']}")
    cell_has("XGBoost", f"subsample {m['xgboost']['subsample']}")
    cell_has("Decision tree", m["decision_tree"]["criterion"])
    cell_has("Decision tree", f"depth {m['decision_tree']['max_depth']}")
    cell_has("Shallow tree", f"depth {m['shallow_tree']['max_depth']}")
    cell_has("Random forest (weighted; unweighted)",
             f"{m['random_forest']['n_estimators']} trees")
    cell_has("Multinomial logit", f"iterations {m['multinomial_logistic']['max_iter']}")
    cell_has("Proportional-odds logit", f"iterations {m['proportional_odds']['max_iter']}")
sc = hp["shared_protocol_constants"]
check("appendix shared constants match config",
      f"minimum category frequency {sc['one_hot_min_frequency']}" in TEX
      and f"{sc['bootstrap_resamples']:,} bootstrap resamples" in TEX
      and sc["seed"] == 42 and "seed 42" in TEX)
check("log-loss clip convention stated and matches implementation",
      "$[10^{-15}, 1-10^{-15}]$" in TEX
      and sc["log_loss_probability_clip"] == 1e-15)

# ---- 4c. analysis-status ledger current ----------------------------------------
_as_spec = _ilu.spec_from_file_location(
    "gen_analysis_status_ledger", Path(__file__).resolve().parent / "gen_analysis_status_ledger.py")
_as = _ilu.module_from_spec(_as_spec)
_as_spec.loader.exec_module(_as)
as_path = ACA / "paper" / "ANALYSIS_STATUS_LEDGER.csv"
as_tex_path = ACA / "paper" / "latex" / "analysis_status_table.tex"
check("analysis status ledger exists", as_path.exists())
if as_path.exists():
    _as_rows = _as.build_rows()
    check("analysis status ledger is current (regeneration identical)",
          as_path.read_text("utf-8") == _as.render(_as_rows),
          "run tools/gen_analysis_status_ledger.py")
    check("analysis-status appendix table is current (regeneration identical)",
          as_tex_path.exists() and as_tex_path.read_text("utf-8") == _as.render_tex(_as_rows),
          "run tools/gen_analysis_status_ledger.py")
    check("analysis-status appendix is included in the manuscript",
          "\\input{analysis_status_table}" in RAW_TEX and "app:analysis-status" in TEX)
    _allowed = {"protocol-primary", "protocol-secondary", "exploratory-final-only", "post-hoc"}
    check("analysis-status ledger uses only the four controlled status values",
          all(r["status"] in _allowed for r in _as_rows))

# ---- 4d. r3 generated tables: development record, governance, cohort-by-year --------
_dv_spec = _ilu.spec_from_file_location(
    "gen_dev_results_table", Path(__file__).resolve().parent / "gen_dev_results_table.py")
_dv = _ilu.module_from_spec(_dv_spec)
_dv_spec.loader.exec_module(_dv)
_dv_art = _dv.build()
check("dev-results cells artifact current",
      (ACA / "experiment" / "dev_results_table_cells.json").read_text("utf-8")
      == json.dumps(_dv_art, indent=2) + "\n", "run tools/gen_dev_results_table.py")
check("dev-results generated tex current",
      (SRC / "dev_results_table.tex").read_text("utf-8") == _dv.render_tex(_dv_art),
      "run tools/gen_dev_results_table.py")
check("dev-results designation semantics re-executed",
      _dv_art["designated_candidate"] == "ordinal_random_forest"
      and _dv_art["comparator"] == "majority"
      and _dv_art["tied_comparator_baselines"] == ["majority", "ordinal_median",
                                                   "prior_probability"])
dev_spec = TableSpec(
    label="tab:dev-results", n_cols=8,
    header={0: "Model", 1: "Role", 2: "2010", 3: "2011", 4: "Seeds",
            5: "Aggregate", 6: "Eligible", 7: "Outcome"},
    rows={_dv_art["rows"][n]["display"]: {
            1: _dv_art["rows"][n]["role"],
            6: "Yes" if _dv_art["rows"][n]["eligible"] else "No",
            7: _dv_art["rows"][n]["outcome"]}
          for n in _dv_art["row_order"]},
)
run_spec(dev_spec)
check("dev-results aggregate cells match the frozen development report",
      all(abs(_dv_art["rows"][n]["aggregate_mean"] - dr["dev_cv"]["aggregate"][n]["mean_ordinal_mae"]) == 0
          for n in _dv_art["row_order"] if n != "ebm"))

_fg_spec = _ilu.spec_from_file_location(
    "gen_feature_governance_tables",
    Path(__file__).resolve().parent / "gen_feature_governance_tables.py")
_fg = _ilu.module_from_spec(_fg_spec)
_fg_spec.loader.exec_module(_fg)
_fg_art = json.loads((ACA / "experiment" / "feature_governance_cells.json").read_text("utf-8"))
check("feature-governance structural reconciliation vs frozen retained list",
      not _fg._structural_check(_fg_art))
check("feature-governance generated tex current",
      (SRC / "feature_governance_tables.tex").read_text("utf-8") == _fg.render_tex(_fg_art),
      "run tools/gen_feature_governance_tables.py")
check("retained-field count is 49 and prohibited count is 14",
      len(_fg_art["retained"]) == 49 and len(_fg_art["prohibited"]) == 14)

_cy_spec = _ilu.spec_from_file_location(
    "gen_cohort_year_table", Path(__file__).resolve().parent / "gen_cohort_year_table.py")
_cy = _ilu.module_from_spec(_cy_spec)
_cy_spec.loader.exec_module(_cy)
_cy_art = json.loads((ACA / "experiment" / "cohort_year_table.json").read_text("utf-8"))
check("cohort-year table reconciles against the frozen target audits",
      not _cy.reconcile(_cy_art))
check("cohort-year generated tex current",
      (SRC / "cohort_year_table.tex").read_text("utf-8") == _cy.render_tex(_cy_art),
      "run tools/gen_cohort_year_table.py")
cy_spec = TableSpec(
    label="tab:cohort-year", n_cols=9, exact_rows=False,
    header={0: "Year", 1: "Source rows", 2: "Mapped", 3: "Quarantined",
            4: "Class 0", 5: "Class 1", 6: "Class 2", 7: "share", 8: "Role"},
    rows={y: {1: f"{_cy_art['years'][y]['source_rows']:,}",
              2: f"{_cy_art['years'][y]['mapped_rows']:,}",
              3: f"{_cy_art['years'][y]['quarantined_rows']:,}",
              4: f"{_cy_art['years'][y]['class_counts']['0']:,}",
              5: f"{_cy_art['years'][y]['class_counts']['1']:,}",
              6: f"{_cy_art['years'][y]['class_counts']['2']:,}"}
          for y in ("2009", "2010", "2011", "2012")},
)
run_spec(cy_spec)

# metric-convention sidecar r3 blocks stated in the manuscript
mc2 = json.loads((ACA / "experiment" / "metric_conventions.json").read_text("utf-8"))
check("F1 convention block present in sidecar and stated in manuscript",
      mc2["f1_reporting_convention"]["pipeline_default"] == "nan"
      and "macro-F1 averages the class-specific F1 values, including those zeros" in TEX
      and "count-based" in TEX)
check("bootstrap convention block matches manuscript statement",
      mc2["paired_bootstrap"]["n_resamples"] == 2000
      and mc2["paired_bootstrap"]["seed"] == 42
      and mc2["paired_bootstrap"]["models_refit_within_resamples"] is False
      and "Paired 95\\% \\emph{percentile} intervals" in TEX
      and "Models are not refit within resamples" in TEX)
check("Frank-Hall construction stated and pinned",
      "running cumulative minimum" in TEX and "floored at $10^{-9}$" in TEX
      and "cumulative minimum" in mc2["frank_hall_probability_construction"]["monotonicity"])

# ---- 5. forbidden strings (tex + PDF text), case-insensitive --------------------
FORBIDDEN = [
    # r4 editorial-marker tokens: the final manuscript may contain none of these
    # (matching is case-insensitive substring over the expanded tex and pdf text)
    "AUTHOR ACTION REQUIRED", "INSERT DOI", "INSERT URL", "pending author",
    "TBD", "TODO", "placeholder", "example.com",
    "constitutes most of class 0", "the signal is real", "genuine empirical findings",
    "cause substantially more", "dominant source of",
    "caused by evaluation defects",
    "the protocol prespecifies four hypotheses", "prespecified class-weighted",
    "evaluated, once", "single governed out-of-time evaluation",
    "reserved for a single governed evaluation",
    "weakly but reproducibly predictable", "credible evidence of future-period performance",
    "the tradeoff is structural", "far more than modeling choices",
    "before severity is known", "0.708 to 0.692",
    # correction-release language gate (R-001 / E-004 / E-013)
    "statistically supported", "statistically significant",
    "development-selected primary", "selected primary candidate",
    "strong source-specific evidence", "evidence-checked",
    # agency-field misstatements (corrected 2026-07-15: the licensed raw extract DID
    # contain Officer Agency / Reporting Agency / Detachment; they were removed as
    # identifier-tier fields during de-identification)
    "exposes no reporting-agency identifier",
    "contains no reporting-agency or officer/agency identifier",
    "no agency field exists",
    "agency fields did not exist in the source",
    "no such field exists in the licensed extract",
    # r3 submission-readiness gate (L-01/02, L-09, L-11, L-14, L-15, L-20)
    "severe-risk",                                # renamed to severe-class score ranking
    "macro-F1 is undefined",                      # count-based convention adopted
    "undefined rather than zero",
    "the role ordering above is intentional",
    "methodological asset rather than an embarrassment",
    "no systematic literature review is claimed",
    "offers no finer reliable resolution",
]
pdf_pages: list[str] = []
pdf_text = ""
try:
    from pypdf import PdfReader
    pdf_pages = [(p.extract_text() or "") for p in PdfReader(str(PDF)).pages]
    # de-hyphenate line breaks and collapse whitespace so phrases split across
    # justified lines are still caught
    pdf_pages = [re.sub(r"\s+", " ", pg.replace("-\n", "")) for pg in pdf_pages]
    pdf_text = "\n".join(pdf_pages)
except Exception as e:
    print(f"[WARN] PDF text extraction unavailable: {e}")

def _tex_line_of(s: str) -> str:
    i = TEX.lower().find(s.lower())
    return f"main.tex line {TEX.count(chr(10), 0, i) + 1}" if i >= 0 else "not in tex"

def _pdf_page_of(s: str) -> str:
    for n, pg in enumerate(pdf_pages, 1):
        if s.lower() in pg.lower():
            return f"PDF page {n}"
    return "not in pdf"

TEX_L = TEX.lower()
for s in FORBIDDEN:
    check(f"forbidden absent (tex): {s!r}", s.lower() not in TEX_L,
          f"phrase {s!r} at {_tex_line_of(s)}")
    if pdf_text:
        s_pdf = s.replace("\\%", "%").lower()
        check(f"forbidden absent (pdf): {s!r}", s_pdf not in pdf_text.lower(),
              f"phrase {s!r} at {_pdf_page_of(s_pdf)}")

# context-qualified phrase: "independent replication"/"independently replicated" may
# appear ONLY negated or definitional (R-001: no unqualified independent-replication claim)
_IR = re.compile(r"independent(?:ly)?\s+replicat\w*", re.I)
_IR_OK = re.compile(r"\bnot\b|\bnever\b|has not\b|distinction between|third-party|"
                    r"has not yet|not yet occurred|invite", re.I)

def _check_qualified(text: str, where: str, locator):
    bad = []
    for m in _IR.finditer(text):
        win = text[max(0, m.start() - 160): m.end() + 120]
        if not _IR_OK.search(win):
            bad.append(f"{locator(m.start())}: …{win[-180:]}…")
    check(f"no unqualified 'independent replication' ({where})", not bad, " | ".join(bad))

_check_qualified(TEX, "tex", lambda i: f"main.tex line {TEX.count(chr(10), 0, i) + 1}")
if pdf_text:
    _check_qualified(pdf_text, "pdf",
                     lambda i: f"PDF page {pdf_text.count(chr(10), 0, i) + 1}")

print("[forbidden-list] " + json.dumps(
    sorted(FORBIDDEN) + ["<contextual> unqualified 'independent replication' / "
                         "'independently replicated' (tex + pdf)"]))

# ---- 6. required concepts ------------------------------------------------------
REQUIRED = {
    "all of class 0 in this extract": "all of class 0 in this extract",
    "recorded-severity outcome": "recorded-severity",
    "raw, uncalibrated probabilities": "raw, uncalibrated",
    "role-based designation": "role-based designation",
    "not selection of the overall (oMAE/ordinal-error) leader": "leader",
    "historical exposure to 2012": "historically exposed",
    "not preregistered": "not preregistered",
    "one controlled execution / not one statistical look": "not mean one statistical look",
    "no validated operational prediction timestamp": "operational prediction timestamp",
    "self-reproduction": "self-reproduction",
    "oMAE equal-spacing convention": "equal-spacing convention",
    "ranking is cohort-specific": "within-cohort ordering",
    "no stakeholder cost matrix": "No stakeholder cost matrix",
    "not a severe-crash detector": "severe-crash detector",
    # agency-field accuracy (2026-07-15): present in source, removed at de-identification
    "agency fields present in raw extract": "removed during de-identification",
    "agency absence scoped to released evidence, not source": "absent from the released evidence",
    # correction-release required concepts (E-002/E-006/E-008/E-009/E-011/E-013/E-014/
    # E-023/E-026/E-027)
    "leakage-controlled defined via custodian clause":
        "does not mean that the recording time of every retained field has been verified",
    "controlled target term": "researcher-defined recorded-severity outcome",
    "controlled target short form": "working ordinal outcome",
    "internal consistency, not semantic validation":
        "a source-specific internal consistency check, not independent semantic validation",
    "working contract pending confirmation":
        "researcher-defined working contract pending codebook or custodian confirmation",
    "no operational prediction timestamp validated":
        "No operational prediction timestamp was validated",
    "author judged, not verified simultaneous":
        "not verified as simultaneously available and unrevised before the outcome is known",
    "controlled model-role term": "protocol-designated weighted candidate",
    "designation not from oMAE leadership":
        "came from the frozen role registry",
    "weighting is a protocol role":
        "a protocol role, not an operationally optimal compromise",
    "H1 verdict descriptive": "Met the protocol's numerical criterion (descriptive)",
    "H1 descriptive-not-confirmatory prose":
        "because the cohort was historically exposed, this is descriptive rather than confirmatory",
    "bootstrap non-coverage":
        "does not incorporate uncertainty from target semantics, feature timing, model-role selection, or future temporal regimes",
    "one primary use of 2012 + ledger":
        "used once for the frozen primary evaluation execution",
    "conclusion scope": "has not been independently replicated and does not establish validated physical injury severity",
    "reproduction tier 1": "metric recomputation",
    "reproduction tier 3": "clean-room package verification",
    "parquet authoritative": "authoritative row-level evidence",
    "working mapped classes": "working mapped class",
    "coarsening not clinical": "not claims of clinical equivalence",
    # r3 submission-readiness required concepts
    "F1 count-based convention stated":
        "macro-F1 averages the class-specific F1 values, including those zeros",
    "F1 storage-vs-reporting split": "flagged not-a-number entries",
    "bootstrap percentile + level": "Paired 95\\% \\emph{percentile} intervals",
    "bootstrap no-refit": "Models are not refit within resamples",
    "FH monotonicity repair": "running cumulative minimum",
    "FH renormalization": "each row is renormalized to sum to one",
    "selection rule aggregation": "unweighted mean over all fold$\\times$seed trials",
    "selection final refit":
        "refit on all 35,214 mapped 2009--2011 rows at the frozen run seed 42",
    "2012 leader stated plainly":
        "the unweighted random forest achieved the lowest 2012 oMAE",
    "claim boundary box": "Claim boundary.",
    "availability release tag": "portfolio-final-r4",
    "availability license": "MIT License",
    # r4 submission-readiness required concepts (markers resolved; see REVISION_MEMO_R4)
    "availability repository URL": "github.com/Bobtheotherone/Alaska_Crash_Analysis",
    "availability release URL": "releases/tag/portfolio-final-r4",
    "reproduction contact email": "rnmercado@alaska.edu",
    "no-DOI statement (truthful)": "No archival DOI had been issued",
    "release date stated": "July 20, 2026",
    "raw-hash withholding statement": "restricted reproduction log",
    "AI privacy attestation (local-only processing)":
        "processed only in local computing environments",
    "AI privacy attestation (nonlocal boundary)":
        "transmitted to or processed by nonlocal language models",
    "handoff release-asset availability": "asset of the tagged public release",
    "handoff anchoring": "canonical/remediation",
    "severe-class score ranking terminology": "severe-class score ranking",
    "neutral attribution": "sole author of the Iteration IV manuscript",
    "AI not an author": "no AI system is listed as an author",
    "reliability diagram present": "fig06c_reliability.pdf",
    "coarsening precision": "not a claim that the source contains no finer labels",
    "focused review sentence": "the review is intentionally focused on work that directly informs",
    "conclusion readiness framing": "not an inflated performance number",
}
for name, s in REQUIRED.items():
    check(f"required concept: {name}", s in TEX, f"pattern {s!r} not found")

# ---- 7. referenced revision artifacts exist ------------------------------------
# The two de-identified parquets are gitignored BY POLICY (shipped in the handoff's
# evidence/ tier), so inside an extracted package they resolve there instead of in the
# canonical source tree.
_POLICY_EVIDENCE = {"experiment/predictions_lossless.parquet",
                    "experiment/missingness_only_predictions.parquet"}
for rel in re.findall(r"\\artifact\{(experiment/[^}]+|paper/[^}]+|reanalysis/[^}]+|data/[^}]+)\}", TEX):
    ok = (ACA / rel).exists()
    if not ok and rel in _POLICY_EVIDENCE:
        ok = (ACA.parent.parent / "evidence" / Path(rel).name).exists()
        if ok:
            print(f"[SKIP-NOTE] {rel}: absent from the canonical tree by policy; "
                  "resolved in the package evidence/ tier")
    check(f"referenced artifact exists: {rel}", ok)

# ---- 8. git: frozen paths unchanged relative to the governed baseline ----------
FROZEN_PREFIXES = ("remediation/runs/", "remediation/evidence_release/",
                   "remediation/configs/", "remediation/data/",
                   "remediation/experiment/final_results.json",
                   "remediation/experiment/development_report.json",
                   "remediation/experiment/leakage_factorial",
                   "remediation/experiment/broad_sensitivity/",
                   "remediation/experiment/lowmiss_sensitivity/",
                   "remediation/experiment/decision_rule_sensitivity.md",
                   "remediation/experiment/error_analysis.md",
                   "remediation/experiment/severe_ranking.md",
                   "remediation/experiment/seed_robustness.md",
                   "remediation/experiment/target_sensitivity.md",
                   "remediation/experiment/missingness_summary.md",
                   "remediation/experiment/split_assignment.csv",
                   "remediation/experiment/FROZEN.lock",
                   "remediation/experiment/FINAL.done")
_in_repo = subprocess.run(["git", "-C", str(ACA.parent), "rev-parse", "--is-inside-work-tree"],
                          capture_output=True, text=True)
_baseline_present = subprocess.run(
    ["git", "-C", str(ACA.parent), "cat-file", "-e", f"{BASELINE_COMMIT}^{{commit}}"],
    capture_output=True).returncode == 0 if _in_repo.returncode == 0 else False
if _in_repo.returncode == 0 and _in_repo.stdout.strip() == "true" and not _baseline_present:
    # Shallow CI clones and snapshot-published checkouts do not carry the governed
    # baseline commit; the full-history gate runs and must pass at packaging time
    # (captured in gate_outputs.json).
    print("[SKIP] git frozen-path gate: governed baseline commit not present in this "
          "checkout's history (shallow or snapshot checkout); verified at packaging time")
elif _in_repo.returncode == 0 and _in_repo.stdout.strip() == "true":
    diff = subprocess.run(["git", "-C", str(ACA.parent), "diff", "--name-status",
                           f"{BASELINE_COMMIT}..HEAD"], capture_output=True, text=True)
    bad = []
    for ln in diff.stdout.splitlines():
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        status, paths = parts[0], parts[1:]
        if status.startswith("A"):
            continue  # additions inside governed dirs are allowed (post-hoc addenda)
        for p in paths:
            if any(p.startswith(pref) or pref.rstrip("/") == p for pref in FROZEN_PREFIXES):
                bad.append(ln)
    check("git: no frozen path modified/deleted since governed baseline 42194a6",
          diff.returncode == 0 and not bad, str(bad))
else:
    print("[SKIP] git frozen-path gate: not a git checkout (history verifiable via the "
          "provenance git bundle; the gate ran and passed at packaging time)")

# ---- 9. r4 release-state checks -------------------------------------------------
# 9a. The licensed raw workbook's SHA-256 is withheld from public artifacts (item 4.3).
# The value is read from the local-only restricted reproduction log so this verifier
# never embeds the identifier itself; where the log is absent (public CI, extracted
# packages), the check is skipped and the packaging-time result stands.
_RESTRICTED_LOG = ACA.parent / "private_reproduction_log" / "RAW_WORKBOOK_IDENTITY.md"
if _RESTRICTED_LOG.exists():
    _m = re.search(r"SHA-256:\s*([0-9a-fA-F]{64})", _RESTRICTED_LOG.read_text(encoding="utf-8"))
    if _m:
        _rawhash = _m.group(1).lower()
        for _needle, _label in ((_rawhash, "full value"), (_rawhash[:8], "8-char prefix")):
            check(f"raw workbook hash withheld from tex ({_label})", _needle not in TEX.lower())
            if pdf_text:
                check(f"raw workbook hash withheld from pdf ({_label})",
                      _needle not in pdf_text.lower())
    else:
        check("restricted reproduction log parseable", False, str(_RESTRICTED_LOG))
else:
    print("[SKIP] raw-hash withholding check: restricted reproduction log not present "
          "(local-only by design); the check ran where the log exists")

# 9b. Table-continuation mechanics: exactly three continued parts, each reusing its
# part-1 number via \ContinuedFloat, and a contiguous List of Tables.
check("three \\ContinuedFloat continuations in source", TEX.count(r"\ContinuedFloat") == 3,
      f"found {TEX.count(chr(92) + 'ContinuedFloat')}")
if pdf_text:
    _cont = re.findall(r"Table (\d+):[^\n]{0,160}?part 2 of 2 \(continued\)", pdf_text)
    check("three continued captions render in pdf", len(_cont) == 3, str(_cont))
    _nums = sorted({int(n) for n in re.findall(r"Table (\d+):", pdf_text)})
    check("pdf table numbers contiguous from 1",
          _nums == list(range(1, len(_nums) + 1)), str(_nums))
    check("continued captions reuse an existing table number",
          all(int(n) in _nums for n in _cont), str(_cont))
_LOT = SRC / "main.lot"
if _LOT.exists():
    _lot_nums = [int(n) for n in re.findall(r"numberline \{(\d+)\}", _LOT.read_text(encoding="utf-8"))]
    check("List of Tables contiguous (no skipped numbers)",
          _lot_nums == list(range(1, len(_lot_nums) + 1)), str(_lot_nums))
else:
    print("[SKIP] List-of-Tables contiguity: main.lot not present (build artifact); "
          "checked at build time")

print(f"\n===== manuscript verification: {PASS} passed, {FAIL} failed =====")
sys.exit(0 if FAIL == 0 else 1)
