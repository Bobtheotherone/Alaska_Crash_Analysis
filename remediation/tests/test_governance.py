"""Governance guard tests (CAL-SELECT-001, DECISION-LOSS-001, TEST-GOV-001).

These reproduce the historical governance failure modes so they cannot silently return:
  * a calibration candidate chosen using final-test outcomes, and
  * a decision rule inconsistent with the declared ordinal loss.
"""
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crashsev import cli as CLI


def _spec(kind):
    return SimpleNamespace(kind=kind)


# ---- CAL-SELECT-001: calibration candidate is chosen on development evidence only ----

def test_select_best_candidate_uses_dev_cv():
    cv = {"aggregate": {
        "ordinal_random_forest": {"mean_ordinal_mae": 0.31},
        "random_forest": {"mean_ordinal_mae": 0.34},
        "majority": {"mean_ordinal_mae": 0.35},          # a baseline: must be ignored
    }}
    reg = {
        "ordinal_random_forest": _spec("candidate"),
        "random_forest": _spec("candidate"),
        "majority": _spec("baseline"),
    }
    assert CLI.select_best_candidate(cv, reg) == "ordinal_random_forest"


def test_select_best_candidate_ignores_baselines_and_returns_none_when_no_candidate():
    cv = {"aggregate": {"majority": {"mean_ordinal_mae": 0.1}}}   # only a baseline present
    reg = {"majority": _spec("baseline")}
    assert CLI.select_best_candidate(cv, reg) is None


def test_calibration_selection_cannot_see_final_results():
    # Structural guard: the selection function takes ONLY (cv, reg). It is impossible for final-test
    # results or labels to drive which model is calibrated.
    params = set(inspect.signature(CLI.select_best_candidate).parameters)
    assert params == {"cv", "reg"}
    assert not (params & {"results", "final", "y_test", "test_results"})


def test_dev_only_calibrate_takes_no_test_labels():
    # The calibrator fit may take X_test (to transform) but must NOT receive final-test labels.
    params = list(inspect.signature(CLI.dev_only_calibrate).parameters)
    assert "y_dev" in params
    assert not any(p in params for p in ("y_test", "y_final", "test_labels"))


# ---- DECISION-LOSS-001: a decision rule consistent with the ordinal (absolute) loss ----

def _posterior_median(proba):
    c = np.cumsum(np.asarray(proba, dtype=float), axis=1)
    return (c < 0.5).sum(axis=1)          # smallest k with cumulative prob >= 0.5


def _argmax(proba):
    return np.asarray(proba, dtype=float).argmax(axis=1)


def test_posterior_median_matches_ordinal_loss_better_than_argmax():
    # Skewed distribution: argmax picks class 0, but the ordinal-optimal decision is class 1.
    proba = np.array([[0.45, 0.40, 0.15]])
    y_true = np.array([2])
    med, arg = _posterior_median(proba), _argmax(proba)
    assert med[0] == 1 and arg[0] == 0                                  # cumsum = [.45,.85,1.0] -> k=1
    assert abs(int(med[0]) - int(y_true[0])) <= abs(int(arg[0]) - int(y_true[0]))


def test_posterior_median_equals_argmax_when_distribution_is_peaked():
    proba = np.array([[0.9, 0.07, 0.03], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])
    assert list(_posterior_median(proba)) == list(_argmax(proba)) == [0, 1, 2]


# ---- GOV-001: development artifacts must not carry the held-out cohort's outcomes ----

def test_develop_split_view_strips_final_test_prevalence():
    manifest = {
        "n_development": 35214, "n_final_test": 11630, "group_overlap_count": 0,
        "assignment_sha256": "abc",
        "class_prevalence": {
            "development": {"0": 0.686, "1": 0.277, "2": 0.037},
            "final_test": {"0": 0.677, "1": 0.284, "2": 0.039},   # 2012 OUTCOMES — must not leak
        },
        "notes": "Final test ... sealed ...",
    }
    view = CLI.develop_split_view(manifest)
    # the held-out outcome distribution is gone; nothing in the view exposes 2012 prevalence
    assert "final_test" not in view["class_prevalence"]
    assert view["class_prevalence"]["development"] == manifest["class_prevalence"]["development"]
    assert "0.677" not in str(view) and "0.284" not in str(view)
    assert "sealed" not in view["notes"].lower()
    # structural counts (not outcomes) may remain for reconciliation
    assert view["n_final_test"] == 11630 and view["group_overlap_count"] == 0
    # the original manifest is not mutated
    assert "final_test" in manifest["class_prevalence"]


# ---- WGT-001: matched unweighted controls exist and do not perturb the headline selection ----

def test_registry_has_matched_unweighted_controls():
    from crashsev import models as M
    reg = M.build_registry(include_optional=False, seed=42)
    for weighted, control in [("random_forest", "random_forest_unweighted"),
                              ("ordinal_random_forest", "ordinal_random_forest_unweighted")]:
        assert control in reg, f"missing matched control {control}"
        assert reg[control].class_weight_mode == "none"
        assert reg[weighted].class_weight_mode == "builtin"
        assert reg[control].kind == "ablation"          # not a comparator/candidate


def test_unweighted_controls_do_not_compete_for_selection():
    # An ablation model, even with the best dev oMAE, must never be chosen as the primary candidate.
    cv = {"aggregate": {
        "ordinal_random_forest_unweighted": {"mean_ordinal_mae": 0.10},  # best, but ablation
        "ordinal_random_forest": {"mean_ordinal_mae": 0.31},
    }}
    reg = {"ordinal_random_forest_unweighted": _spec("ablation"),
           "ordinal_random_forest": _spec("candidate")}
    assert CLI.select_best_candidate(cv, reg) == "ordinal_random_forest"


# ---- METRIC-001: undefined precision/recall are NaN + status, never a silent 0.0 ----

def test_undefined_precision_is_nan_not_zero():
    from crashsev import metrics as MET
    # Majority-style prediction: everything predicted class 0. Classes 1 and 2 are never predicted,
    # so their precision is UNDEFINED (not measured zero). Rows = true, cols = predicted.
    cm = [[70, 0, 0], [20, 0, 0], [10, 0, 0]]
    prf = MET.per_class_prf_from_cm(cm)
    assert np.isnan(prf[1]["precision"]) and np.isnan(prf[2]["precision"])
    assert prf[1]["status"].startswith("precision_undefined")
    assert prf[0]["precision"] == 70 / 100        # class 0 precision is defined
    assert prf[1]["recall"] == 0.0                # recall IS defined (class 1 has support, 0 hits)
    # macro-F1 is undefined when a class F1 is undefined; it must not silently become a number
    assert np.isnan(MET.macro_f1_from_cm(cm))


# ---- SPLIT-001: the temporal split enforces earlier-only, contiguous development years ----

def _toy_split(years):
    import pandas as pd
    n = len(years)
    df = pd.DataFrame({"Year": years, "Crash Number": [f"c{i}" for i in range(n)]})
    y = pd.Series([i % 3 for i in range(n)])
    from crashsev import splits as S
    return S.chronological_group_split(
        df, y, year_col="Year", group_col="Crash Number",
        final_test_years=[2012], row_id_col="Crash Number")


def test_split_accepts_contiguous_earlier_development():
    # 2009-2011 develop, 2012 final: the intended, valid contract.
    dev_idx, test_idx, manifest, _ = _toy_split([2009, 2010, 2011, 2012, 2012])
    assert len(dev_idx) == 3 and len(test_idx) == 2


def test_split_rejects_future_year_in_development():
    import pytest
    with pytest.raises(ValueError, match="SPLIT-001"):
        _toy_split([2009, 2010, 2013, 2012])       # 2013 would enter development


def test_split_rejects_gap_year():
    import pytest
    with pytest.raises(ValueError, match="SPLIT-001"):
        _toy_split([2009, 2011, 2012])             # 2010 gap between dev and final


# ---- AUTH-001: no duplicate feature declaration in the schema authority ----

def test_feature_ledger_has_no_duplicate_names():
    from crashsev import schema as SC
    names = [fs.name for fs in SC.FEATURE_LEDGER]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate FeatureSpec names in schema authority: {sorted(dupes)}"


# ===========================================================================================
# v4 protocol tests (third-round remediation): structural outcome isolation, strict tier,
# loss-consistent decision rule, deterministic prior baseline, byte-exact bundles.
# ===========================================================================================

import json as _json

import pandas as _pd
import pytest as _pytest

from crashsev import contracts as K
from crashsev import metrics as MET2
from crashsev import models as MODELS
from crashsev import synth as SYNTH


def _real_contract_frame(n=900, seed=11, poison_final=False, final_year=2012):
    """A synthetic frame padded to satisfy the REAL data contract (schema.json), with severity
    letters rewritten to the real extract's display labels. With ``poison_final`` the final
    year's severity values become a token that is NOT in the codebook - so any code path that
    validates or maps final-year outcomes must fail loudly."""
    df = SYNTH.make_synthetic_crash_df(n=n, seed=seed)
    df = df[_pd.to_numeric(df["Year"], errors="coerce") <= final_year].reset_index(drop=True).copy()
    disp = {"O": "", "C": "Possible", "B": "Non-Incapacitating",
            "A": "Incapacitating", "K": "Fatal"}
    df["Crash Severity"] = df["Crash Severity"].map(lambda v: disp.get(str(v), str(v)))
    schema = K.load_schema()
    for c in schema.required_columns:
        if c not in df.columns:
            df[c] = 0 if c in schema.numeric_sentinels else "x"
    if "DateTime" not in df.columns:
        df["DateTime"] = "2009-01-01 00:00"
    if poison_final:
        df.loc[_pd.to_numeric(df["Year"], errors="coerce") == final_year,
               "Crash Severity"] = "POISON_NOT_IN_CODEBOOK"
    return df


def _v4_cfg(final_year=2012):
    cfg = dict(CLI.DEFAULT_CONFIG)
    cfg["final_test_years"] = [final_year]
    cfg["feature_tier"] = "broad"          # synth lacks most strict-tier columns
    cfg["decision_rule"] = "posterior_median"
    return cfg


# ---- GOV-001 (v4): develop NEVER interprets final-year outcome values ----------------------

def test_develop_survives_poisoned_final_outcomes_that_full_prepare_rejects(tmp_path):
    """THE structural-isolation sentinel. The final year's severities are poisoned with an
    undocumented token: prepare_development must succeed (it never validates/maps them), while
    the full prepare() used by evaluate-final must REJECT the same file at Gate 0. If a
    regression makes develop read final-year outcomes again, this test fails on one side."""
    df = _real_contract_frame(poison_final=True)
    p = tmp_path / "poisoned.csv"
    df.to_csv(p, index=False)
    cfg = _v4_cfg()

    prep = CLI.prepare_development(str(p), cfg)              # must NOT raise
    n_dev_rows = int((_pd.to_numeric(df["Year"], errors="coerce") != 2012).sum())
    n_final_rows = int((_pd.to_numeric(df["Year"], errors="coerce") == 2012).sum())
    assert prep.audit.n_total == n_dev_rows                  # audit universe == dev rows ONLY
    assert prep.n_final_rows_raw == n_final_rows

    # nothing serialized by develop may carry the poison token (a final-year outcome value)
    blob = _json.dumps({"audit": prep.audit.to_dict(), "split": prep.dev_split,
                        "sentinel": prep.sentinel_flagged}, default=str)
    assert "POISON" not in blob

    with _pytest.raises(K.ContractViolation):
        CLI.prepare(str(p), cfg, seal_final=True)            # the full path MUST reject it


def test_develop_report_split_carries_no_final_outcome_distribution(tmp_path):
    """v4 develop artifacts must make the v3 subtraction attack impossible: the audit reconciles
    WITHIN the development years, and no final-side outcome quantity exists anywhere."""
    df = _real_contract_frame(poison_final=False)
    p = tmp_path / "clean.csv"
    df.to_csv(p, index=False)
    prep = CLI.prepare_development(str(p), _v4_cfg())

    a = prep.audit.to_dict()
    s = prep.dev_split
    assert a["n_total"] == s["n_dev_year_rows"]                       # dev universe only
    assert a["n_mapped"] + a["n_quarantined"] == a["n_total"]          # reconciles inside dev
    assert list(s["class_prevalence"].keys()) == ["development"]       # no final prevalence key
    assert s["n_development"] + s["n_dev_quarantined"] == s["n_dev_year_rows"]
    # the only final-side facts are the config years and the raw row count (plus the
    # identifier-based zero-overlap check, which carries no outcome)
    final_keys = sorted(k for k in s if "final" in k)
    assert final_keys == ["final_test_years", "n_final_test_rows_raw"]


def test_v4_freeze_pins_dev_side_assignment_and_final_recomputes_it(tmp_path):
    """The frozen split object is the outcome-free DEV-side assignment hash, and the full
    prepare() recomputes the identical hash from the whole file."""
    df = _real_contract_frame(poison_final=False)
    p = tmp_path / "clean2.csv"
    df.to_csv(p, index=False)
    cfg = _v4_cfg()
    dev = CLI.prepare_development(str(p), cfg)
    full = CLI.prepare(str(p), cfg, seal_final=True)
    assert dev.dev_assignment_sha256 == full.dev_assignment_sha256
    assert dev.dev_split["dev_assignment_sha256"] == dev.dev_assignment_sha256


# ---- OBJ-001 / DECISION-LOSS-001 (v4): loss-consistent primary decision rule ---------------

def test_posterior_median_helper_is_bayes_rule_for_absolute_loss():
    import numpy as _np
    proba = _np.array([[0.45, 0.40, 0.15], [0.9, 0.07, 0.03], [0.1, 0.2, 0.7]])
    med = MET2.posterior_median(proba)
    assert list(med) == [1, 0, 2]
    # skewed row: argmax says 0, the ordinal-optimal decision is 1
    assert int(_np.argmax(proba[0])) == 0 and med[0] == 1


def test_decide_labels_respects_configured_rule():
    import numpy as _np
    proba = _np.array([[0.45, 0.40, 0.15]])
    arg = _np.array([0])
    assert list(CLI.decide_labels(proba, arg, {"decision_rule": "posterior_median"})) == [1]
    assert list(CLI.decide_labels(proba, arg, {"decision_rule": "argmax"})) == [0]
    assert list(CLI.decide_labels(None, arg, {"decision_rule": "posterior_median"})) == [0]


# ---- BASE-001 (v4): deterministic prior-probability baseline -------------------------------

def test_prior_probability_baseline_is_deterministic_prevalence_forecast():
    import numpy as _np
    reg = MODELS.build_registry(include_optional=False, seed=1)
    assert "prior_probability" in reg and "empirical_prior" not in reg
    est = reg["prior_probability"].make_estimator()
    X = _np.zeros((10, 2)); y = _np.array([0]*7 + [1]*2 + [2]*1)
    est.fit(X, y)
    P = est.predict_proba(_np.zeros((3, 2)))
    assert _np.allclose(P, _np.array([[0.7, 0.2, 0.1]] * 3))          # the training prevalence
    est2 = reg["prior_probability"].make_estimator().fit(X, y)
    assert _np.allclose(P, est2.predict_proba(_np.zeros((3, 2))))     # deterministic


# ---- FEAT-001 (v4): strict scene tier is fail-closed and conservative ----------------------

def test_strict_tier_is_subset_and_excludes_flagged_families():
    led = K.load_ledger()
    strict = set(led.strict_allowed()); broad = set(led.allowed())
    assert strict < broad
    for banned in ("Unit 1 Person 1 Test Given", "Unit 1 Person 1 Insurance Coverage",
                   "Unit 1 Primary Contributing Circumstance", "Unit 1 Secondary Sequence of Events",
                   "Unit 1 Primary Damage Location", "Unit 1 Person 1 Restraint",
                   "Unit 1 Person 1 Ejected", "Unit 1 Person 1 Seat Location"):
        assert banned in broad and banned not in strict
    for kept in ("Weather", "Lighting", "Posted Speed", "Road Surface"):
        assert kept in strict


def test_strict_tier_fails_closed_without_ledger_column():
    led = K.load_ledger()
    df = led.df.drop(columns=["strict_scene_tier"])
    with _pytest.raises(K.ContractViolation):
        K.Ledger(df=df, source_path="in-memory").strict_allowed()


# ---- PO-001 (v4): proportional-odds convergence is recorded --------------------------------

def test_proportional_odds_records_convergence_state():
    import numpy as _np
    rng = _np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    latent = X @ _np.array([1.0, -0.5, 0.25]) + rng.normal(scale=0.5, size=300)
    y = _np.digitize(latent, [-0.5, 0.8])
    po = MODELS.ProportionalOddsClassifier(max_iter=200).fit(X, y)
    assert po.optimizer_success_ is True
    assert po.optimizer_n_iter_ > 0
    assert _np.isfinite(po.optimizer_grad_norm_)


def test_drop_uninformative_masks_missing_tokens():
    """MISS-001: a column consisting ONLY of documented missing tokens (e.g. 'Rural Urban' =
    'Unknown'/'Null value') is uninformative and must be dropped; a column with real values plus
    tokens is kept."""
    df = _pd.DataFrame({
        "all_tokens": ["Unknown"] * 5 + ["Null value"] * 5,
        "real_mixed": ["A", "B", "Unknown", "A", "B", "A", "B", "A", "Unknown", "B"],
        "constant": ["same"] * 10,
        "numeric_ok": list(range(10)),
    })
    toks = ["unknown", "null value", "missing"]
    kept, dropped = CLI.drop_uninformative(df, list(df.columns), toks)
    assert dropped == ["all_tokens", "constant"]
    assert kept == ["real_mixed", "numeric_ok"]


def test_source_tree_dirty_exempts_output_prefixes(monkeypatch):
    """The freeze/evaluate dirty-gate must exempt the phase-output dirs even for the FIRST
    porcelain line, whose leading status space is removed by _git()'s blob-level strip (the v3
    fixed-offset parser silently mis-sliced that line's path and flagged outputs as source)."""
    import crashsev.cli as C
    monkeypatch.setattr(C, "_git", lambda args:
                        "M remediation/experiment/development_report.json")   # first-line strip case
    assert C.source_tree_dirty() is False
    monkeypatch.setattr(C, "_git", lambda args:
                        " M remediation/experiment/development_report.json\n"
                        "?? remediation/runs/final_x/predictions_majority.csv")
    assert C.source_tree_dirty() is False
    monkeypatch.setattr(C, "_git", lambda args: "M remediation/crashsev/cli.py")
    assert C.source_tree_dirty() is True
