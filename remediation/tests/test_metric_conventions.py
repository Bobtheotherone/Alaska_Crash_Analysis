"""Guards against a silent metric-convention switch (R-007 / R-008 / E-019).

The manuscript states: RPS is UNNORMALIZED (twice the /(K-1) alternative for K=3);
log loss clips probabilities at the implementation epsilon; ECE is top-label with the
frozen binning rule. These tests pin each statement to the implementation and to the
committed sidecar ``experiment/metric_conventions.json``, and — when the de-identified
lossless parquet is present — recompute a stored governed value under the declared
convention.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from crashsev import calibration as C
from crashsev import metrics as M

REM = Path(__file__).resolve().parents[1]
SIDECAR = REM / "experiment" / "metric_conventions.json"


def _sidecar() -> dict:
    return json.loads(SIDECAR.read_text("utf-8"))


def _build():
    spec = importlib.util.spec_from_file_location(
        "gen_metric_conventions", REM / "tools" / "gen_metric_conventions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def test_sidecar_exists_and_is_current():
    assert SIDECAR.exists(), "run tools/gen_metric_conventions.py"
    assert _sidecar() == _build(), "sidecar stale; run tools/gen_metric_conventions.py"


def test_rps_is_the_unnormalized_convention():
    # y = [0, 2] with both rows forecast one-hot class 0:
    #   row 1: cumulative error 0                      -> contribution 0
    #   row 2: (1-0)^2 + (1-0)^2 over the 2 thresholds -> contribution 2
    # unnormalized mean = 1.0; the /(K-1) convention would give 0.5.
    P = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    y = np.array([0, 2])
    val = M.ranked_probability_score(y, P, n_classes=3)
    assert val == pytest.approx(1.0), (
        f"RPS convention changed: got {val}; the manuscript and metric_conventions.json "
        "declare the UNNORMALIZED cumulative form")
    sc = _sidecar()
    assert sc["rps_normalization"] == "none"
    assert sc["alternative_normalized_divisor"] == 2


def test_log_loss_epsilon_matches_sidecar_and_hyperparameter_ledger():
    eps = inspect.signature(M.multiclass_log_loss).parameters["eps"].default
    assert _sidecar()["log_loss_epsilon"] == eps == 1e-15
    hp = json.loads((REM / "experiment" / "hyperparameter_ledger.json").read_text("utf-8"))
    assert hp["shared_protocol_constants"]["log_loss_probability_clip"] == eps
    # unclipped one-hot behavior the sidecar documents: -log(0) is infinite
    P = np.array([[1.0, 0.0, 0.0]])
    with np.errstate(divide="ignore"):
        assert np.isinf(-np.log(P[0, 1]))


def test_ece_binning_matches_sidecar():
    n_bins = inspect.signature(C.expected_calibration_error).parameters["n_bins"].default
    sc = _sidecar()
    assert sc["ece_n_bins"] == n_bins == 10
    assert "top-label" in sc["ece_definition"]
    # a perfectly calibrated two-bin toy example scores 0 under the declared rule
    y = np.array([0, 1, 0, 0])
    P = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    assert C.expected_calibration_error(y, P) == pytest.approx(0.0)


def test_f1_convention_block_matches_implementation():
    """F1-CONV-001: the sidecar's reporting/storage split matches the code; the pipeline
    default MUST stay 'nan' so frozen bundles keep self-reproducing byte-for-byte."""
    sc = _sidecar()["f1_reporting_convention"]
    assert sc["pipeline_default"] == "nan"
    assert inspect.signature(M.per_class_prf_from_cm).parameters["zero_division"].default == "nan"
    # reporting convention behaviour on a majority-only confusion matrix
    cm = [[9, 0, 0], [6, 0, 0], [1, 0, 0]]
    assert np.isnan(M.macro_f1_from_cm(cm))                                # storage
    rep = M.per_class_prf_from_cm(cm, zero_division="zero")                # reporting
    assert rep[2]["f1"] == 0.0 and np.isnan(rep[2]["precision"])
    assert M.macro_f1_from_cm(cm, zero_division="zero") == pytest.approx(
        (2 * 9 / (2 * 9 + 7)) / 3)


def test_bootstrap_convention_block_matches_implementation():
    """BOOT-CONV-001: paired percentile interval, level, seed, resample count, and the
    manual percentile recomputation all agree with crashsev/uncertainty.py."""
    from crashsev import uncertainty as U
    sc = _sidecar()["paired_bootstrap"]
    assert sc["interval_type"].startswith("percentile")
    assert sc["confidence_level"] == pytest.approx(0.95)
    assert sc["n_resamples"] == 2000 and sc["seed"] == 42
    assert sc["models_refit_within_resamples"] is False
    # behavioural pin: the implementation's interval equals a manual percentile of the
    # replicate differences under the same RNG stream
    rng_y = np.random.default_rng(7)
    y = rng_y.integers(0, 3, 400)
    a = np.clip(y + (rng_y.random(400) < 0.2), 0, 2)
    b = np.clip(y + (rng_y.random(400) < 0.4), 0, 2)
    mfn = lambda yt, yp: float(np.mean(np.abs(np.asarray(yt) - np.asarray(yp))))
    got = U.paired_difference_ci(y, a, b, mfn, n_resamples=200, seed=42)
    diffs = []
    for rows in U.grouped_bootstrap_indices(np.arange(400), 200, 42):
        diffs.append(mfn(y[rows], a[rows]) - mfn(y[rows], b[rows]))
    assert got["ci_low"] == pytest.approx(float(np.nanpercentile(diffs, 2.5)))
    assert got["ci_high"] == pytest.approx(float(np.nanpercentile(diffs, 97.5)))
    assert got["difference"] == pytest.approx(mfn(y, a) - mfn(y, b))


def test_frank_hall_convention_block_matches_implementation():
    """FH-CONV-001: the sidecar's construction description is pinned to the code, and an
    adversarial non-monotone base estimator still yields valid probabilities."""
    from crashsev import models as Mo
    sc = _sidecar()["frank_hall_probability_construction"]
    assert "cumulative minimum" in sc["monotonicity"]
    assert sc["row_renormalization"] is True
    src = inspect.getsource(Mo.FrankHallOrdinalClassifier)
    assert "np.minimum.accumulate" in src and "1e-9" in src

    class NonMonotoneBase:
        """predict_proba returns p(y_bin=1) that VIOLATES monotonicity across thresholds."""
        def __init__(self):
            self.calls = 0
        def fit(self, X, y):
            return self
        def get_params(self, deep=True):
            return {}
        def set_params(self, **kw):
            return self
        def predict_proba(self, X):
            n = X.shape[0]
            NonMonotoneBase.CALLS = getattr(NonMonotoneBase, "CALLS", 0) + 1
            # first threshold gets LOW exceedance, second gets HIGH -> raw g0 < g1 (invalid)
            p1 = np.full(n, 0.2 if NonMonotoneBase.CALLS % 2 == 1 else 0.9)
            return np.column_stack([1 - p1, p1])

    NonMonotoneBase.CALLS = 0
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = np.array([0, 1, 2] * 20)
    fh = Mo.FrankHallOrdinalClassifier(NonMonotoneBase()).fit(X, y)
    P = fh.predict_proba(X)
    assert np.isfinite(P).all() and (P >= 0).all()
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-9)
    cum = np.cumsum(P, axis=1)
    assert (np.diff(cum, axis=1) >= -1e-12).all(), "reconstructed cumulative must be monotone"


def test_frank_hall_degenerate_training_targets_stay_valid():
    """FH-CONV-001 degenerate-target rules. In the governed pipeline a development fold
    missing one of the three classes is SKIPPED by the CV loop (cli.dev_cv checks
    nunique < n_classes; no such fold occurred in the frozen runs). The estimator itself
    additionally (a) handles a reduced class set by fitting K-1 thresholds over the
    observed classes only, (b) handles a single-class target with a valid constant
    distribution, and (c) retains a constant-exceedance fallback for a single-sided
    threshold as defence-in-depth (unreachable when classes derive from y, pinned at
    source level)."""
    from crashsev import models as Mo
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))

    # (a) two observed classes -> one threshold, valid distribution over 2 classes
    y2 = np.array([0] * 30 + [1] * 20)
    fh2 = Mo.FrankHallOrdinalClassifier().fit(X, y2)
    P2 = fh2.predict_proba(X)
    assert P2.shape == (50, 2) and np.isfinite(P2).all() and (P2 >= 0).all()
    assert np.allclose(P2.sum(axis=1), 1.0, atol=1e-9)

    # (b) single-class target -> constant, valid distribution
    y1 = np.zeros(50, dtype=int)
    fh1 = Mo.FrankHallOrdinalClassifier().fit(X, y1)
    P1 = fh1.predict_proba(X)
    assert P1.shape == (50, 1) and np.allclose(P1, 1.0)

    # (c) the constant-exceedance fallback exists in source (defence-in-depth)
    src = inspect.getsource(Mo.FrankHallOrdinalClassifier.fit)
    assert "trivial_" in src and "y_bin.mean()" in src

    # the governed CV loop's skip rule exists (a class-incomplete fold never fits)
    from crashsev import cli
    assert 'nunique() < cfg["n_classes"]' in inspect.getsource(cli.dev_cv)


def test_governed_rps_and_log_loss_reproduce_from_lossless_parquet():
    pq_path = REM / "experiment" / "predictions_lossless.parquet"
    if not pq_path.exists():
        pytest.skip("de-identified lossless parquet not present in this checkout "
                    "(shipped in the handoff evidence/ tier)")
    pd = pytest.importorskip("pandas")
    fr = json.loads((REM / "experiment" / "final_results.json").read_text("utf-8"))
    df = pd.read_parquet(pq_path)
    for model in ("prior_probability", "ordinal_random_forest"):
        d = df[df["model"] == model]
        P = d[["proba_0", "proba_1", "proba_2"]].to_numpy()
        y = d["y_true"].to_numpy(int)
        rps = M.ranked_probability_score(y, P, n_classes=3)
        ll = M.multiclass_log_loss(y, P, n_classes=3)
        assert rps == pytest.approx(fr["results"][model]["rps"], abs=1e-12), model
        assert ll == pytest.approx(fr["results"][model]["log_loss"], abs=1e-12), model
        # a silent switch to the normalized convention would halve the stored value
        normalized = rps / 2
        assert abs(normalized - fr["results"][model]["rps"]) > 1e-3, (
            "normalized and unnormalized RPS indistinguishable — guard is vacuous")
