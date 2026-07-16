"""Model-semantics and metric-contract tests (AA2-010/012/013/015/016)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crashsev import metrics as M, models as Mo, preprocessing as pp


# ---- metric input contracts (AA2-010) ----

def test_metrics_reject_negative_labels():
    with pytest.raises(ValueError):
        M.confusion_matrix_from_preds([0, 1, -1], [0, 1, 2], 3)   # negative-index wrap guard


def test_metrics_reject_unequal_length():
    with pytest.raises(ValueError):
        M.confusion_matrix_from_preds([0, 1], [0, 1, 2], 3)


def test_metrics_reject_out_of_range_label():
    with pytest.raises(ValueError):
        M.confusion_matrix_from_preds([0, 1, 3], [0, 1, 2], 3)


def test_proba_contracts_reject_bad_shapes_and_rowsums():
    with pytest.raises(ValueError):
        M.multiclass_log_loss([0, 1], [[0.5, 0.5, 0.0]], 3)          # rows != n
    with pytest.raises(ValueError):
        M.multiclass_brier([0, 1], [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]], 3)  # rows don't sum to 1
    with pytest.raises(ValueError):
        M.ranked_probability_score([0], [[0.5, 0.5]], 3)            # wrong n_classes width


def test_confusion_matrix_correct():
    cm = M.confusion_matrix_from_preds([0, 0, 1, 2, 2], [0, 1, 1, 2, 0], 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]


# ---- F1 zero-division conventions (F1-CONV-001 vs METRIC-001 storage) ----

def test_majority_only_classifier_f1_conventions():
    """A classifier that predicts ONLY the majority class (TP=0, FP=0, FN>0 for both
    minority classes): precision is undefined under both conventions; recall is a
    measured zero; F1 is NaN under the frozen storage convention ('nan') but a measured
    zero under the count-based reporting convention ('zero'); macro-F1 averages the
    zeros under 'zero' and stays undefined under 'nan'."""
    cm = [[70, 0, 0], [25, 0, 0], [5, 0, 0]]  # all rows predicted class 0

    # storage convention (frozen-run behaviour, default)
    prf_nan = M.per_class_prf_from_cm(cm)
    for c in (1, 2):
        assert np.isnan(prf_nan[c]["precision"])
        assert prf_nan[c]["recall"] == 0.0
        assert np.isnan(prf_nan[c]["f1"])
        assert prf_nan[c]["status"] == "precision_undefined_class_never_predicted"
    assert np.isnan(M.macro_f1_from_cm(cm))

    # count-based reporting convention (F1-CONV-001)
    prf_zero = M.per_class_prf_from_cm(cm, zero_division="zero")
    for c in (1, 2):
        assert np.isnan(prf_zero[c]["precision"])           # TP+FP=0 -> undefined
        assert prf_zero[c]["recall"] == 0.0                  # TP=0, FN>0 -> measured 0
        assert prf_zero[c]["f1"] == 0.0                      # 2TP/(2TP+FP+FN) = 0
    f1_0 = 2 * 70 / (2 * 70 + 30 + 0)                        # count-based class-0 F1
    assert prf_zero[0]["f1"] == pytest.approx(f1_0)
    assert M.macro_f1_from_cm(cm, zero_division="zero") == pytest.approx((f1_0 + 0.0 + 0.0) / 3)

    # the two conventions agree exactly wherever both are defined
    cm_full = [[50, 10, 2], [8, 30, 6], [1, 4, 9]]
    a = M.per_class_prf_from_cm(cm_full)
    b = M.per_class_prf_from_cm(cm_full, zero_division="zero")
    for c in range(3):
        assert a[c]["f1"] == pytest.approx(b[c]["f1"])
    assert M.macro_f1_from_cm(cm_full) == pytest.approx(M.macro_f1_from_cm(cm_full, zero_division="zero"))


def test_f1_convention_rejects_unknown_value():
    with pytest.raises(ValueError):
        M.per_class_prf_from_cm([[1, 0], [0, 1]], zero_division="warn")


def test_f1_zero_convention_no_support_no_predictions_is_nan():
    """Count-based F1 is undefined only when a class has neither predictions nor support."""
    cm = [[3, 0, 0], [1, 2, 0], [0, 0, 0]]  # class 2 absent entirely
    prf = M.per_class_prf_from_cm(cm, zero_division="zero")
    assert np.isnan(prf[2]["f1"]) and np.isnan(prf[2]["precision"]) and np.isnan(prf[2]["recall"])
    assert np.isnan(M.macro_f1_from_cm(cm, zero_division="zero"))


# ---- proportional-odds ordered logit (AA2-013/015) ----

def _poc_objective(X, y, d, K, l2=1e-3):
    """Replicate the ProportionalOddsClassifier objective for an independent gradient check."""
    def thr(a):
        return np.cumsum(np.concatenate([[a[0]], np.logaddexp(0.0, a[1:])]))
    def fn(p):
        beta, a = p[:d], p[d:]; theta = thr(a); eta = X @ beta; n = len(y)
        F = np.empty((n, K + 1)); F[:, 0] = 0; F[:, K] = 1
        for k in range(K - 1):
            F[:, k + 1] = 1 / (1 + np.exp(-np.clip(theta[k] - eta, -35, 35)))
        P = np.clip(F[np.arange(n), y + 1] - F[np.arange(n), y], 1e-12, 1)
        nll = -np.sum(np.log(P)) + l2 * np.sum(beta * beta)
        sp = np.zeros((n, K + 1))
        for k in range(K - 1):
            f = F[:, k + 1]; sp[:, k + 1] = f * (1 - f)
        up, lo = y + 1, y
        dP = -sp[np.arange(n), up] + sp[np.arange(n), lo]; coef = -(1.0 / P)
        gb = X.T @ (coef * dP) + 2 * l2 * beta
        gth = np.array([np.sum(coef * (np.where(up == k + 1, sp[:, k + 1], 0)
                                       - np.where(lo == k + 1, sp[:, k + 1], 0)))
                        for k in range(K - 1)])
        suf = np.cumsum(gth[::-1])[::-1]
        ga = np.empty(K - 1); ga[0] = suf[0]; ga[1:] = suf[1:] * (1 / (1 + np.exp(-a[1:])))
        return nll, np.concatenate([gb, ga])
    return fn


def test_proportional_odds_gradient_matches_numeric():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 3)); y = rng.integers(0, 3, 120)
    fn = _poc_objective(X, y, d=3, K=3)
    p = rng.normal(size=5) * 0.4
    _, ga = fn(p)
    gn = approx_fprime(p, lambda q: fn(q)[0], 1e-6)
    assert np.max(np.abs(ga - gn)) < 1e-4


def test_proportional_odds_recovers_ordering():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3)); beta = np.array([1.5, -1.0, 0.0]); eta = X @ beta
    th = np.array([-0.4, 1.1])
    p0 = 1 / (1 + np.exp(-(th[0] - eta))); p1 = 1 / (1 + np.exp(-(th[1] - eta)))
    P = np.column_stack([p0, p1 - p0, 1 - p1])
    y = np.array([rng.choice(3, p=pi / pi.sum()) for pi in P])
    poc = Mo.ProportionalOddsClassifier(class_weight=None).fit(X, y)
    assert poc.optimizer_success_
    assert poc.theta_[0] < poc.theta_[1]                 # thresholds stay ordered
    assert np.max(np.abs(poc.theta_ - th)) < 0.4         # roughly recovered


def test_frank_hall_probabilities_are_coherent():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3)); y = rng.integers(0, 3, 200)
    from sklearn.linear_model import LogisticRegression
    fh = Mo.FrankHallOrdinalClassifier(LogisticRegression(max_iter=500)).fit(X, y)
    P = fh.predict_proba(X)
    assert np.allclose(P.sum(axis=1), 1.0) and (P >= 0).all()


# ---- seed propagation (AA2-012) ----

def test_seed_changes_stochastic_predictions():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    # mild signal so the forest is not a constant predictor
    y = pd.Series(((X["a"] + 0.5 * rng.normal(size=300)) > 0).astype(int)
                  + ((X["b"] + 0.5 * rng.normal(size=300)) > 0.8).astype(int))
    num = list("abcd")
    r1 = Mo.build_registry(include_optional=False, seed=1)["random_forest"]
    r2 = Mo.build_registry(include_optional=False, seed=2)["random_forest"]
    proba1 = Mo.make_pipeline(r1, num, []).fit(X, y).predict_proba(X)
    proba2 = Mo.make_pipeline(r2, num, []).fit(X, y).predict_proba(X)
    # different seeds -> different bootstrap samples -> different fitted probabilities
    assert not np.allclose(proba1, proba2)


def test_registry_has_true_ordered_logit_and_renamed_frank_hall():
    reg = Mo.build_registry(include_optional=False, seed=1)
    assert "proportional_odds" in reg and "frank_hall_logistic" in reg
    assert "ordinal_logistic" not in reg       # the misleading name is gone
    assert "proportional" in reg["proportional_odds"].note.lower()


# ---- sparse one-hot (AA2-016) ----

def test_preprocessor_can_emit_sparse():
    import scipy.sparse as sp
    rng = np.random.default_rng(0)
    # categorical-heavy (like the real data: few numerics, many high-cardinality categoricals)
    # so the encoded matrix is below the ColumnTransformer sparse_threshold and stays sparse.
    cats = {f"c{i}": rng.integers(0, 10, 200).astype(str) for i in range(6)}
    X = pd.DataFrame({"n": np.arange(200.0), **cats})
    cat_cols = list(cats)
    Xt = pp.build_preprocessor(["n"], cat_cols, sparse=True).fit_transform(X)
    assert sp.issparse(Xt)
    assert not sp.issparse(pp.build_preprocessor(["n"], cat_cols, sparse=False).fit_transform(X))
