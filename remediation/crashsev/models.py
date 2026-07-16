"""
crashsev.models — baselines and candidate models under one shared contract (EVAL-001).

Every model is an sklearn ``Pipeline([("pre", preprocessor), ("clf", estimator)])`` so the
preprocessor is fit on training rows only, and every model consumes the *same* split ids
and the *same* allowed feature set.

Baselines (the statistical floor the original study never reported):
  * majority            — constant most-frequent class (the 67.6% accuracy trap)
  * ordinal_median      — constant median class (relevant to ordinal MAE)
  * prior_probability   — deterministic training-prevalence probability forecast (BASE-001)
  * multinomial_logistic
  * ordinal_logistic    — Frank & Hall cumulative ordinal logistic (dependency-free,
                          coherent ordinal probabilities); stands in for proportional-odds
  * shallow_tree        — depth-limited decision tree

Candidates (retained from the original project, given fair, identical treatment):
  * decision_tree       — the repo's DecisionTree configuration
  * random_forest       — ordinary RF comparator
  * ordinal_random_forest — Frank & Hall ordinal RF with COHERENT predict_proba; this is
                          the corrected replacement for the repo's MultiLevelRandomForest
                          (METH-005: the repo MRF has no predict_proba and hard-codes
                          3-class thresholds).
  * xgboost             — if xgboost is installed
  * ebm                 — if interpret is installed

Class imbalance is handled with class weights where supported; final evaluation is always
on natural prevalence (never resampled test data).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import preprocessing as pp

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    XGBClassifier = None
    _HAS_XGB = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    _HAS_EBM = True
except Exception:  # pragma: no cover
    ExplainableBoostingClassifier = None
    _HAS_EBM = False


# ---------------------------------------------------------------------------
# Custom estimators
# ---------------------------------------------------------------------------


class OrdinalMedianClassifier(BaseEstimator, ClassifierMixin):
    """Constant predictor that always returns the (weighted) median class.

    For an ordinal target this is a stronger trivial baseline than the majority class,
    because it minimises expected absolute error under the training label distribution.
    """

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        # weighted median of the ordinal labels
        order = np.sort(y)
        self.median_class_ = int(order[len(order) // 2])
        return self

    def predict(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n, self.median_class_, dtype=int)

    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        P = np.zeros((n, len(self.classes_)))
        j = int(np.where(self.classes_ == self.median_class_)[0][0])
        P[:, j] = 1.0
        return P


class FrankHallOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    Frank & Hall (2001) ordinal decomposition with coherent probabilities.

    For ordered classes 0..K-1, fit K-1 binary classifiers estimating P(y > k). Class
    probabilities are reconstructed as differences of cumulative probabilities and clipped
    to be non-negative and normalised, guaranteeing a valid distribution (unlike the repo's
    MultiLevelRandomForestClassifier, which exposes no predict_proba and can produce
    incoherent threshold decisions).
    """

    def __init__(self, base_estimator: Optional[BaseEstimator] = None):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.k_ = len(self.classes_)
        base = self.base_estimator if self.base_estimator is not None else LogisticRegression(max_iter=1000)
        self.estimators_ = []
        self.trivial_ = []  # for degenerate binary targets in a fold
        for k in range(self.k_ - 1):
            y_bin = (y > self.classes_[k]).astype(int)
            if len(np.unique(y_bin)) < 2:
                # all rows on one side of the threshold; store the constant probability
                self.estimators_.append(None)
                self.trivial_.append(float(y_bin.mean()))
            else:
                est = clone(base)
                est.fit(X, y_bin)
                self.estimators_.append(est)
                self.trivial_.append(None)
        return self

    def _cum_gt(self, X) -> np.ndarray:
        """Return array (n, K-1) of P(y > k), enforced monotone non-increasing in k."""
        n = X.shape[0]
        cols = []
        for k in range(self.k_ - 1):
            est = self.estimators_[k]
            if est is None:
                cols.append(np.full(n, self.trivial_[k]))
            else:
                cols.append(est.predict_proba(X)[:, 1])
        P_gt = np.column_stack(cols) if cols else np.zeros((n, 0))
        # enforce monotonicity P(y>0) >= P(y>1) >= ...
        P_gt = np.minimum.accumulate(P_gt, axis=1)
        return np.clip(P_gt, 0.0, 1.0)

    def predict_proba(self, X):
        n = X.shape[0]
        P_gt = self._cum_gt(X)  # (n, K-1)
        P = np.zeros((n, self.k_))
        prev = np.ones(n)  # P(y > -1) = 1
        for k in range(self.k_ - 1):
            P[:, k] = prev - P_gt[:, k]
            prev = P_gt[:, k]
        P[:, self.k_ - 1] = prev
        P = np.clip(P, 1e-9, None)
        P = P / P.sum(axis=1, keepdims=True)
        return P

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ProportionalOddsClassifier(BaseEstimator, ClassifierMixin):
    """
    A **true** proportional-odds (ordered logit) model — a single linear predictor with
    ordered thresholds, fit by maximum likelihood — as distinct from the Frank-Hall
    decomposition (which fits K-1 *separate* binary classifiers). Provides the genuine
    ordered-logit baseline the audit asked for (AA2-013/015).

    Model: P(y <= k) = sigma(theta_k - x·beta), with theta_0 < theta_1 < ... The thresholds
    are kept ordered by an unconstrained softplus reparameterisation, and the negative
    log-likelihood is minimised with L-BFGS-B using an analytic gradient. Sparse or dense X
    are both accepted; if the encoded design is very wide the caller may prefer another
    baseline (this one is intended for the modest post-OHE dimensionality of this study).
    """

    def __init__(self, max_iter: int = 200, l2: float = 1e-4, class_weight: Optional[str] = "balanced"):
        self.max_iter = max_iter
        self.l2 = l2
        self.class_weight = class_weight

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def _thresholds(self, a):
        # theta_0 = a[0]; theta_k = theta_{k-1} + softplus(a[k])  -> strictly increasing
        inc = np.concatenate([[a[0]], np.logaddexp(0.0, a[1:])])
        return np.cumsum(inc)

    def fit(self, X, y):
        from scipy.optimize import minimize
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        self.k_ = K
        cls_index = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([cls_index[v] for v in y], dtype=int)
        Xd = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype="float64")
        n, d = Xd.shape

        # per-sample weights for class imbalance (applied to the likelihood)
        if self.class_weight == "balanced":
            counts = np.bincount(yi, minlength=K).astype("float64")
            w_class = n / (K * np.maximum(counts, 1))
            w = w_class[yi]
        else:
            w = np.ones(n)

        def unpack(p):
            beta = p[:d]
            a = p[d:]
            theta = self._thresholds(a)
            return beta, a, theta

        def nll_and_grad(p):
            beta, a, theta = unpack(p)
            eta = Xd @ beta
            # cumulative F_k = sigma(theta_k - eta), k=0..K-2 ; F_{-1}=0, F_{K-1}=1
            F = np.empty((n, K + 1))
            F[:, 0] = 0.0
            F[:, K] = 1.0
            for k in range(K - 1):
                F[:, k + 1] = self._sigmoid(theta[k] - eta)
            P = F[np.arange(n), yi + 1] - F[np.arange(n), yi]
            P = np.clip(P, 1e-12, 1.0)
            nll = -np.sum(w * np.log(P)) + self.l2 * np.sum(beta * beta)

            # gradients
            # dF_k/deta = -s'(theta_k-eta); dF_k/dtheta_k = s'(theta_k-eta)
            sprime = np.zeros((n, K + 1))
            for k in range(K - 1):
                f = F[:, k + 1]
                sprime[:, k + 1] = f * (1.0 - f)
            up = yi + 1
            lo = yi
            dP_deta = -sprime[np.arange(n), up] + sprime[np.arange(n), lo]
            coef = -(w / P)  # d(-log P)/dP * w
            g_eta = coef * dP_deta                      # (n,)
            grad_beta = Xd.T @ g_eta + 2.0 * self.l2 * beta
            # threshold grads
            grad_theta = np.zeros(K - 1)
            for k in range(K - 1):
                # dP/dtheta_k nonzero when k == lo (as upper bound) or k == lo-1 (as lower)
                dP_dtheta_k = np.where(up == k + 1, sprime[:, k + 1], 0.0) \
                    - np.where(lo == k + 1, sprime[:, k + 1], 0.0)
                grad_theta[k] = np.sum(coef * dP_dtheta_k)
            # chain through softplus reparam: theta = cumsum(inc), inc_0=a0, inc_k=softplus(a_k)
            # dtheta_j/da_k = 1 (k=0) or sigmoid(a_k) (k>=1), for j>=k
            suffix = np.cumsum(grad_theta[::-1])[::-1]  # suffix[k] = sum_{j>=k} grad_theta[j]
            grad_a = np.empty(K - 1)
            grad_a[0] = suffix[0]
            if K - 1 > 1:
                grad_a[1:] = suffix[1:] * self._sigmoid(a[1:])
            return nll, np.concatenate([grad_beta, grad_a])

        p0 = np.zeros(d + (K - 1))
        # initialise thresholds at the empirical cumulative logits
        cum = np.clip(np.cumsum(np.bincount(yi, minlength=K)[:-1]) / n, 1e-3, 1 - 1e-3)
        th0 = np.log(cum / (1 - cum))
        p0[d] = th0[0]
        if K - 1 > 1:
            p0[d + 1:] = np.log(np.expm1(np.maximum(np.diff(th0), 1e-3)))
        res = minimize(nll_and_grad, p0, jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.coef_, self.a_, self.theta_ = unpack(res.x)
        # PO-001: convergence is RECORDED here and SURFACED by the CLI into the run results
        # (a non-converged fit is flagged and excluded from any superlative claim).
        self.optimizer_success_ = bool(res.success)
        self.optimizer_n_iter_ = int(getattr(res, "nit", -1))
        try:
            self.optimizer_grad_norm_ = float(np.linalg.norm(np.asarray(res.jac, dtype="float64")))
        except Exception:
            self.optimizer_grad_norm_ = float("nan")
        self.n_features_in_ = d
        return self

    def predict_proba(self, X):
        Xd = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype="float64")
        eta = Xd @ self.coef_
        n = Xd.shape[0]
        K = self.k_
        F = np.empty((n, K + 1))
        F[:, 0] = 0.0
        F[:, K] = 1.0
        for k in range(K - 1):
            F[:, k + 1] = self._sigmoid(self.theta_[k] - eta)
        P = np.diff(F, axis=1)
        P = np.clip(P, 1e-12, None)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str
    kind: str  # "baseline" or "candidate"
    make_estimator: Callable[[], BaseEstimator]
    scale_numeric: bool = False
    supports_proba: bool = True
    # how class imbalance is handled: "builtin" (estimator's own class_weight),
    # "sample_weight" (balanced weights passed to fit -> clf__sample_weight), or "none".
    class_weight_mode: str = "none"
    dense_required: bool = False  # estimator cannot consume a sparse design matrix
    note: str = ""


def _rf(seed: int, **kw):
    params = dict(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed)
    params.update(kw)
    return RandomForestClassifier(**params)


def build_registry(include_optional: bool = True, seed: int = 42) -> Dict[str, ModelSpec]:
    """Build the model registry with ``seed`` propagated into every stochastic estimator
    (AA2-012). ``seed`` must come from the run config; nothing is hard-coded to 42 at runtime."""
    reg: Dict[str, ModelSpec] = {}

    # --- Baselines (the statistical floor + a true ordered logit) ---
    reg["majority"] = ModelSpec(
        "majority", "baseline",
        lambda: DummyClassifier(strategy="most_frequent"),
        note="constant most-frequent class (the accuracy trap)",
    )
    reg["ordinal_median"] = ModelSpec(
        "ordinal_median", "baseline",
        lambda: OrdinalMedianClassifier(),
        note="constant median class (minimises expected |error| under the prior)",
    )
    reg["prior_probability"] = ModelSpec(
        "prior_probability", "baseline",
        lambda: DummyClassifier(strategy="prior"),
        note="DETERMINISTIC training-prevalence probability forecast — the honest trivial "
             "probabilistic floor (BASE-001; replaces the v3 'empirical_prior', which was "
             "stochastic hard-label draws misdescribed as a probability baseline)",
    )
    reg["multinomial_logistic"] = ModelSpec(
        "multinomial_logistic", "baseline",
        lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        scale_numeric=True, class_weight_mode="builtin",
        note="regularised multinomial logistic (nominal, ignores order)",
    )
    reg["proportional_odds"] = ModelSpec(
        "proportional_odds", "baseline",
        lambda: ProportionalOddsClassifier(class_weight="balanced"),
        scale_numeric=True, dense_required=True, class_weight_mode="builtin",
        note="TRUE proportional-odds ordered logit (single latent + ordered thresholds, MLE)",
    )
    reg["frank_hall_logistic"] = ModelSpec(
        "frank_hall_logistic", "baseline",
        lambda: FrankHallOrdinalClassifier(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        ),
        scale_numeric=True, class_weight_mode="builtin",
        note="Frank-Hall ordinal decomposition: K-1 binary cumulative logits (NOT proportional odds)",
    )
    reg["shallow_tree"] = ModelSpec(
        "shallow_tree", "baseline",
        lambda: DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=seed),
        class_weight_mode="builtin", note="depth-4 transparent tree",
    )

    # --- Candidates (retained from the original project, given identical treatment) ---
    reg["decision_tree"] = ModelSpec(
        "decision_tree", "candidate",
        lambda: DecisionTreeClassifier(
            criterion="entropy", max_depth=10, min_samples_split=2,
            min_samples_leaf=1, class_weight="balanced", random_state=seed,
        ),
        class_weight_mode="builtin", note="repo DecisionTree config (entropy, depth 10)",
    )
    reg["random_forest"] = ModelSpec(
        "random_forest", "candidate",
        lambda: _rf(seed),
        class_weight_mode="builtin", note="ordinary RF comparator (nominal)",
    )
    reg["ordinal_random_forest"] = ModelSpec(
        "ordinal_random_forest", "candidate",
        lambda: FrankHallOrdinalClassifier(_rf(seed)),
        class_weight_mode="builtin",
        note="Frank-Hall ordinal RF with coherent predict_proba (supersedes repo MultiLevelRF)",
    )

    # --- Matched weighting ablation (WGT-001) ---
    # Identical to random_forest / ordinal_random_forest EXCEPT class_weight=None, so the ONLY
    # difference is balanced class weighting. kind="ablation": these do not compete for the primary
    # candidate/comparator slot (select_best_candidate/select_primary_baseline ignore them); they
    # exist to measure what class weighting changes with model, seed, features, and folds held fixed.
    reg["random_forest_unweighted"] = ModelSpec(
        "random_forest_unweighted", "ablation",
        lambda: _rf(seed, class_weight=None),
        class_weight_mode="none",
        note="RF, class_weight=None — matched unweighted control for random_forest (WGT-001)",
    )
    reg["ordinal_random_forest_unweighted"] = ModelSpec(
        "ordinal_random_forest_unweighted", "ablation",
        lambda: FrankHallOrdinalClassifier(_rf(seed, class_weight=None)),
        class_weight_mode="none",
        note="ordinal RF, class_weight=None — matched unweighted control for ordinal_random_forest (WGT-001)",
    )

    if include_optional and _HAS_XGB:
        reg["xgboost"] = ModelSpec(
            "xgboost", "candidate",
            lambda: XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                objective="multi:softprob", tree_method="hist", random_state=seed, n_jobs=-1,
            ),
            # XGBoost has no class_weight; imbalance handled by balanced sample_weight at fit,
            # which the CLI ACTUALLY passes (clf__sample_weight) and records (AA2-015).
            class_weight_mode="sample_weight",
            note="XGBoost (CPU hist); balanced sample_weight applied at fit",
        )
    if include_optional and _HAS_EBM:
        reg["ebm"] = ModelSpec(
            "ebm", "exploratory",
            lambda: ExplainableBoostingClassifier(random_state=seed),
            class_weight_mode="sample_weight", dense_required=True,
            note="Explainable Boosting Machine (final-only EXPLORATORY: excluded from development CV, "
                 "so it never competes for the primary candidate; reported for interest only) (MODEL-FAMILY-001)",
        )

    return reg


def make_pipeline(
    spec: ModelSpec, numeric_cols, categorical_cols,
    min_frequency: float = 0.01, sparse: bool = True,
) -> Pipeline:
    pre = pp.build_preprocessor(
        numeric_cols, categorical_cols,
        scale_numeric=spec.scale_numeric, min_frequency=min_frequency,
        sparse=sparse and not spec.dense_required,
    )
    return Pipeline([("pre", pre), ("clf", spec.make_estimator())])


def availability() -> Dict[str, bool]:
    return {"xgboost": _HAS_XGB, "ebm": _HAS_EBM}
