"""
crashsev.metrics — ordinal-aware evaluation metrics (EVAL-002).

Accuracy is not an adequate headline for this imbalanced *ordinal* problem. The primary
metric is ordinal mean absolute error (MAE in severity-class steps); a full secondary
suite is reported alongside it.

Every metric can be computed two ways so the same code serves both:
  * the re-analysis of the paper's published confusion matrices (``*_from_cm``), and
  * the corrected pipeline's own predictions (``compute_all``).

Definitions are standard; QWK follows Cohen's quadratic-weighted kappa.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Confusion-matrix based metrics (rows = true class, cols = predicted class)
# ---------------------------------------------------------------------------


def _as_cm(cm) -> np.ndarray:
    cm = np.asarray(cm, dtype="float64")
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion matrix must be square 2D; got shape {cm.shape}")
    return cm


def accuracy_from_cm(cm) -> float:
    cm = _as_cm(cm)
    total = cm.sum()
    return float(np.trace(cm) / total) if total else float("nan")


def per_class_prf_from_cm(cm, zero_division: str = "nan") -> Dict[int, Dict[str, float]]:
    """Per-class precision/recall/F1 with an explicit division convention.

    ``zero_division`` selects the F1 convention for a class that is never predicted
    (TP = 0, FP = 0, FN > 0):

    * ``"nan"`` — METRIC-001 storage convention: F1 is NaN with a status flag, never a
      silent 0.0. This is what the frozen governed runs stored; it is preserved as the
      default so a pipeline rerun reproduces the frozen artifacts byte-for-byte.
    * ``"zero"`` — the standard count-based reporting convention (F1-CONV-001):
      F1 = 2TP / (2TP + FP + FN), which is 0 whenever TP = 0 and FP + FN > 0, and is
      undefined (NaN) only when the class has neither predictions nor support
      (2TP + FP + FN = 0). This matches scikit-learn's ``zero_division=0`` behaviour
      and is the convention the manuscript reports.

    Under BOTH conventions precision is undefined (NaN) when TP + FP = 0 and recall is
    undefined only when the class has no support (TP + FN = 0); recall is a measured 0
    when TP = 0 and FN > 0.
    """
    if zero_division not in ("nan", "zero"):
        raise ValueError(f"zero_division must be 'nan' or 'zero', got {zero_division!r}")
    cm = _as_cm(cm)
    n = cm.shape[0]
    out: Dict[int, Dict[str, float]] = {}
    col_sums = cm.sum(axis=0)
    row_sums = cm.sum(axis=1)
    for i in range(n):
        tp = cm[i, i]
        prec = float(tp / col_sums[i]) if col_sums[i] > 0 else float("nan")
        rec = float(tp / row_sums[i]) if row_sums[i] > 0 else float("nan")
        if zero_division == "zero":
            denom = 2 * tp + (col_sums[i] - tp) + (row_sums[i] - tp)  # 2TP + FP + FN
            f1 = float(2 * tp / denom) if denom > 0 else float("nan")
        else:
            if np.isnan(prec) or np.isnan(rec):
                f1 = float("nan")
            elif (prec + rec) > 0:
                f1 = float(2 * prec * rec / (prec + rec))
            else:
                f1 = 0.0
        status = "ok"
        if np.isnan(prec):
            status = "precision_undefined_class_never_predicted"
        elif np.isnan(rec):
            status = "recall_undefined_no_support"
        out[i] = {"precision": prec, "recall": rec, "f1": f1,
                  "support": int(row_sums[i]), "status": status}
    return out


def macro_f1_from_cm(cm, zero_division: str = "nan") -> float:
    """Macro-F1 under the selected convention (see ``per_class_prf_from_cm``).

    ``"nan"`` (storage default): NaN when any class F1 is NaN — the frozen runs do not
    paper over a never-predicted class with 0.0 or nanmean.
    ``"zero"`` (F1-CONV-001 reporting): averages the class F1 values *including* the
    zeros of never-predicted classes; NaN only if some class F1 is itself undefined
    under the count-based definition (no predictions and no support)."""
    vals = [v["f1"] for v in per_class_prf_from_cm(cm, zero_division=zero_division).values()]
    return float(np.mean(vals)) if not any(np.isnan(x) for x in vals) else float("nan")


def balanced_accuracy_from_cm(cm) -> float:
    """Mean per-class recall. Recall is defined whenever a class has support, so this is defined
    for every model on a cohort where all classes are present; NaN only if a class has no support."""
    vals = [v["recall"] for v in per_class_prf_from_cm(cm).values()]
    return float(np.mean(vals)) if not any(np.isnan(x) for x in vals) else float("nan")


def ordinal_mae_from_cm(cm) -> float:
    """Mean absolute error in class steps: sum |i-j| * cm[i,j] / N. Primary metric."""
    cm = _as_cm(cm)
    n = cm.shape[0]
    idx = np.arange(n)
    dist = np.abs(idx.reshape(-1, 1) - idx.reshape(1, -1))
    total = cm.sum()
    return float((dist * cm).sum() / total) if total else float("nan")


def within_one_accuracy_from_cm(cm) -> float:
    cm = _as_cm(cm)
    n = cm.shape[0]
    idx = np.arange(n)
    within = (np.abs(idx.reshape(-1, 1) - idx.reshape(1, -1)) <= 1).astype(float)
    total = cm.sum()
    return float((within * cm).sum() / total) if total else float("nan")


def two_step_error_rate_from_cm(cm) -> float:
    """Fraction of predictions off by exactly 2 severity levels (the worst ordinal error)."""
    cm = _as_cm(cm)
    n = cm.shape[0]
    idx = np.arange(n)
    two = (np.abs(idx.reshape(-1, 1) - idx.reshape(1, -1)) == 2).astype(float)
    total = cm.sum()
    return float((two * cm).sum() / total) if total else float("nan")


def quadratic_weighted_kappa_from_cm(cm) -> float:
    """Cohen's quadratic weighted kappa."""
    O = _as_cm(cm)
    n = O.shape[0]
    total = O.sum()
    if total == 0:
        return float("nan")
    idx = np.arange(n)
    W = (idx.reshape(-1, 1) - idx.reshape(1, -1)) ** 2 / float((n - 1) ** 2)
    row_marg = O.sum(axis=1)
    col_marg = O.sum(axis=0)
    E = np.outer(row_marg, col_marg) / total
    denom = float((W * E).sum())
    if denom == 0:
        return 0.0
    return float(1.0 - (W * O).sum() / denom)


def predicted_distribution_from_cm(cm) -> Dict[int, float]:
    cm = _as_cm(cm)
    total = cm.sum()
    col_sums = cm.sum(axis=0)
    return {int(i): float(col_sums[i] / total) if total else float("nan") for i in range(cm.shape[0])}


def metrics_from_cm(cm, severe_class: int = 2, zero_division: str = "nan") -> Dict[str, object]:
    """Full ordinal metric suite from a confusion matrix (used by the re-analysis).

    ``zero_division`` selects the F1 convention (see ``per_class_prf_from_cm``): ``"nan"``
    is the frozen storage convention; ``"zero"`` is the count-based reporting convention
    (F1-CONV-001) used by the manuscript tables."""
    cm = _as_cm(cm)
    prf = per_class_prf_from_cm(cm, zero_division=zero_division)
    pred_dist = predicted_distribution_from_cm(cm)
    return {
        "n": int(cm.sum()),
        "accuracy": accuracy_from_cm(cm),
        "ordinal_mae": ordinal_mae_from_cm(cm),
        "qwk": quadratic_weighted_kappa_from_cm(cm),
        "macro_f1": macro_f1_from_cm(cm, zero_division=zero_division),
        "balanced_accuracy": balanced_accuracy_from_cm(cm),
        "within_one_accuracy": within_one_accuracy_from_cm(cm),
        "two_step_error_rate": two_step_error_rate_from_cm(cm),
        "per_class": {int(k): v for k, v in prf.items()},
        "severe_precision": prf[severe_class]["precision"],
        "severe_recall": prf[severe_class]["recall"],
        "severe_f1": prf[severe_class]["f1"],
        "predicted_distribution": pred_dist,
        "predicted_class0_share": pred_dist.get(0, float("nan")),
        "confusion_matrix": cm.astype(int).tolist(),
    }


# ---------------------------------------------------------------------------
# Strict input contracts (AA2-010, AA2-011)
# ---------------------------------------------------------------------------


def _validate_labels(y_true, y_pred, n_classes: int):
    """Fail explicitly on the failure modes the original metric code accepted silently:
    unequal lengths (zip truncation), non-integer or negative labels (numpy negative-index
    wrap in the confusion matrix), and labels outside ``[0, n_classes-1]``."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"y_true and y_pred have unequal length ({yt.shape[0]} vs {yp.shape[0]})")
    if yt.size == 0:
        raise ValueError("empty label vectors")
    yt_i = yt.astype(int)
    yp_i = yp.astype(int)
    if not (np.all(yt_i == yt) and np.all(yp_i == yp)):
        raise ValueError("labels must be integer-valued")
    for name, arr in (("y_true", yt_i), ("y_pred", yp_i)):
        lo, hi = int(arr.min()), int(arr.max())
        if lo < 0 or hi >= n_classes:
            raise ValueError(f"{name} has labels outside [0, {n_classes-1}] (min={lo}, max={hi})")
    return yt_i, yp_i


def _validate_proba(y_true, proba, n_classes: int, row_sum_tol: float = 1e-6):
    """Validate a probability matrix: shape ``(n, n_classes)``, finite, non-negative, and rows
    summing to 1 (within tolerance); ``y_true`` in range. Prevents silently scoring malformed
    probabilities (AA2-010)."""
    yt = np.asarray(y_true)
    P = np.asarray(proba, dtype="float64")
    if P.ndim != 2 or P.shape[1] != n_classes:
        raise ValueError(f"proba must have shape (n, {n_classes}); got {P.shape}")
    if P.shape[0] != yt.shape[0]:
        raise ValueError(f"proba rows ({P.shape[0]}) != len(y_true) ({yt.shape[0]})")
    if not np.all(np.isfinite(P)):
        raise ValueError("proba contains non-finite values")
    if np.any(P < -row_sum_tol):
        raise ValueError("proba contains negative values")
    rs = P.sum(axis=1)
    if np.any(np.abs(rs - 1.0) > 1e-3):
        raise ValueError(f"proba rows do not sum to 1 (max deviation {float(np.abs(rs-1).max()):.2e})")
    yt_i = yt.astype(int)
    if yt_i.min() < 0 or yt_i.max() >= n_classes:
        raise ValueError(f"y_true has labels outside [0, {n_classes-1}]")
    return yt_i, P


# ---------------------------------------------------------------------------
# Prediction-based metrics (for the corrected pipeline)
# ---------------------------------------------------------------------------


def posterior_median(proba) -> np.ndarray:
    """Hard decision under ABSOLUTE ordinal loss (DECISION-LOSS-001 / OBJ-001): the smallest
    class k whose cumulative probability reaches 0.5. Bayes optimal when applied to meaningful
    posterior probabilities; for class-weighted or miscalibrated models it remains loss matched
    to the model's stated probability vector, which need not equal a natural-prevalence
    posterior. For a peaked (near one-hot) distribution this coincides with argmax; for skewed
    distributions it can differ, and it is the decision rule the primary metric (ordinal MAE)
    actually implies. The v4 protocol prespecifies this as the PRIMARY hard-decision rule;
    argmax is reported as a sensitivity."""
    P = np.asarray(proba, dtype="float64")
    if P.ndim != 2:
        raise ValueError(f"proba must be 2-D, got shape {P.shape}")
    c = np.cumsum(P, axis=1)
    return (c < 0.5).sum(axis=1).astype(int)


def confusion_matrix_from_preds(y_true, y_pred, n_classes: int) -> np.ndarray:
    y_true, y_pred = _validate_labels(y_true, y_pred, n_classes)
    cm = np.zeros((n_classes, n_classes), dtype="int64")
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def compute_all(y_true, y_pred, n_classes: int, severe_class: int = 2) -> Dict[str, object]:
    cm = confusion_matrix_from_preds(y_true, y_pred, n_classes)
    return metrics_from_cm(cm, severe_class=severe_class)


# ---------------------------------------------------------------------------
# Probabilistic / proper scoring rules (require calibrated-ish probabilities)
# ---------------------------------------------------------------------------


def multiclass_log_loss(y_true, proba, n_classes: int, eps: float = 1e-15) -> float:
    y_true, P = _validate_proba(y_true, proba, n_classes)
    P = np.clip(P, eps, 1 - eps)
    P = P / P.sum(axis=1, keepdims=True)
    ll = -np.log(P[np.arange(len(y_true)), y_true])
    return float(np.mean(ll))


def multiclass_brier(y_true, proba, n_classes: int) -> float:
    y_true, P = _validate_proba(y_true, proba, n_classes)
    Y = np.zeros_like(P)
    Y[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def ranked_probability_score(y_true, proba, n_classes: int) -> float:
    """RPS for ordinal targets: mean squared error between cumulative predicted and
    cumulative true distributions. Lower is better; respects ordinal distance."""
    y_true, P = _validate_proba(y_true, proba, n_classes)
    cum_pred = np.cumsum(P, axis=1)
    Y = np.zeros_like(P)
    Y[np.arange(len(y_true)), y_true] = 1.0
    cum_true = np.cumsum(Y, axis=1)
    # sum over the first n-1 thresholds (the last cumulative is always 1)
    return float(np.mean(np.sum((cum_pred[:, :-1] - cum_true[:, :-1]) ** 2, axis=1)))
