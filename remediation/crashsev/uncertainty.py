"""
crashsev.uncertainty — paired crash-level bootstrap confidence intervals (STAT-002).

Model comparisons must account for run-to-run and sampling variability. Because the models
predict the *same* evaluation observations, comparisons are paired. The bootstrap resamples at
the *crash* level (crash id). Note: the analytical unit is the crash and each crash is exactly one
row, so crash-level and row-level resampling coincide — this is a case/row bootstrap under a
**cross-crash independence approximation**. Residual dependence between crashes (e.g. same corridor,
day, or reporting batch) is *not* modelled because the extract exposes no such cluster id; the
approximation is stated wherever intervals are reported (see paper §5.6, §10). This is not a
"dependence-aware" guarantee.

We report:
  * a 95% CI for each model's primary metric, and
  * a 95% CI for the paired difference between each model and the primary baseline.

A model is only called "better" when the paired-difference CI excludes zero in the
improving direction (and the effect exceeds a prespecified practical margin — enforced by
the caller / protocol, not here).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def grouped_bootstrap_indices(
    groups: Sequence, n_resamples: int, seed: int
) -> List[np.ndarray]:
    """Yield row-index arrays by resampling whole groups with replacement."""
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    # precompute row indices per group
    group_to_rows = {g: np.where(groups == g)[0] for g in unique_groups}
    rng = _rng(seed)
    out = []
    n_groups = len(unique_groups)
    for _ in range(n_resamples):
        sampled = rng.choice(unique_groups, size=n_groups, replace=True)
        rows = np.concatenate([group_to_rows[g] for g in sampled])
        out.append(rows)
    return out


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    groups: Optional[Sequence] = None,
    n_resamples: int = 2000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if groups is None:
        groups = np.arange(len(y_true))
    point = float(metric_fn(y_true, y_pred))
    samples = []
    for rows in grouped_bootstrap_indices(groups, n_resamples, seed):
        samples.append(metric_fn(y_true[rows], y_pred[rows]))
    samples = np.asarray(samples, dtype="float64")
    lo = float(np.nanpercentile(samples, 100 * alpha / 2))
    hi = float(np.nanpercentile(samples, 100 * (1 - alpha / 2)))
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_resamples": n_resamples}


def paired_difference_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    groups: Optional[Sequence] = None,
    n_resamples: int = 2000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    95% CI for metric(a) - metric(b) using the SAME resampled groups for both models
    (paired). For an error metric like ordinal MAE, a negative difference (a < b) means
    model a is better.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    if groups is None:
        groups = np.arange(len(y_true))
    point = float(metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b))
    diffs = []
    for rows in grouped_bootstrap_indices(groups, n_resamples, seed):
        da = metric_fn(y_true[rows], y_pred_a[rows])
        db = metric_fn(y_true[rows], y_pred_b[rows])
        diffs.append(da - db)
    diffs = np.asarray(diffs, dtype="float64")
    lo = float(np.nanpercentile(diffs, 100 * alpha / 2))
    hi = float(np.nanpercentile(diffs, 100 * (1 - alpha / 2)))
    # For an error metric, "a improves over b" iff the whole CI is < 0.
    excludes_zero = (lo < 0 and hi < 0) or (lo > 0 and hi > 0)
    return {
        "difference": point,
        "ci_low": lo,
        "ci_high": hi,
        "excludes_zero": bool(excludes_zero),
        "n_resamples": n_resamples,
    }
