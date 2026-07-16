"""
crashsev.calibration — probability calibration diagnostics (EVAL-002, benchmark B3).

A classifier that assigns labels acceptably can still produce unreliable probabilities.
For risk-style use, probabilities must be calibrated. This module computes:

  * multiclass Expected Calibration Error (ECE) on the max-probability, and
  * per-class reliability curve points (mean predicted vs empirical frequency),

on the final test set, alongside the proper scores in crashsev.metrics (log loss, Brier,
ranked probability score). Calibration is fit on development data only when applied.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Top-label ECE: bin by predicted confidence, average |confidence - accuracy|."""
    y_true = np.asarray(y_true, dtype=int)
    P = np.asarray(proba, dtype="float64")
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - acc)
    return float(ece)


def reliability_points(y_true, proba, class_index: int, n_bins: int = 10) -> List[Dict[str, float]]:
    """Per-class reliability: for the predicted probability of ``class_index``, mean
    predicted probability vs empirical frequency of that class, per confidence bin."""
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype="float64")[:, class_index]
    y_bin = (y_true == class_index).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    pts = []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (p > lo) & (p <= hi) if b > 0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        pts.append({
            "bin_low": float(lo),
            "bin_high": float(hi),
            "mean_predicted": float(p[mask].mean()),
            "empirical_frequency": float(y_bin[mask].mean()),
            "count": int(mask.sum()),
        })
    return pts
