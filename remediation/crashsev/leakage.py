"""
crashsev.leakage — pre-specified feature-availability denylist (METH-001).

The original project detects leakage with ``find_near_perfect_predictors(X, y)`` run on
the *full* dataset before the split — i.e. supervised feature selection on the test set,
the exact leakage it claims to prevent. Name heuristics (``suggest_by_name``) also cannot
establish *when* a feature becomes available.

Correct approach:

1. A deterministic, reviewed denylist derived from the feature-availability ledger
   (``schema.FEATURE_LEDGER``). Outcome-derived columns are excluded regardless of their
   statistical association with the target. This requires no access to ``y`` and is applied
   identically to every partition.
2. An OPTIONAL statistical sentinel used only as a *diagnostic*, and only ever fit inside a
   development fold (never on the final test), to catch columns the ledger missed. Its
   findings are surfaced for human review; they do not silently alter the model.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from . import schema


def denylist_columns(columns: Iterable[str], use_case: str = "post_crash_triage") -> Set[str]:
    """Columns to drop because they are outcome-derived / identifiers / unknown-timing.

    Deterministic and target-free: identical on train, development, and final test.
    Unknown-timing columns (not in the ledger) are excluded by default and must be
    reviewed and added to the ledger before use (fail-closed).
    """
    classified = schema.classify_columns(list(columns), use_case=use_case)
    return set(classified["prohibited"]) | set(classified["unknown_timing"])


def allowed_feature_columns(columns: Iterable[str], use_case: str = "post_crash_triage") -> List[str]:
    """The complement of the denylist among the provided columns, preserving order."""
    deny = denylist_columns(columns, use_case=use_case)
    target = schema.TARGET_COLUMN.strip().lower()
    return [c for c in columns if c not in deny and str(c).strip().lower() != target]


def near_perfect_sentinel(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    min_accuracy: float = 0.98,
    max_unique: int = 50,
) -> List[Tuple[str, float]]:
    """
    DIAGNOSTIC ONLY. Single-column majority-vote leakage probe, fit on TRAINING rows only.

    For each low-cardinality column, map each value to its training-majority class and
    measure resubstitution accuracy on the training fold. Columns exceeding ``min_accuracy``
    are *flagged for human review* — they are not automatically dropped, because a strong
    legitimate predictor and an outcome-derived leak look identical to this probe. The
    authoritative control is the ledger denylist.

    Threshold defaults to 0.98 (strict) rather than the original 0.90, to reduce false
    positives on legitimately strong predictors.
    """
    flagged: List[Tuple[str, float]] = []
    y = pd.Series(np.asarray(y_train)).reset_index(drop=True)
    for col in X_train.columns:
        s = X_train[col].reset_index(drop=True)
        if s.nunique(dropna=True) > max_unique:
            continue
        df = pd.DataFrame({"f": s.astype("object"), "y": y})
        # majority target per feature value (train only)
        maj = df.groupby("f")["y"].agg(lambda v: v.value_counts().idxmax())
        yhat = df["f"].map(maj)
        acc = float((yhat == df["y"]).mean())
        if acc >= min_accuracy:
            flagged.append((str(col), acc))
    return sorted(flagged, key=lambda t: -t[1])
