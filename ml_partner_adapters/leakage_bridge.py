"""
Non-interactive wrappers around Peyton's data-leakage utilities.
"""

from __future__ import annotations

from typing import Hashable, Sequence, Set

import pandas as pd

from peyton_original import leakage_column_utils as peyton_leakage


def find_leakage_columns_noninteractive(
    X: pd.DataFrame,
    y,
    use_near_perfect_check: bool = True,
    min_accuracy: float = 0.9,
    max_unique: int = 50,
) -> Set[Hashable]:
    """
    Heuristic data-leakage detection without any interactive prompts.

    This function mirrors Peyton's :func:`find_leakage_columns` logic,
    but skips the CLI-based selection step. Instead it returns the union
    of:

    * Name-based suggestions from :func:`suggest_by_name`, and
    * Optional near-perfect predictors from
      :func:`find_near_perfect_predictors`.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target vector, passed through to
        :func:`find_near_perfect_predictors` when enabled.
    use_near_perfect_check:
        Whether to run the near-perfect predictor search.
    min_accuracy:
        Minimum predictive accuracy required for a column to be
        considered a near-perfect predictor.
    max_unique:
        Columns with more than this number of unique values are skipped
        in the near-perfect predictor step.

    Returns
    -------
    Set[Hashable]
        Set of column names that should be treated as leakage.
    """
    name_suggestions = peyton_leakage.suggest_by_name(X.columns)

    near_perfect_cols: Set[Hashable] = set()
    if use_near_perfect_check:
        suspicious = peyton_leakage.find_near_perfect_predictors(
            X,
            y,
            min_accuracy=min_accuracy,
            max_unique=max_unique,
        )
        # ``find_near_perfect_predictors`` returns a list of
        # (column_name, accuracy) tuples.
        near_perfect_cols = {col for col, _ in suspicious}

    return set(name_suggestions) | near_perfect_cols


def warn_suspicious_importances(
    feature_names: Sequence[str],
    importances: Sequence[float],
    importance_threshold: float = 0.2,
    dominance_ratio: float = 2.0,
    top_n: int = 10,
) -> Sequence[str]:
    """
    Thin wrapper around Peyton's ``warn_suspicious_importances``.

    The underlying implementation prints warnings to stdout but does not
    prompt the user. This helper simply forwards the call and returns
    the list of suspicious feature names that Peyton's function already
    computes.
    """
    return peyton_leakage.warn_suspicious_importances(
        feature_names,
        importances,
        importance_threshold=importance_threshold,
        dominance_ratio=dominance_ratio,
        top_n=top_n,
    )
