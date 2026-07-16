"""
crashsev.preprocessing — train-only fit/transform pipeline (VAL-001, METH-003).

Every learned transformation lives inside an sklearn ``Pipeline`` / ``ColumnTransformer``
so that ``fit`` sees only training rows and ``transform`` applies frozen state to
development/final-test rows. This makes preprocessing leakage structurally impossible:
imputation medians, one-hot vocabularies, and infrequent-category buckets are all learned
from training data only.

Contrast with the integrated web pipeline (``analysis/ml_core/cleaning.py``), which calls
``pd.get_dummies`` and fills medians on the full dataframe before the split.

Unseen categories at inference are handled deterministically (``handle_unknown="ignore"``
plus an infrequent bucket), and feature names are recoverable for interpretation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


MIN_TRAIN_ROWS = 50  # below this, fail closed rather than evaluate on training data


class InsufficientDataError(RuntimeError):
    """Raised when there are too few rows/groups to train or evaluate honestly."""


def split_feature_types(
    X: pd.DataFrame, categorical_max_cardinality: int = 100
) -> Tuple[List[str], List[str], List[str]]:
    """
    Partition columns into (numeric, categorical, dropped_high_cardinality).

    High-cardinality categoricals (> ``categorical_max_cardinality`` distinct values, e.g.
    free-text street names, ids) are dropped: one-hot encoding them explodes dimensionality
    and mostly encodes identity. This is a *pre-specified structural* rule (not target-aware)
    and is applied via the training-set cardinality only (see build_preprocessor caller).
    """
    numeric, categorical, dropped = [], [], []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        else:
            if s.nunique(dropna=True) > categorical_max_cardinality:
                dropped.append(col)
            else:
                categorical.append(col)
    return numeric, categorical, dropped


def build_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    scale_numeric: bool = False,
    min_frequency: float = 0.01,
    sparse: bool = True,
) -> ColumnTransformer:
    """
    ColumnTransformer:
      * numeric   -> median imputation (+ optional standardisation for linear models)
      * categorical -> most-frequent imputation -> string cast -> one-hot with an
        infrequent bucket and unknown-category ignoring.

    All state is fit on training rows only when this transformer is the first step of a
    Pipeline that is ``fit`` on X_train.

    When ``sparse`` is True the one-hot block stays sparse and (to keep the whole matrix
    sparse) numeric standardisation is centre-free (``with_mean=False``), avoiding the dense
    densification of a wide one-hot design on real-scale data (AA2-016).
    """
    num_steps: List[Tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler(with_mean=not sparse)))
    numeric_pipe = Pipeline(num_steps)

    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("to_str", FunctionTransformer(_as_str, feature_names_out="one-to-one")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    dtype=np.float32,
                    sparse_output=sparse,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(numeric_cols)),
            ("cat", categorical_pipe, list(categorical_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=0.3 if sparse else 0.0,
    )


def _as_str(X):
    """Cast every cell to str so the one-hot encoder sees consistent dtypes."""
    return pd.DataFrame(X).astype(str).to_numpy()


def check_sufficient(X_train: pd.DataFrame, y_train: pd.Series, min_rows: int = MIN_TRAIN_ROWS) -> None:
    """Fail closed on too-small training data or a missing class (removes the
    ``len(X) < 10 -> train == test`` fallback in the original code)."""
    if len(X_train) < min_rows:
        raise InsufficientDataError(
            f"Only {len(X_train)} training rows (< {min_rows}); refusing to train. "
            f"The original pipeline would silently train and test on the same rows."
        )
    if pd.Series(y_train).nunique() < 2:
        raise InsufficientDataError("Training target has < 2 classes; cannot train a classifier.")
