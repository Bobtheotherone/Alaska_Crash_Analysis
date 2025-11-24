
"""
Model training utilities for the analysis app.

This module contains importable training functions for the different
"official" models supported by the API.  Each training function follows
the same high–level contract:

    train_xxx(df, cleaning_params, model_params) -> dict

Where the returned dict contains at least:

    {
        "model": <fitted model instance>,
        "metrics": {...},               # JSON-serialisable metrics
        "feature_importances": {...},   # mapping feature -> importance
        "cleaning_meta": {...},         # metadata from build_ml_ready_dataset
        "model_params": {...},          # fully-resolved model params
    }

The worker process is responsible for deciding which parts of this
dictionary are persisted to the database (e.g. metrics, top importances,
leakage warnings, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:  # Optional dependency, but expected to be installed via requirements.txt
    from interpret.glassbox import ExplainableBoostingClassifier
except Exception:  # pragma: no cover - handled at runtime
    ExplainableBoostingClassifier = None  # type: ignore

try:  # Optional dependency, but expected to be installed via requirements.txt
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - handled at runtime
    XGBClassifier = None  # type: ignore

from .cleaning import build_ml_ready_dataset, warn_suspicious_importances

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_cleaning_params(cleaning_params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Normalise cleaning parameters into kwargs for build_ml_ready_dataset.

    We deliberately accept a very permissive shape here because the API
    exposes "parameters.cleaning" as a free-form JSON object.  Unknown
    keys are simply ignored.
    """
    params = dict(cleaning_params or {})

    return {
        "severity_col": params.get("severity_col"),
        "base_unknowns": params.get("base_unknowns"),
        "unknown_threshold": params.get("unknown_threshold"),
        "yes_no_threshold": params.get("yes_no_threshold"),
        "columns_to_drop": params.get("columns_to_drop"),
    }


def _evaluate_classifier(
    y_true_train: pd.Series,
    y_pred_train: np.ndarray,
    y_true_test: Optional[pd.Series] = None,
    y_pred_test: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute a small, JSON-safe set of classification metrics.

    If no test set is supplied, we compute "overall_*" metrics only.
    """
    metrics: Dict[str, Any] = {}

    # Train / overall metrics
    metrics["train_accuracy"] = float(accuracy_score(y_true_train, y_pred_train))
    metrics["train_f1_macro"] = float(f1_score(y_true_train, y_pred_train, average="macro"))
    metrics["train_balanced_accuracy"] = float(
        balanced_accuracy_score(y_true_train, y_pred_train)
    )
    metrics["train_classification_report"] = classification_report(
        y_true_train, y_pred_train, output_dict=True
    )

    # Test metrics (optional)
    if y_true_test is not None and y_pred_test is not None:
        metrics["test_accuracy"] = float(accuracy_score(y_true_test, y_pred_test))
        metrics["test_f1_macro"] = float(
            f1_score(y_true_test, y_pred_test, average="macro")
        )
        metrics["test_balanced_accuracy"] = float(
            balanced_accuracy_score(y_true_test, y_pred_test)
        )
        metrics["test_classification_report"] = classification_report(
            y_true_test, y_pred_test, output_dict=True
        )

    # Basic label distribution info (for debugging / sanity checks)
    metrics["label_distribution"] = (
        y_true_train.value_counts(normalize=True).sort_index().to_dict()
    )
    if y_true_test is not None:
        metrics["label_distribution_test"] = (
            y_true_test.value_counts(normalize=True).sort_index().to_dict()
        )

    return metrics


def _extract_feature_importances(
    model: Any, feature_names: Sequence[str]
) -> Dict[str, float]:
    """
    Extract feature importances for a fitted model, falling back to zeros
    if the model does not expose importances.

    All values are normalised to sum to 1.0 to make comparison between
    models easier.
    """
    importances: Optional[np.ndarray] = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(getattr(model, "feature_importances_"), dtype="float64")
    elif hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_"), dtype="float64")
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            importances = np.mean(np.abs(coef), axis=0)
    elif hasattr(model, "term_importances_"):
        # Interpret's ExplainableBoostingClassifier exposes term_importances_.
        importances = np.asarray(getattr(model, "term_importances_"), dtype="float64")

    if importances is None:
        # Graceful fallback: no notion of feature importance for this model.
        logger.warning(
            "Model of type %s does not expose feature importances; "
            "falling back to zeros.",
            type(model).__name__,
        )
        return {name: 0.0 for name in feature_names}

    if importances.shape[0] != len(feature_names):
        # Extremely defensive: shape mismatch usually indicates something
        # odd in how the model represents terms.
        logger.warning(
            "Feature importance vector has length %s but there are %s feature names. "
            "Truncating/padding with zeros.",
            importances.shape[0],
            len(feature_names),
        )
        # Truncate or pad with zeros to match.
        if importances.shape[0] > len(feature_names):
            importances = importances[: len(feature_names)]
        else:
            pad_width = len(feature_names) - importances.shape[0]
            importances = np.pad(importances, (0, pad_width), mode="constant")

    total = float(importances.sum())
    if not np.isfinite(total) or total <= 0.0:
        total = 1.0

    normalised = importances / total
    return {name: float(val) for name, val in zip(feature_names, normalised)}


def _train_test_split_if_possible(
    X: pd.DataFrame, y: pd.Series, *, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a robust train/test split that gracefully falls back to using
    the full dataset as "train" if we don't have enough rows or labels.
    """
    n_samples = len(X)
    n_classes = y.nunique(dropna=True)

    if n_samples < 10 or n_classes < 2:
        # Too small or degenerate – just return the full dataset as train
        # and an empty test split.
        logger.warning(
            "Dataset too small or not enough label variety for a proper "
            "train/test split (n_samples=%s, n_classes=%s). "
            "Using the full dataset as train only.",
            n_samples,
            n_classes,
        )
        return X, pd.DataFrame(columns=X.columns), y, pd.Series(dtype=y.dtype)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=y if n_classes > 1 else None,
        )
    except ValueError as exc:
        logger.warning(
            "Falling back to no train/test split due to ValueError: %s", exc
        )
        return X, pd.DataFrame(columns=X.columns), y, pd.Series(dtype=y.dtype)

    return X_train, X_test, y_train, y_test


def _base_result(
    model: Any,
    feature_names: Sequence[str],
    cleaning_meta: Mapping[str, Any],
    model_params: Mapping[str, Any],
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Assemble the standard result dictionary for all training functions."""
    # Predictions for metrics
    y_pred_train = model.predict(X_train)
    y_pred_test = None
    if X_test is not None and not X_test.empty and y_test is not None and len(y_test) > 0:
        y_pred_test = model.predict(X_test)

    metrics = _evaluate_classifier(y_train, y_pred_train, y_test, y_pred_test)

    feature_importances = _extract_feature_importances(model, feature_names)

    # Detect suspicious / potentially leaky features.
    suspicious_features: List[str] = warn_suspicious_importances(
        list(feature_importances.keys()),
        list(feature_importances.values()),
    )

    leakage_columns: List[str] = list(
        cleaning_meta.get("leakage_columns", [])  # type: ignore[call-arg]
    )

    return {
        "model": model,
        "metrics": metrics,
        "feature_importances": feature_importances,
        "cleaning_meta": dict(cleaning_meta),
        "model_params": dict(model_params),
        "leakage_warnings": {
            "leakage_columns": leakage_columns,
            "suspicious_features": suspicious_features,
        },
    }


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------


def train_crash_severity_decision_tree(
    df: pd.DataFrame,
    cleaning_params: Optional[Mapping[str, Any]] = None,
    model_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train the baseline crash severity risk model using a DecisionTreeClassifier.

    This function is intended to back the ``crash_severity_risk_v1`` entry
    in the model registry.
    """
    cleaning_kwargs = _ensure_cleaning_params(cleaning_params)
    X, y, cleaning_meta = build_ml_ready_dataset(df, **cleaning_kwargs)

    default_model_params: Dict[str, Any] = {
        "criterion": "gini",
        "max_depth": 6,
        "min_samples_leaf": 50,
        "random_state": 42,
    }
    final_model_params = {**default_model_params, **(model_params or {})}

    model = DecisionTreeClassifier(**final_model_params)

    X_train, X_test, y_train, y_test = _train_test_split_if_possible(X, y)
    model.fit(X_train, y_train)

    return _base_result(
        model=model,
        feature_names=list(X.columns),
        cleaning_meta=cleaning_meta,
        model_params=final_model_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test if not X_test.empty else None,
        y_test=y_test if len(y_test) > 0 else None,
    )


def train_ebm(
    df: pd.DataFrame,
    cleaning_params: Optional[Mapping[str, Any]] = None,
    model_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train an Explainable Boosting Machine (EBM) classifier.

    Backs the ``ebm_v1`` entry in the registry.
    """
    if ExplainableBoostingClassifier is None:
        raise RuntimeError(
            "interpret is not installed or ExplainableBoostingClassifier "
            "is unavailable. Ensure the 'interpret' dependency is installed."
        )

    cleaning_kwargs = _ensure_cleaning_params(cleaning_params)
    X, y, cleaning_meta = build_ml_ready_dataset(df, **cleaning_kwargs)

    default_model_params: Dict[str, Any] = {
        "interactions": 10,
        "max_bins": 256,
        "outer_bags": 8,
        "inner_bags": 0,
        "learning_rate": 0.01,
        "random_state": 42,
    }
    final_model_params = {**default_model_params, **(model_params or {})}

    model = ExplainableBoostingClassifier(**final_model_params)

    X_train, X_test, y_train, y_test = _train_test_split_if_possible(X, y)
    model.fit(X_train, y_train)

    return _base_result(
        model=model,
        feature_names=list(X.columns),
        cleaning_meta=cleaning_meta,
        model_params=final_model_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test if not X_test.empty else None,
        y_test=y_test if len(y_test) > 0 else None,
    )


def train_mrf(
    df: pd.DataFrame,
    cleaning_params: Optional[Mapping[str, Any]] = None,
    model_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train a (monotonic) Random Forest classifier.

    The original ML repo refers to this family as "MRF".  Here we
    implement it using sklearn's RandomForestClassifier.
    """
    cleaning_kwargs = _ensure_cleaning_params(cleaning_params)
    X, y, cleaning_meta = build_ml_ready_dataset(df, **cleaning_kwargs)

    default_model_params: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 10,
        "n_jobs": -1,
        "random_state": 42,
        "class_weight": "balanced",
    }
    final_model_params = {**default_model_params, **(model_params or {})}

    model = RandomForestClassifier(**final_model_params)

    X_train, X_test, y_train, y_test = _train_test_split_if_possible(X, y)
    model.fit(X_train, y_train)

    return _base_result(
        model=model,
        feature_names=list(X.columns),
        cleaning_meta=cleaning_meta,
        model_params=final_model_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test if not X_test.empty else None,
        y_test=y_test if len(y_test) > 0 else None,
    )


def train_xgb(
    df: pd.DataFrame,
    cleaning_params: Optional[Mapping[str, Any]] = None,
    model_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train an XGBoost classifier for crash severity risk.

    Backs the ``xgb_v1`` entry in the registry.
    """
    if XGBClassifier is None:
        raise RuntimeError(
            "xgboost is not installed or XGBClassifier is unavailable. "
            "Ensure the 'xgboost' dependency is installed."
        )

    cleaning_kwargs = _ensure_cleaning_params(cleaning_params)
    X, y, cleaning_meta = build_ml_ready_dataset(df, **cleaning_kwargs)

    default_model_params: Dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    final_model_params = {**default_model_params, **(model_params or {})}

    model = XGBClassifier(**final_model_params)

    X_train, X_test, y_train, y_test = _train_test_split_if_possible(X, y)
    model.fit(X_train, y_train)

    return _base_result(
        model=model,
        feature_names=list(X.columns),
        cleaning_meta=cleaning_meta,
        model_params=final_model_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test if not X_test.empty else None,
        y_test=y_test if len(y_test) > 0 else None,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


TrainFunc = Callable[[pd.DataFrame, Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]], Dict[str, Any]]


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a supported model."""

    name: str
    description: str
    trainer: TrainFunc
    default_model_params: Mapping[str, Any]


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "crash_severity_risk_v1": ModelSpec(
        name="crash_severity_risk_v1",
        description="Baseline crash severity risk model (DecisionTree v1).",
        trainer=train_crash_severity_decision_tree,
        default_model_params={
            "criterion": "gini",
            "max_depth": 6,
            "min_samples_leaf": 50,
            "random_state": 42,
        },
    ),
    "ebm_v1": ModelSpec(
        name="ebm_v1",
        description="Explainable Boosting Machine (EBM) v1.",
        trainer=train_ebm,
        default_model_params={
            "interactions": 10,
            "max_bins": 256,
            "outer_bags": 8,
            "inner_bags": 0,
            "learning_rate": 0.01,
            "random_state": 42,
        },
    ),
    "mrf_v1": ModelSpec(
        name="mrf_v1",
        description="Monotonic Random Forest (MRF) v1.",
        trainer=train_mrf,
        default_model_params={
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 10,
            "n_jobs": -1,
            "random_state": 42,
            "class_weight": "balanced",
        },
    ),
    "xgb_v1": ModelSpec(
        name="xgb_v1",
        description="XGBoost crash risk model v1.",
        trainer=train_xgb,
        default_model_params={
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
}
