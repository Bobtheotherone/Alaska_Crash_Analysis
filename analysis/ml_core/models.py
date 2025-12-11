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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  # still imported; may be used elsewhere
from sklearn.utils.class_weight import compute_class_weight

from ml_partner_adapters import model_configs as partner_model_configs
from ml_partner_adapters.mrf_ordinal import MultiLevelRandomForestClassifier

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
        # Peyton-style manual leakage list; passed through to build_ml_ready_dataset
        "leakage_columns": params.get("leakage_columns"),
    }


def _train_test_split_if_possible(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.25
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Utility wrapper around train_test_split with defensive defaults.

    When there are too few rows to sensibly split, returns (X, X, y, y)
    so that downstream code can still train and evaluate without
    additional branching.
    """
    if len(X) < 10:
        # Too few rows; just train+test on the same data.
        return X, X, y, y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None,
    )
    return X_train, X_test, y_train, y_test


def _compute_classification_metrics(
    y_true_train: pd.Series,
    y_pred_train: np.ndarray,
    y_true_test: Optional[pd.Series],
    y_pred_test: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    Compute a standard set of classification metrics for train/test splits.

    Returns a dict with:

        {
            "train": {...},
            "test": {...},  # optional if no test split
        }

    Additionally, this function populates flattened keys such as
    ``train_classification_report`` and ``test_classification_report`` for
    compatibility with the frontend, which expects those top-level fields.
    """
    metrics: Dict[str, Any] = {}

    def _per_split(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        rep = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        return {
            "accuracy": float(acc),
            "classification_report": rep,
            "confusion_matrix": cm.tolist(),
        }

    # Nested train / test blocks used by Python code
    metrics["train"] = _per_split(y_true_train, y_pred_train)

    if y_true_test is not None and y_pred_test is not None:
        metrics["test"] = _per_split(y_true_test, y_pred_test)
    else:
        metrics["test"] = None

    # For convenience, include the class distribution for train/test.
    metrics["class_distribution"] = {
        "train": y_true_train.value_counts(normalize=True).sort_index().to_dict(),
    }

    if y_true_test is not None:
        metrics["class_distribution"]["test"] = (
            y_true_test.value_counts(normalize=True).sort_index().to_dict()
        )

    # ------------------------------------------------------------------
    # FLATTENED KEYS for the React UI (back-compat shim)
    # ------------------------------------------------------------------
    train_block = metrics.get("train") or {}
    test_block = metrics.get("test") or {}

    if "classification_report" in train_block:
        metrics.setdefault(
            "train_classification_report", train_block["classification_report"]
        )
    if "confusion_matrix" in train_block:
        metrics.setdefault("train_confusion_matrix", train_block["confusion_matrix"])
    if "accuracy" in train_block:
        metrics.setdefault("train_accuracy", train_block["accuracy"])

    if isinstance(test_block, dict) and test_block:
        if "classification_report" in test_block:
            metrics.setdefault(
                "test_classification_report", test_block["classification_report"]
            )
        if "confusion_matrix" in test_block:
            metrics.setdefault("test_confusion_matrix", test_block["confusion_matrix"])
        if "accuracy" in test_block:
            metrics.setdefault("test_accuracy", test_block["accuracy"])

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
    elif hasattr(model, "get_feature_importances"):
        # Fallback for model types that expose a custom getter.
        try:
            raw = model.get_feature_importances()
            importances = np.asarray(raw, dtype="float64")
        except Exception:  # noqa: BLE001
            importances = None

    if importances is None:
        # Fallback: equal weights.
        importances = np.ones(len(feature_names), dtype="float64")

    # Normalise to sum to 1.0; guard against all-zero arrays.
    total = float(np.sum(importances))
    if total <= 0:
        importances = np.ones_like(importances, dtype="float64")
        total = float(np.sum(importances))

    norm = importances / total
    return {str(name): float(val) for name, val in zip(feature_names, norm)}


def _base_result(
    model: Any,
    feature_names: Sequence[str],
    cleaning_meta: Mapping[str, Any],
    model_params: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame],
    y_test: Optional[pd.Series],
) -> Dict[str, Any]:
    """
    Helper to assemble the standard result dict from a fitted model and
    its training data.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test) if X_test is not None else None

    metrics = _compute_classification_metrics(
        y_true_train=y_train,
        y_pred_train=y_pred_train,
        y_true_test=y_test,
        y_pred_test=y_pred_test,
    )

    feature_importances = _extract_feature_importances(model, feature_names)

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
    Train a DecisionTreeClassifier for crash severity risk.

    Backs the ``crash_severity_risk_v1`` entry in the registry.
    """
    cleaning_kwargs = _ensure_cleaning_params(cleaning_params)
    X, y, cleaning_meta = build_ml_ready_dataset(df, **cleaning_kwargs)

    default_model_params: Dict[str, Any] = dict(
        partner_model_configs.DECISION_TREE_BASE_PARAMS
    )
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


def train_mrf(
    df: pd.DataFrame,
    cleaning_params: Optional[Mapping[str, Any]] = None,
    model_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train a (monotonic) Random Forest classifier.

    The original ML repo refers to this family as "MRF".  Here we
    implement it using Peyton's MultiLevelRandomForestClassifier.
    """
    cleaning_kwargs = _ensure_cleaning_params(cleaning_params)
    X, y, cleaning_meta = build_ml_ready_dataset(df, **cleaning_kwargs)

    # Default hyperparameters mirror Peyton's multilevel RF configuration
    # via the ml_partner_adapters.model_configs module.
    default_model_params: Dict[str, Any] = dict(
        partner_model_configs.MRF_BASE_PARAMS
    )
    final_model_params = {**default_model_params, **(model_params or {})}

    model = MultiLevelRandomForestClassifier(**final_model_params)

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
    Train an Explainable Boosting Machine classifier for crash severity risk.

    Backs the ``ebm_v1`` entry in the registry.
    """
    if ExplainableBoostingClassifier is None:
        raise RuntimeError(
            "InterpretML is not installed or ExplainableBoostingClassifier is unavailable. "
            "Ensure the 'interpret' dependency is installed."
        )

    cleaning_kwargs = _ensure_cleaning_params(cleaning_params)
    X, y, cleaning_meta = build_ml_ready_dataset(df, **cleaning_kwargs)

    default_model_params: Dict[str, Any] = dict(
        partner_model_configs.EBM_BASE_PARAMS
    )
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


def train_xgb(
    df: pd.DataFrame,
    cleaning_params: Optional[Mapping[str, Any]] = None,
    model_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train an XGBoost classifier for crash severity risk.

    Backs the ``xgb_v1`` entry in the registry.

    This implementation aligns with Peyton's XGBoost training in the
    aspects that matter most for behaviour:

      * Use Peyton's canonical default hyperparameters from
        ``ml_partner_adapters.model_configs.XGB_BASE_PARAMS``.
      * Compute class-balanced sample weights (so the model pays more
        attention to serious crashes).
      * Prefer GPU acceleration (tree_method='gpu_hist', device='cuda')
        and fall back to a CPU configuration if that fails.

    We intentionally **do not** run the full RandomizedSearchCV here,
    because 20x3-fold CV is too heavy for the web worker / dev server
    environment and can exhaust system resources on Windows.
    """
    if XGBClassifier is None:
        raise RuntimeError(
            "xgboost is not installed or XGBClassifier is unavailable. "
            "Ensure the 'xgboost' dependency is installed."
        )

    # Normalise cleaning params and build the ML-ready dataset.
    cleaning_kwargs = _ensure_cleaning_params(cleaning_params)
    X, y, cleaning_meta = build_ml_ready_dataset(df, **cleaning_kwargs)

    # Default hyperparameters mirror Peyton's XGBoost configuration
    # via the ml_partner_adapters.model_configs module.
    default_model_params: Dict[str, Any] = dict(
        partner_model_configs.XGB_BASE_PARAMS
    )
    base_model_params: Dict[str, Any] = {
        **default_model_params,
        **(model_params or {}),
    }

    # Drop legacy 'predictor' parameter if present; modern XGBoost
    # selects the appropriate predictor based on 'device'.
    base_model_params.pop("predictor", None)

    # For alignment with Peyton's scripts, use a 20% test set.
    X_train, X_test, y_train, y_test = _train_test_split_if_possible(
        X, y, test_size=0.20
    )

    # ------------------------------------------------------------------
    # Class-balanced sample weights (Peyton-style)
    # ------------------------------------------------------------------
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    weight_map = dict(zip(classes, class_weights))

    # Optional extra boost for most severe class (left commented, as in Peyton's script).
    # if 2 in weight_map:
    #     weight_map[2] = weight_map[2] * 1.5

    sample_weight = y_train.map(weight_map)

    # ------------------------------------------------------------------
    # 1) Prefer GPU: gpu_hist + device='cuda' where supported
    # ------------------------------------------------------------------
    gpu_params: Dict[str, Any] = dict(base_model_params)

    # Only override tree_method if the caller hasn't explicitly set it.
    tree_method = str(gpu_params.get("tree_method", "")).lower()
    if not tree_method or tree_method == "auto":
        gpu_params["tree_method"] = "gpu_hist"

    # Newer xgboost versions accept a 'device' param; this will be ignored or
    # error on older builds, which we handle in the fallback branch.
    gpu_params.setdefault("device", "cuda")

    model: XGBClassifier
    final_model_params: Dict[str, Any]

    try:
        model = XGBClassifier(**gpu_params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        final_model_params = gpu_params
        logger.info(
            "Trained XGBoost with GPU acceleration (tree_method=%r, device=%r).",
            final_model_params.get("tree_method"),
            final_model_params.get("device"),
        )
    except Exception as exc:
        # ------------------------------------------------------------------
        # 2) Fallback: CPU configuration (hist / auto predictor)
        # ------------------------------------------------------------------
        logger.warning(
            "GPU-accelerated XGBoost unavailable (%r); falling back to CPU configuration.",
            exc,
        )

        cpu_params: Dict[str, Any] = dict(base_model_params)

        # If caller or GPU-params requested a GPU-only tree_method, revert to a CPU-safe default.
        if str(cpu_params.get("tree_method", "")).lower().startswith("gpu"):
            cpu_params["tree_method"] = "hist"

        # Remove GPU-specific knobs that might confuse a CPU-only build.
        cpu_params.pop("predictor", None)

        device_val = str(cpu_params.get("device", "")).lower()
        if device_val.startswith("cuda"):
            cpu_params.pop("device", None)

        model = XGBClassifier(**cpu_params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        final_model_params = cpu_params
        logger.info(
            "Trained XGBoost on CPU (tree_method=%r).",
            final_model_params.get("tree_method"),
        )

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


TrainFunc = Callable[
    [pd.DataFrame, Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]],
    Dict[str, Any],
]


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
        description="Decision tree crash risk model v1.",
        trainer=train_crash_severity_decision_tree,
        default_model_params=partner_model_configs.DECISION_TREE_BASE_PARAMS,
    ),
    "ebm_v1": ModelSpec(
        name="ebm_v1",
        description="Explainable Boosting Machine (EBM) crash risk model v1.",
        trainer=train_ebm,
        default_model_params=partner_model_configs.EBM_BASE_PARAMS,
    ),
    "mrf_v1": ModelSpec(
        name="mrf_v1",
        description="Monotonic Random Forest (MRF) v1.",
        trainer=train_mrf,
        default_model_params=partner_model_configs.MRF_BASE_PARAMS,
    ),
    "xgb_v1": ModelSpec(
        name="xgb_v1",
        description="XGBoost crash risk model v1.",
        trainer=train_xgb,
        default_model_params=partner_model_configs.XGB_BASE_PARAMS,
    ),
}
