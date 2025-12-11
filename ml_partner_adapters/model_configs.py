"""
Canonical default hyperparameters for Peyton's ML models, extracted from the
original training scripts under ``peyton_original/Machine Learning Models``.

These are intentionally kept as plain dicts so that the Django engine can
reference them without importing the training scripts themselves.
"""

from __future__ import annotations

from typing import Any, Dict

# Decision Tree defaults from DecisionTree/decisionTree.py
DECISION_TREE_BASE_PARAMS: Dict[str, Any] = {
    "random_state": 42,
    "criterion": "entropy",
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
}

# Explainable Boosting Machine defaults from EBM/EBM.py
EBM_BASE_PARAMS: Dict[str, Any] = {
    "interactions": 0,
    "max_rounds": 5000,
    "learning_rate": 0.01,
    "max_bins": 256,
    "max_leaves": 3,
    "random_state": 42,
    "n_jobs": -1,
}

# XGBoost defaults from XGBoost/XGB.py
# Updated for newer XGBoost GPU usage:
#   - tree_method='hist'  (GPU-friendly histogram algorithm)
#   - device='cuda'       (run training on the GPU)
#
# NOTE:
# We intentionally **do not** set the legacy "predictor" parameter here.
# In modern XGBoost (2.x+ / 3.x), the predictor is inferred from "device",
# and forcing "gpu_predictor" can lead to
# "Parameters: {\"predictor\"} are not used" warnings while providing no
# additional benefit.
XGB_BASE_PARAMS: Dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0.0,
    "reg_lambda": 1.0,
    "eval_metric": "mlogloss",
    "objective": "multi:softmax",
    "random_state": 42,
    "n_jobs": -1,
    # GPU configuration for XGBoost 3.x
    "tree_method": "hist",  # hist algorithm; GPU is selected via device='cuda'
    "device": "cuda",       # run training on the GPU
}

# Multilevel Random Forest defaults from MultilevelRandomForest/MRF.py
MRF_BASE_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
    "thresholds": [0.5, 0.2],
}

__all__ = [
    "DECISION_TREE_BASE_PARAMS",
    "EBM_BASE_PARAMS",
    "XGB_BASE_PARAMS",
    "MRF_BASE_PARAMS",
]
