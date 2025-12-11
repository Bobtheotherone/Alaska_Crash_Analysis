"""
Adapter layer between the Alaska crash analysis backend and the frozen
Peyton ML / cleaning snapshot under ``peyton_original``.

This package is intentionally thin and non-interactive. It exposes safe
wrappers that can be called from web/worker code, without having to import
the Peyton modules directly.
"""

from __future__ import annotations

from .config_bridge import UNKNOWN_THRESHOLD, YES_NO_THRESHOLD, UNKNOWN_STRINGS
from .unknown_bridge import discover_unknown_placeholders_web
from .severity_mapping_bridge import find_severity_mapping_noninteractive
from .leakage_bridge import (
    find_leakage_columns_noninteractive,
    warn_suspicious_importances,
)
from .model_configs import (
    DECISION_TREE_BASE_PARAMS,
    EBM_BASE_PARAMS,
    XGB_BASE_PARAMS,
    MRF_BASE_PARAMS,
)
from .mrf_ordinal import MultiLevelRandomForestClassifier

__all__ = [
    "UNKNOWN_THRESHOLD",
    "YES_NO_THRESHOLD",
    "UNKNOWN_STRINGS",
    "discover_unknown_placeholders_web",
    "find_severity_mapping_noninteractive",
    "find_leakage_columns_noninteractive",
    "warn_suspicious_importances",
    "DECISION_TREE_BASE_PARAMS",
    "EBM_BASE_PARAMS",
    "XGB_BASE_PARAMS",
    "MRF_BASE_PARAMS",
    "MultiLevelRandomForestClassifier",
]
