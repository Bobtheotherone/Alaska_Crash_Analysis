from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from ml_partner_adapters.config_bridge import (
    UNKNOWN_THRESHOLD,
    YES_NO_THRESHOLD,
    UNKNOWN_STRINGS as PEYTON_UNKNOWN_STRINGS,
)
from ml_partner_adapters.unknown_bridge import discover_unknown_placeholders_web
from ml_partner_adapters.severity_mapping_bridge import (
    find_severity_mapping_noninteractive,
)
from ml_partner_adapters.leakage_bridge import (
    find_leakage_columns_noninteractive,
    warn_suspicious_importances as adapter_warn_suspicious_importances,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration values (adapted from Peyton's DataCleaning/config.py)
# ---------------------------------------------------------------------------

# Canonical "unknown" tokens come directly from the partner repository via the
# config bridge.  We always work with a *lower-cased* version when matching.
DEFAULT_UNKNOWN_STRINGS: Set[str] = {str(s).strip().lower() for s in PEYTON_UNKNOWN_STRINGS}


def _normalize_str(val: Any) -> str:
    """Convert arbitrary value to a normalised, lower-cased string."""
    if val is None:
        return ""
    return str(val).strip().lower()


# ---------------------------------------------------------------------------
# Validation helpers for config values
# ---------------------------------------------------------------------------


def validate_config_values(
    unknown_threshold: float = UNKNOWN_THRESHOLD,
    yes_no_threshold: float = YES_NO_THRESHOLD,
) -> None:
    """
    Validate that thresholds coming from config are within [0, 100].

    Peyton's original configuration expresses UNKNOWN_THRESHOLD and
    YES_NO_THRESHOLD as *percentages*, so we keep that convention here.
    """

    def _ok(x: float) -> bool:
        try:
            xf = float(x)
        except Exception:
            return False
        return 0.0 <= xf <= 100.0

    if not _ok(unknown_threshold):
        raise ValueError(f"UNKNOWN_THRESHOLD must be 0–100, got {unknown_threshold!r}")
    if not _ok(yes_no_threshold):
        raise ValueError(f"YES_NO_THRESHOLD must be 0–100, got {yes_no_threshold!r}")


# ---------------------------------------------------------------------------
# Unknown placeholder discovery (delegating to ml_partner_adapters)
# ---------------------------------------------------------------------------


def discover_unknown_placeholders(
    df: pd.DataFrame,
    base_unknowns: Iterable[str] | None = None,
    *,
    min_count: int = 2,
    max_token_length: int = 80,
) -> Set[str]:
    """
    Thin wrapper around the ML partner's unknown placeholder discovery.

    Parameters
    ----------
    df:
        Input dataframe.
    base_unknowns:
        Known "unknown" tokens.  If omitted, Peyton's canonical
        UNKNOWN_STRINGS are used.
    min_count:
        Minimum frequency required for a token to be considered an
        additional "unknown".
    max_token_length:
        Safety cap on token length to ignore pathological values.

    Returns
    -------
    Set[str]
        Augmented set of unknown tokens, lower-cased.
    """
    if base_unknowns is None:
        base_unknowns = DEFAULT_UNKNOWN_STRINGS
    else:
        base_unknowns = {str(s).strip().lower() for s in base_unknowns}

    try:
        discovered = discover_unknown_placeholders_web(
            df,
            base_unknowns,
            min_freq=min_count,
            max_token_length=max_token_length,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "discover_unknown_placeholders_web failed (%r); falling back to base_unknowns only.",
            exc,
        )
        discovered = set()

    augmented: Set[str] = set(base_unknowns)
    augmented.update({str(v).strip().lower() for v in discovered})
    return augmented


# ---------------------------------------------------------------------------
# Column profiling – adapted from Peyton's Script to Clean.py
# ---------------------------------------------------------------------------


YES_TOKENS: Set[str] = {"yes", "y", "true", "t"}
NO_TOKENS: Set[str] = {"no", "n", "false", "f"}


def profile_columns(
    df: pd.DataFrame,
    unknown_strings: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Profile each column to compute the statistics used by Peyton's cleaning
    script, but in a non-interactive, reusable way.

    For every column this returns:

        {
          "total": int,
          "known": int,
          "unknown_pct": float,
          "nunique": int,
          "dominant_pct": float,
          "dominant_count": int,
          "yes_cnt": int,
          "no_cnt": int,
          "yes_pct": float,
          "no_pct": float,
          "yesno_total": int,
        }
    """
    tokens = {str(s).strip().lower() for s in unknown_strings}
    stats: Dict[str, Dict[str, Any]] = {}

    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        total = int(len(series))

        # Fast path for completely empty columns
        if total == 0:
            stats[col] = {
                "total": 0,
                "known": 0,
                "unknown_pct": 0.0,
                "nunique": 0,
                "dominant_pct": 0.0,
                "dominant_count": 0,
                "yes_cnt": 0,
                "no_cnt": 0,
                "yes_pct": 0.0,
                "no_pct": 0.0,
                "yesno_total": 0,
            }
            continue

        mask_not_null = ~series.isna()
        if not mask_not_null.any():
            # All values are NaN
            stats[col] = {
                "total": total,
                "known": 0,
                "unknown_pct": 100.0,
                "nunique": 0,
                "dominant_pct": 0.0,
                "dominant_count": 0,
                "yes_cnt": 0,
                "no_cnt": 0,
                "yes_pct": 0.0,
                "no_pct": 0.0,
                "yesno_total": 0,
            }
            continue

        # Normalise text to lower-case strings
        series_norm = series[mask_not_null].astype(str).str.strip().str.lower()

        # Exclude known-unknown tokens
        series_clean = series_norm[~series_norm.isin(tokens)]
        known = int(series_clean.size)
        unknown_pct = 100.0 * (total - known) / float(total) if total else 0.0

        if known == 0:
            stats[col] = {
                "total": total,
                "known": 0,
                "unknown_pct": round(unknown_pct, 2),
                "nunique": 0,
                "dominant_pct": 0.0,
                "dominant_count": 0,
                "yes_cnt": 0,
                "no_cnt": 0,
                "yes_pct": 0.0,
                "no_pct": 0.0,
                "yesno_total": 0,
            }
            continue

        # Dominant value and unique count
        value_counts = series_clean.value_counts(dropna=False)
        nunique = int(value_counts.size)
        dominant_count = int(value_counts.iloc[0])
        dominant_pct = 100.0 * dominant_count / float(known)

        # Yes/no statistics
        yes_cnt = sum(int(value_counts.get(y, 0)) for y in YES_TOKENS)
        no_cnt = sum(int(value_counts.get(n, 0)) for n in NO_TOKENS)
        yesno_total = yes_cnt + no_cnt
        if yesno_total > 0:
            yes_pct = 100.0 * yes_cnt / float(yesno_total)
            no_pct = 100.0 * no_cnt / float(yesno_total)
        else:
            yes_pct = 0.0
            no_pct = 0.0

        stats[col] = {
            "total": total,
            "known": known,
            "unknown_pct": round(unknown_pct, 2),
            "nunique": nunique,
            "dominant_pct": round(dominant_pct, 2),
            "dominant_count": dominant_count,
            "yes_cnt": yes_cnt,
            "no_cnt": no_cnt,
            "yes_pct": round(yes_pct, 2),
            "no_pct": round(no_pct, 2),
            "yesno_total": yesno_total,
        }

    return stats


def suggest_columns_to_drop(
    df: pd.DataFrame,
    column_stats: Mapping[str, Mapping[str, Any]],
    *,
    unknown_threshold: Optional[float] = None,
    yes_no_threshold: Optional[float] = None,
    yesno_coverage_min: float = 50.0,
    protected_columns: Iterable[str] | None = None,
) -> Set[str]:
    """
    Non-interactive replica of the Script to Clean column-drop rules.

    Parameters
    ----------
    df:
        Original dataframe (used only for row count).
    column_stats:
        Output of :func:`profile_columns`.
    unknown_threshold:
        Percentage threshold (0–100).  Columns whose unknown_pct is
        greater than or equal to this value are dropped.
    yes_no_threshold:
        Percentage threshold (0–100).  For columns that are dominated
        by yes/no-like values, if either yes_pct or no_pct falls below
        this threshold the column is dropped.
    yesno_coverage_min:
        Minimum percentage (0–100) of rows that must be covered by
        yes/no tokens before we consider dropping on yes/no imbalance.
    protected_columns:
        Columns that should never be dropped automatically (typically
        includes the severity column).

    Returns
    -------
    Set[str]
        Column names to drop.
    """
    if unknown_threshold is None:
        unknown_threshold = float(UNKNOWN_THRESHOLD)
    if yes_no_threshold is None:
        yes_no_threshold = float(YES_NO_THRESHOLD)

    validate_config_values(unknown_threshold, yes_no_threshold)

    protected: Set[str] = {c for c in (protected_columns or [])}

    to_drop: Set[str] = set()
    n_rows = float(len(df))
    yn_coverage_min = float(yesno_coverage_min)

    for col, stat in column_stats.items():
        if col in protected:
            continue

        unknown_pct = float(stat.get("unknown_pct", 0.0))
        nunique = int(stat.get("nunique", 0))
        dominant_pct = float(stat.get("dominant_pct", 0.0))
        dominant_count = int(stat.get("dominant_count", 0))
        known = int(stat.get("known", 0))
        yes_pct = float(stat.get("yes_pct", 0.0))
        no_pct = float(stat.get("no_pct", 0.0))
        yesno_total = float(stat.get("yesno_total", 0.0))

        # 1) High proportion of unknowns
        if unknown_pct >= unknown_threshold:
            to_drop.add(col)
            continue

        # 2) Imbalanced yes/no columns, provided they cover enough rows
        if yesno_total > 0 and n_rows > 0:
            if yesno_total >= (yn_coverage_min / 100.0) * n_rows:
                if yes_pct < yes_no_threshold or no_pct < yes_no_threshold:
                    to_drop.add(col)
                    continue

        # 3) Extreme uniqueness (constant or almost row-unique)
        if nunique <= 1 or nunique >= n_rows:
            to_drop.add(col)
            continue

        # 4) Near-constant columns (dominant >= 99.5% and at least 25 minority rows)
        if yesno_total == 0:  # avoid double-handling yes/no columns
            minority_count = known - dominant_count
            if dominant_pct >= 99.5 and minority_count >= 25:
                to_drop.add(col)
                continue

    return to_drop


# ---------------------------------------------------------------------------
# Core cleaning pipeline
# ---------------------------------------------------------------------------


def clean_crash_dataframe_for_import(
    df: pd.DataFrame,
    *,
    base_unknowns: Iterable[str] | None = None,
    protected_columns: Iterable[str] | None = None,
    unknown_threshold: float | None = None,
    yes_no_threshold: float | None = None,
    columns_to_drop: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    End-to-end cleaning step used by both the import_crash_records management
    command and the model-training pipeline.

    This function is the library equivalent of Peyton's "Script to Clean":
    it discovers unknown placeholders, profiles columns, decides which
    columns to drop, and normalises unknown tokens to NaN.

    It intentionally does *not* perform feature encoding; that happens
    later in :func:`build_ml_ready_dataset`.
    """
    if base_unknowns is None:
        base_unknowns = DEFAULT_UNKNOWN_STRINGS
    else:
        base_unknowns = {str(s).strip().lower() for s in base_unknowns}

    if unknown_threshold is None:
        unknown_threshold = float(UNKNOWN_THRESHOLD)
    if yes_no_threshold is None:
        yes_no_threshold = float(YES_NO_THRESHOLD)

    validate_config_values(unknown_threshold, yes_no_threshold)

    protected: Set[str] = {c for c in (protected_columns or [])}

    # 1) Discover additional unknown placeholders via the partner adapter.
    augmented_unknowns = discover_unknown_placeholders(df, base_unknowns)

    # 2) Profile columns using the augmented unknown set.
    column_stats = profile_columns(df, augmented_unknowns)

    # 3) Decide which columns to drop based on the Script-to-Clean rules.
    auto_drop = suggest_columns_to_drop(
        df,
        column_stats,
        unknown_threshold=unknown_threshold,
        yes_no_threshold=yes_no_threshold,
        protected_columns=protected,
    )

    user_specified_drops: Set[str] = set(columns_to_drop or [])
    user_specified_drops.difference_update(protected)

    drop_cols = auto_drop | user_specified_drops

    # 4) Create a working copy and normalise unknown tokens to NaN
    cleaned = df.copy()

    unknown_token_set = {str(u).strip().lower() for u in augmented_unknowns}

    for col in cleaned.columns:
        s = cleaned[col]
        if s.dtype == "O" or pd.api.types.is_categorical_dtype(s):
            norm = s.astype(str).map(_normalize_str)
            mask_unknown = norm.isin(unknown_token_set)
            if mask_unknown.any():
                cleaned.loc[mask_unknown, col] = np.nan

    # 5) Drop the selected columns
    if drop_cols:
        cleaned = cleaned.drop(columns=list(drop_cols), errors="ignore")

    meta: Dict[str, Any] = {
        "unknown_values": sorted(unknown_token_set),
        "dropped_columns": sorted(drop_cols),
        "column_stats": column_stats,
        "input_shape": (int(df.shape[0]), int(df.shape[1])),
        "output_shape": (int(cleaned.shape[0]), int(cleaned.shape[1])),
        "cleaning_config": {
            "unknown_threshold": float(unknown_threshold),
            "yes_no_threshold": float(yes_no_threshold),
        },
        "user_specified_drops": sorted(user_specified_drops),
    }

    return cleaned, meta


# ---------------------------------------------------------------------------
# Severity mapping and leakage utilities
# ---------------------------------------------------------------------------


# Optional passthrough helpers for severity mapping
# -------------------------------------------------
# Some of Peyton-aligned tests import `map_numeric_severity` and
# `map_text_severity` from this module.  The canonical implementations
# live in the ml_partner_adapters.severity_mapping_bridge module, but
# those helpers may or may not be present depending on the adapter
# version.  We expose thin wrappers here that delegate to the adapter
# when available and raise a clear error otherwise.

try:  # pragma: no cover - adapter may omit these helpers
    from ml_partner_adapters.severity_mapping_bridge import (
        map_numeric_severity as _adapter_map_numeric_severity,
        map_text_severity as _adapter_map_text_severity,
    )
except Exception:  # ImportError, AttributeError, etc.
    _adapter_map_numeric_severity = None  # type: ignore[assignment]
    _adapter_map_text_severity = None  # type: ignore[assignment]


def map_numeric_severity(values: Sequence[Any]) -> Mapping[Any, int]:
    """Delegate to the ML partner's numeric severity mapping helper.

    This is primarily provided for compatibility with tests that expect
    `analysis.ml_core.cleaning` to expose Peyton-style helpers.  If the
    adapter does not implement the underlying function, a RuntimeError
    is raised with a clear message.
    """
    if _adapter_map_numeric_severity is None:
        raise RuntimeError(
            "map_numeric_severity is not available from the severity mapping adapter."
        )
    return _adapter_map_numeric_severity(values)


def map_text_severity(values: Sequence[Any]) -> Mapping[Any, int]:
    """Delegate to the ML partner's text severity mapping helper.

    See :func:`map_numeric_severity` for behaviour when the adapter does
    not implement the helper.
    """
    if _adapter_map_text_severity is None:
        raise RuntimeError(
            "map_text_severity is not available from the severity mapping adapter."
        )
    return _adapter_map_text_severity(values)


def guess_severity_column(df: pd.DataFrame) -> str:
    """
    Heuristic for choosing the crash-severity column when the caller does
    not specify one explicitly.

    Preference order:
      1. A column literally named "severity" (case-insensitive).
      2. The first column whose name contains the substring "severity".
    """
    lower_map = {c.lower(): c for c in df.columns}
    if "severity" in lower_map:
        return lower_map["severity"]

    candidates = [c for c in df.columns if "severity" in c.lower()]
    if not candidates:
        raise ValueError(
            "Could not determine the crash severity column. "
            "Pass `severity_col` explicitly in cleaning parameters."
        )
    if len(candidates) > 1:
        logger.info(
            "Multiple candidate severity columns found (%s); using %s",
            ", ".join(candidates),
            candidates[0],
        )
    return candidates[0]


def warn_suspicious_importances(
    feature_names: Sequence[str],
    importances: Sequence[float],
    importance_threshold: float = 0.2,
    dominance_ratio: float = 2.0,
) -> Sequence[str]:
    """
    Convenience wrapper around the ML partner's importance-leakage heuristic.

    We delegate to :func:`adapter_warn_suspicious_importances` and fall back
    to a simple local heuristic if the adapter raises.
    """
    try:
        suspicious = adapter_warn_suspicious_importances(
            feature_names,
            importances,
            importance_threshold=importance_threshold,
            dominance_ratio=dominance_ratio,
        )
        if suspicious is not None:
            return list(suspicious)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "adapter_warn_suspicious_importances failed (%r); falling back to local heuristic.",
            exc,
        )

    # Simple local fallback: pick any feature whose importance is both above the
    # absolute threshold *and* at least `dominance_ratio` times larger than the
    # second most important feature.
    if not feature_names or len(feature_names) != len(importances):
        return []

    arr = np.asarray(importances, dtype="float64")
    if arr.size == 0:
        return []

    total = float(arr.sum())
    if total <= 0.0:
        return []

    pct = arr / total
    idx_sorted = np.argsort(pct)[::-1]
    top_idx = idx_sorted[0]
    top_score = float(pct[top_idx])
    top_name = feature_names[top_idx]

    if len(idx_sorted) > 1:
        second_score = float(pct[idx_sorted[1]])
    else:
        second_score = 0.0

    suspicious: list[str] = []
    if top_score >= importance_threshold and second_score > 0.0:
        if top_score / second_score >= dominance_ratio:
            suspicious.append(top_name)

    return suspicious


# ---------------------------------------------------------------------------
# Build ML-ready (X, y, meta) from a raw crash-level dataframe
# ---------------------------------------------------------------------------


def build_ml_ready_dataset(
    df: pd.DataFrame,
    *,
    severity_col: str | None = None,
    base_unknowns: Iterable[str] | None = None,
    unknown_threshold: float | None = None,
    yes_no_threshold: float | None = None,
    leakage_columns: Iterable[str] | None = None,
    columns_to_drop: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Produce (X, y, meta) suitable for model training.

    This function wires together:
      * the Script-to-Clean-style column pruning and unknown handling;
      * Peyton's severity mapping (via the adapter);
      * Peyton's leakage detection (via the adapter); and
      * a simple one-hot encoding of any remaining categorical features so
        that scikit-learn models only see numeric inputs.

    The goal is to stay faithful to the intent of the original scripts
    while providing a clean, non-interactive API for the web service.
    """
    if base_unknowns is None:
        base_unknowns = DEFAULT_UNKNOWN_STRINGS

    if unknown_threshold is None:
        unknown_threshold = float(UNKNOWN_THRESHOLD)
    if yes_no_threshold is None:
        yes_no_threshold = float(YES_NO_THRESHOLD)

    validate_config_values(unknown_threshold, yes_no_threshold)

    # Determine severity column if not provided.
    if severity_col is None:
        severity_col = guess_severity_column(df)

    # Run the core cleaning pipeline, protecting the severity column from drop.
    cleaned_df, cleaning_meta = clean_crash_dataframe_for_import(
        df,
        base_unknowns=base_unknowns,
        protected_columns={severity_col},
        unknown_threshold=unknown_threshold,
        yes_no_threshold=yes_no_threshold,
        columns_to_drop=columns_to_drop,
    )

    # Ensure the severity column still exists
    if severity_col not in cleaned_df.columns:
        raise KeyError(
            f"Severity column '{severity_col}' is missing after cleaning; "
            f"check cleaning configuration."
        )

    # Map severity labels to ordinal integers using Peyton's adapter.
    sev_mapping = find_severity_mapping_noninteractive(cleaned_df, severity_col)
    y_raw = cleaned_df[severity_col]
    y = y_raw.map(sev_mapping)

    mask = y.notna()
    y = y.loc[mask].astype(int)

    # Drop the severity column from features; filter rows to those with valid y.
    X = cleaned_df.loc[mask].drop(columns=[severity_col], errors="ignore")

    # ------------------------------------------------------------------
    # Data-leakage handling (Peyton-style)
    # ------------------------------------------------------------------
    # Always run the detector so we can surface suggestions/warnings.
    auto_leak_cols = find_leakage_columns_noninteractive(X, y)

    if leakage_columns is not None:
        # UI / caller provided an explicit list (interactive flow).
        leak_cols_set: Set[str] = {c for c in leakage_columns if c in X.columns}
    else:
        leak_cols_set = set(auto_leak_cols)

    X_no_leak = X.drop(columns=list(leak_cols_set), errors="ignore")

    # One-hot encode any remaining categorical columns so that scikit-learn
    # sees a purely numeric feature matrix.
    numeric_part = X_no_leak.select_dtypes(include=[np.number])
    non_numeric_part = X_no_leak.select_dtypes(exclude=[np.number])

    if not non_numeric_part.empty:
        dummies = pd.get_dummies(non_numeric_part, dummy_na=True)
        X_final = pd.concat([numeric_part, dummies], axis=1)
    else:
        X_final = numeric_part.copy()

    # Ensure no NaNs remain in X_final (fill numeric NaNs with column medians).
    for col in X_final.columns:
        s = X_final[col]
        if s.isna().any():
            if pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s):
                fill_value = float(s.median(skipna=True))
                X_final[col] = s.fillna(fill_value)
            else:
                # Should not happen because get_dummies produces numeric dtypes,
                # but we guard anyway.
                X_final[col] = s.fillna(0)

    unknown_values = set(cleaning_meta.get("unknown_values", []))

    meta: Dict[str, Any] = {
        "severity_column": severity_col,
        "severity_mapping": sev_mapping,
        "unknown_values": sorted(unknown_values),
        # Columns actually dropped as leakage
        "leakage_columns": sorted(leak_cols_set),
        # What the non-interactive detector suggested automatically
        "auto_leakage_suggestions": sorted(set(auto_leak_cols)),
        "n_rows_before_target_filter": int(df.shape[0]),
        "n_rows_after_target_filter": int(X_final.shape[0]),
        "n_features_before_leakage": int(X.shape[1]),
        "n_features_after_leakage": int(X_final.shape[1]),
        "cleaning_meta": cleaning_meta,
    }

    return X_final, y, meta
