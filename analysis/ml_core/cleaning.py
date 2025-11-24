from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration values (adapted from the ML team's DataCleaning/config.py)
# ---------------------------------------------------------------------------

UNKNOWN_THRESHOLD: float = 10.0  # percentage threshold for unknowns in a column
YES_NO_THRESHOLD: float = 1.0    # percentage threshold for yes/no balance in a column

# Base set of tokens that should always be treated as "unknown"
DEFAULT_UNKNOWN_STRINGS: Set[str] = {
    "unknown",
    "missing",
    "unspecified",
    "not specified",
    "not applicable",
    "n/a",
    "na",
    "null",
    "blank",
    "tbd",
    "tba",
    "to be determined",
    "refused",
    "prefer not to say",
    "no data",
    "no value",
}

# Generic substrings we treat as "unknown-like" when scanning free-text values.
GENERIC_UNKNOWN_SUBSTRINGS: Set[str] = {
    "unknown",
    "missing",
    "not specified",
    "not applicable",
    "n/a",
    "none",
    "unspecified",
    "tbd",
    "tba",
    "no data",
    "no value",
    "refused",
    "prefer not to say",
}


def validate_config_values(
    unknown_threshold: float = UNKNOWN_THRESHOLD,
    yes_no_threshold: float = YES_NO_THRESHOLD,
) -> None:
    """Validate percentage thresholds are in the expected 0–100 range."""

    def _ok(x: float) -> bool:
        return isinstance(x, (int, float)) and 0.0 <= float(x) <= 100.0

    if not _ok(unknown_threshold):
        raise ValueError(
            f"UNKNOWN_THRESHOLD must be 0–100, got {unknown_threshold}"
        )
    if not _ok(yes_no_threshold):
        raise ValueError(
            f"YES_NO_THRESHOLD must be 0–100, got {yes_no_threshold}"
        )


# ---------------------------------------------------------------------------
# Unknown discovery (adapted from DataCleaning/unknown_discovery.py)
# ---------------------------------------------------------------------------


def discover_unknown_placeholders(
    df: pd.DataFrame,
    base_unknowns: Iterable[str] | None = None,
    generic_substrings: Iterable[str] | None = None,
    min_count: int = 2,
    max_token_length: int = 80,
) -> Set[str]:
    """Scan a DataFrame and automatically augment the set of 'unknown' tokens.

    The original ML script used simple heuristics:
    - focus on string / object columns,
    - look for values that contain generic unknown-like substrings,
    - only keep tokens that appear at least ``min_count`` times.
    """

    if base_unknowns is None:
        base_unknowns = DEFAULT_UNKNOWN_STRINGS
    if generic_substrings is None:
        generic_substrings = GENERIC_UNKNOWN_SUBSTRINGS

    known_unknowns: Set[str] = {
        str(v).strip().lower() for v in base_unknowns if isinstance(v, str)
    }
    counts: Dict[str, int] = {}

    for col in df.columns:
        series = df[col]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        for raw in series.dropna().unique():
            text = str(raw).strip()
            if not text:
                continue

            lower = text.lower()
            if lower in known_unknowns:
                continue
            if len(lower) > max_token_length:
                continue

            if any(sub in lower for sub in generic_substrings):
                counts[lower] = counts.get(lower, 0) + 1

    new_unknowns = {tok for tok, c in counts.items() if c >= min_count}
    augmented: Set[str] = set(known_unknowns)
    augmented.update(new_unknowns)
    return augmented


# ---------------------------------------------------------------------------
# Column profiling (adapted from DataCleaning/Script to Clean.py)
# ---------------------------------------------------------------------------


def percent_unknowns_per_column(
    df: pd.DataFrame,
    unknown_strings: Iterable[str],
) -> Dict[str, float]:
    """Return {column -> percent of cells that are 'unknown'}."""
    unknown_set = {str(u).lower() for u in unknown_strings}
    result: Dict[str, float] = {}

    for col in df.columns:
        series = df[col]
        total = len(series)
        if total == 0:
            result[col] = 0.0
            continue

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            lower = series.astype(str).str.lower()
            mask_unknown = lower.isin(unknown_set)
        else:
            mask_unknown = pd.Series([False] * total, index=series.index)

        pct = 100.0 * mask_unknown.sum() / float(total)
        result[col] = float(pct)

    return result


def yes_no(df: pd.DataFrame, unknown_strings: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Find columns that behave like Yes/No flags and compute coverage stats.

    Returns a dict of the form::

        {
            "col_name": {
                "yes_pct": float,
                "no_pct": float,
                "yesno_total": int,   # number of non-unknown yes/no values
            },
            ...
        }
    """
    unknown_set = {str(u).lower() for u in unknown_strings}
    stats: Dict[str, Dict[str, float]] = {}

    yes_like = {"yes", "y", "true", "t", "1"}
    no_like = {"no", "n", "false", "f", "0"}

    for col in df.columns:
        series = df[col]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        lower = series.astype(str).str.lower()
        mask_unknown = lower.isin(unknown_set)
        known = lower[~mask_unknown]

        if known.empty:
            continue

        yes_mask = known.isin(yes_like)
        no_mask = known.isin(no_like)

        yes_count = int(yes_mask.sum())
        no_count = int(no_mask.sum())

        total_yesno = yes_count + no_count
        total_known = len(known)

        if total_yesno == 0:
            continue

        yes_pct = 100.0 * yes_count / float(total_known)
        no_pct = 100.0 * no_count / float(total_known)

        stats[col] = {
            "yes_pct": float(yes_pct),
            "no_pct": float(no_pct),
            "yesno_total": float(total_yesno),
        }

    return stats


def profile_columns(
    df: pd.DataFrame,
    unknown_strings: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """Return a rich per-column profile used to decide which features to drop.

    This mirrors the ML group's Script-to-Clean logic in a non-interactive form.
    """
    unknown_percent = percent_unknowns_per_column(df, unknown_strings)
    yes_no_stats = yes_no(df, unknown_strings)

    stats: Dict[str, Dict[str, Any]] = {}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        col_stats: Dict[str, Any] = {}
        col_stats["unknown_pct"] = float(unknown_percent.get(col, 0.0))
        col_stats["total"] = int(n_rows)
        col_stats["nunique"] = int(series.nunique(dropna=True))

        non_null = series.dropna()
        col_stats["known"] = int(len(non_null))

        # dominant value ratio (excluding NaNs)
        if not non_null.empty:
            value_counts = non_null.value_counts()
            dominant_count = int(value_counts.iloc[0])
            col_stats["dominant_count"] = dominant_count
            col_stats["dominant_pct"] = 100.0 * dominant_count / float(len(non_null))
        else:
            col_stats["dominant_count"] = 0
            col_stats["dominant_pct"] = 0.0

        # yes/no stats if available
        yn = yes_no_stats.get(col, None)
        if yn is not None:
            col_stats["yes_pct"] = float(yn["yes_pct"])
            col_stats["no_pct"] = float(yn["no_pct"])
            col_stats["yesno_total"] = float(yn["yesno_total"])
        else:
            col_stats["yes_pct"] = 0.0
            col_stats["no_pct"] = 0.0
            col_stats["yesno_total"] = 0.0

        stats[col] = col_stats

    return stats


def suggest_columns_to_drop(
    df: pd.DataFrame,
    column_stats: Mapping[str, Mapping[str, Any]],
    *,
    unknown_threshold: float | None = None,
    yes_no_threshold: float | None = None,
    yesno_coverage_min: float = 50.0,
    protected_columns: Iterable[str] | None = None,
) -> Set[str]:
    """Heuristic feature-pruning logic using the profiling stats.

    This is a direct, non-interactive refactor of the Script-to-Clean rules.
    """
    if unknown_threshold is None:
        unknown_threshold = UNKNOWN_THRESHOLD
    if yes_no_threshold is None:
        yes_no_threshold = YES_NO_THRESHOLD

    validate_config_values(unknown_threshold, yes_no_threshold)

    protected = {c for c in (protected_columns or [])}
    n_rows = len(df)
    to_drop: Set[str] = set()

    for col, st in column_stats.items():
        if col in protected:
            continue

        unknown_pct = float(st.get("unknown_pct", 0.0))
        nunique = int(st.get("nunique", 0))
        yes_pct = float(st.get("yes_pct", 0.0))
        no_pct = float(st.get("no_pct", 0.0))
        yesno_total = float(st.get("yesno_total", 0.0))
        dominant_pct = float(st.get("dominant_pct", 0.0))
        dominant_count = int(st.get("dominant_count", 0))

        # 1) Too many unknowns
        if unknown_pct >= unknown_threshold:
            to_drop.add(col)
            continue

        # 2) Yes/No columns that are massively imbalanced
        if yesno_total >= yesno_coverage_min:
            if yes_pct < yes_no_threshold or no_pct < yes_no_threshold:
                to_drop.add(col)
                continue

        # 3) Columns with too few or too many unique values
        if nunique <= 1 or nunique >= n_rows:
            to_drop.add(col)
            continue

        # 4) Almost-constant columns (non yes/no) with enough minority support
        if yesno_total == 0.0 and dominant_pct >= 99.5:
            # require at least 25 non-dominant values so we do not drop
            # columns that are genuinely tiny.
            minority = max(int(st.get("known", 0)) - dominant_count, 0)
            if minority >= 25:
                to_drop.add(col)
                continue

    return to_drop


# ---------------------------------------------------------------------------
# Severity mapping (adapted from severity_mapping_utils.py)
# ---------------------------------------------------------------------------


def map_numeric_severity(unique_values: Sequence[Any]) -> Dict[Any, int]:
    """Map numeric severities into {0, 1, 2} buckets.

    0 = lowest severity, 2 = highest severity.
    """
    cleaned = []
    for v in unique_values:
        try:
            cleaned.append(float(v))
        except Exception:
            return {}

    if not cleaned:
        return {}

    lo = min(cleaned)
    hi = max(cleaned)
    mid = (lo + hi) / 2.0

    mapping: Dict[Any, int] = {}
    for raw, num in zip(unique_values, cleaned):
        if num <= mid and num != hi:
            mapping[raw] = 0
        elif num >= mid and num != lo:
            mapping[raw] = 2
        else:
            mapping[raw] = 1

    return mapping


def map_text_severity(unique_values: Sequence[Any]) -> Dict[Any, int]:
    """Map textual severities into {0, 1, 2} buckets using keywords."""
    severity_mapping: Dict[Any, int] = {}

    low_keywords = [
        "no injury",
        "property damage only",
        "pdo",
        "no apparent injury",
        "possible injury",
        "minor injury",
        "non-incapacitating",
    ]

    mid_keywords = [
        "suspected minor injury",
        "possible injury",
        "moderate injury",
        "non-serious injury",
    ]

    high_keywords = [
        "fatal",
        "death",
        "killed",
        "serious injury",
        "severe injury",
        "incapacitating injury",
        "hospitalized",
        "life threatening",
        "critical",
    ]

    for raw in unique_values:
        text = str(raw).strip().lower()

        score = None
        if any(k in text for k in high_keywords):
            score = 2
        elif any(k in text for k in mid_keywords):
            score = 1
        elif any(k in text for k in low_keywords):
            score = 0

        if score is not None:
            severity_mapping[raw] = score

    return severity_mapping


def guess_severity_column(
    df: pd.DataFrame,
    candidate_names: Sequence[str] | None = None,
) -> str | None:
    """Best-effort guess of the severity column name in a crash DataFrame."""
    if candidate_names is None:
        candidate_names = [
            "severity",
            "crash_severity",
            "Crash Severity",
            "Crash_Severity",
        ]

    for name in candidate_names:
        if name in df.columns:
            return name

    # fallback: any column containing the word severity
    for col in df.columns:
        if "severity" in col.lower():
            return col

    return None


def find_severity_mapping(df: pd.DataFrame, severity_col: str) -> Dict[Any, int]:
    """Non-interactive version of the ML team's find_severity_mapping.

    For MMUCC-style KABCO codes (K/A/B/C/O) we map to 3 buckets:

        K or A -> 2 (high)
        B or C -> 1 (medium)
        O      -> 0 (low)

    For other datasets we fall back to numeric or keyword-based mappings.
    """
    if severity_col not in df.columns:
        raise KeyError(f"Severity column {severity_col!r} not found in DataFrame.")

    series = df[severity_col].dropna()
    unique_vals = list(series.unique())

    if not unique_vals:
        raise ValueError("Severity column has no non-null values.")

    # Special-case KABCO-style single-letter codes
    normalized = [str(v).strip().upper() for v in unique_vals]
    kabco_set = {"K", "A", "B", "C", "O"}
    if set(normalized).issubset(kabco_set):
        mapping: Dict[Any, int] = {}
        for raw, norm in zip(unique_vals, normalized):
            if norm in {"K", "A"}:
                mapping[raw] = 2
            elif norm in {"B", "C"}:
                mapping[raw] = 1
            elif norm == "O" or norm == "0" or norm == "NO" or norm == "NONE":
                mapping[raw] = 0
        if mapping:
            return mapping

    # Try numeric mapping
    numeric = pd.to_numeric(series, errors="coerce")
    frac_numeric = float(numeric.notna().mean())
    if frac_numeric >= 0.9:
        mapping = map_numeric_severity(unique_vals)
        if mapping:
            return mapping

    # Try keyword-based text mapping
    text_mapping = map_text_severity(unique_vals)
    if text_mapping:
        return text_mapping

    raise ValueError(
        "Could not automatically determine a severity mapping for column "
        f"{severity_col!r}. Provide a cleaner severity column if needed."
    )


# ---------------------------------------------------------------------------
# Leakage detection (adapted from leakage_column_utils.py)
# ---------------------------------------------------------------------------


def suggest_leakage_by_name(columns: Sequence[str]) -> Set[str]:
    """Suggest leakage columns via simple keyword matches on column names."""
    keywords = [
        "fatal",
        "fatalities",
        "death",
        "dead",
        "killed",
        "injury",
        "injuries",
        "injured",
        "severity",
        "severe",
        "serious",
        "k_count",
        "killed_cnt",
        "inj_cnt",
    ]

    suggestions: Set[str] = set()
    for col in columns:
        lower = col.lower()
        if any(kw in lower for kw in keywords):
            suggestions.add(col)
    return suggestions


def find_near_perfect_predictors(
    X: pd.DataFrame,
    y: pd.Series,
    min_accuracy: float = 0.98,
    max_unique: int = 50,
) -> Sequence[Tuple[str, float]]:
    """Return columns that almost perfectly predict y on their own."""
    suspicious: list[Tuple[str, float]] = []

    for col in X.columns:
        series = X[col]

        if series.nunique(dropna=True) > max_unique:
            continue

        df_col = pd.DataFrame({"feature": series, "target": y})
        df_col = df_col.dropna(subset=["feature", "target"])
        if df_col.empty:
            continue

        mapping = (
            df_col.groupby("feature")["target"]
            .agg(lambda s: s.value_counts().idxmax())
        )

        y_hat = df_col["feature"].map(mapping)
        acc = float((y_hat == df_col["target"]).mean())
        if acc >= min_accuracy:
            suspicious.append((col, acc))

    return suspicious


def find_leakage_columns(
    X: pd.DataFrame,
    y: pd.Series,
    use_near_perfect_check: bool = True,
    min_accuracy: float = 0.9,
    max_unique: int = 50,
) -> Set[str]:
    """Non-interactive leakage detection.

    Combines:
      1) name-based suggestions; and
      2) near-perfect single-column predictors.
    """
    name_suggestions = suggest_leakage_by_name(list(X.columns))

    near_perfect: Sequence[Tuple[str, float]] = []
    if use_near_perfect_check:
        near_perfect = find_near_perfect_predictors(
            X,
            y,
            min_accuracy=min_accuracy,
            max_unique=max_unique,
        )

    leak_cols: Set[str] = set(name_suggestions)
    leak_cols.update(col for col, _ in near_perfect)

    if leak_cols:
        logger.info(
            "Detected potential leakage columns: %s",
            ", ".join(sorted(leak_cols)),
        )

    return leak_cols


def warn_suspicious_importances(
    feature_names: Sequence[str],
    importances: Sequence[float],
    importance_threshold: float = 0.2,
    dominance_ratio: float = 2.0,
) -> Sequence[str]:
    """Post-training helper that warns about unusually dominant features."""
    if not feature_names or len(feature_names) != len(importances):
        logger.warning(
            "warn_suspicious_importances: feature_names and importances size mismatch."
        )
        return []

    pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    if not pairs:
        return []

    max_name, max_imp = pairs[0]
    second_imp = pairs[1][1] if len(pairs) > 1 else 0.0

    suspicious: list[str] = []

    if second_imp == 0:
        dominance = float("inf") if max_imp > 0 else 0.0
    else:
        dominance = float(max_imp) / float(second_imp)

    if (max_imp >= importance_threshold) or (dominance >= dominance_ratio):
        suspicious.append(max_name)

    if suspicious:
        logger.warning(
            "Potential leakage features due to high importance: %s",
            ", ".join(suspicious),
        )

    return suspicious


# ---------------------------------------------------------------------------
# High-level helpers for ETL and model building
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
    """End-to-end cleaning step used by the import_crash_records command.

    This keeps the DataFrame row-aligned with the original dataset while:
      * normalising unknown tokens to NaN,
      * dropping obviously bad or redundant columns, and
      * optionally honouring per-job cleaning configuration.

    The *threshold* arguments are expressed as percentages in the 0–100 range.
    If omitted they fall back to the module-level UNKNOWN_THRESHOLD /
    YES_NO_THRESHOLD defaults so existing callers continue to behave as before.
    """
    if base_unknowns is None:
        base_unknowns = DEFAULT_UNKNOWN_STRINGS

    # Decide effective thresholds, falling back to the module defaults.
    if unknown_threshold is None:
        unknown_threshold = UNKNOWN_THRESHOLD
    if yes_no_threshold is None:
        yes_no_threshold = YES_NO_THRESHOLD

    validate_config_values(
        unknown_threshold=unknown_threshold,
        yes_no_threshold=yes_no_threshold,
    )

    # Discover unknown-like placeholders and normalise them to NaN.
    unknown_values = discover_unknown_placeholders(df, base_unknowns)
    df_clean = df.replace(list(unknown_values), np.nan)

    # Profile columns once so we can drive all heuristics from the same stats.
    column_stats = profile_columns(df_clean, unknown_values)

    # Protect core identifiers and geo columns by default; callers can extend.
    default_protected: Set[str] = {
        # Core MMUCC schema columns
        "crash_id",
        "crash_date",
        "severity",
        "kabco",
        "latitude",
        "longitude",
    }
    if protected_columns is not None:
        default_protected.update(protected_columns)

    # First, use the heuristic suggestions.
    to_drop = suggest_columns_to_drop(
        df_clean,
        column_stats,
        unknown_threshold=unknown_threshold,
        yes_no_threshold=yes_no_threshold,
        protected_columns=default_protected,
    )

    # Then, honour any explicit user-specified drops (still respecting protection).
    user_specified_drops: Set[str] = set()
    if columns_to_drop is not None:
        for col in columns_to_drop:
            if not isinstance(col, str):
                continue
            col_name = col
            if col_name in default_protected:
                # Never drop protected columns, even if requested.
                logger.warning(
                    "Requested to drop protected column %r; ignoring.",
                    col_name,
                )
                continue
            user_specified_drops.add(col_name)

    if user_specified_drops:
        to_drop.update(user_specified_drops)

    cleaned = df_clean.drop(columns=list(to_drop), errors="ignore")

    meta: Dict[str, Any] = {
        "unknown_values": sorted(unknown_values),
        "dropped_columns": sorted(to_drop),
        "protected_columns": sorted(default_protected),
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



def build_ml_ready_dataset(
    df: pd.DataFrame,
    *,
    severity_col: str | None = None,
    base_unknowns: Iterable[str] | None = None,
    unknown_threshold: float | None = None,
    yes_no_threshold: float | None = None,
    columns_to_drop: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Produce (X, y, meta) suitable for model training.

    This wraps the ML pipeline:

      - unknown discovery (via clean_crash_dataframe_for_import),
      - severity mapping, and
      - basic leakage detection.

    The same cleaning configuration knobs used by the ETL pipeline are
    exposed so that model-training jobs can see an identical view of the
    data (unknown_threshold, yes_no_threshold, columns_to_drop, etc.).
    """
    if base_unknowns is None:
        base_unknowns = DEFAULT_UNKNOWN_STRINGS

    # Best-effort guess of the severity column if one is not provided.
    if severity_col is None:
        severity_col = guess_severity_column(df)
    if severity_col is None:
        raise ValueError("Could not infer a severity column from the dataset.")

    # Reuse the crash-cleaning pipeline so training matches ETL behaviour.
    cleaned_df, cleaning_meta = clean_crash_dataframe_for_import(
        df,
        base_unknowns=base_unknowns,
        protected_columns={severity_col},
        unknown_threshold=unknown_threshold,
        yes_no_threshold=yes_no_threshold,
        columns_to_drop=columns_to_drop,
    )

    unknown_values = set(cleaning_meta.get("unknown_values", []))

    sev_mapping = find_severity_mapping(cleaned_df, severity_col)
    y_raw = cleaned_df[severity_col]
    y = y_raw.map(sev_mapping)

    mask = y.notna()
    X = cleaned_df.loc[mask].drop(columns=[severity_col], errors="ignore")
    y = y.loc[mask].astype(int)

    leak_cols = find_leakage_columns(X, y)

    X_final = X.drop(columns=list(leak_cols), errors="ignore")

    meta: Dict[str, Any] = {
        "severity_column": severity_col,
        "severity_mapping": sev_mapping,
        "unknown_values": sorted(unknown_values),
        "leakage_columns": sorted(leak_cols),
        "n_rows_before_target_filter": int(df.shape[0]),
        "n_rows_after_target_filter": int(X_final.shape[0]),
        "n_features_before_leakage": int(X.shape[1]),
        "n_features_after_leakage": int(X_final.shape[1]),
        "cleaning_meta": cleaning_meta,
    }

    return X_final, y, meta
