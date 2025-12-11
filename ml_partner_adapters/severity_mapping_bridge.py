"""
Non-interactive adaptation of Peyton's severity-mapping utilities.

The original CLI helper (``find_severity_mapping``) uses interactive
prompts to confirm or manually override the mapping. For web/worker
use we instead expose functions that run the same automatic heuristics
and return the proposed mappings directly.
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable

import pandas as pd

from peyton_original import severity_mapping_utils as peyton_severity


def map_numeric_severity(values: Iterable[Hashable]) -> Dict[Hashable, int]:
    """
    Thin wrapper around Peyton's numeric severity mapping helper.

    Given the distinct values that appear in a severity column, return
    a mapping from each raw value to an ordinal severity score
    (0 = least severe, 2 = most severe). The exact heuristics are
    delegated to :func:`peyton_severity.map_numeric_severity`.
    """
    # The underlying helper is robust to value types; it just expects an
    # iterable of unique values.
    return peyton_severity.map_numeric_severity(list(values))


def map_text_severity(values: Iterable[Hashable]) -> Dict[Hashable, int]:
    """
    Thin wrapper around Peyton's text-based severity mapping helper.

    This function applies keyword heuristics (e.g. "fatal", "serious",
    "minor") to produce a 0/1/2 mapping for free-text severity labels.
    """
    return peyton_severity.map_text_severity(list(values))


def find_severity_mapping_noninteractive(
    df: pd.DataFrame,
    severity_col: str,
) -> Dict[Hashable, int]:
    """
    Infer a 0/1/2 severity mapping for ``severity_col`` without any
    interactive prompts.

    The logic mirrors Peyton's ``find_severity_mapping``:

    * If at least ~90% of the values in ``severity_col`` can be parsed
      as numbers, use :func:`map_numeric_severity`.
    * Otherwise, attempt a keyword-based mapping via
      :func:`map_text_severity`.

    If neither heuristic is able to produce a mapping, an empty dict is
    returned instead of falling back to manual input.
    """
    sev_vals = df[severity_col].dropna()
    unique_sev_vals = sev_vals.unique().tolist()

    # Replicate Peyton's numeric vs text branching.
    numeric_sev = pd.to_numeric(sev_vals, errors="coerce")
    num_valid = numeric_sev.notna().sum()
    frac_valid = float(num_valid) / len(sev_vals) if len(sev_vals) > 0 else 0.0

    # 1) Try numeric mapping if most values look numeric.
    if frac_valid >= 0.9:
        mapping = map_numeric_severity(unique_sev_vals)
        if mapping:
            # Original code would call ``confirm_or_edit_mapping`` here,
            # which is interactive; for non-interactive use we simply
            # accept the proposed mapping.
            return mapping

    # 2) Fallback to keyword-based text mapping.
    text_mapping = map_text_severity(unique_sev_vals)
    if text_mapping:
        return text_mapping

    # 3) In the CLI tool this would fall back to manual input; in a
    # non-interactive context we instead return an empty mapping.
    return {}
