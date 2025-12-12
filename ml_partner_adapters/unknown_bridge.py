"""
Non-interactive wrappers around Peyton's unknown-value discovery helpers.
"""

from __future__ import annotations

from typing import Iterable, Set

import pandas as pd

from peyton_original.DataCleaning import unknown_discovery as peyton_unknown


def discover_unknown_placeholders_web(
    df: pd.DataFrame,
    unknown_strings: Iterable[str],
    min_freq: int = 2,
    max_token_length: int = 80,
) -> Set[str]:
    """
    Thin, non-interactive wrapper around
    :func:`peyton_original.DataCleaning.unknown_discovery.discover_unknown_placeholders`.

    Parameters
    ----------
    df:
        DataFrame to scan.
    unknown_strings:
        Baseline collection of values that should always be treated as
        "unknown". In most call sites this should be
        :data:`ml_partner_adapters.config_bridge.UNKNOWN_STRINGS`.
    min_freq:
        Minimum number of occurrences required for an automatically
        discovered placeholder to be accepted.
    max_token_length:
        Maximum length of a candidate token that will be considered.

    Returns
    -------
    Set[str]
        Combined set of the normalized baseline unknowns and any
        automatically discovered placeholders.
    """
    base_unknowns = list(unknown_strings)
    return peyton_unknown.discover_unknown_placeholders(
        df=df,
        base_unknowns=base_unknowns,
        min_freq=min_freq,
        max_token_len=max_token_length,
    )
