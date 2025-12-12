# ingestion/excel_support.py

from __future__ import annotations

import io
import logging
from typing import Callable, Iterable

import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)


def _normalise_ext_list(raw_exts: object) -> list[str]:
    """
    Turn whatever is in settings.INGESTION_ALLOWED_EXTENSIONS into
    a normalised list of lowercase extensions (including the leading dot).
    """
    if isinstance(raw_exts, str):
        items = [p.strip() for p in raw_exts.split(",") if p.strip()]
    elif isinstance(raw_exts, (list, tuple, set)):
        items = [str(p).strip() for p in raw_exts if str(p).strip()]
    else:
        logger.warning(
            "Excel support: unexpected type for INGESTION_ALLOWED_EXTENSIONS: %r",
            type(raw_exts),
        )
        items = []

    normalised: list[str] = []
    for item in items:
        lower = item.lower()
        # Ensure a leading dot, so "xlsx" becomes ".xlsx"
        if not lower.startswith("."):
            lower = "." + lower
        if lower not in normalised:
            normalised.append(lower)
    return normalised


def apply() -> None:
    """Install the Excel-aware dataframe loader and widen allowed extensions.

    Safe to call multiple times.
    """
    try:
        from ingestion import validation as v
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Excel support: failed to import ingestion.validation: %s", exc)
        return

    original: Callable[[bytes, str], pd.DataFrame] = v.load_dataframe_from_bytes  # type: ignore[assignment]

    def load_dataframe_from_bytes_with_excel(file_bytes: bytes, extension: str) -> pd.DataFrame:
        """Wrapper that adds Excel handling then defers to the original loader."""
        ext = (extension or "").lower()
        if ext in {".xlsx", ".xls", ".xlsm"}:
            try:
                # Use BytesIO so pandas can seek within the stream.
                # Let pandas pick the appropriate engine (openpyxl for .xlsx).
                return pd.read_excel(io.BytesIO(file_bytes))
            except Exception as exc:
                # Normalise all Excel parsing issues into a ValueError so
                # callers see a clean error message.
                raise ValueError(f"Failed to parse Excel file: {exc}") from exc

        # Fallback to the original behaviour for CSV / Parquet (and any
        # other extensions it already understands).
        return original(file_bytes, extension)

    # Install the wrapper (idempotent – don't re-wrap if already patched).
    if getattr(v, "_excel_support_installed", False) is False:
        v.load_dataframe_from_bytes = load_dataframe_from_bytes_with_excel  # type: ignore[assignment]
        v._excel_support_installed = True

    # Ensure the upload endpoint treats Excel as an allowed extension.
    raw_exts = getattr(settings, "INGESTION_ALLOWED_EXTENSIONS", ".csv,.parquet")
    exts = _normalise_ext_list(raw_exts)

    # Always include .xlsx; optionally also allow .xls / .xlsm.
    for e in (".xlsx", ".xls", ".xlsm"):
        if e not in exts:
            exts.append(e)

    settings.INGESTION_ALLOWED_EXTENSIONS = exts

    logger.info(
        "Excel ingestion support initialised; INGESTION_ALLOWED_EXTENSIONS=%r",
        getattr(settings, "INGESTION_ALLOWED_EXTENSIONS", None),
    )
