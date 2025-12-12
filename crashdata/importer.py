from __future__ import annotations

import os
from datetime import datetime
import logging
import time
from typing import Any, Iterable, Tuple

import pandas as pd
from django.contrib.gis.geos import Point
from django.utils import timezone

from crashdata.models import CrashRecord
from ingestion.models import UploadedDataset
from ingestion.validation import load_dataframe_from_bytes
from ingestion.validation import _apply_column_aliases


class ImportError(Exception):
    """Raised when crash record import cannot be completed."""

    def __init__(self, message: str, meta: dict | None = None):
        super().__init__(message)
        self.meta = meta or {}


logger = logging.getLogger(__name__)

def _first_non_null(series: pd.Series, n: int = 5) -> list[str]:
    return (
        series.dropna()
        .astype(str)
        .map(lambda s: s.strip())
        .loc[lambda s: s != ""]
        .head(n)
        .tolist()
    )


def _detect_datetime_columns(df: pd.DataFrame) -> tuple[pd.Series | None, dict]:
    """Find and parse a crash datetime column with fallbacks."""
    meta: dict[str, Any] = {}
    lower_map = {c.lower(): c for c in df.columns}

    def _find(names: list[str]) -> str | None:
        for name in names:
            if name in lower_map:
                return lower_map[name]
        for col_l, orig in lower_map.items():
            for name in names:
                if name in col_l:
                    return orig
        return None

    datetime_col = _find(
        [
            "crash_datetime",
            "crash date/time",
            "crash date time",
            "crash date_time",
            "crashdate",
        ]
    )
    date_col = _find(["crash_date", "crash date"])
    time_col = _find(["crash_time", "crash time"])

    raw_series: pd.Series | None = None
    if datetime_col:
        raw_series = df[datetime_col]
        meta["datetime_source"] = datetime_col
    elif date_col and time_col:
        raw_series = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(
            str
        ).str.strip()
        meta["datetime_source"] = f"{date_col}+{time_col}"
    elif date_col:
        raw_series = df[date_col]
        meta["datetime_source"] = date_col
    else:
        return None, meta

    meta["sample_raw_crash_date"] = _first_non_null(raw_series, 5)

    def _try_parse(series: pd.Series, infer: bool = False) -> pd.Series:
        return pd.to_datetime(series, errors="coerce", utc=True, infer_datetime_format=infer)

    parsed = _try_parse(raw_series, infer=False)
    parsed_ok = parsed.notna().sum()
    if parsed_ok == 0 and date_col and time_col:
        parsed = _try_parse(
            df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
            infer=True,
        )
        parsed_ok = parsed.notna().sum()
        meta["datetime_source"] = f"{date_col}+{time_col}"
    if parsed_ok == 0:
        parsed = _try_parse(raw_series, infer=True)
        parsed_ok = parsed.notna().sum()

    meta["parsed_datetime_ok"] = int(parsed_ok)
    meta["parsed_datetime_total"] = int(len(parsed))
    meta["parsed_datetime_pct"] = float(parsed_ok) / float(len(parsed) or 1) * 100.0

    return parsed, meta


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_location(lon: Any, lat: Any):
    try:
        if lon is None or lat is None:
            return None
        if pd.isna(lon) or pd.isna(lat):
            return None
        return Point(float(lon), float(lat), srid=4326)
    except Exception:
        return None


def _coerce_severity(val: Any) -> str:
    raw = "" if val is None else str(val).strip().lower()
    if not raw or raw == "nan":
        return "O"

    kabco_map = {
        "k": "K",
        "fatal": "K",
        "fatality": "K",
        "a": "A",
        "serious": "A",
        "major": "A",
        "b": "B",
        "non-incapacitating": "B",
        "minor": "B",
        "c": "C",
        "possible": "C",
        "o": "O",
        "pdo": "O",
        "property": "O",
        "property damage only": "O",
        "0": "O",
        "1": "C",
        "2": "B",
        "3": "A",
        "4": "K",
    }
    if raw.upper() in {"K", "A", "B", "C", "O"}:
        return raw.upper()
    if raw in kabco_map:
        return kabco_map[raw]
    # Numeric strings
    if raw.isdigit():
        if raw == "0":
            return "O"
        if raw == "1":
            return "C"
        if raw == "2":
            return "B"
        if raw == "3":
            return "A"
        if raw == "4":
            return "K"
    return "O"


def _lightweight_clean_crash_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Minimal, import-focused cleaning that avoids the heavy ML profiling path.

    - Applies column aliases (same as ingestion).
    - Ensures required columns exist.
    - Normalises crash_id/severity text and parses crash_datetime to UTC.
    - Coerces lon/lat to numeric; invalid values become NaN and are skipped later.
    """
    meta: dict[str, Any] = {}
    logger.info(
        "Import dataframe initial snapshot",
        extra={
            "shape": (int(df.shape[0]), int(df.shape[1])),
            "columns_sample": list(df.columns)[:80],
        },
    )

    df = _apply_column_aliases(df)
    logger.info(
        "After column aliasing",
        extra={
            "shape": (int(df.shape[0]), int(df.shape[1])),
            "columns_sample": list(df.columns)[:80],
        },
    )

    required_cols: set[str] = {"crash_id", "severity"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ImportError(
            "Uploaded data is missing required columns: " + ", ".join(sorted(missing)),
            meta={"missing_columns": sorted(missing)},
        )

    working = df.copy()

    # crash_id
    working["crash_id"] = working["crash_id"].astype(str).str.strip()
    mask_missing_crash_id = working["crash_id"].isna() | (working["crash_id"] == "") | (
        working["crash_id"].str.lower() == "nan"
    )
    meta["sample_raw_crash_id"] = _first_non_null(working["crash_id"], 5)

    # severity
    raw_severity = working["severity"].copy()
    meta["sample_raw_severity"] = _first_non_null(raw_severity, 10)
    t_sev_start = time.monotonic()
    working["severity"] = working["severity"].map(_coerce_severity)
    meta["severity_coerced_to_O"] = int((working["severity"] == "O").sum())
    meta["severity_valid_kabco"] = int(
        (working["severity"].isin({"K", "A", "B", "C", "O"})).sum()
    )
    meta["severity_total"] = int(len(working))
    meta["severity_normalize_sec"] = round(time.monotonic() - t_sev_start, 3)

    # crash_date/crash_datetime detection and parse with fallback
    t_dt_start = time.monotonic()
    parsed_dt, dt_meta = _detect_datetime_columns(working)
    meta.update(dt_meta)
    meta["crash_date_parse_sec"] = round(time.monotonic() - t_dt_start, 3)
    if parsed_dt is None:
        raise ImportError(
            "No crash date/datetime column found. Expected one of crash_datetime, crash_date (+ optional crash_time).",
            meta=meta,
        )
    working["crash_datetime"] = parsed_dt

    # Normalise lon/lat if present.
    for col in ("longitude", "latitude"):
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    # Filter to rows with required fields present/valid.
    mask_required = (~mask_missing_crash_id) & working["crash_datetime"].notna()
    cleaned = working.loc[mask_required].copy()
    meta["rows_before_filter"] = len(df)
    meta["rows_after_filter"] = len(cleaned)
    meta["dropped_missing_crash_id"] = int(mask_missing_crash_id.sum())
    meta["dropped_unparseable_datetime"] = int(
        (~mask_missing_crash_id & working["crash_datetime"].isna()).sum()
    )
    if cleaned.empty:
        sample_crash_date = (
            working["crash_datetime"].dropna().astype(str).head(5).tolist()
        )
        sample_raw_dates = meta.get("sample_raw_crash_date", [])
        raise ImportError(
            "Cleaning pipeline produced an empty DataFrame; no rows had both crash_id and a parseable crash_datetime.",
            meta={
                **meta,
                "sample_parsed_crash_date": sample_crash_date,
                "sample_raw_crash_date": sample_raw_dates,
                "sample_severity": working["severity"].head(10).tolist(),
            },
        )
    return cleaned, meta


def import_crash_records_for_dataset(
    dataset: UploadedDataset,
    *,
    dry_run: bool = False,
) -> Tuple[int, int, dict]:
    """
    Import CrashRecord rows for a dataset using the same cleaning pipeline as the CLI command.

    Returns (imported_count, mappable_count). Raises ImportError on failure.
    """
    if not dataset.raw_file:
        raise ImportError("UploadedDataset has no raw_file attached.")

    file_field = dataset.raw_file
    file_name = file_field.name or dataset.original_filename
    ext = os.path.splitext(file_name)[1].lower() or ".csv"

    overall_start = time.monotonic()
    logger.info(
        "Import crash records start",
        extra={"dataset_id": str(dataset.id), "filename": file_name},
    )

    file_field.open("rb")
    try:
        raw_bytes = file_field.read()
    finally:
        file_field.close()
    if not raw_bytes:
        raise ImportError("UploadedDataset.raw_file is empty.")

    t0 = time.monotonic()
    df = load_dataframe_from_bytes(raw_bytes, ext)
    logger.info(
        "Loaded dataframe from upload",
        extra={
          "dataset_id": str(dataset.id),
          "shape": (int(df.shape[0]), int(df.shape[1])),
          "duration_sec": round(time.monotonic() - t0, 3),
        },
    )

    t1 = time.monotonic()
    cleaned_df, meta = _lightweight_clean_crash_dataframe(df)
    logger.info(
        "Lightweight cleaning complete",
        extra={
          "dataset_id": str(dataset.id),
          "shape": (int(cleaned_df.shape[0]), int(cleaned_df.shape[1])),
          "duration_sec": round(time.monotonic() - t1, 3),
          **meta,
        },
    )

    if cleaned_df.empty:
        raise ImportError(
            "Cleaning pipeline produced an empty DataFrame; nothing to import.",
            meta=meta,
        )

    required_cols: Iterable[str] = {"crash_id", "crash_datetime", "severity"}
    missing = [c for c in required_cols if c not in cleaned_df.columns]
    if missing:
        raise ImportError(
            "Cleaned dataset is missing required columns for CrashRecord mapping: "
            + ", ".join(missing)
        )

    if dry_run:
        return 0, 0, meta

    existing_qs = CrashRecord.objects.filter(dataset=dataset)
    if existing_qs.exists():
        existing_qs.delete()

    records: list[CrashRecord] = []

    t2 = time.monotonic()
    for _, row in cleaned_df.iterrows():
        crash_id = str(row.get("crash_id", "")).strip()
        if not crash_id:
            continue

        crash_date_raw = row.get("crash_datetime")
        crash_dt = None
        if isinstance(crash_date_raw, pd.Timestamp):
            crash_dt = crash_date_raw.to_pydatetime()
        elif isinstance(crash_date_raw, datetime):
            crash_dt = crash_date_raw
        else:
            parsed = pd.to_datetime(crash_date_raw, errors="coerce")
            if isinstance(parsed, pd.Series):
                parsed = parsed.iloc[0]
            if not pd.isna(parsed):
                crash_dt = parsed.to_pydatetime()
        if crash_dt is None:
            continue
        if timezone.is_naive(crash_dt):
            crash_dt = timezone.make_aware(crash_dt, timezone=timezone.utc)

        severity_raw = row.get("severity")
        severity = str(severity_raw).strip().upper() if severity_raw is not None else ""
        if severity not in {"K", "A", "B", "C", "O"}:
            continue

        location = _get_location(row.get("longitude"), row.get("latitude"))

        record = CrashRecord(
            dataset=dataset,
            crash_id=crash_id,
            crash_datetime=crash_dt,
            severity=severity,
            location=location,
            roadway_name="" if pd.isna(row.get("roadway_name")) else str(row.get("roadway_name", "")),
            municipality="" if pd.isna(row.get("municipality")) else str(row.get("municipality", "")),
            posted_speed_limit=_safe_int(row.get("posted_speed_limit")),
            vehicle_count=_safe_int(row.get("vehicle_count")),
            person_count=_safe_int(row.get("person_count")),
        )
        records.append(record)

    if not records:
        raise ImportError(
            "No CrashRecord rows were constructed from the cleaned dataset. "
            "Check that severity codes and crash_date values are valid."
        )

    build_duration = time.monotonic() - t2
    CrashRecord.objects.bulk_create(records, batch_size=1000)
    insert_duration = time.monotonic() - t2 - build_duration
    mappable_count = CrashRecord.objects.filter(dataset=dataset, location__isnull=False).count()

    logger.info(
        "Import crash records complete",
        extra={
          "dataset_id": str(dataset.id),
          "imported": len(records),
          "mappable": mappable_count,
          "build_records_sec": round(build_duration, 3),
          "db_insert_and_count_sec": round(insert_duration, 3),
          "overall_sec": round(time.monotonic() - overall_start, 3),
        },
    )
    meta.update(
        {
            "imported": len(records),
            "mappable": mappable_count,
            "build_records_sec": round(build_duration, 3),
            "db_insert_and_count_sec": round(insert_duration, 3),
        }
    )
    return len(records), mappable_count, meta
