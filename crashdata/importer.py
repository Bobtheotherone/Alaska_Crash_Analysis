from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Iterable, Tuple

import pandas as pd
from django.contrib.gis.geos import Point
from django.utils import timezone

from crashdata.models import CrashRecord
from ingestion.models import UploadedDataset
from ingestion.validation import load_dataframe_from_bytes
from analysis.ml_core import cleaning as ml_cleaning


class ImportError(Exception):
    """Raised when crash record import cannot be completed."""


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


def import_crash_records_for_dataset(
    dataset: UploadedDataset,
    *,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Import CrashRecord rows for a dataset using the same cleaning pipeline as the CLI command.

    Returns (imported_count, mappable_count). Raises ImportError on failure.
    """
    if not dataset.raw_file:
        raise ImportError("UploadedDataset has no raw_file attached.")

    file_field = dataset.raw_file
    file_name = file_field.name or dataset.original_filename
    ext = os.path.splitext(file_name)[1].lower() or ".csv"

    file_field.open("rb")
    try:
        raw_bytes = file_field.read()
    finally:
        file_field.close()
    if not raw_bytes:
        raise ImportError("UploadedDataset.raw_file is empty.")

    df = load_dataframe_from_bytes(raw_bytes, ext)
    cleaned_df, meta = ml_cleaning.clean_crash_dataframe_for_import(df)

    if cleaned_df.empty:
        raise ImportError("Cleaning pipeline produced an empty DataFrame; nothing to import.")

    required_cols: Iterable[str] = {"crash_id", "crash_date", "severity"}
    missing = [c for c in required_cols if c not in cleaned_df.columns]
    if missing:
        raise ImportError(
            "Cleaned dataset is missing required columns for CrashRecord mapping: "
            + ", ".join(missing)
        )

    if dry_run:
        return 0, 0

    existing_qs = CrashRecord.objects.filter(dataset=dataset)
    if existing_qs.exists():
        existing_qs.delete()

    records: list[CrashRecord] = []

    for _, row in cleaned_df.iterrows():
        crash_id = str(row.get("crash_id", "")).strip()
        if not crash_id:
            continue

        crash_date_raw = row.get("crash_date")
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

    CrashRecord.objects.bulk_create(records, batch_size=1000)
    mappable_count = CrashRecord.objects.filter(dataset=dataset, location__isnull=False).count()
    return len(records), mappable_count
