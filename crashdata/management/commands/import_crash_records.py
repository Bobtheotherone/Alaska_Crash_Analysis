from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
from django.contrib.gis.geos import Point
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from crashdata.models import CrashRecord
from ingestion.models import UploadedDataset
from ingestion.validation import load_dataframe_from_bytes
from analysis.ml_core import cleaning as ml_cleaning


class Command(BaseCommand):
    help = "Import cleaned crash records for a given UploadedDataset using the ML cleaning pipeline."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "upload_id",
            type=str,
            help="ID of the UploadedDataset to import.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Run the cleaning pipeline and show a summary without writing CrashRecord rows.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        upload_id = options["upload_id"]
        dry_run = bool(options["dry_run"])

        try:
            dataset = UploadedDataset.objects.get(id=upload_id)
        except UploadedDataset.DoesNotExist:
            raise CommandError(f"UploadedDataset with id={upload_id!r} does not exist.")

        if not dataset.raw_file:
            raise CommandError("UploadedDataset has no raw_file attached.")

        file_field = dataset.raw_file
        file_name = file_field.name or dataset.original_filename
        ext = os.path.splitext(file_name)[1].lower() or ".csv"

        raw_bytes = file_field.read()
        if not raw_bytes:
            raise CommandError("UploadedDataset.raw_file is empty.")

        # Re-parse using the same helper as the ingestion gateway so semantics match.
        df = load_dataframe_from_bytes(raw_bytes, ext)

        # Run the refactored, non-interactive cleaning pipeline.
        cleaned_df, meta = ml_cleaning.clean_crash_dataframe_for_import(df)

        if cleaned_df.empty:
            raise CommandError("Cleaning pipeline produced an empty DataFrame; nothing to import.")

        # Log a small summary to the console so ML teammates can sanity-check.
        self.stdout.write(
            self.style.SUCCESS(
                f"Cleaned dataframe shape: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns"
            )
        )
        dropped = meta.get("dropped_columns", [])
        if dropped:
            self.stdout.write("Dropped columns (likely high-unknown / low-signal / near-constant):")
            for name in dropped:
                self.stdout.write(f"  - {name}")

        if dry_run:
            self.stdout.write(self.style.SUCCESS("Dry run complete; no CrashRecord rows were written."))
            return

        # Basic mapping assumptions:
        #   * crash_id       -> CrashRecord.crash_id
        #   * crash_date     -> CrashRecord.crash_datetime (midnight in the current timezone)
        #   * severity       -> CrashRecord.severity (MMUCC KABCO codes)
        #   * latitude/longitude -> CrashRecord.location (WGS84 point)
        #   * posted_speed_limit, vehicle_count, person_count if available
        required_cols = {"crash_id", "crash_date", "severity"}
        missing = [c for c in required_cols if c not in cleaned_df.columns]
        if missing:
            raise CommandError(
                "Cleaned dataset is missing required columns for CrashRecord mapping: "
                + ", ".join(missing)
            )

        # If we re-import the same dataset, replace its CrashRecords to avoid duplicates.
        existing_qs = CrashRecord.objects.filter(dataset=dataset)
        existing_count = existing_qs.count()
        if existing_count:
            self.stdout.write(
                self.style.WARNING(
                    f"Dataset {dataset.id} already has {existing_count} CrashRecord rows; "
                    "they will be deleted and replaced."
                )
            )
            existing_qs.delete()

        records: list[CrashRecord] = []

        for _, row in cleaned_df.iterrows():
            try:
                crash_id = str(row["crash_id"])
            except KeyError:
                # Should not happen because we already checked required_cols above.
                continue

            crash_date_raw = row.get("crash_date")
            if isinstance(crash_date_raw, pd.Timestamp):
                crash_dt = crash_date_raw.to_pydatetime()
            elif isinstance(crash_date_raw, datetime):
                crash_dt = crash_date_raw
            else:
                # Let pandas parse strings or date-like objects.
                crash_dt = pd.to_datetime(crash_date_raw, errors="coerce")
                if isinstance(crash_dt, pd.Series):
                    # to_datetime on scalar sometimes returns Series in older pandas;
                    # guard against that by taking the first element.
                    crash_dt = crash_dt.iloc[0]

            if pd.isna(crash_dt):
                # If we cannot parse the date, skip this row rather than crashing the import.
                continue

            if timezone.is_naive(crash_dt):
                crash_dt = timezone.make_aware(crash_dt, timezone.get_current_timezone())

            severity_raw = row.get("severity")
            severity = str(severity_raw).strip().upper() if severity_raw is not None else ""

            # Only accept MMUCC KABCO severity codes; skip rows with invalid codes.
            if severity not in {"K", "A", "B", "C", "O"}:
                continue

            lat = row.get("latitude")
            lon = row.get("longitude")
            location = None
            try:
                if lat is not None and lon is not None and not (pd.isna(lat) or pd.isna(lon)):
                    location = Point(float(lon), float(lat), srid=4326)
            except Exception:
                # Leave location as None if parsing fails.
                location = None

            posted_speed_limit = None
            if "posted_speed_limit" in cleaned_df.columns:
                val = row.get("posted_speed_limit")
                if val is not None and not pd.isna(val):
                    try:
                        posted_speed_limit = int(val)
                    except (TypeError, ValueError):
                        posted_speed_limit = None

            vehicle_count = None
            if "vehicle_count" in cleaned_df.columns:
                val = row.get("vehicle_count")
                if val is not None and not pd.isna(val):
                    try:
                        vehicle_count = int(val)
                    except (TypeError, ValueError):
                        vehicle_count = None

            person_count = None
            if "person_count" in cleaned_df.columns:
                val = row.get("person_count")
                if val is not None and not pd.isna(val):
                    try:
                        person_count = int(val)
                    except (TypeError, ValueError):
                        person_count = None

            roadway_name = ""
            if "roadway_name" in cleaned_df.columns:
                rn = row.get("roadway_name")
                roadway_name = "" if rn is None or pd.isna(rn) else str(rn)

            municipality = ""
            if "municipality" in cleaned_df.columns:
                m = row.get("municipality")
                municipality = "" if m is None or pd.isna(m) else str(m)

            record = CrashRecord(
                dataset=dataset,
                crash_id=crash_id,
                crash_datetime=crash_dt,
                severity=severity,
                location=location,
                roadway_name=roadway_name,
                municipality=municipality,
                posted_speed_limit=posted_speed_limit,
                vehicle_count=vehicle_count,
                person_count=person_count,
            )
            records.append(record)

        if not records:
            raise CommandError(
                "No CrashRecord rows were constructed from the cleaned dataset. "
                "Check that severity codes and crash_date values are valid."
            )

        CrashRecord.objects.bulk_create(records, batch_size=1000)

        self.stdout.write(
            self.style.SUCCESS(
                f"Imported {len(records)} CrashRecord rows for UploadedDataset {dataset.id}."
            )
        )
