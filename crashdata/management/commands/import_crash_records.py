from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand, CommandError

from crashdata.importer import ImportError, import_crash_records_for_dataset
from ingestion.models import UploadedDataset


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

        try:
            imported, mappable = import_crash_records_for_dataset(dataset, dry_run=dry_run)
        except ImportError as exc:
            raise CommandError(str(exc))

        if dry_run:
            self.stdout.write(self.style.SUCCESS("Dry run complete; no CrashRecord rows were written."))
            return

        self.stdout.write(
            self.style.SUCCESS(
                f"Imported {imported} CrashRecord rows for UploadedDataset {dataset.id} "
                f"({mappable} with valid coordinates)."
            )
        )
