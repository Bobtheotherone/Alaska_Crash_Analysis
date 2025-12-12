
from __future__ import annotations

import logging
from typing import Any

from django.core.management.base import BaseCommand

from analysis.ml_core.worker import run_next_queued_job

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "Process ModelJob instances with status='queued'. "
        "By default this command processes a single job and exits. "
        "Use --loop to keep polling for new jobs."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--loop",
            action="store_true",
            help="Continuously poll for queued jobs instead of exiting "
            "after processing a single job.",
        )
        parser.add_argument(
            "--sleep",
            type=int,
            default=5,
            help="Number of seconds to sleep between polling iterations "
            "when --loop is enabled (default: 5).",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        loop = options["loop"]
        sleep_seconds = options["sleep"]

        import time

        if not loop:
            job_id = run_next_queued_job()
            if job_id:
                self.stdout.write(self.style.SUCCESS(f"Processed ModelJob {job_id}"))
            else:
                self.stdout.write("No queued ModelJob instances to process.")
            return

        self.stdout.write(
            self.style.WARNING(
                f"Entering polling loop (sleep={sleep_seconds}s). "
                "Press Ctrl+C to stop."
            )
        )

        try:
            while True:
                job_id = run_next_queued_job()
                if job_id:
                    self.stdout.write(
                        self.style.SUCCESS(f"Processed ModelJob {job_id}")
                    )
                else:
                    logger.debug("No queued jobs found, sleeping for %s seconds.", sleep_seconds)
                time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("Stopping run_model_jobs loop."))
