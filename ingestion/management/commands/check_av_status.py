from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from ingestion.antivirus import scan_bytes_with_clamav


class Command(BaseCommand):
    help = "Check ClamAV availability by scanning a small payload."

    def handle(self, *args, **options):
        result = scan_bytes_with_clamav(b"healthcheck")
        status = result.get("status")
        details = result.get("details", "")

        self.stdout.write(f"ClamAV scan status: {status}")
        if details:
            self.stdout.write(f"Details: {details}")

        require_av = getattr(settings, "INGESTION_REQUIRE_AV", False)
        if status == "failed":
            raise CommandError("ClamAV detected malware during health check.")
        if status == "skipped" and require_av:
            raise CommandError("Antivirus scanning is required but was skipped.")
