from django.contrib.gis.db import models as gis_models
from django.contrib.gis.db.models import indexes as gis_indexes
from django.db import models

from ingestion.models import UploadedDataset


class CrashRecord(gis_models.Model):
    """Crash-level record aligned with core MMUCC / KABCO fields.

    This model is deliberately minimal; additional fields can be added by the
    data engineering or ML teams as needed.
    """

    SEVERITY_CHOICES = [
        ("K", "Fatal (K)"),
        ("A", "Suspected serious injury (A)"),
        ("B", "Suspected minor injury (B)"),
        ("C", "Possible injury (C)"),
        ("O", "Property damage only (O)"),
    ]

    dataset = models.ForeignKey(
        UploadedDataset,
        on_delete=models.CASCADE,
        related_name="crash_records",
    )
    crash_id = models.CharField(max_length=64)
    crash_datetime = models.DateTimeField()
    severity = models.CharField(max_length=1, choices=SEVERITY_CHOICES)

    # Geospatial location; stored in PostGIS.
    location = gis_models.PointField(
        geography=True,
        null=True,
        blank=True,
        help_text="Crash location in WGS84 (lon/lat).",
    )

    roadway_name = models.CharField(max_length=128, blank=True)
    municipality = models.CharField(max_length=128, blank=True)

    posted_speed_limit = models.PositiveIntegerField(null=True, blank=True)
    vehicle_count = models.PositiveIntegerField(null=True, blank=True)
    person_count = models.PositiveIntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["crash_datetime"]),
            models.Index(fields=["severity"]),
            models.Index(fields=["dataset"]),
            gis_indexes.GiSTIndex(fields=["location"]),
        ]

    def __str__(self) -> str:
        return f"Crash {self.crash_id} ({self.get_severity_display()})"
