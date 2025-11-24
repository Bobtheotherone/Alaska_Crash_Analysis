import uuid

from django.conf import settings
from django.db import models


class UploadedDataset(models.Model):
    """Raw uploaded crash datasets plus their validation metadata.

    The ML / data-cleaning team can pick these up for deeper processing.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        ACCEPTED = "accepted", "Accepted"
        REJECTED = "rejected", "Rejected"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="uploaded_datasets",
    )
    original_filename = models.CharField(max_length=255)
    size_bytes = models.BigIntegerField()
    mime_type = models.CharField(max_length=255, blank=True)
    # Schema configuration used at validation time (e.g. "mmucc-alaska-v1").
    schema_version = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    raw_file = models.FileField(upload_to="uploaded_datasets/")
    # Full structured status report returned from the ingestion endpoint.
    validation_report = models.JSONField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(fields=["owner"]),
            models.Index(fields=["status"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.original_filename} ({self.id})"
