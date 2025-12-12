import uuid

from django.conf import settings
from django.db import models

from ingestion.models import UploadedDataset


class ModelJob(models.Model):
    """Track long-running model / analysis jobs.

    This model intentionally does *not* contain raw prediction payloads; those
    should live in separate, model-specific tables managed by the ML team.
    """

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    upload = models.ForeignKey(
        UploadedDataset,
        on_delete=models.CASCADE,
        related_name="model_jobs",
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="model_jobs",
    )
    model_name = models.CharField(max_length=128)
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.QUEUED,
    )
    parameters = models.JSONField(blank=True, default=dict)
    result_metadata = models.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["owner"]),
            models.Index(fields=["upload"]),
            models.Index(fields=["status"]),
        ]

    def __str__(self) -> str:
        return f"{self.model_name} on {self.upload_id} ({self.status})"
