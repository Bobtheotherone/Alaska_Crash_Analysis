from django.contrib import admin

from .models import UploadedDataset


@admin.register(UploadedDataset)
class UploadedDatasetAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "owner",
        "original_filename",
        "schema_version",
        "size_bytes",
        "mime_type",
        "status",
        "created_at",
    )
    list_filter = ("status", "schema_version", "created_at")
    search_fields = ("original_filename", "owner__username", "schema_version")
