from django.contrib import admin

from .models import CrashRecord


@admin.register(CrashRecord)
class CrashRecordAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "dataset",
        "crash_id",
        "crash_datetime",
        "severity",
        "municipality",
    )
    list_filter = ("severity", "crash_datetime", "municipality")
    search_fields = ("crash_id", "dataset__original_filename", "municipality", "roadway_name")
