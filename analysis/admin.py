from django.contrib import admin

from .models import ModelJob


@admin.register(ModelJob)
class ModelJobAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "model_name",
        "status",
        "owner",
        "upload",
        "created_at",
    )
    list_filter = ("status", "model_name", "created_at")
    search_fields = ("id", "owner__username", "upload__id", "model_name")
