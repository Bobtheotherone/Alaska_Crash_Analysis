from django.urls import path

from . import views

urlpatterns = [
    path("upload/", views.upload_dataset, name="ingest-upload"),
    path("uploads/", views.list_uploads, name="ingest-upload-list"),
    path(
        "uploads/<uuid:upload_id>/",
        views.get_upload_status,
        name="ingest-upload-status",
    ),
    path(
        "uploads/<uuid:upload_id>/export/validation.csv",
        views.export_validation_csv,
        name="ingest-validation-export",
    ),
]
