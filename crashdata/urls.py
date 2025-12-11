from django.urls import path

from . import views

urlpatterns = [
    path(
        "severity-histogram/",
        views.severity_histogram_view,
        name="crashdata-severity-histogram",
    ),
    path(
        "crashes-within-bbox/",
        views.crashes_within_bbox_view,
        name="crashdata-crashes-within-bbox",
    ),
    path(
        "heatmap/",
        views.heatmap_view,
        name="crashdata-heatmap",
    ),
    path(
        "exports/crashes.csv",
        views.export_crashes_csv,
        name="crashdata-export-crashes-csv",
    ),
    path(
        "datasets/<uuid:upload_id>/stats/",
        views.dataset_stats_view,
        name="crashdata-dataset-stats",
    ),
    path(
        "datasets/<uuid:upload_id>/import-crash-records/",
        views.import_crash_records_view,
        name="crashdata-import-crash-records",
    ),
]
