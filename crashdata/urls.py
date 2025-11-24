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
]
