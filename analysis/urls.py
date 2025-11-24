from django.urls import path
from . import views

urlpatterns = [
    # Legacy/simple upload + summary endpoint
    path("upload/", views.upload_and_analyze, name="upload-and-analyze"),
    # Static example payload for frontend smoke tests
    path("summary/", views.summary_example, name="summary-example"),
    # Abstract ML/model endpoints; to be implemented by the ML team.
    path("models/run/", views.model_run, name="models-run"),
    path("models/results/<uuid:job_id>/", views.model_results, name="models-results"),
]
