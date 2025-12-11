from django.urls import path

from . import views

urlpatterns = [
    # Legacy/simple upload + summary endpoint used by the GUI.
    path("upload/", views.upload_and_analyze, name="upload-and-analyze"),

    # Lightweight authenticated health-check used by the login form.
    path("auth/ping/", views.auth_ping, name="auth-ping"),

    # Static/example payload for frontend smoke tests; falls back to a
    # synthetic dataframe if the example CSV is not present.
    path("summary/", views.summary_example, name="summary-example"),

    # Abstract ML/model endpoints.
    path("models/run/", views.model_run, name="models-run"),
    path("models/results/<uuid:job_id>/", views.model_results, name="models-results"),
]
