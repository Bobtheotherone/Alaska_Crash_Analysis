from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.http import JsonResponse
from django.urls import path, include


def healthcheck(_request):
    """Simple readiness/liveness probe used by deployment."""
    return JsonResponse({"status": "ok"})


urlpatterns = [
    path("admin/", admin.site.urls),
    # Upload gateway / ingestion endpoints
    path("api/ingest/", include("ingestion.urls")),
    # Crashdata / visualization endpoints
    path("api/crashdata/", include("crashdata.urls")),
    # Analysis / model endpoints
    path("api/", include("analysis.urls")),
    # Healthcheck
    path("health/", healthcheck, name="healthcheck"),
    # Frontend SPA
    path("", include("frontend.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
