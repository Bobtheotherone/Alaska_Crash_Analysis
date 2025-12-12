import csv
import io
import logging
from datetime import datetime, time

from django.conf import settings
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from django.db.models import Count, Min, Max, Q

from ingestion.models import UploadedDataset

from .importer import ImportError, import_crash_records_for_dataset
from . import queries
from .models import CrashRecord

logger = logging.getLogger(__name__)

DANGEROUS_CSV_PREFIXES = ("=", "+", "-", "@")


def _safe_csv_value(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    if value.startswith(DANGEROUS_CSV_PREFIXES):
        return "'" + value
    return value


def _user_is_admin(user) -> bool:
    return bool(
        getattr(user, "is_superuser", False)
        or getattr(user, "is_staff", False)
        or user.groups.filter(name="Admin").exists()
    )


def _get_dataset_for_user(upload_id, user) -> UploadedDataset:
    dataset = get_object_or_404(UploadedDataset, id=upload_id)
    if dataset.owner_id != getattr(user, "id", None) and not _user_is_admin(user):
        raise Http404("No such upload.")
    return dataset


def _parse_datetime_param(raw: str | None):
    if not raw:
        return None

    dt = parse_datetime(raw)
    if dt is not None:
        if timezone.is_naive(dt):
            dt = timezone.make_aware(dt, timezone=timezone.utc)
        return dt

    d = parse_date(raw)
    if d is not None:
        dt = datetime.combine(d, time.min)
        return timezone.make_aware(dt, timezone=timezone.utc)

    return None


def _parse_optional_datetime_param(raw: str | None, field_name: str):
    """Parse an optional ISO datetime/date. Returns None when absent, raises on bad strings."""
    if not raw:
        return None

    dt = _parse_datetime_param(raw)
    if dt is None:
        raise ValueError(f"{field_name} must be ISO-8601 datetimes or dates.")
    return dt


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def severity_histogram_view(request):
    """Return a severity histogram for a given upload (or all uploads)."""
    upload_id = request.query_params.get("upload_id")
    municipality = request.query_params.get("municipality") or None
    try:
        start_dt = _parse_optional_datetime_param(
            request.query_params.get("start_datetime"), "start_datetime"
        )
        end_dt = _parse_optional_datetime_param(
            request.query_params.get("end_datetime"), "end_datetime"
        )
    except ValueError as exc:
        return JsonResponse({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    dataset = None
    if upload_id:
        dataset = _get_dataset_for_user(upload_id, request.user)

    histogram_qs = queries.severity_histogram(
        dataset=dataset,
        municipality=municipality,
        start_datetime=start_dt,
        end_datetime=end_dt,
    )
    data = [{"severity": row["severity"], "count": row["count"]} for row in histogram_qs]

    return JsonResponse(
        {
            "upload_id": str(dataset.id) if dataset else None,
            "filters": {
                "municipality": municipality,
                "start_datetime": start_dt.isoformat() if start_dt else None,
                "end_datetime": end_dt.isoformat() if end_dt else None,
            },
            "histogram": data,
        }
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def crashes_within_bbox_view(request):
    """Return crashes inside a bounding box as a GeoJSON FeatureCollection."""
    upload_id = request.query_params.get("upload_id")
    required_params = ["min_lon", "min_lat", "max_lon", "max_lat"]
    try:
        min_lon = float(request.query_params["min_lon"])
        min_lat = float(request.query_params["min_lat"])
        max_lon = float(request.query_params["max_lon"])
        max_lat = float(request.query_params["max_lat"])
    except KeyError as missing:
        return JsonResponse(
            {"detail": f"Missing required parameter: {missing.args[0]}", "required": required_params},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except ValueError:
        return JsonResponse(
            {"detail": "Bounding box parameters must be valid floats."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    dataset = None
    if upload_id:
        dataset = _get_dataset_for_user(upload_id, request.user)

    severity_param = request.query_params.get("severity") or ""
    if severity_param:
        severity = [s.strip() for s in severity_param.split(",") if s.strip()]
    else:
        severity = None

    municipality = request.query_params.get("municipality") or None
    try:
        start_dt = _parse_optional_datetime_param(
            request.query_params.get("start_datetime"), "start_datetime"
        )
        end_dt = _parse_optional_datetime_param(
            request.query_params.get("end_datetime"), "end_datetime"
        )
    except ValueError as exc:
        return JsonResponse({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    try:
        limit = int(request.query_params.get("limit", "5000"))
    except ValueError:
        return JsonResponse(
            {"detail": "limit must be an integer."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if limit <= 0 or limit > 50000:
        limit = 5000

    try:
        qs = queries.crashes_within_bbox(
            dataset=dataset,
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            severity=severity,
            municipality=municipality,
            start_datetime=start_dt,
            end_datetime=end_dt,
        ).select_related("dataset")
    except Exception:
        logger.exception(
            "Failed to query crashes within bbox",
            extra={
                "upload_id": upload_id,
                "bbox": [min_lon, min_lat, max_lon, max_lat],
                "start_dt": start_dt,
                "end_dt": end_dt,
            },
        )
        return JsonResponse(
            {"detail": "Internal error while querying crashes."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    features = []
    for crash in qs[:limit]:
        if not crash.location:
            continue
        lon, lat = crash.location.coords
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "id": crash.id,
                    "crash_id": crash.crash_id,
                    "severity": crash.severity,
                    "crash_datetime": crash.crash_datetime.isoformat(),
                    "roadway_name": crash.roadway_name,
                    "municipality": crash.municipality,
                    "posted_speed_limit": crash.posted_speed_limit,
                    "dataset_id": str(crash.dataset_id),
                },
            }
        )

    return JsonResponse(
        {
            "type": "FeatureCollection",
            "count": len(features),
            "limit": limit,
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "features": features,
        }
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def heatmap_view(request):
    """Return a lightweight grid-aggregated heatmap for crashes in a bbox."""
    upload_id = request.query_params.get("upload_id")
    try:
        min_lon = float(request.query_params["min_lon"])
        min_lat = float(request.query_params["min_lat"])
        max_lon = float(request.query_params["max_lon"])
        max_lat = float(request.query_params["max_lat"])
    except KeyError as missing:
        return JsonResponse(
            {
                "detail": f"Missing required parameter: {missing.args[0]}",
                "required": ["min_lon", "min_lat", "max_lon", "max_lat"],
            },
            status=status.HTTP_400_BAD_REQUEST,
        )
    except ValueError:
        return JsonResponse(
            {"detail": "Bounding box parameters must be valid floats."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    dataset = None
    if upload_id:
        dataset = _get_dataset_for_user(upload_id, request.user)

    try:
        grid_size = float(request.query_params.get("grid_size", "0.05"))
    except ValueError:
        return JsonResponse(
            {"detail": "grid_size must be a float representing degrees."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if grid_size <= 0:
        grid_size = 0.05

    severity_param = request.query_params.get("severity") or ""
    if severity_param:
        severity = [s.strip() for s in severity_param.split(",") if s.strip()]
    else:
        severity = None

    municipality = request.query_params.get("municipality") or None
    try:
        start_dt = _parse_optional_datetime_param(
            request.query_params.get("start_datetime"), "start_datetime"
        )
        end_dt = _parse_optional_datetime_param(
            request.query_params.get("end_datetime"), "end_datetime"
        )
    except ValueError as exc:
        return JsonResponse({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    try:
        qs = queries.crashes_within_bbox(
            dataset=dataset,
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            severity=severity,
            municipality=municipality,
            start_datetime=start_dt,
            end_datetime=end_dt,
        ).only("location")
    except Exception:
        logger.exception(
            "Failed to query heatmap within bbox",
            extra={
                "upload_id": upload_id,
                "bbox": [min_lon, min_lat, max_lon, max_lat],
                "start_dt": start_dt,
                "end_dt": end_dt,
            },
        )
        return JsonResponse(
            {"detail": "Internal error while querying heatmap."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # Aggregate into simple grid cells.
    buckets: dict[tuple[int, int], int] = {}
    for crash in qs:
        if not crash.location:
            continue
        lon, lat = crash.location.coords
        x = int(lon // grid_size)
        y = int(lat // grid_size)
        key = (x, y)
        buckets[key] = buckets.get(key, 0) + 1

    cells = []
    for (x, y), count in buckets.items():
        center_lon = (x + 0.5) * grid_size
        center_lat = (y + 0.5) * grid_size
        cells.append(
            {
                "count": count,
                "center": [center_lon, center_lat],
                "grid_size": grid_size,
            }
        )

    return JsonResponse(
        {
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "grid_size": grid_size,
            "cells": cells,
        }
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def export_crashes_csv(request):
    """Export a filtered subset of crashes as CSV, with a hard row cap."""
    upload_id = request.query_params.get("upload_id")
    if not upload_id:
        return JsonResponse(
            {"detail": "upload_id is required for crash exports."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    dataset = _get_dataset_for_user(upload_id, request.user)

    severity_param = request.query_params.get("severity") or ""
    if severity_param:
        severity = [s.strip() for s in severity_param.split(",") if s.strip()]
    else:
        severity = None

    municipality = request.query_params.get("municipality") or None
    try:
        start_dt = _parse_optional_datetime_param(
            request.query_params.get("start_datetime"), "start_datetime"
        )
        end_dt = _parse_optional_datetime_param(
            request.query_params.get("end_datetime"), "end_datetime"
        )
    except ValueError as exc:
        return JsonResponse({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    qs = CrashRecord.objects.filter(dataset=dataset)
    if severity:
        qs = qs.filter(severity__in=severity)
    if municipality:
        qs = qs.filter(municipality__iexact=municipality)
    if start_dt is not None:
        qs = qs.filter(crash_datetime__gte=start_dt)
    if end_dt is not None:
        qs = qs.filter(crash_datetime__lte=end_dt)

    try:
        max_rows = int(request.query_params.get("max_rows", "100000"))
    except ValueError:
        return JsonResponse(
            {"detail": "max_rows must be an integer."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if max_rows <= 0 or max_rows > 500000:
        max_rows = 100000

    row_count = qs.count()
    if row_count > max_rows:
        return JsonResponse(
            {
                "detail": (
                    "Refusing to export more than max_rows rows in a single request."
                ),
                "row_count": row_count,
                "max_rows": max_rows,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    buffer = io.StringIO()
    writer = csv.writer(buffer)

    header = [
        "id",
        "dataset_id",
        "crash_id",
        "crash_datetime",
        "severity",
        "roadway_name",
        "municipality",
        "posted_speed_limit",
        "vehicle_count",
        "person_count",
        "lon",
        "lat",
    ]
    writer.writerow(header)

    for crash in qs.iterator():
        lon = lat = ""
        if crash.location:
            lon, lat = crash.location.coords
        writer.writerow(
            [
                crash.id,
                crash.dataset_id,
                _safe_csv_value(crash.crash_id),
                crash.crash_datetime.isoformat(),
                crash.severity,
                _safe_csv_value(crash.roadway_name),
                _safe_csv_value(crash.municipality),
                crash.posted_speed_limit,
                crash.vehicle_count,
                crash.person_count,
                lon,
                lat,
            ]
        )

    csv_bytes = buffer.getvalue().encode("utf-8")
    response = HttpResponse(csv_bytes, content_type="text/csv")
    response["Content-Disposition"] = (
        f"attachment; filename=crashes_{dataset.id}.csv"
    )
    return response


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def dataset_stats_view(request, upload_id: str):
    """Return basic stats for a dataset to inform the map UI."""
    dataset = _get_dataset_for_user(upload_id, request.user)
    qs = CrashRecord.objects.filter(dataset=dataset)
    agg = qs.aggregate(
        crashrecord_count=Count("id"),
        mappable_count=Count("id", filter=Q(location__isnull=False)),
        min_dt=Min("crash_datetime"),
        max_dt=Max("crash_datetime"),
    )
    response = JsonResponse(
        {
            "upload_id": str(dataset.id),
            "crashrecord_count": agg["crashrecord_count"] or 0,
            "mappable_count": agg["mappable_count"] or 0,
            "min_crash_datetime": agg["min_dt"].isoformat() if agg["min_dt"] else None,
            "max_crash_datetime": agg["max_dt"].isoformat() if agg["max_dt"] else None,
        }
    )
    version = getattr(settings, "SPECTACULAR_SETTINGS", {}).get("VERSION")
    if version:
        response["X-Alaska-Backend-Version"] = str(version)
    return response


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def import_crash_records_view(request, upload_id: str):
    """Run the crash record import for a dataset so the map has data."""
    start = timezone.now()
    dataset = _get_dataset_for_user(upload_id, request.user)
    try:
        imported, mappable = import_crash_records_for_dataset(dataset)
    except ImportError as exc:
        return JsonResponse({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            "Unexpected error importing crash records",
            extra={"dataset_id": str(upload_id)},
        )
        return JsonResponse(
            {"detail": "Internal error importing crash records."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    response = JsonResponse(
        {
            "status": "completed",
            "upload_id": str(dataset.id),
            "imported": imported,
            "mappable": mappable,
            "duration_sec": (timezone.now() - start).total_seconds(),
        }
    )
    response["X-Alaska-Import-Duration"] = str(
        round((timezone.now() - start).total_seconds(), 3)
    )
    return response


# Throttle scope for exports
export_crashes_csv.throttle_scope = "exports"  # type: ignore[attr-defined]
