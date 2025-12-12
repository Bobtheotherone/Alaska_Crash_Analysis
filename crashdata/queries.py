from datetime import datetime
from typing import Iterable, Optional

from django.contrib.gis.geos import Polygon
from django.db.models import Count, QuerySet

from ingestion.models import UploadedDataset

from .models import CrashRecord


def _apply_common_filters(
    qs: QuerySet,
    severity: Optional[Iterable[str]] = None,
    municipality: str | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
) -> QuerySet:
    if severity:
        if isinstance(severity, (list, tuple, set)):
            qs = qs.filter(severity__in=list(severity))
        else:
            qs = qs.filter(severity=severity)
    if municipality:
        qs = qs.filter(municipality__iexact=municipality)
    if start_datetime is not None:
        qs = qs.filter(crash_datetime__gte=start_datetime)
    if end_datetime is not None:
        qs = qs.filter(crash_datetime__lte=end_datetime)
    return qs


def severity_histogram(
    dataset: Optional[UploadedDataset] = None,
    *,
    municipality: str | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
):
    """Aggregate crashes by KABCO severity for a given dataset.

    Returns a queryset of dicts: [{"severity": "K", "count": 123}, ...]
    """
    qs: QuerySet = CrashRecord.objects.all()
    if dataset is not None:
        qs = qs.filter(dataset=dataset)

    qs = _apply_common_filters(
        qs,
        municipality=municipality,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )
    return qs.values("severity").annotate(count=Count("id")).order_by("severity")


def crashes_within_bbox(
    dataset: Optional[UploadedDataset],
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    *,
    severity: Optional[Iterable[str]] = None,
    municipality: str | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
) -> QuerySet:
    """Return crash records that fall within the provided bounding box."""
    bbox = Polygon.from_bbox((min_lon, min_lat, max_lon, max_lat))
    bbox.srid = 4326
    qs: QuerySet = CrashRecord.objects.filter(location__within=bbox)
    if dataset is not None:
        qs = qs.filter(dataset=dataset)

    qs = _apply_common_filters(
        qs,
        severity=severity,
        municipality=municipality,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )
    return qs
