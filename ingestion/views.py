import csv
import io
import logging
import os
from typing import Any, Dict, List

from django.conf import settings
from django.core.files.base import ContentFile
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated

from .antivirus import scan_bytes_with_clamav
from .models import UploadedDataset
from . import validation

logger = logging.getLogger(__name__)

DANGEROUS_CSV_PREFIXES = ("=", "+", "-", "@")


def _safe_csv_value(value: Any) -> str:
    """Defend against CSV formula injection by prefixing dangerous cells."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    if value.startswith(DANGEROUS_CSV_PREFIXES):
        return "'" + value
    return value


def _step(
    step: str,
    step_status: str,
    details: str,
    *,
    severity: str | None = None,
    meta: Dict[str, Any] | None = None,
    code: str | None = None,
    is_hard_fail: bool | None = None,
) -> Dict[str, Any]:
    """Build a single validation step entry for the upload report.

    - status: "passed" | "failed" | "skipped"
    - severity: "error" | "warning" | "info"
      * "error" + is_hard_fail=True => upload will be rejected if this step fails
      * "warning" => non-fatal data quality issues (upload still accepted)
      * "info" => purely informational/housekeeping steps
    """
    item: Dict[str, Any] = {
        "step": step,
        "status": step_status,
        "details": details,
    }
    if severity is not None:
        item["severity"] = severity
    if is_hard_fail is not None:
        item["is_hard_fail"] = is_hard_fail
    if code is not None:
        item["code"] = code
    if meta is not None:
        item["meta"] = meta
    return item


def _user_is_admin(user) -> bool:
    return bool(
        getattr(user, "is_superuser", False)
        or getattr(user, "is_staff", False)
        or user.groups.filter(name="Admin").exists()
    )


@api_view(["POST"])
@parser_classes([MultiPartParser])
@permission_classes([IsAuthenticated])
def upload_dataset(request):
    """Secure upload gateway endpoint: POST /api/ingest/upload/

    This implements the ingestion pipeline described in the design doc:
    - file extension & size checks
    - MIME sniffing
    - antivirus scan (ClamAV)
    - schema validation
    - basic type/range checks
    - geo bounds checks

    If the upload is accepted, an UploadedDataset row is created with the
    raw file and the full structured validation report.
    """
    user = request.user
    steps: List[Dict[str, Any]] = []

    upload = request.FILES.get("file")
    if upload is None:
        error_code = "PAYLOAD_MISSING_FILE"
        steps.append(
            _step(
                "PAYLOAD",
                "failed",
                "Expected multipart/form-data with a single file field named 'file'.",
                severity="error",
                is_hard_fail=True,
                code=error_code,
            )
        )
        logger.info(
            "Upload rejected (no file provided)",
            extra={
                "event": "upload_rejected",
                "error_code": error_code,
                "user_id": getattr(user, "id", None),
                "username": getattr(user, "username", None),
            },
        )
        return JsonResponse(
            {
                "overall_status": "rejected",
                "error_code": error_code,
                "message": "No file was provided in the 'file' field.",
                "steps": steps,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Payload successfully received
    steps.append(
        _step(
            "PAYLOAD",
            "passed",
            "Received multipart/form-data with a single file field named 'file'.",
            severity="info",
            is_hard_fail=False,
        )
    )

    original_name = upload.name or "uploaded_dataset"
    ext = os.path.splitext(original_name)[1].lower()
    allowed_exts = {
        e.lower().strip()
        for e in getattr(
            settings,
            "INGESTION_ALLOWED_EXTENSIONS",
            [".csv"],
        )
        if e
    }

    if ext not in allowed_exts:
        error_code = "EXTENSION_NOT_ALLOWED"
        steps.append(
            _step(
                "EXTENSION_CHECK",
                "failed",
                (
                    f"Unsupported file extension '{ext}'. "
                    f"Allowed extensions: {', '.join(sorted(allowed_exts))}."
                ),
                severity="error",
                is_hard_fail=True,
                code=error_code,
            )
        )
        logger.info(
            "Upload rejected (extension)",
            extra={
                "event": "upload_rejected",
                "error_code": error_code,
                "user_id": getattr(user, "id", None),
                "username": getattr(user, "username", None),
                "filename": original_name,
                "extension": ext,
            },
        )
        return JsonResponse(
            {
                "overall_status": "rejected",
                "error_code": error_code,
                "message": "File extension is not allowed.",
                "steps": steps,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    steps.append(
        _step(
            "EXTENSION_CHECK",
            "passed",
            f"Extension '{ext}' is allowed.",
            severity="info",
            is_hard_fail=False,
        )
    )

    # Read file into memory once so we can reuse it for AV + validation.
    try:
        file_bytes = upload.read()
    except Exception as exc:  # noqa: BLE001
        error_code = "READ_FAILED"
        steps.append(
            _step(
                "READ_FILE",
                "failed",
                f"Failed to read uploaded file: {exc}",
                severity="error",
                is_hard_fail=True,
                code=error_code,
            )
        )
        logger.warning(
            "Upload rejected (read failure)",
            extra={
                "event": "upload_rejected",
                "error_code": error_code,
                "user_id": getattr(user, "id", None),
                "username": getattr(user, "username", None),
                "filename": original_name,
            },
        )
        return JsonResponse(
            {
                "overall_status": "rejected",
                "error_code": error_code,
                "message": "The uploaded file could not be read.",
                "steps": steps,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    size_bytes = len(file_bytes)
    max_size = getattr(settings, "INGESTION_MAX_FILE_SIZE_BYTES", 10 * 1024 * 1024)
    if size_bytes > max_size:
        error_code = "FILE_TOO_LARGE"
        steps.append(
            _step(
                "FILE_SIZE",
                "failed",
                (
                    f"File is too large: {size_bytes} bytes. "
                    f"The configured maximum is {max_size} bytes."
                ),
                meta={"size_bytes": size_bytes, "max_size_bytes": max_size},
                severity="error",
                is_hard_fail=True,
                code=error_code,
            )
        )
        logger.info(
            "Upload rejected (size)",
            extra={
                "event": "upload_rejected",
                "error_code": error_code,
                "user_id": getattr(user, "id", None),
                "username": getattr(user, "username", None),
                "filename": original_name,
                "size_bytes": size_bytes,
                "max_size_bytes": max_size,
            },
        )
        return JsonResponse(
            {
                "overall_status": "rejected",
                "error_code": error_code,
                "message": "The uploaded file exceeds the configured size limit.",
                "steps": steps,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    steps.append(
        _step(
            "FILE_SIZE",
            "passed",
            f"File size {size_bytes} bytes is within the configured limit.",
            meta={"size_bytes": size_bytes, "max_size_bytes": max_size},
            severity="info",
            is_hard_fail=False,
        )
    )

    # MIME sniffing
    sniff_result = validation.sniff_mime_type(
        file_bytes=file_bytes,
        original_name=original_name,
        declared_mime=getattr(upload, "content_type", None),
    )
    sniff_status = sniff_result["status"]
    mime_step_code = "MIME_MISMATCH" if sniff_status == "failed" else None
    mime_severity = "error" if sniff_status == "failed" else "info"
    mime_is_hard_fail = sniff_status == "failed"
    steps.append(
        _step(
            "MIME_SNIFF",
            sniff_status,
            sniff_result["details"],
            meta={
                "detected_mime_type": sniff_result["detected_mime_type"],
                "declared_mime_type": getattr(upload, "content_type", None),
            },
            severity=mime_severity,
            is_hard_fail=mime_is_hard_fail,
            code=mime_step_code,
        )
    )
    if sniff_status == "failed":
        error_code = "MIME_MISMATCH"
        logger.info(
            "Upload rejected (MIME mismatch)",
            extra={
                "event": "upload_rejected",
                "error_code": error_code,
                "user_id": getattr(user, "id", None),
                "username": getattr(user, "username", None),
                "filename": original_name,
                "detected_mime_type": sniff_result["detected_mime_type"],
                "declared_mime_type": getattr(upload, "content_type", None),
            },
        )
        return JsonResponse(
            {
                "overall_status": "rejected",
                "error_code": error_code,
                "message": "Declared Content-Type does not match the detected MIME type.",
                "steps": steps,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Antivirus scanning
    require_av = getattr(settings, "INGESTION_REQUIRE_AV", False)
    av_result = scan_bytes_with_clamav(file_bytes)
    av_status = av_result["status"]

    if av_status == "failed":
        av_step_code = "UPLOAD_INFECTED"
        av_severity = "error"
        av_is_hard_fail = True
    elif av_status == "skipped" and require_av:
        av_step_code = "AV_UNAVAILABLE_REQUIRED"
        av_severity = "error"
        av_is_hard_fail = True
    elif av_status == "skipped":
        av_step_code = None
        av_severity = "warning"
        av_is_hard_fail = False
    else:
        av_step_code = None
        av_severity = "info"
        av_is_hard_fail = False

    steps.append(
        _step(
            "AV_SCAN",
            av_status,
            av_result["details"],
            severity=av_severity,
            is_hard_fail=av_is_hard_fail,
            code=av_step_code,
        )
    )

    if av_status == "failed" or (require_av and av_status == "skipped"):
        # Malware discovered or AV required but unavailable; hard reject.
        if av_status == "failed":
            error_code = "UPLOAD_INFECTED"
            message = (
                "The uploaded file contains malware and has been rejected."
            )
        else:
            error_code = "AV_UNAVAILABLE_REQUIRED"
            message = (
                "Antivirus scanning is required but could not be completed; "
                "the upload was rejected."
            )

        logger.warning(
            "Upload rejected (antivirus)",
            extra={
                "event": "upload_rejected",
                "error_code": error_code,
                "user_id": getattr(user, "id", None),
                "username": getattr(user, "username", None),
                "filename": original_name,
                "size_bytes": size_bytes,
                "av_status": av_status,
            },
        )
        return JsonResponse(
            {
                "overall_status": "rejected",
                "error_code": error_code,
                "message": message,
                "steps": steps,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Load into pandas
    try:
        df = validation.load_dataframe_from_bytes(
            file_bytes=file_bytes,
            extension=ext,
        )
    except Exception as exc:  # noqa: BLE001
        error_code = "PARSE_FAILED"
        steps.append(
            _step(
                "PARSE_TABLE",
                "failed",
                f"Could not parse uploaded file into a tabular dataset: {exc}",
                severity="error",
                is_hard_fail=True,
                code=error_code,
            )
        )
        logger.warning(
            "Upload rejected (parse failure)",
            extra={
                "event": "upload_rejected",
                "error_code": error_code,
                "user_id": getattr(user, "id", None),
                "username": getattr(user, "username", None),
                "filename": original_name,
            },
        )
        return JsonResponse(
            {
                "overall_status": "rejected",
                "error_code": error_code,
                "message": "The uploaded file could not be parsed as a table.",
                "steps": steps,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    steps.append(
        _step(
            "PARSE_TABLE",
            "passed",
            f"Parsed dataset with {len(df)} rows and {len(df.columns)} columns.",
            severity="info",
            is_hard_fail=False,
        )
    )

    # Schema validation
    schema_result = validation.validate_schema(df)
    schema_status = "passed" if schema_result["is_valid"] else "failed"
    schema_version = schema_result.get("schema_version", validation.SCHEMA_VERSION)
    schema_step_code = "SCHEMA_MISSING_COLUMNS" if schema_status == "failed" else None
    schema_severity = "error" if schema_status == "failed" else "info"
    schema_is_hard_fail = schema_status == "failed"

    steps.append(
        _step(
            "SCHEMA_CHECK",
            schema_status,
            schema_result["details"],
            meta={
                "missing_columns": schema_result["missing_columns"],
                "unknown_columns": schema_result["unknown_columns"],
                "columns": schema_result["columns"],
                "schema_version": schema_version,
            },
            severity=schema_severity,
            is_hard_fail=schema_is_hard_fail,
            code=schema_step_code,
        )
    )
    if schema_status == "failed":
        error_code = "SCHEMA_MISSING_COLUMNS"
        logger.info(
            "Upload rejected (schema)",
            extra={
                "event": "upload_rejected",
                "error_code": error_code,
                "user_id": getattr(user, "id", None),
                "username": getattr(user, "username", None),
                "filename": original_name,
                "missing_columns": schema_result["missing_columns"],
            },
        )
        return JsonResponse(
            {
                "overall_status": "rejected",
                "error_code": error_code,
                "message": "The uploaded file is missing required columns.",
                "steps": steps,
                "schema": {
                    "missing_columns": schema_result["missing_columns"],
                    "unknown_columns": schema_result["unknown_columns"],
                    "columns": schema_result["columns"],
                    "schema_version": schema_version,
                },
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Type & range checks
    value_checks = validation.validate_value_types_and_ranges(df)
    value_severity = "warning" if value_checks["invalid_row_count"] > 0 else "info"
    steps.append(
        _step(
            "TYPE_AND_RANGE_CHECKS",
            "passed",
            value_checks["details"],
            meta=value_checks,
            severity=value_severity,
            is_hard_fail=False,
        )
    )

    # Geo bounds checks
    geo_checks = validation.validate_geo_bounds(df)
    geo_invalid = int(geo_checks.get("invalid_row_count", 0) or 0)
    if not geo_checks.get("has_coordinates"):
        geo_severity = "info"
    else:
        geo_severity = "warning" if geo_invalid > 0 else "info"

    steps.append(
        _step(
            "GEO_CHECKS",
            "passed",
            geo_checks["details"],
            meta=geo_checks,
            severity=geo_severity,
            is_hard_fail=False,
        )
    )

    # At this point the upload is accepted, even if some rows are flagged.
    overall_status = "accepted"

    detected_mime = sniff_result["detected_mime_type"]
    content = ContentFile(file_bytes)

    dataset = UploadedDataset(
        owner=user,
        original_filename=original_name,
        size_bytes=size_bytes,
        mime_type=detected_mime,
        schema_version=schema_version,
        status=UploadedDataset.Status.ACCEPTED,
        validation_report={
            "overall_status": overall_status,
            "schema_version": schema_version,
            "steps": steps,
            "schema": schema_result,
            "value_checks": value_checks,
            "geo_checks": geo_checks,
        },
    )
    dataset.raw_file.save(original_name, content, save=True)
    upload_id = str(dataset.id)

    logger.info(
        "Upload accepted",
        extra={
            "event": "upload_accepted",
            "user_id": getattr(user, "id", None),
            "username": getattr(user, "username", None),
            "upload_id": upload_id,
            "filename": original_name,
            "size_bytes": size_bytes,
            "mime_type": detected_mime,
            "schema_version": schema_version,
        },
    )

    response_payload: Dict[str, Any] = {
        "upload_id": upload_id,
        "overall_status": overall_status,
        "schema_version": schema_version,
        "steps": steps,
        "schema": {
            "missing_columns": schema_result["missing_columns"],
            "unknown_columns": schema_result["unknown_columns"],
            "columns": schema_result["columns"],
            "schema_version": schema_version,
        },
        "row_checks": {
            "total_rows": value_checks["total_rows"],
            "invalid_row_count": value_checks["invalid_row_count"],
            "invalid_geo_row_count": geo_checks.get("invalid_row_count", 0),
        },
    }

    return JsonResponse(response_payload, status=status.HTTP_201_CREATED)


# Mark the view with a DRF throttle scope for configuration in settings.REST_FRAMEWORK.
upload_dataset.throttle_scope = "ingest_upload"  # type: ignore[attr-defined]


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_uploads(request):
    """Return a list of uploads visible to the current user.

    Non-admin users see only their own uploads; admins see everything.
    """
    user = request.user
    is_admin = _user_is_admin(user)

    qs = UploadedDataset.objects.all()
    if not is_admin:
        qs = qs.filter(owner=user)
    qs = qs.order_by("-created_at")

    results: List[Dict[str, Any]] = []
    for dataset in qs:
        report = dataset.validation_report or {}
        # Support both old and new report shapes
        row_checks = report.get("row_checks") or report.get("value_checks") or {}
        geo_checks = report.get("geo_checks") or {}

        total_rows = row_checks.get("total_rows")
        invalid_row_count = row_checks.get("invalid_row_count")
        invalid_geo_row_count = row_checks.get(
            "invalid_geo_row_count", geo_checks.get("invalid_row_count")
        )

        results.append(
            {
                "id": str(dataset.id),
                "original_filename": dataset.original_filename,
                "created_at": dataset.created_at.isoformat(),
                "status": dataset.status,
                "schema_version": dataset.schema_version,
                "mime_type": dataset.mime_type,
                "total_rows": total_rows,
                "invalid_row_count": invalid_row_count,
                "invalid_geo_row_count": invalid_geo_row_count,
            }
        )

    return JsonResponse(
        {
            "count": len(results),
            "results": results,
            "next": None,
            "previous": None,
        }
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_upload_status(request, upload_id):
    """Retrieve the persisted validation report and metadata for a given upload.

    This lets the UI re-display validation results without requiring a re-upload.
    """
    dataset = get_object_or_404(UploadedDataset, id=upload_id)
    if dataset.owner_id != request.user.id and not _user_is_admin(request.user):
        return JsonResponse(
            {"detail": "You do not have permission to view this upload."},
            status=status.HTTP_403_FORBIDDEN,
        )

    report = dataset.validation_report or {}

    payload: Dict[str, Any] = {
        "upload_id": str(dataset.id),
        "original_filename": dataset.original_filename,
        "size_bytes": dataset.size_bytes,
        "mime_type": dataset.mime_type,
        "schema_version": dataset.schema_version,
        "status": dataset.status,
        "created_at": dataset.created_at.isoformat(),
        "validation_report": report,
    }
    return JsonResponse(payload)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def export_validation_csv(request, upload_id):
    """Export a one-row-per-step CSV summary of the validation report."""
    dataset = get_object_or_404(UploadedDataset, id=upload_id)
    if dataset.owner_id != request.user.id and not _user_is_admin(request.user):
        return JsonResponse(
            {"detail": "You do not have permission to export this upload."},
            status=status.HTTP_403_FORBIDDEN,
        )

    report = dataset.validation_report or {}
    steps = report.get("steps", [])
    row_checks = report.get("value_checks", {})
    geo_checks = report.get("geo_checks", {})

    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(
        [
            "step",
            "status",
            "code",
            "details",
            "schema_version",
            "meta_json",
        ]
    )

    schema_version = report.get("schema_version") or dataset.schema_version or ""

    for step in steps:
        writer.writerow(
            [
                _safe_csv_value(step.get("step")),
                _safe_csv_value(step.get("status")),
                _safe_csv_value(step.get("code", "")),
                _safe_csv_value(step.get("details", "")),
                _safe_csv_value(schema_version),
                _safe_csv_value(step.get("meta")),
            ]
        )

    # Append a couple of synthetic rows to capture row/geo summaries.
    writer.writerow([])
    writer.writerow(["summary", "row_checks", "", "", "", ""])
    writer.writerow(
        [
            "total_rows",
            row_checks.get("total_rows", ""),
            "",
            "",
            "",
            "",
        ]
    )
    writer.writerow(
        [
            "invalid_row_count",
            row_checks.get("invalid_row_count", ""),
            "",
            "",
            "",
            "",
        ]
    )
    writer.writerow(
        [
            "invalid_geo_row_count",
            geo_checks.get("invalid_row_count", ""),
            "",
            "",
            "",
            "",
        ]
    )

    csv_bytes = buffer.getvalue().encode("utf-8")
    response = HttpResponse(csv_bytes, content_type="text/csv")
    response["Content-Disposition"] = (
        f"attachment; filename=validation_{dataset.id}.csv"
    )
    return response


# Throttle scope for exports
export_validation_csv.throttle_scope = "exports"  # type: ignore[attr-defined]
