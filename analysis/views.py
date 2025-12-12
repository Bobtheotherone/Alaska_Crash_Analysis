from __future__ import annotations

import io
import logging
import math
import numbers
from typing import Any, Dict, List

from django.contrib.auth import authenticate
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import (
    api_view,
    authentication_classes,
    permission_classes,
    throttle_classes,
)
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response

from ingestion.models import UploadedDataset
from ingestion.validation import validate_dataframe_from_upload
from .models import ModelJob
from .throttling import BurstRateThrottle
from .ml_core.models import MODEL_REGISTRY
from .ml_core.worker import enqueue_model_job

logger = logging.getLogger(__name__)


def _json_safe(obj: Any) -> Any:
  """
  Recursively convert a nested structure into something that can be
  safely encoded as strict JSON.

  - Any non-finite numbers (NaN, ±inf) become None → `null` in JSON.
  - Dict keys are stringified.
  - Tuples become lists.
  - Unknown scalar types fall back to `str(obj)`.
  """
  # Covers int, float, numpy.numeric, etc.
  if isinstance(obj, numbers.Real):
      val = float(obj)
      if math.isfinite(val):
          return val
      return None

  if obj is None or isinstance(obj, (str, bool)):
      return obj

  if isinstance(obj, dict):
      return {str(k): _json_safe(v) for k, v in obj.items()}

  if isinstance(obj, (list, tuple)):
      return [_json_safe(v) for v in obj]

  # Pandas / numpy scalars and anything exotic we haven't special-cased
  try:
      return str(obj)
  except Exception:
      return None


def build_summary_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute a lightweight, JSON-safe statistical summary of a dataframe."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    # Head of the data
    head_df = df.head()
    head = head_df.to_dict(orient="list")

    # Descriptive stats; include all dtypes.
    describe_df = df.describe(include="all")
    describe = describe_df.to_dict()

    # Sample up to 10 non-null values per column
    column_samples: Dict[str, List[Any]] = {}
    for col in df.columns:
        samples = df[col].dropna().unique()[:10].tolist()
        column_samples[col] = samples

    raw_summary: Dict[str, Any] = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "info": info_str,
        "head": head,
        "describe": describe,
        "column_samples": column_samples,
    }
    return _json_safe(raw_summary)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([BurstRateThrottle])
def upload_and_analyze(request):
    """Upload a file, validate it, persist it, and return a quick profile.

    This view now does *three* things:

    1. Uses ``ingestion.validation.validate_dataframe_from_upload`` to run
       the same validation pipeline as the ingestion API (schema + basic
       value checks).
    2. Persists the uploaded file as an :class:`UploadedDataset` row so it
       shows up in the Django admin and can be reused by the CLI and ML
       worker APIs.
    3. Returns a compact JSON summary for the front-end to visualise.

    The response JSON matches the previous summary fields and adds:

    * ``upload_id`` – primary key of the new :class:`UploadedDataset`.
    * ``original_filename`` – for convenience in the UI.
    """
    upload = request.FILES.get("file")
    if not upload:
        return JsonResponse({"detail": "No file uploaded."}, status=400)

    # 1) Validation into a DataFrame
    try:
        df = validate_dataframe_from_upload(upload)
    except ValueError as exc:
        return JsonResponse({"detail": str(exc)}, status=400)

    # 2) Persist the raw file as an UploadedDataset so the dataset can be
    #    accessed later by the poster visualisation script and ML worker.
    try:
        upload.seek(0)
    except Exception:
        # Django's UploadedFile objects are seekable in normal operation,
        # so this is purely defensive.
        pass

    mime_type = getattr(upload, "content_type", "") or ""

    dataset = UploadedDataset.objects.create(
        owner=request.user,
        original_filename=upload.name,
        size_bytes=upload.size,
        mime_type=mime_type,
        status=UploadedDataset.Status.ACCEPTED,
        # We do not currently construct the rich multi-step
        # ``validation_report`` here; leave it null so it is obvious this
        # row came from the quick-analysis endpoint.
        validation_report=None,
    )
    # Save the file to the FileField – this will write into MEDIA_ROOT.
    dataset.raw_file.save(upload.name, upload, save=True)

    # 3) Build the same lightweight summary as before and add identifiers.
    summary = build_summary_from_dataframe(df)
    summary["upload_id"] = str(dataset.id)
    summary["original_filename"] = dataset.original_filename

    return JsonResponse(summary, status=200)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def summary_example(request):
    """
    Return a canned dataframe summary useful for front-end mocks.

    In development we *try* to load ``analysis/example_data/crash_sample.csv``.
    If that file is missing, we fall back to a tiny in-memory dataset so
    the endpoint never fails just because of a local file.
    """
    from pathlib import Path

    example_path = Path(__file__).resolve().parent / "example_data" / "crash_sample.csv"

    try:
        df = pd.read_csv(example_path)
    except FileNotFoundError:
        df = pd.DataFrame(
            {
                "CrashID": [1, 2, 3],
                "Latitude": [61.2, 61.3, 61.4],
                "Longitude": [-149.9, -149.8, -149.7],
                "InjurySeverity": ["Minor", "Serious", "Fatal"],
            }
        )

    summary = build_summary_from_dataframe(df)
    return JsonResponse(summary, status=200)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def auth_ping(request):
    """Lightweight endpoint used by the frontend to verify credentials.

    Returns basic information about the currently authenticated user.
    """
    user = request.user
    return JsonResponse(
        {
            "username": user.get_username(),
            "is_superuser": user.is_superuser,
            "is_staff": user.is_staff,
        },
        status=200,
    )


@api_view(["POST"])
@authentication_classes([])  # Explicitly avoid BasicAuth challenges here
@permission_classes([AllowAny])
def auth_login(request):
    """Username/password login endpoint for the React app.

    Returns 200 on success and 400 on invalid credentials without
    emitting ``WWW-Authenticate`` headers that trigger browser popups.
    """
    payload = request.data or {}
    username = payload.get("username")
    password = payload.get("password")

    if not username or not password:
        return Response(
            {"detail": "Username and password are required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    user = authenticate(request, username=username, password=password)
    if user is None:
        return Response(
            {"detail": "Invalid credentials."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    return Response(
        {
            "detail": "ok",
            "username": user.get_username(),
            "is_superuser": user.is_superuser,
            "is_staff": user.is_staff,
        },
        status=status.HTTP_200_OK,
    )


def _user_is_admin(user) -> bool:
    """
    Treat staff/superusers and users in the "Admin"/"admin" group as admins.

    This mirrors the ingestion + crashdata apps so permissions are
    consistent across all APIs.
    """
    return bool(
        getattr(user, "is_superuser", False)
        or getattr(user, "is_staff", False)
        or user.groups.filter(name__in=["Admin", "admin"]).exists()
    )


# Keep a lightweight, description-only mapping for backwards compatibility.
SUPPORTED_MODELS: Dict[str, str] = {
    name: spec.description for name, spec in MODEL_REGISTRY.items()
}


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def model_run(request):
    """Start a model training run for an existing UploadedDataset.

    Expected JSON body::

        {
            "dataset_id": "<uuid of UploadedDataset>",  # or "upload_id"
            "model_name": "crash_severity_risk_v1",     # or "model"
            "parameters": {...}  # optional, free-form
        }

    Behavioural notes:

    * A ModelJob row is created in the database with status="queued".
    * The worker is enqueued (via a lightweight background thread) to
      process the job asynchronously.
    * The response includes the job id and a URL for polling the results.
    """
    user = request.user

    # Only admins can start model runs (staff / superuser / Admin group).
    if not _user_is_admin(user):
        return JsonResponse(
            {"detail": "Only admin users can start model runs."},
            status=403,
        )

    try:
        payload = request.data
    except Exception:
        payload = {}

    # Accept both "dataset_id" and the earlier "upload_id" spelling.
    dataset_id = payload.get("dataset_id") or payload.get("upload_id")

    # Accept both the newer "model_name" and the original "model" key.
    model_name = payload.get("model_name") or payload.get("model")

    parameters = payload.get("parameters") or {}

    errors: Dict[str, str] = {}
    if not dataset_id:
        errors["dataset_id"] = "This field is required."
    if not model_name:
        errors["model_name"] = "This field is required."
    elif model_name not in MODEL_REGISTRY:
        errors["model_name"] = (
            f"Unsupported model_name '{model_name}'. "
            f"Supported options are: {', '.join(sorted(MODEL_REGISTRY.keys()))}."
        )

    if not isinstance(parameters, dict):
        errors["parameters"] = "Must be a JSON object."

    if errors:
        return JsonResponse({"errors": errors}, status=400)

    dataset = get_object_or_404(UploadedDataset, pk=dataset_id)

    # Defence-in-depth ownership check (in case admin guard changes later).
    if dataset.owner_id != user.id and not _user_is_admin(user):
        return JsonResponse(
            {"detail": "You do not have permission to run models on this upload."},
            status=403,
        )

    job = ModelJob.objects.create(
        owner=user,
        upload=dataset,
        model_name=model_name,
        parameters=parameters,
        status=ModelJob.Status.QUEUED,
    )

    # Fire-and-forget background worker – matches the async contract.
    enqueue_model_job(job.id)

    # Avoid URL name coupling; build the polling URL directly.
    results_path = f"/api/models/results/{job.id}/"
    results_url = request.build_absolute_uri(results_path)

    return JsonResponse(
        {
            "job_id": str(job.id),
            "status": job.status,
            # Provide both spellings for compatibility with the docs + UI.
            "dataset_id": str(dataset.id),
            "upload_id": str(dataset.id),
            "model_name": job.model_name,
            "model": job.model_name,
            "parameters": parameters,
            "results_url": results_url,
            "detail": "Job has been created and queued.",
        },
        status=202,
    )


# Throttling scope used by the custom BurstRateThrottle class.
model_run.throttle_scope = "model_run"  # type: ignore[attr-defined]


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def model_results(request, job_id: str):
    """Fetch the JSON ``result_metadata`` for a :class:`ModelJob`.

    This endpoint is read-only and returns whatever the ML worker stored
    into ``ModelJob.result_metadata``, along with basic job bookkeeping
    fields.

    Status codes follow the documented contract:

    * 202 Accepted – job is queued or running
    * 200 OK – job has succeeded or failed
    """
    job = get_object_or_404(ModelJob, pk=job_id)

    if not _user_is_admin(request.user) and job.upload.owner != request.user:
        return JsonResponse(
            {"detail": "You do not have permission to view this job."},
            status=403,
        )

    payload = {
        "job_id": str(job.id),
        "status": job.status,
        # Provide both names for compatibility with the docs/UI.
        "model_name": job.model_name,
        "model": job.model_name,
        "dataset_id": str(job.upload_id),
        "upload_id": str(job.upload_id),
        "parameters": job.parameters or {},
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "result_metadata": job.result_metadata or {},
    }

    if job.status in (ModelJob.Status.QUEUED, ModelJob.Status.RUNNING):
        http_status = 202
    else:
        http_status = 200

    return JsonResponse(payload, status=http_status)
