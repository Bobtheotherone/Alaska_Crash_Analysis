
import io
import logging
from typing import Any, Dict, List

import pandas as pd
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated

from ingestion.models import UploadedDataset
from ingestion.validation import validate_dataframe_from_upload, load_dataframe_from_bytes
from .models import ModelJob
from .throttling import BurstRateThrottle
from .ml_core.models import MODEL_REGISTRY
from .ml_core.worker import enqueue_model_job

logger = logging.getLogger(__name__)


def build_summary_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute a lightweight statistical summary of a dataframe."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    head = df.head().to_dict(orient="list")
    describe = df.describe(include="all").to_dict()

    column_samples: Dict[str, List[Any]] = {}
    for col in df.columns:
        column_samples[col] = df[col].dropna().unique()[:10].tolist()

    return {
        "shape": df.shape,
        "info": info_str,
        "head": head,
        "describe": describe,
        "column_samples": column_samples,
    }


@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([BurstRateThrottle])
def upload_and_analyze(request):
    """Upload a CSV/Excel file and return a basic profile for quick inspection."""
    upload = request.FILES.get("file")
    if not upload:
        return JsonResponse({"detail": "No file uploaded."}, status=400)

    try:
        df = validate_dataframe_from_upload(upload)
    except ValueError as exc:
        return JsonResponse({"detail": str(exc)}, status=400)

    summary = build_summary_from_dataframe(df)
    return JsonResponse(summary, status=200)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def summary_example(request):
    """Return a canned dataframe summary useful for front-end mocks."""
    from pathlib import Path

    example_path = Path(__file__).resolve().parent / "example_data" / "crash_sample.csv"
    try:
        df = pd.read_csv(example_path)
    except FileNotFoundError:
        return JsonResponse(
            {"detail": f"Example file not found at {example_path}."}, status=500
        )

    summary = build_summary_from_dataframe(df)
    return JsonResponse(summary, status=200)


def _user_is_admin(user) -> bool:
    return user.is_superuser or user.groups.filter(name="admin").exists()


# Keep a lightweight, description-only mapping for backwards compatibility.
SUPPORTED_MODELS: Dict[str, str] = {
    name: spec.description for name, spec in MODEL_REGISTRY.items()
}


@api_view(["POST"])
@permission_classes([IsAuthenticated])
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def model_run(request):
    """
    Start a model training run for an existing UploadedDataset.

    Expected JSON body:
    {
        "dataset_id": "<uuid of UploadedDataset>",
        "model_name": "ebm_v1",
        "parameters": {...}  # optional, free-form
    }

    Behavioural notes:

    * A ModelJob row is created in the database with status="queued".
    * The worker is enqueued (via a lightweight background thread) to
      process the job asynchronously.
    * The response includes the job id and a URL for polling the results.
    """

    if not _user_is_admin(request.user):
        return JsonResponse(
            {"detail": "Only admin users can start model runs."}, status=403
        )

    try:
        payload = request.data
    except Exception:
        payload = {}

    dataset_id = payload.get("dataset_id")
    model_name = payload.get("model_name")
    parameters = payload.get("parameters") or {}

    errors = {}
    if not dataset_id:
        errors["dataset_id"] = "This field is required."
    if not model_name:
        errors["model_name"] = "This field is required."
    elif model_name not in MODEL_REGISTRY:
        errors["model_name"] = (
            f"Unsupported model_name. Supported: {sorted(MODEL_REGISTRY.keys())}"
        )

    if errors:
        return JsonResponse({"errors": errors}, status=400)

    dataset = get_object_or_404(UploadedDataset, pk=dataset_id)

    # Basic validation of parameters.cleaning.* to avoid confusing the ML pipeline
    cleaning_params = parameters.get("cleaning") or {}
    if not isinstance(cleaning_params, dict):
        return JsonResponse(
            {"errors": {"parameters.cleaning": "Must be an object/dict."}},
            status=400,
        )

    for key in ["unknown_threshold", "yes_no_threshold"]:
        if key in cleaning_params and not isinstance(cleaning_params[key], (int, float)):
            return JsonResponse(
                {
                    "errors": {
                        "parameters.cleaning": f'"{key}" must be numeric (int/float).'
                    }
                },
                status=400,
            )

    if "columns_to_drop" in cleaning_params and not isinstance(
        cleaning_params["columns_to_drop"], list
    ):
        return JsonResponse(
            {
                "errors": {
                    "parameters.cleaning": '"columns_to_drop" must be a list of strings.'
                }
            },
            status=400,
        )

    job = ModelJob.objects.create(
        upload=dataset,
        model_name=model_name,
        parameters=parameters,
        status=ModelJob.Status.QUEUED,
    )

    # Enqueue the worker.  The current implementation uses a lightweight
    # background thread so that the HTTP request can return quickly.  In
    # a production deployment this can be replaced with a Celery task,
    # RQ job, or other task runner.
    try:
        enqueue_model_job(job.id)
    except Exception as exc:  # pragma: no cover - failure is recorded on the job
        logger.exception("Failed to enqueue worker for ModelJob %s: %s", job.id, exc)
        job.status = ModelJob.Status.FAILED
        job.result_metadata = {
            "error": f"Failed to enqueue worker: {exc}",
        }
        job.save(update_fields=["status", "result_metadata"])

    # Provide a canonical URL for polling results.  We avoid depending on
    # URL names here to keep this view decoupled from URLconf details.
    results_path = f"/api/models/results/{job.id}/"
    results_url = request.build_absolute_uri(results_path)

    return JsonResponse(
        {
            "job_id": str(job.id),
            "status": job.status,
            "dataset_id": str(dataset.id),
            "model_name": job.model_name,
            "results_url": results_url,
        },
        status=202,
    )


model_run.throttle_scope = "model_run"


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def model_results(request, job_id: str):
    """
    Fetch the JSON result_metadata for a ModelJob.

    This endpoint is read-only and returns whatever the ML worker stored
    into ModelJob.result_metadata, along with basic job bookkeeping fields.
    """
    job = get_object_or_404(ModelJob, pk=job_id)

    if not _user_is_admin(request.user) and job.upload.owner != request.user:
        return JsonResponse(
            {"detail": "You do not have permission to view this job."}, status=403
        )

    payload = {
        "job_id": str(job.id),
        "status": job.status,
        "model_name": job.model_name,
        "dataset_id": str(job.upload_id),
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "result_metadata": job.result_metadata or {},
    }
    return JsonResponse(payload, status=200)
