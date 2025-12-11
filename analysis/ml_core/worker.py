"""
ModelJob worker process utilities.

This module provides the core "worker" implementation that can be used
from:

* a background thread (see enqueue_model_job),
* a Django management command, or
* an external task runner (Celery / RQ) that simply calls run_model_job.

The responsibilities here are:

* Load the UploadedDataset into a pandas DataFrame.
* Look up the correct training function from MODEL_REGISTRY.
* Invoke the training function with cleaning + model parameters.
* Persist metrics, top feature importances, and leakage warnings into
  ModelJob.result_metadata.
* Update the ModelJob status to succeeded/failed.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from typing import Any, Dict, Optional
from uuid import UUID

import pandas as pd
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from ingestion.models import UploadedDataset
from ingestion.validation import load_dataframe_from_bytes
from analysis.models import ModelJob

from .models import MODEL_REGISTRY, ModelSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _load_dataframe_for_job(upload: UploadedDataset) -> pd.DataFrame:
    """Load the raw_file from an UploadedDataset into a DataFrame."""
    file_field = upload.raw_file
    if not file_field:
        raise ValueError("UploadedDataset has no raw_file attached.")

    # Reset pointer and read bytes.
    file_field.open("rb")
    try:
        raw_bytes = file_field.read()
    finally:
        file_field.close()

    # Infer extension from the filename when possible.
    filename = file_field.name or upload.original_filename or ""
    _, ext = os.path.splitext(filename)
    ext = ext.lower() or ".csv"

    logger.info("Loading dataframe for UploadedDataset %s (ext=%s)", upload.pk, ext)

    return load_dataframe_from_bytes(raw_bytes, ext)


def _build_result_metadata(
    job: ModelJob,
    spec: ModelSpec,
    trainer_output: Dict[str, Any],
    *,
    started_at,
    finished_at,
) -> Dict[str, Any]:
    """
    Normalise the training output into a JSON-serialisable metadata
    structure suitable for storing in ModelJob.result_metadata.
    """
    metrics = trainer_output.get("metrics") or {}
    feature_importances = trainer_output.get("feature_importances") or {}
    cleaning_meta = trainer_output.get("cleaning_meta") or {}
    model_params = trainer_output.get("model_params") or {}
    leakage_warnings = trainer_output.get("leakage_warnings") or {}

    # Sort and capture the top N feature importances.
    items = list(feature_importances.items())
    items.sort(key=lambda kv: kv[1], reverse=True)
    top_n = [
        {"feature": name, "importance": float(score)}
        for name, score in items[:50]
    ]

    duration_seconds: Optional[float] = None
    try:
        duration_seconds = float((finished_at - started_at).total_seconds())
    except Exception:
        # Defensive: if timestamps are weird, don't crash the worker.
        duration_seconds = None

    return {
        "job_id": str(job.id),
        "model_name": job.model_name,
        "spec_name": spec.name,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration_seconds,
        "metrics": metrics,
        "feature_importances": {
            "all": feature_importances,
            "top_n": top_n,
        },
        "cleaning_meta": cleaning_meta,
        "model_params": model_params,
        "leakage_warnings": leakage_warnings,
    }


# ---------------------------------------------------------------------------
# Core job runner
# ---------------------------------------------------------------------------


def run_model_job(job_id: str | UUID) -> None:
    """
    Run a single ModelJob end-to-end.

    This is the main entry point that can be invoked by a Celery task,
    RQ job, Django management command, or the lightweight in-process
    enqueue_model_job helper below.
    """
    job_id_str = str(job_id)
    logger.info("Starting ModelJob worker for job_id=%s", job_id_str)

    # Take a row-level lock and mark the job as running.
    with transaction.atomic():
        try:
            job = (
                ModelJob.objects.select_for_update()
                .select_related("upload")
                .get(pk=job_id)
            )
        except ModelJob.DoesNotExist:
            logger.error("ModelJob %s does not exist; aborting.", job_id_str)
            return

        if job.status != ModelJob.Status.QUEUED:
            logger.warning(
                "ModelJob %s has status=%s, not 'queued'; refusing to run.",
                job_id_str,
                job.status,
            )
            return

        job.status = ModelJob.Status.RUNNING
        job.updated_at = timezone.now()
        job.save(update_fields=["status", "updated_at"])

    # Outside the transaction from here on.
    started_at = timezone.now()

    try:
        # Reload job with its upload relation in case anything changed.
        job = ModelJob.objects.select_related("upload").get(pk=job_id)
        upload = job.upload

        df = _load_dataframe_for_job(upload)

        try:
            spec = MODEL_REGISTRY[job.model_name]
        except KeyError:
            raise ValueError(
                f"Unknown model_name '{job.model_name}'. "
                f"Supported: {sorted(MODEL_REGISTRY.keys())}"
            )

        # Split parameters into cleaning + model-specific parameters.
        parameters = job.parameters or {}
        cleaning_params = parameters.get("cleaning") or {}
        model_params = parameters.get("model_params") or parameters.get("model") or {}

        # Merge user-supplied params over the registry defaults.
        merged_model_params = {
            **spec.default_model_params,
            **(model_params or {}),
        }

        logger.info(
            "Running model '%s' for job %s with %d cleaning params and %d model params.",
            job.model_name,
            job_id_str,
            len(cleaning_params),
            len(merged_model_params),
        )

        trainer_output = spec.trainer(
            df,
            cleaning_params=cleaning_params,
            model_params=merged_model_params,
        )

        finished_at = timezone.now()
        metadata = _build_result_metadata(
            job=job,
            spec=spec,
            trainer_output=trainer_output,
            started_at=started_at,
            finished_at=finished_at,
        )

        job.status = ModelJob.Status.SUCCEEDED
        job.result_metadata = metadata
        job.updated_at = finished_at
        job.save(update_fields=["status", "result_metadata", "updated_at"])

    except Exception as exc:  # pragma: no cover - error path
        finished_at = timezone.now()
        tb_str = traceback.format_exc()
        logger.exception("ModelJob %s failed: %s", job_id_str, exc)

        # Best-effort update of the job record with failure details.
        try:
            job = ModelJob.objects.get(pk=job_id)
            job.status = ModelJob.Status.FAILED
            job.result_metadata = {
                "error": str(exc),
                "traceback": tb_str,
                "failed_at": finished_at.isoformat(),
            }
            job.updated_at = finished_at
            job.save(update_fields=["status", "result_metadata", "updated_at"])
        except Exception:  # pragma: no cover - last resort
            logger.error(
                "Failed to update ModelJob %s with failure metadata.", job_id_str
            )


# ---------------------------------------------------------------------------
# Queue helpers / management-command friendly API
# ---------------------------------------------------------------------------


def run_next_queued_job() -> Optional[str]:
    """
    Pop the next queued ModelJob (if any) and run it synchronously.

    Returns the job ID that was processed, or None if there were no
    queued jobs.
    """
    job = (
        ModelJob.objects.filter(status=ModelJob.Status.QUEUED)
        .select_related("upload")
        .order_by("created_at")
        .first()
    )
    if not job:
        logger.info("No queued ModelJob instances found.")
        return None

    job_id = str(job.id)
    run_model_job(job_id)
    return job_id


def enqueue_model_job(job_id: str | UUID) -> None:
    """
    Lightweight "enqueue" helper used by the /api/models/run/ view.

    In development (DEBUG=True) we execute the job synchronously in the
    current process so that the dev server and front-end polling do not
    get out of sync.  In non-debug environments, this function preserves
    the original asynchronous behaviour by spawning a daemon thread.
    """
    if getattr(settings, "DEBUG", False):
        logger.info(
            "DEBUG mode: running ModelJob %s synchronously in-process.", job_id
        )
        run_model_job(job_id)
        return

    def _target() -> None:
        # Small delay so that the HTTP response has a chance to be sent
        # before heavy work begins (helpful for local dev and proxies).
        time.sleep(0.1)
        run_model_job(job_id)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
