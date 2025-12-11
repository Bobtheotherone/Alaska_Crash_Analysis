
"""
Non-interactive adaptation of Peyton's severity-mapping utilities.

The original CLI helper (``find_severity_mapping``) uses interactive
prompts to confirm or manually override the mapping. For web/worker
use we instead expose functions that run the same automatic heuristics
and return the proposed mappings directly.
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable

import pandas as pd

from peyton_original import severity_mapping_utils as peyton_severity


def map_numeric_severity(values: Iterable[Hashable]) -> Dict[Hashable, int]:
    """
    Thin wrapper around Peyton's numeric severity mapping helper.

    Given the distinct values that appear in a severity column, return
    a mapping from each raw value to an ordinal severity score
    (0 = least severe, 2 = most severe). The exact heuristics are
    delegated to :func:`peyton_severity.map_numeric_severity`.
    """
    # The underlying helper is robust to value types; it just expects an
    # iterable of unique values.
    return peyton_severity.map_numeric_severity(list(values))


def map_text_severity(values: Iterable[Hashable]) -> Dict[Hashable, int]:
    """
    Thin wrapper around Peyton's text-based severity mapping helper.

    This function applies keyword heuristics (e.g. "fatal", "serious",
    "minor") to produce a 0/1/2 mapping for free-text severity labels.
    """
    return peyton_severity.map_text_severity(list(values))


def find_severity_mapping_noninteractive(
    df: pd.DataFrame,
    severity_col: str,
) -> Dict[Hashable, int]:
    """
    Infer a 0/1/2 severity mapping for ``severity_col`` without any
    interactive prompts.

    The logic mirrors Peyton's ``find_severity_mapping``:

    * If at least ~90% of the values in ``severity_col`` can be parsed
      as numbers, use :func:`map_numeric_severity`.
    * Otherwise, attempt a keyword-based mapping via
      :func:`map_text_severity`.

    If neither heuristic is able to produce a mapping, an empty dict is
    returned instead of falling back to manual input.
    """
    sev_vals = df[severity_col].dropna()
    unique_sev_vals = sev_vals.unique().tolist()

    # Replicate Peyton's numeric vs text branching.
    numeric_sev = pd.to_numeric(sev_vals, errors="coerce")
    num_valid = numeric_sev.notna().sum()
    frac_valid = float(num_valid) / len(sev_vals) if len(sev_vals) > 0 else 0.0

    # 1) Try numeric mapping if most values look numeric.
    if frac_valid >= 0.9:
        mapping = map_numeric_severity(unique_sev_vals)
        if mapping:
            # Original code would call ``confirm_or_edit_mapping`` here,
            # which is interactive; for non-interactive use we simply
            # accept the proposed mapping.
            return mapping

    # 2) Fallback to keyword-based text mapping.
    text_mapping = map_text_severity(unique_sev_vals)
    if text_mapping:
        return text_mapping

    # 3) In the CLI tool this would fall back to manual input; in a
    # non-interactive context we instead return an empty mapping.
    return {}
```

---

### `docs/model_api.md` 

````markdown
# Model API Contract

This document defines the non-ML interface for running crash analysis models via
the Django backend. The ML / serving team is responsible for wiring these
endpoints into their own pipelines.

## Endpoints

### `POST /api/models/run/`

Start a new model job.

**Authentication:** required (session or token).

**Request body (JSON):**

```json
{
  "upload_id": "uuid-of-UploadedDataset",
  "model": "crash_severity_risk_v1",
  "parameters": {
    "exclude_invalid_rows": true,
    "geography_filter": {
      "municipality": "Anchorage"
    }
  }
}
````

* `upload_id` (string, required)
  UUID of an `ingestion.UploadedDataset` with `status="accepted"`.

* `model` (string, required)
  Identifier of the model to run. Must be one of:

  * `crash_severity_risk_v1`
  * `ebm_v1`
  * `mrf_v1`
  * `xgb_v1`

  This list is defined in `analysis.views.SUPPORTED_MODELS` and can be extended
  in a backwards-compatible way.

* `parameters` (object, optional)
  Free-form JSON object interpreted by the ML pipeline. The backend stores this
  verbatim in `ModelJob.parameters`, with light validation for an optional
  `cleaning` sub-object:

  * `cleaning.unknown_threshold` (number, optional, 0–100)
  * `cleaning.yes_no_threshold` (number, optional, 0–100)
  * `cleaning.columns_to_drop` (array of strings, optional)

**Responses:**

* `202 Accepted` – job created

  ```json
  {
    "job_id": "00000000-0000-0000-0000-000000000000",
    "status": "queued",
    "upload_id": "uuid-of-UploadedDataset",
    "model": "crash_severity_risk_v1",
    "parameters": { "...": "..." },
    "detail": "Job has been created and queued."
  }
  ```

* `400 Bad Request` – validation error (missing fields, unsupported model, non-object parameters)

* `403 Forbidden` – upload is not owned by the caller

* `404 Not Found` – upload does not exist

---

### `GET /api/models/results/<job_id>/`

Retrieve the status and high-level metadata for a model job.

**Authentication:** required.

**Path parameters:**

* `job_id` – UUID of the `ModelJob` row created by the `run` endpoint.

**Response body (JSON):**

```json
{
  "job_id": "00000000-0000-0000-0000-000000000000",
  "upload_id": "uuid-of-UploadedDataset",
  "model": "crash_severity_risk_v1",
  "status": "queued",
  "parameters": { "...": "..." },
  "result_metadata": {
    "message": "Job created and queued. Hook this into your task queue or model-serving pipeline."
  },
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

**Status codes:**

* `202 Accepted` – job is `queued` or `running`
* `200 OK` – job is `succeeded` or `failed`
* `403 Forbidden` – job is not visible to the caller
* `404 Not Found` – job id unknown

---

## Responsibilities

* **App / API team (this repo):**

  * Maintain `ModelJob` schema and migrations.
  * Enforce permissions (only owners/Admin can see jobs for a given upload).
  * Provide stable request/response contracts for the ML team.

* **ML / data-cleaning team:**

  * Attach a task queue / worker that consumes `ModelJob` rows with status
    `queued`, runs models, writes results to storage of their choice, and
    updates `ModelJob.status` and `ModelJob.result_metadata` (e.g., summary
    metrics, links to result tables, etc.).
  * Keep the logic that interprets `parameters` in ML-owned code, not in
    the Django views.


