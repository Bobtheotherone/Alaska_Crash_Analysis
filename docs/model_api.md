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
```

- `upload_id` (string, required)
  - UUID of an `ingestion.UploadedDataset` with `status="accepted"`.
- `model` (string, required)
  - Identifier of the model to run. Must be one of:
    - `crash_severity_risk_v1`
    - `ebm_v1`
  - This list is defined in `analysis.views.SUPPORTED_MODELS` and can be extended
    in a backwards-compatible way.
- `parameters` (object, optional)
  - Free-form JSON object interpreted by the ML pipeline. The backend stores
    this verbatim in `ModelJob.parameters`, with light validation for an
    optional `cleaning` sub-object:
    - `cleaning.unknown_threshold` (number, optional, 0–100)
    - `cleaning.yes_no_threshold` (number, optional, 0–100)
    - `cleaning.columns_to_drop` (array of strings, optional)

**Responses:**

- `202 Accepted` – job created

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

- `400 Bad Request` – validation error (missing fields, unsupported model, non-object parameters)
- `403 Forbidden` – upload is not owned by the caller
- `404 Not Found` – upload does not exist

### `GET /api/models/results/<job_id>/`

Retrieve the status and high-level metadata for a model job.

**Authentication:** required.

**Path parameters:**

- `job_id` – UUID of the `ModelJob` row created by the `run` endpoint.

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

- `202 Accepted` – job is `queued` or `running`
- `200 OK` – job is `succeeded` or `failed`
- `403 Forbidden` – job is not visible to the caller
- `404 Not Found` – job id unknown

## Responsibilities

- **App / API team (this repo):**
  - Maintain `ModelJob` schema and migrations.
  - Enforce permissions (only owners/Admin can see jobs for a given upload).
  - Provide stable request/response contracts for the ML team.

- **ML / data-cleaning team:**
  - Attach a task queue / worker that consumes `ModelJob` rows with status
    `queued`, runs models, writes results to storage of their choice, and
    updates `ModelJob.status` and `ModelJob.result_metadata` (e.g., summary
    metrics, links to result tables, etc.).
  - Keep the logic that interprets `parameters` in ML-owned code, not in
    the Django views.
