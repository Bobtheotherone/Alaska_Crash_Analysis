# Alaska Car Crash Analysis

This repository contains a Django + React application for exploring and summarizing
police-reported crash data. The backend follows an **App Server + Upload Gateway**
pattern with clear separation of responsibilities and a security-first upload flow.

## Backend (Django)

The Django project is split into three main apps:

- **`ingestion`** – upload gateway / validation engine
  - `POST /api/ingest/upload/`
    - Accepts authenticated uploads of crash datasets (CSV and Parquet by
      default; additional formats can be enabled via config).
    - Enforces file size limits (`INGESTION_MAX_FILE_SIZE_BYTES`).
    - Restricts extensions to a configurable whitelist
      (`INGESTION_ALLOWED_EXTENSIONS`).
    - Performs MIME sniffing (via `python-magic` when available).
    - Runs an antivirus scan via ClamAV (when configured).
    - Validates the header against a MMUCC-aligned schema loaded from
      `ingestion/config/mmucc_schema.yml` (including KABCO severity and core
      location fields). See `docs/schema_config.md` for non-developer editing
      guidance.
    - Runs basic type/range checks and Alaska-specific geo bounding-box checks.
    - Produces a structured per-step status report plus row-level summary stats.
    - Returns a `schema_version` so the UI can display which schema was applied.
  - `GET /api/ingest/uploads/`
    - Returns uploads visible to the current user (uploads owned by the current user; admins see all).
    - Each entry includes `id`, `original_filename`, `created_at`, `status`, `schema_version`, `mime_type`,
      and row summary counts (`total_rows`, `invalid_row_count`, `invalid_geo_row_count`).
  - `GET /api/ingest/uploads/<upload_id>/`
    - Returns the persisted validation report and metadata for a given upload.
    - Useful for re-displaying validation status in the UI without re-uploading.
  - `GET /api/ingest/uploads/<upload_id>/export/validation.csv`
    - CSV export of the validation steps + row summaries, with formula-injection
      defenses applied.
  - Persists each successful upload in `ingestion.UploadedDataset` with:
    - `owner` (uploading user)
    - `original_filename`, `size_bytes`, `mime_type`
    - `schema_version` (from the MMUCC config file)
    - `raw_file` (stored under `uploaded_datasets/`)
    - `status` (`pending | accepted | rejected`)
    - `validation_report` (JSON copy of the status report returned
      to the client)
  - All “hard fail” stages in the pipeline now return machine-readable
    `error_code`s (e.g. `EXTENSION_NOT_ALLOWED`, `FILE_TOO_LARGE`,
    `MIME_MISMATCH`, `UPLOAD_INFECTED`, `AV_UNAVAILABLE_REQUIRED`,
    `SCHEMA_MISSING_COLUMNS`). Error codes are documented in
    `docs/ingestion_errors.md`.

- **`crashdata`** – domain models & query helpers
  - Owns the `CrashRecord` model, which captures core MMUCC/KABCO fields and
    a PostGIS `PointField` for the crash location with a **GiST index** for
    fast spatial queries.
  - Includes helpers in `crashdata.queries` for:
    - severity histograms (`severity_histogram`)
    - spatial queries (`crashes_within_bbox`), with support for filters such as
      severity, municipality and date range.
  - Exposes read-only APIs for visualization and exports:
    - `GET /api/crashdata/severity-histogram/`
    - `GET /api/crashdata/within-bbox/`
    - `GET /api/crashdata/export/`
  - Most endpoints return a `count` and a list of results, suitable for
    paginated tables and charts.

- **`models`** – integration surface for ML / statistical models
  - Provides a thin wrapper around long-running model jobs driven by
    uploaded crash datasets.
  - Exposes:
    - `POST /api/models/run/`
    - `GET /api/models/results/<job_id>/`
  - Uses a `ModelJob` Django model to track jobs:
    - `id` (UUID), `upload`, `owner`, `model_name`, `status`
    - `parameters` (JSON), `result_metadata` (JSON)
    - timestamps (`created_at`, `updated_at`)
  - `POST /api/models/run/` validates the request body:
    - required: `upload_id`, `model`
    - optional: `parameters` object
    - `model` must be one of the enum values in `SUPPORTED_MODELS`
      (e.g. `crash_severity_risk_v1`, `ebm_v1`).
    - creates a `ModelJob` with status `queued` and returns 202 + `job_id`.
    - designed to be wired into a task queue by the ML team.
  - `GET /api/models/results/<job_id>/` returns the current job status plus any
    high-level `result_metadata`. For queued/running jobs it returns 202;
    for finished jobs it returns 200 and a `result_metadata` payload.

## Frontend (React)

The frontend (`alaska_ui`) is a Vite + React app that consumes the Django APIs
and provides:

- An upload page for CSV/Parquet datasets with a validation results panel.
- A map-based view for filtering and exploring crashes.
- A simple model runner UI for triggering backend model jobs and viewing
  high-level results.

## Running the project locally

See `docs/deployment.md` for detailed instructions on running the full stack
(with or without Docker).

In short:

- Start Postgres (and PostGIS).
- Run migrations (`python manage.py migrate`).
- Start the backend (`python manage.py runserver`).
- Run the frontend dev server (`npm run dev` inside `alaska_ui`).

## Ingestion pipeline details

The ingestion pipeline is designed to be:

- **Secure by default** – strict file type/size checks, MIME sniffing, and
  antivirus integration.
- **Configurable** – schema, allowed extensions, and size limits are all driven
  by configuration rather than hard-coded.
- **Observable** – every major step is recorded in a structured `steps` array
  along with row-level summary counts to help the UI explain what happened.

### File formats

By default, the ingestion pipeline accepts:

- **CSV** – parsed with `pandas.read_csv` using UTF-8 decoding.
- **Parquet** – parsed with `pandas.read_parquet` (requires `pyarrow` or `fastparquet`,
  which are already included in the backend dependencies).

The set of allowed extensions is controlled by the `INGESTION_ALLOWED_EXTENSIONS`
environment variable, which should be a comma-separated list of lower-cased
extensions such as:

```bash
INGESTION_ALLOWED_EXTENSIONS=".csv,.parquet"
```

If the environment variable is not set, the default is `.csv,.parquet`.

Additional formats can be enabled by extending `INGESTION_ALLOWED_EXTENSIONS`
**and** updating the loader in
`ingestion.validation.load_dataframe_from_bytes` to handle the new format.

The key knobs are controlled by environment variables:

- `INGESTION_ALLOWED_EXTENSIONS` (default: `.csv,.parquet`)
- `INGESTION_MAX_FILE_SIZE_BYTES` (default: `10MB` – intentionally small so
  production must explicitly opt into larger uploads)
- `INGESTION_SCHEMA_CONFIG_PATH` (optional override of the MMUCC schema path)

### Antivirus

Antivirus integration is handled via **ClamAV** and the `clamd` Python
library. Configuration is controlled through environment variables:

- `CLAMAV_UNIX_SOCKET` – path to the ClamAV Unix socket
- `CLAMAV_TCP_HOST` / `CLAMAV_TCP_PORT` – host/port for TCP connections
- `INGESTION_REQUIRE_AV` – when set to `"true"`, any upload where the AV
  step is skipped will be treated as a hard failure.

### Ingestion status report shape

`POST /api/ingest/upload/` returns a JSON payload of the form:

```json
{
  "upload_id": "uuid-of-UploadedDataset",
  "overall_status": "accepted",
  "schema_version": "mmucc-alaska-v1",
  "steps": [
    {
      "step": "PAYLOAD",
      "status": "passed",
      "severity": "info",
      "is_hard_fail": false,
      "details": "Received multipart/form-data with a single file field named 'file'."
    },
    {
      "step": "EXTENSION_CHECK",
      "status": "passed",
      "severity": "info",
      "is_hard_fail": false,
      "details": "Extension '.csv' is allowed."
    },
    {
      "step": "FILE_SIZE",
      "status": "passed",
      "severity": "info",
      "is_hard_fail": false,
      "details": "File size is within the configured limit.",
      "meta": { "size_bytes": 12345, "max_size_bytes": 10485760 }
    },
    {
      "step": "MIME_SNIFF",
      "status": "passed",
      "severity": "info",
      "is_hard_fail": false,
      "details": "Declared Content-Type matches detected MIME type.",
      "meta": { "detected_mime_type": "text/csv", "declared_mime_type": "text/csv" }
    },
    {
      "step": "AV_SCAN",
      "status": "passed",
      "severity": "info",
      "is_hard_fail": false,
      "details": "ClamAV did not detect malware in the upload."
    },
    {
      "step": "PARSE_TABLE",
      "status": "passed",
      "severity": "info",
      "is_hard_fail": false,
      "details": "Parsed dataset with 1234 rows and 25 columns."
    },
    {
      "step": "SCHEMA_CHECK",
      "status": "passed",
      "severity": "info",
      "is_hard_fail": false,
      "details": "All required MMUCC columns are present.",
      "meta": {
        "missing_columns": [],
        "unknown_columns": [],
        "columns": ["crash_id", "crash_date", "severity", "..."],
        "schema_version": "mmucc-alaska-v1"
      }
    },
    {
      "step": "TYPE_AND_RANGE_CHECKS",
      "status": "passed",
      "severity": "warning",
      "is_hard_fail": false,
      "details": "Some rows have out-of-range or invalid values.",
      "meta": {
        "total_rows": 1234,
        "invalid_row_count": 12,
        "details": "..."
      }
    },
    {
      "step": "GEO_CHECKS",
      "status": "passed",
      "severity": "warning",
      "is_hard_fail": false,
      "details": "Some rows have coordinates outside configured Alaska bounds.",
      "meta": {
        "has_coordinates": true,
        "invalid_row_count": 3,
        "details": "..."
      }
    }
  ],
  "schema": {
    "schema_version": "mmucc-alaska-v1",
    "missing_columns": [],
    "unknown_columns": [],
    "columns": ["crash_id", "crash_date", "severity", "..."]
  },
  "row_checks": {
    "total_rows": 1234,
    "invalid_row_count": 12,
    "invalid_geo_row_count": 3
  }
}
```

If the upload is rejected for a "hard" reason (extension, size, MIME,
antivirus (when required), or schema), `overall_status` is `"rejected"`,
`upload_id` is omitted, and `error_code` / `message` describe the reason.

- `status` describes the outcome of the step itself (`"passed"`, `"failed"`,
  `"skipped"`).
- `severity` describes how serious the outcome is for the upload as a whole:
  - `"error"` + `is_hard_fail: true` → the upload is rejected.
  - `"warning"` → upload is accepted, but there are data quality issues.
  - `"info"` → purely informational, no impact on acceptance.

## Security & authentication

- All ingestion and model endpoints require authentication (`IsAuthenticated`).
- Access control is enforced so that users can only see uploads and jobs they
  own, unless they are in the `Admin` group or have staff/superuser status.
- CSRF protection is enabled for browser-based sessions; API clients should use
  token or session authentication as appropriate.

