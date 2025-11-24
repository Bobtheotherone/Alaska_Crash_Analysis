# Ingestion schema configuration (`mmucc_schema.yml`)

The upload gateway uses a configurable schema file at:

```text
ingestion/config/mmucc_schema.yml
```

This file describes the MMUCC / KABCO–aligned schema that is enforced at **upload
time**. Non-developers can safely adjust required columns, labels, and basic
ranges by editing this YAML file and restarting Django – no Python changes are
needed.

> ⚠️ **Important:** Invalid YAML syntax will prevent Django from starting.
> Always keep a backup of the previous version and make small, incremental
> changes.

---

## Top‑level structure

The file has three main sections:

```yaml
schema_version: "2024.1"
columns:
  crash_id:
    label: "Crash ID"
    required: true
    type: string
  crash_date:
    label: "Crash Date"
    required: true
    type: date
  # ...
geo_bounds:
  latitude:
    min: 51.0
    max: 72.0
  longitude:
    min: -170.0
    max: -130.0
```

### `schema_version`

- A human‑readable version string (e.g. `"2024.1"`).
- Returned in the upload response and stored on `UploadedDataset.schema_version`.
- Changing this string is a simple way to track schema revisions.

### `columns`

A mapping from **column key** → configuration object. The column key must match
the exact column name in the uploaded CSV / Parquet file (case sensitive).

Each column has the following fields:

- `label` (string)
  - A friendly, human‑readable name used in UI and reports.
- `required` (boolean)
  - `true` → this column **must** be present in uploads.
  - If any `required: true` columns are missing, the `SCHEMA_CHECK` step fails,
    the upload is **rejected** with error code `SCHEMA_MISSING_COLUMNS`, and
    `overall_status` is `"rejected"`.
  - `false` → optional column. Missing optional columns are allowed; they will
    be recorded in the `unknown_columns` / `missing_columns` metadata but do
    not block ingestion.
- `type` (string)
  - Controls how type / range validation interprets the values.
  - Supported values include:
    - `string`
    - `int`
    - `float`
    - `date`
    - `category`
  - If the type does not match (e.g. a non‑numeric value for an `int` column),
    the row is counted as invalid in the value/type checks. These issues are
    **soft‑fail**: the upload is still accepted, but the validation report
    will contain warnings.
- Optional `min` / `max` (numeric)
  - For numeric types (`int`, `float`), define allowed value ranges.
  - Rows with values outside this range increase `invalid_row_count` in the
    value/type checks but do **not** cause a hard rejection.

The validation report exposes these schema details under:

```json
"schema": {
  "schema_version": "...",
  "columns": { ... },
  "missing_columns": [ ... ],
  "unknown_columns": [ ... ]
}
```

### `geo_bounds`

Defines latitude / longitude ranges used by the GEO_CHECKS step:

```yaml
geo_bounds:
  latitude:
    min: 51.0
    max: 72.0
  longitude:
    min: -170.0
    max: -130.0
```

- `latitude.min` / `latitude.max`
- `longitude.min` / `longitude.max`

Rows with coordinates outside these ranges are counted as invalid in the
geo‑bounds check. This is also a **soft‑fail**: the upload is still accepted,
but the GEO_CHECKS step will have `severity: "warning"` and non‑zero
`invalid_row_count`.

If an upload has no usable latitude/longitude columns, the GEO_CHECKS step is
still recorded but treated as informational.

---

## Common edits

### 1. Adding a new optional column

Suppose you want to add a new optional `driver_phone` field as a string:

```yaml
columns:
  driver_phone:
    label: "Driver Phone"
    required: false
    type: string
```

- Uploads that include `driver_phone` will have the values validated as
  strings.
- Uploads that omit `driver_phone` are still accepted.

### 2. Making an existing column required

To require `posted_speed_limit` to be present in all uploads:

```yaml
columns:
  posted_speed_limit:
    label: "Posted Speed Limit"
    required: true
    type: int
    min: 0
    max: 120
```

Impact on the ingestion pipeline:

- If `posted_speed_limit` is missing from the uploaded file:
  - `SCHEMA_CHECK` step → `status: "failed"`, `severity: "error"`,
    `is_hard_fail: true`, `code: "SCHEMA_MISSING_COLUMNS"`.
  - `overall_status` becomes `"rejected"` and no dataset is stored.
- If the column is present but some rows have out‑of‑range values:
  - The upload is **accepted**.
  - Value/range checks count invalid rows and report them via
    `row_checks.invalid_row_count` and a `TYPE_AND_RANGE_CHECKS` step with
    `severity: "warning"`.

### 3. Adjusting numeric ranges

To tighten the allowed age range:

```yaml
columns:
  driver_age:
    label: "Driver Age"
    required: false
    type: int
    min: 16
    max: 100
```

- Any row with `driver_age` < 16 or > 100 will:
  - Increase `invalid_row_count` in the value/range checks.
  - Leave the overall upload **accepted**, but with warnings.

### 4. Updating geo bounds

To widen the acceptable longitude range:

```yaml
geo_bounds:
  latitude:
    min: 51.0
    max: 72.0
  longitude:
    min: -180.0
    max: -130.0
```

- Future uploads will use the new bounds when computing
  `geo_checks.invalid_row_count`.

---

## When do changes take effect?

- The schema file is loaded at Django startup.
- After editing `mmucc_schema.yml`, you **must restart** the Django server
  (or the worker process if running under a process manager) for changes to
  take effect.

---

## Relationship to validation output

A few key fields in the validation JSON are directly driven by this file:

- `schema_version`
  - Comes from the YAML `schema_version`.
- `schema.missing_columns`
  - Any `required: true` columns that are absent from the upload.
- `schema.unknown_columns`
  - Columns present in the upload but not listed under `columns` in the YAML.
- `row_checks.total_rows` / `row_checks.invalid_row_count`
  - Computed using the `type`, `min`, and `max` configuration for each column.
- `geo_checks.invalid_row_count`
  - Computed using `geo_bounds`.

By keeping `mmucc_schema.yml` aligned with the MMUCC spec and local policy,
you control what the upload gateway considers "required" versus "nice to have"
and how strict basic data quality checks should be.
