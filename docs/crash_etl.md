# Crash ETL and ML workflow

This document ties together the ingestion app, the crash ETL step, and the
machine‑learning worker. It assumes the Alaska MMUCC‑style schema defined in
`ingestion/config/mmucc_schema.yml`.

The high‑level flow is:

1. **Raw CSV/Parquet upload → `UploadedDataset` (ingestion app).**
2. **`import_crash_records` management command → cleaned, normalized `CrashRecord`.**
3. **Model worker → uses those cleaned records (and the same cleaning utilities)
   to build ML‑ready feature matrices and train/evaluate models.**

---

## 1. Ingestion: raw file → `UploadedDataset`

The ingestion gateway (FR1) is responsible for:

* Accepting an uploaded CSV or Parquet file.
* Validating it against the configured schema (e.g. `mmucc-alaska-v1`).
* Persisting the file and validation metadata into an `UploadedDataset` row.

Key pieces:

* **Model:** `ingestion.models.UploadedDataset`
* **Parser:** `ingestion.validation.load_dataframe_from_bytes`

The `UploadedDataset.raw_file` field always contains the original bytes; the
cleaning/ML stack never mutates this file in place.

---

## 2. Crash ETL: `python manage.py import_crash_records <upload_id>`

The new management command lives at
`crashdata/management/commands/import_crash_records.py`.

### What the command does

1. **Lookup the dataset** by `upload_id` using `UploadedDataset.objects.get(id=...)`.
2. **Re‑parse the raw file** using
   `ingestion.validation.load_dataframe_from_bytes` so the ETL step sees the
   same column types that the ingestion endpoint validated.
3. **Run the refactored cleaning pipeline** from
   `analysis.ml_core.cleaning.clean_crash_dataframe_for_import`:

   * `discover_unknown_placeholders(df, base_unknowns)`
      * Scans string/object columns for terms that look like "unknown" values
        using the configured `DEFAULT_UNKNOWN_STRINGS` and
        `GENERIC_UNKNOWN_SUBSTRINGS` sets.
      * Returns an augmented set of tokens that should be treated as missing.

   * `profile_columns(df, unknown_strings)`
      * Computes rich per‑column diagnostics:
        * `% unknown` values,
        * number of unique values,
        * dominant value frequency,
        * whether a column behaves like a Yes/No flag and how balanced it is.

   * `suggest_columns_to_drop(df, column_stats, ...)`
      * Applies heuristic rules to mark columns as low‑value for modeling:
        * drop columns where the share of unknowns exceeds `UNKNOWN_THRESHOLD`,
        * drop Yes/No columns that are extremely imbalanced based on
          `YES_NO_THRESHOLD`,
        * drop columns with either only one unique value or as many uniques as
          there are rows,
        * drop nearly‑constant columns where one value dominates ≥99.5% of
          non‑missing rows but there are still at least 25 non‑dominant values.
      * A core set of MMUCC columns is always **protected** from being dropped:
        `crash_id`, `crash_date`, `severity`, `kabco`, `latitude`, `longitude`.

   * The command uses the cleaned DataFrame returned by
     `clean_crash_dataframe_for_import` to map into `CrashRecord` rows and logs
     which columns were dropped. This cleaned DataFrame is also "model‑ready"
     in the sense that obviously bad or useless columns have been stripped out
     while keeping row alignment with the original data.

4. **Map the cleaned rows into `CrashRecord` objects** and bulk‑create them.

### Field mapping

For each row in the cleaned DataFrame, `import_crash_records` creates a
`CrashRecord` with:

* `dataset` – the `UploadedDataset` instance corresponding to `<upload_id>`.
* `crash_id` – from the `crash_id` column (coerced to string).
* `crash_datetime` – parsed from `crash_date`. If only a date is present it is
  interpreted as midnight in the Django project's current timezone.
* `severity` – from the `severity` column, **normalized to upper‑case
  MMUCC KABCO codes**. Rows where the severity is not one of `{"K","A","B","C","O"}`
  are skipped so that the database only contains valid records.
* `location` – built from `longitude`/`latitude` (WGS84, SRID 4326). If either
  coordinate is missing or invalid, the `CrashRecord` is still created with
  `location=None`.
* `roadway_name` – optional, from the `roadway_name` column if present,
  otherwise empty string.
* `municipality` – optional, from the `municipality` column if present,
  otherwise empty string.
* `posted_speed_limit` – optional integer from `posted_speed_limit` if present
  and parsable.
* `vehicle_count` – optional integer from `vehicle_count` if present and
  parsable.
* `person_count` – optional integer from `person_count` if present and
  parsable.

Idempotency / overwrite behaviour:

* If a dataset already has `CrashRecord` rows, the command **deletes** them
  before bulk‑creating the new set. This makes repeated imports for the same
  upload safe and avoids duplicate rows.

### Running the command

From the Django project root:

```bash
python manage.py import_crash_records <upload_id>
```

You can inspect the ingestion tests (`ingestion/tests/test_ingestion_gateway.py`)
for an example of a minimal, valid CSV; the same shape will work with this
command.

To see what would happen without writing to the database, use:

```bash
python manage.py import_crash_records <upload_id> --dry-run
```

This runs the cleaning pipeline and prints a summary (including dropped
columns), but does not create any `CrashRecord` rows.

---

## 3. Model worker: building ML‑ready datasets

The model API and worker are described in more detail in `docs/model_api.md`.
This section explains how the **same cleaning stack** is intended to be used
from a training / inference worker.

The central entrypoint for modeling is
`analysis.ml_core.cleaning.build_ml_ready_dataset(df, ...)`, which wraps four
key steps:

1. **Unknown discovery** – `discover_unknown_placeholders`
2. **Severity mapping** – `find_severity_mapping`
3. **Leakage handling** – `find_leakage_columns`
4. (Optionally) post‑hoc leakage inspection – `warn_suspicious_importances`

### 3.1. Building `(X, y)` from a cleaned crash table

Given a pandas DataFrame `df` (for example built from a queryset against
`CrashRecord` or by re‑reading `UploadedDataset.raw_file`), the worker calls:

```python
from analysis.ml_core import cleaning

X, y, meta = cleaning.build_ml_ready_dataset(df)
```

Under the hood this performs:

1. **Unknown normalisation**

   ```python
   unknown_values = cleaning.discover_unknown_placeholders(df, cleaning.DEFAULT_UNKNOWN_STRINGS)
   df_clean = df.replace(list(unknown_values), pandas.NA)
   ```

2. **Severity column selection + mapping**

   * `guess_severity_column(df_clean)` tries a list of common names
     (`"severity"`, `"Crash Severity"`, `"crash_severity"`, etc.) and falls
     back to any column containing `"severity"`.
   * `find_severity_mapping(df_clean, severity_col)` then maps values in that
     column to the numeric labels `{0, 1, 2}`:
       * For MMUCC KABCO codes:
         * `K` or `A` → `2` (high severity),
         * `B` or `C` → `1` (medium),
         * `O` → `0` (low / property‑damage‑only).
       * For purely numeric severities it buckets into low/medium/high based on
         the observed range.
       * For textual severities it searches for fatal/serious/minor keywords.

   Rows where the severity cannot be mapped are dropped from the modeling
   dataset (but not from the `CrashRecord` table).

3. **Feature matrix construction**

   * The raw severity column is dropped from the feature matrix `X`.
   * The mapped target values become `y` (integer dtype).

4. **Leakage detection and removal**

   * `find_leakage_columns(X, y)` performs two non‑interactive checks:
       * **Name‑based scan** via `suggest_leakage_by_name`, looking for column
         names that contain keywords like `"fatal"`, `"injury"`, `"severity"`,
         etc.
       * **Near‑perfect predictors** via `find_near_perfect_predictors`, which
         looks for low‑cardinality features that almost deterministically
         predict the severity labels on their own.
   * The union of these suggestions is dropped from `X` before training.

5. **Metadata for reproducibility**

   * The returned `meta` dictionary includes:
       * the chosen severity column and mapping,
       * the full list of discovered unknown tokens,
       * the set of dropped leakage columns,
       * row/column counts before and after filtering.

A typical model worker implementation would stash `meta` into
`ModelJob.result_metadata` (see `docs/model_api.md`) alongside the usual
metrics so that future teammates can reconstruct exactly how the dataset was
formed.

### 3.2. Post‑training leakage sanity‑check

After fitting a model that exposes feature importances, the worker can call:

```python
suspicious = cleaning.warn_suspicious_importances(feature_names, importances)
```

If a single feature dominates the importance scores (either absolutely or
relative to the runner‑up), this helper logs a warning suggesting that the
feature may be a leakage column (for example, `"Number of Fatalities"` or a
duplicate severity code).

---

## 4. Putting it together

* **Where does cleaning happen?**  
  In the shared `analysis.ml_core.cleaning` module. Both the
  `import_crash_records` ETL step and the model worker call into the same
  functions (`discover_unknown_placeholders`, `profile_columns`,
  `find_severity_mapping`, `find_leakage_columns`).

* **Where is normalization to `CrashRecord` defined?**  
  In `crashdata/management/commands/import_crash_records.py`. That command
  takes the cleaned, unknown‑normalised DataFrame and maps it to the
  `CrashRecord` schema (one row per crash, with coordinates, severity, and a
  small set of core attributes).

* **How does this feed modeling?**  
  A model worker can either:
  * read from the `CrashRecord` table, convert to a DataFrame, and then call
    `build_ml_ready_dataset`, or
  * re‑read `UploadedDataset.raw_file`, call
    `clean_crash_dataframe_for_import`, and then `build_ml_ready_dataset` on
    the result.

In both cases the worker relies on the same refactored, non‑interactive
cleaning utilities, so any future teammate can follow the pipeline end‑to‑end
from raw CSV → `UploadedDataset` → `CrashRecord` → `(X, y)` without having to
reverse‑engineer ad‑hoc scripts.
