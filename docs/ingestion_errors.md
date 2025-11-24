# Ingestion error codes

The upload gateway uses **stable, machine‑readable error codes** for any
hard‑fail condition. These codes appear:

- As the top‑level `error_code` field when `overall_status == "rejected"`.
- On one of the `steps[i].code` entries for the step that caused the failure.

This document is the source of truth for codes used by FR1 (data ingestion &
validation). When adding new hard‑fail conditions in the future, extend this
table and keep codes stable.

---

## Error code reference

| error_code              | Step            | Typical severity | Description (for developers)                                              | Suggested user action                                                  |
| ----------------------- | --------------- | ---------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `PAYLOAD_MISSING_FILE`  | `PAYLOAD`       | error            | No file was sent in the `file` field of the multipart/form-data request. | Re‑upload with a file attached under the `file` field.                 |
| `EXTENSION_NOT_ALLOWED` | `EXTENSION_CHECK` | error          | File extension is not in `INGESTION_ALLOWED_EXTENSIONS`.                  | Convert the file to an allowed type (e.g. `.csv` / `.parquet`) or ask an admin to update the configuration. |
| `READ_FAILED`           | `READ_FILE`     | error            | Django could not read the uploaded file stream.                           | Retry the upload; if it recurs, check client/network and server logs.  |
| `FILE_TOO_LARGE`        | `FILE_SIZE`     | error            | File size exceeds `INGESTION_MAX_FILE_SIZE_BYTES`.                        | Split the file or lower its size, or have an admin raise the limit.    |
| `MIME_MISMATCH`         | `MIME_SNIFF`    | error            | Declared `Content-Type` does not match the detected MIME type.           | Save/export the file again so the MIME type matches its actual content, then re‑upload. |
| `UPLOAD_INFECTED`       | `AV_SCAN`       | error            | ClamAV reports the file as infected (malware detected).                   | Clean or regenerate the file, verify it is virus‑free, then re‑upload. |
| `AV_UNAVAILABLE_REQUIRED` | `AV_SCAN`     | error            | Antivirus scanning is required but was skipped (e.g., AV offline).       | Retry later or contact an administrator to restore AV scanning.        |
| `PARSE_FAILED`          | `PARSE_TABLE`   | error            | Pandas failed to parse the file into a tabular dataset.                  | Ensure the file is valid CSV/Parquet (no truncation or mixed formats) and try again. |
| `SCHEMA_MISSING_COLUMNS`| `SCHEMA_CHECK`  | error            | One or more `required: true` columns from `mmucc_schema.yml` are missing. | Update the dataset to include all required columns, then re‑upload.    |

---

## Notes

- All of the codes above correspond to **hard‑fail** conditions in the pipeline.
  When one of these occurs:
  - `overall_status` is `"rejected"`.
  - The upload is **not** persisted as an `UploadedDataset`.
- Other validation steps (type/range and geo checks) are **soft‑fail**:
  - They only affect per‑step `severity` (`"warning"`) and row counts.
  - They never set a top‑level `error_code` or cause the upload to be rejected.
- The UI can rely on:
  - `steps[i].severity == "error"` and `steps[i].is_hard_fail == true` to mark
    blocking issues.
  - The top‑level `error_code` to render user‑friendly error screens or
    contextual documentation.

When adding new hard‑fail conditions, make sure to:

1. Introduce a new, descriptive, UPPER_SNAKE_CASE error code.
2. Set it both on the failing step (`code`) and at the top level (`error_code`).
3. Document it in the table above.
