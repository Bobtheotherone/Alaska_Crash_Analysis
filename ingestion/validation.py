import io
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml
from django.conf import settings

try:
    import magic  # type: ignore[import]
except Exception:  # noqa: BLE001
    magic = None  # type: ignore[assignment]

# -----------------------------------------------------------------------
# MMUCC-aligned schema configuration (externalized for non-devs)
# -----------------------------------------------------------------------


def _get_schema_config_path() -> Path:
    configured = getattr(settings, "INGESTION_SCHEMA_CONFIG_PATH", None)
    if configured:
        return Path(configured)
    base_dir = getattr(settings, "BASE_DIR", Path(__file__).resolve().parent.parent)
    return Path(base_dir) / "ingestion" / "config" / "mmucc_schema.yml"


def _load_schema_config() -> Dict[str, Any]:
    config_path = _get_schema_config_path()
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    return data


_SCHEMA_CONFIG: Dict[str, Any] = _load_schema_config()
SCHEMA_VERSION: str = _SCHEMA_CONFIG.get("schema_version", "unknown")
COLUMN_SPECS: Dict[str, Dict[str, Any]] = _SCHEMA_CONFIG.get("columns", {})

REQUIRED_COLUMNS = sorted(
    name for name, spec in COLUMN_SPECS.items() if spec.get("required")
)
KNOWN_COLUMNS = set(COLUMN_SPECS.keys())

# Column aliases to support real-world header names without modifying raw data.
# These are applied in-memory before schema validation for analysis uploads.
COLUMN_ALIASES: Dict[str, str] = {
    # User's dataset headers -> canonical MMUCC names
    "Crash Number": "crash_id",
    "DateTime": "crash_date",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Crash Severity": "severity",
}

_GEO = _SCHEMA_CONFIG.get("geo_bounds", {})
ALASKA_LAT_RANGE: Tuple[float, float] = (
    float(_GEO.get("latitude", {}).get("min", 51.0)),
    float(_GEO.get("latitude", {}).get("max", 72.0)),
)
ALASKA_LON_RANGE: Tuple[float, float] = (
    float(_GEO.get("longitude", {}).get("min", -170.0)),
    float(_GEO.get("longitude", {}).get("max", -129.0)),
)


def load_dataframe_from_bytes(file_bytes: bytes, extension: str) -> pd.DataFrame:
    """Load the uploaded file into a pandas DataFrame based on its extension."""
    ext = extension.lower()
    if ext == ".csv":
        text = file_bytes.decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(text))
    if ext == ".parquet":
        # Requires pyarrow or fastparquet to be installed. pyarrow is
        # declared in requirements.txt to make this work out-of-the-box.
        return pd.read_parquet(io.BytesIO(file_bytes))

    raise ValueError(f"Unsupported file type for dataframe load: {extension!r}")


def sniff_mime_type(
    file_bytes: bytes, original_name: str, declared_mime: str | None
) -> Dict[str, Any]:
    """Use python-magic when available, otherwise fall back to filename-based detection.

    Returns a dict with:
    - status: "passed" or "failed"
    - detected_mime_type: str
    - details: human-friendly description
    """
    detected: str | None = None
    details: List[str] = []

    # Try python-magic first
    if magic is not None:
        try:
            detected = magic.from_buffer(file_bytes, mime=True)
            details.append(f"python-magic detected MIME type {detected!r}.")
        except Exception as exc:  # noqa: BLE001
            details.append(f"python-magic failed to sniff MIME type: {exc!r}.")

    # Fall back to filename / declared type
    if not detected:
        guessed, _ = mimetypes.guess_type(original_name)
        detected = guessed or declared_mime or "application/octet-stream"
        details.append(
            "Falling back to filename-based MIME detection and/or declared Content-Type."
        )

    # Normalise (strip charset etc.)
    declared_norm = (declared_mime or "").split(";", 1)[0].strip()
    detected_norm = (detected or "").split(";", 1)[0].strip()

    # Treat common CSV-ish types as equivalent. Browsers and libmagic often
    # disagree here, but they're all “texty CSV upload” in practice.
    CSV_LIKE_MIME_TYPES = {
        "text/csv",
        "text/plain",
        "application/vnd.ms-excel",
        "application/octet-stream",
    }

    status = "passed"

    if declared_norm and detected_norm:
        if declared_norm in CSV_LIKE_MIME_TYPES and detected_norm in CSV_LIKE_MIME_TYPES:
            # Soft-accept near-equivalent CSV types
            details.append(
                f"Declared MIME type {declared_norm!r} and detected {detected_norm!r} "
                "are both CSV-like; allowing upload."
            )
        elif declared_norm != detected_norm:
            # Real mismatch (e.g. HTML, binary, etc.) -> fail
            status = "failed"
            details.append(
                f"Declared MIME type {declared_norm!r} does not match detected {detected_norm!r}."
            )
        else:
            details.append(
                f"Declared MIME type {declared_norm!r} is consistent with detected {detected_norm!r}."
            )
    else:
        details.append(
            f"Using detected MIME type {detected_norm!r}; no declared Content-Type from client."
        )

    return {
        "status": status,
        "detected_mime_type": detected_norm or detected,
        "details": " ".join(details),
    }


def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare the uploaded columns against the configured MMUCC-aligned schema.

    Column aliases from COLUMN_ALIASES are treated as satisfying their
    canonical counterparts. For example, a CSV with "Crash Number" will be
    accepted as providing the required "crash_id" field.
    """
    # Original headers as provided by the upload
    columns = [str(c) for c in df.columns]
    actual = set(columns)

    # Work out which canonical schema columns are effectively present once
    # we account for aliases and exact canonical names.
    canonical_present: set[str] = set()
    for col in columns:
        stripped = col.strip()
        if stripped in COLUMN_ALIASES:
            canonical_present.add(COLUMN_ALIASES[stripped])
        elif stripped in KNOWN_COLUMNS:
            canonical_present.add(stripped)

    # Required columns that are missing even after alias mapping
    missing = sorted(list(set(REQUIRED_COLUMNS) - canonical_present))

    # "Unknown" columns = headers that are neither canonical nor an alias source.
    known_or_alias_sources = set(KNOWN_COLUMNS) | set(COLUMN_ALIASES.keys())
    unknown = sorted(list(actual - known_or_alias_sources))

    is_valid = len(missing) == 0
    if is_valid:
        details = "All required MMUCC-aligned columns are present."
    else:
        details = (
            "Dataset is missing required MMUCC-aligned columns: "
            + ", ".join(missing)
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "is_valid": is_valid,
        "missing_columns": missing,
        "unknown_columns": unknown,
        "columns": columns,
        "details": details,
    }


def validate_value_types_and_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    """Basic type & range checks for configured columns.

    Rows with violations are *flagged* (counted) but not dropped. The
    dataset can still be accepted for downstream cleaning.
    """
    total_rows = int(len(df))
    invalid_row_indexes: set[int] = set()
    column_summaries: Dict[str, Dict[str, Any]] = {}

    for name, spec in COLUMN_SPECS.items():
        if name not in df.columns:
            continue

        series = df[name]
        summary: Dict[str, Any] = {
            "non_null_values": int(series.notna().sum()),
            "missing_values": int(series.isna().sum()),
            "invalid_type_count": 0,
            "invalid_range_count": 0,
        }

        col_type = spec.get("type")
        if col_type in {"int", "float"}:
            numeric = pd.to_numeric(series, errors="coerce")
            invalid_type_mask = series.notna() & numeric.isna()
            summary["invalid_type_count"] = int(invalid_type_mask.sum())
            invalid_row_indexes.update(series[invalid_type_mask].index.tolist())

            lower = spec.get("min")
            upper = spec.get("max")
            if lower is not None or upper is not None:
                range_mask = pd.Series(False, index=series.index)
                if lower is not None:
                    range_mask |= numeric < lower
                if upper is not None:
                    range_mask |= numeric > upper

                range_mask &= numeric.notna()
                summary["invalid_range_count"] = int(range_mask.sum())
                invalid_row_indexes.update(series[range_mask].index.tolist())

        elif col_type == "date":
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
            invalid_type_mask = series.notna() & parsed.isna()
            summary["invalid_type_count"] = int(invalid_type_mask.sum())
            invalid_row_indexes.update(series[invalid_type_mask].index.tolist())

        # "category" and "string" are not constrained beyond presence here.

        column_summaries[name] = summary

    invalid_row_count = len(invalid_row_indexes)
    if invalid_row_count == 0:
        details = "All configured columns passed type and range checks."
    else:
        details = (
            f"{invalid_row_count} rows have at least one type or range issue, "
            "but the dataset was accepted for downstream cleaning."
        )

    return {
        "total_rows": total_rows,
        "invalid_row_count": invalid_row_count,
        "columns": column_summaries,
        "details": details,
    }


def _resolve_column_name(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def validate_geo_bounds(df: pd.DataFrame) -> Dict[str, Any]:
    """Ensure that coordinates (if present) fall within the Alaska bounding box.

    Rows that fall outside the bounding box are counted but do not cause rejection.
    """
    lat_col = _resolve_column_name(
        df, ["latitude", "Latitude", "LATITUDE", "lat", "Lat", "LAT"]
    )
    lon_col = _resolve_column_name(
        df, ["longitude", "Longitude", "LONGITUDE", "lon", "Lon", "LON"]
    )

    if lat_col is None or lon_col is None:
        return {
            "has_coordinates": False,
            "invalid_row_count": 0,
            "details": (
                "Dataset does not contain both latitude and longitude columns; "
                "geo bounds checks were skipped."
            ),
        }

    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    in_lat_range = lat.between(ALASKA_LAT_RANGE[0], ALASKA_LAT_RANGE[1])
    in_lon_range = lon.between(ALASKA_LON_RANGE[0], ALASKA_LON_RANGE[1])

    has_coords = lat.notna() & lon.notna()
    in_bounds = has_coords & in_lat_range & in_lon_range
    out_of_bounds_mask = has_coords & ~in_bounds

    invalid_row_count = int(out_of_bounds_mask.sum())

    if invalid_row_count == 0:
        details = "All rows with coordinates fall within the Alaska bounding box."
    else:
        details = (
            f"{invalid_row_count} rows fall outside the Alaska bounding box; "
            "they will be flagged as geo-invalid for downstream processing."
        )

    return {
        "has_coordinates": True,
        "invalid_row_count": invalid_row_count,
        "details": details,
        "lat_column": lat_col,
        "lon_column": lon_col,
        "lat_range": ALASKA_LAT_RANGE,
        "lon_range": ALASKA_LON_RANGE,
    }


# -----------------------------------------------------------------------
# Helpers for analysis app: load + validate dataframe from generic upload
# -----------------------------------------------------------------------


def _extract_file_bytes_and_extension(upload: Any) -> Tuple[bytes, str]:
    """
    Normalize different upload objects into (bytes, extension).

    Supports:
    - ingestion.models.UploadedDataset instances (with .raw_file and original_filename)
    - Django UploadedFile-like objects (request.FILES["file"])
    """
    # Case 1: UploadedDataset model instance
    if hasattr(upload, "raw_file"):
        file_obj = upload.raw_file
        original_name = getattr(upload, "original_filename", file_obj.name)
    # Case 2: plain UploadedFile or file-like object
    elif hasattr(upload, "read"):
        file_obj = upload
        original_name = getattr(upload, "name", "uploaded")
    else:
        raise ValueError(
            "Unsupported upload object; expected UploadedDataset or UploadedFile."
        )

    # Read bytes
    file_bytes = file_obj.read()
    # Reset pointer if possible so the caller can re-read later
    try:
        file_obj.seek(0)
    except Exception:
        pass

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    # Determine extension
    ext = Path(str(original_name)).suffix
    if not ext:
        # Fallback: try using declared content_type if available
        mime = getattr(upload, "content_type", None)
        if mime:
            guessed_ext = mimetypes.guess_extension(mime)
            if guessed_ext:
                ext = guessed_ext

    if not ext:
        raise ValueError(
            f"Could not determine file extension for uploaded file {original_name!r}."
        )

    return file_bytes, ext


def _apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename known alias columns to their canonical MMUCC names.

    This does NOT modify the raw uploaded file; it only adjusts the in-memory
    DataFrame so downstream schema validation and ML logic see the expected names.
    """
    if not COLUMN_ALIASES:
        return df

    rename_map: Dict[str, str] = {}

    for original in df.columns:
        stripped = original.strip()
        # Apply explicit alias if we know this header.
        if stripped in COLUMN_ALIASES:
            rename_map[original] = COLUMN_ALIASES[stripped]
        # Optionally normalize whitespace, keeping the same logical name.
        elif stripped != original:
            rename_map[original] = stripped

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def validate_dataframe_from_upload(upload: Any) -> pd.DataFrame:
    """
    Convenience helper used by the analysis app.

    - Accepts either a Django UploadedFile or an UploadedDataset.
    - Loads the underlying bytes into a pandas DataFrame.
    - Applies column alias mapping for known non-MMUCC header names
      (e.g. 'Crash Number' -> 'crash_id').
    - Runs the same schema checks used in the ingestion pipeline.
    - Raises ValueError if required MMUCC columns are missing.
    - Does NOT drop rows for type/range/geo issues; those are only flagged.
    """
    file_bytes, ext = _extract_file_bytes_and_extension(upload)

    # Load the dataframe from the bytes using the existing helper.
    df = load_dataframe_from_bytes(file_bytes, ext)

    # Normalize / alias column headers before schema validation so that
    # real-world datasets with friendlier names can still pass and be
    # processed by the cleaning and ML pipeline.
    df = _apply_column_aliases(df)

    # Schema validation is strict here: missing required columns = hard error.
    schema_result = validate_schema(df)
    if not schema_result.get("is_valid", False):
        missing = schema_result.get("missing_columns", [])
        details = schema_result.get("details", "")
        msg = f"Schema validation failed. Missing required columns: {', '.join(missing)}."
        if details:
            msg += f" {details}"
        raise ValueError(msg)

    # Run soft checks for additional info (no exceptions here).
    _ = validate_value_types_and_ranges(df)
    _ = validate_geo_bounds(df)

    return df
