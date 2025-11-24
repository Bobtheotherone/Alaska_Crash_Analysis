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

    if magic is not None:
        try:
            detected = magic.from_buffer(file_bytes, mime=True)
            details.append(f"python-magic detected MIME type {detected!r}.")
        except Exception as exc:  # noqa: BLE001
            details.append(f"python-magic failed to sniff MIME type: {exc!r}.")

    if not detected:
        guessed, _ = mimetypes.guess_type(original_name)
        detected = guessed or declared_mime or "application/octet-stream"
        details.append(
            "Falling back to filename-based MIME detection and/or declared Content-Type."
        )

    status = "passed"
    if declared_mime and detected and detected != declared_mime:
        status = "failed"
        details.append(
            f"Declared MIME type {declared_mime!r} does not match detected {detected!r}."
        )
    else:
        details.append(
            f"Declared MIME type {declared_mime!r} is consistent with detected {detected!r}."
        )

    return {
        "status": status,
        "detected_mime_type": detected,
        "details": " ".join(details),
    }


def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Compare the uploaded columns against the configured MMUCC-aligned schema."""
    columns = list(df.columns)
    actual = set(columns)
    missing = sorted(list(set(REQUIRED_COLUMNS) - actual))
    unknown = sorted(list(actual - KNOWN_COLUMNS))

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
