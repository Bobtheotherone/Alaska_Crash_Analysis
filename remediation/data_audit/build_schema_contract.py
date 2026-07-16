"""Emit the executable data-schema contract ``data/schema.json`` (AA2-003).

Reads the raw extract once to record, per column: the expected dtype family, the
feature-availability tier (from the ledger), and — for numeric feature columns — any
NULL sentinels (e.g. AADT's int32-min ``-2147483648``) so preprocessing can neutralise them
instead of feeding absurd magnitudes to a model. The resulting JSON is what
``crashsev.contracts`` loads to accept a valid extract and reject an impostor.

Run: ``python data_audit/build_schema_contract.py``
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
DEFAULT_XLSX = str(DATA_DIR / "raw" / "Crash Level 09-12 (1).xlsx")  # repo-relative; supply a lawful copy
XLSX = Path(os.environ.get("CRASH_XLSX_09_12", DEFAULT_XLSX))

INT32_MIN = -2147483648  # numpy int32 sentinel used as NULL in the source system

STRING_MISSING_TOKENS = [
    "", "null value", "no data", "missing", "not reported", "not available",
    "not applicable", "unknown", "n/a", "na", "none given", "not stated",
]

# Domain rules for numeric feature columns (beyond the generic INT32_MIN sentinel).
NUMERIC_DOMAIN = {
    "AADT": {"valid_min": 1},                     # traffic volume must be positive
    "Distance From Intersection": {"valid_min": 0, "valid_max": 52800},  # >10 mi from a junction = sentinel
    "Temperature": {"valid_min": -80, "valid_max": 120},  # plausible Alaska °F band
}


def _dtype_family(s: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        return "numeric"
    return "string"


def main() -> None:
    ledger = pd.read_csv(DATA_DIR / "feature_availability_ledger.csv")
    tier = dict(zip(ledger["column"], ledger["availability"]))
    allowed = dict(zip(ledger["column"], ledger["allowed_post_crash_triage"]))

    df = pd.read_excel(XLSX, engine="openpyxl")

    columns: dict[str, dict] = {}
    numeric_sentinels: dict[str, dict] = {}
    for c in df.columns:
        fam = _dtype_family(df[c])
        columns[c] = {
            "dtype_family": fam,
            "tier": tier.get(c, "UNCLASSIFIED"),
            "allowed_feature": bool(allowed.get(c, False)),
        }
        if fam == "numeric" and allowed.get(c, False):
            vals = pd.to_numeric(df[c], errors="coerce")
            nulls = []
            if (vals == INT32_MIN).any():
                nulls.append(INT32_MIN)
            rule = {"null_values": nulls}
            rule.update(NUMERIC_DOMAIN.get(c, {}))
            # record how many rows the rule would neutralise (transparency)
            bad = (vals.isin(nulls)) if nulls else pd.Series(False, index=vals.index)
            if "valid_min" in rule:
                bad = bad | (vals < rule["valid_min"])
            if "valid_max" in rule:
                bad = bad | (vals > rule["valid_max"])
            rule["n_flagged"] = int(bad.sum())
            rule["observed_min"] = float(np.nanmin(vals.values))
            rule["observed_max"] = float(np.nanmax(vals.values))
            if nulls or "valid_min" in rule or "valid_max" in rule:
                numeric_sentinels[c] = rule

    allowed_features = sorted(c for c in df.columns if allowed.get(c, False))
    required = sorted(set(allowed_features) | {"Crash Severity", "Crash Number", "Year", "DateTime"})

    schema = {
        "schema_version": "1.0",
        "source_extract": "Crash Level 09-12",
        "unit": "crash",
        "row_id_col": "Crash Number",
        "group_col": "Crash Number",
        "year_col": "Year",
        "datetime_col": "DateTime",
        "target_col": "Crash Severity",
        "n_documented_columns": int(df.shape[1]),
        "required_columns": required,
        "allowed_feature_columns": allowed_features,
        "columns": columns,
        "numeric_sentinels": numeric_sentinels,
        "string_missing_tokens": STRING_MISSING_TOKENS,
        "target_domain": {
            "mappable_labels": ["Fatal", "Incapacitating", "Non-Incapacitating", "Possible"],
            "blank_maps_to": "O",
            "quarantine_labels": ["Unknown", "Not Reported", "Null value"],
            "note": "Any target value outside mappable ∪ quarantine ∪ blank is UNDOCUMENTED and aborts the run.",
        },
    }
    (DATA_DIR / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"wrote data/schema.json: {len(required)} required columns, "
          f"{len(allowed_features)} allowed features, {len(numeric_sentinels)} sentinel rules")
    for c, r in numeric_sentinels.items():
        print(f"  sentinel[{c}]: {r}")


if __name__ == "__main__":
    main()
