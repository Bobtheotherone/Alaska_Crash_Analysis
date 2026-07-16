"""
crashsev.contracts — the executable Gate-0 precondition (AA2-003, AA2-004).

The original runner read an arbitrary CSV and proceeded; a three-column impostor ran to
completion. This module makes the data/target/feature contract an *executable precondition*:

* ``load_target_mapping`` parses and **validates** ``data/target_mapping.yml`` (the single
  target authority). Malformed / incomplete / ambiguous mappings, or an unevidenced blank
  override, abort here — the mapping can no longer silently fail to load.
* ``load_schema`` parses ``data/schema.json`` (required columns, dtypes, sentinels, target
  domain). ``load_ledger`` parses the feature-availability ledger.
* ``validate_dataframe`` blocks a run unless the frame satisfies the schema: required columns
  present, target values inside the documented domain, id/year/datetime present. A
  three-column impostor fails **before** any split or model fit; the real extract passes.
* ``hash_contract_inputs`` records the SHA-256 of every contract file so a run manifest pins
  the exact contract it was validated against.

Nothing here reads or fabricates data values; it only checks structure and domain.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
KABCO_LETTERS = {"K", "A", "B", "C", "O"}


class ContractViolation(Exception):
    """Raised when data or a contract file fails validation (fail-closed)."""


# ---------------------------------------------------------------------------
# Target mapping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetMapping:
    target_column: str
    text_to_kabco: Dict[str, str]
    numeric_code_map: Dict[str, str]
    quarantine_labels: List[str]
    blank_maps_to: Optional[str]
    kabco_to_ordinal: Dict[str, int]
    kabco_rank: Dict[str, int]
    class_labels: Dict[int, str]
    n_classes: int
    codebook_evidence: Optional[str]
    source_path: str

    def normalized_text_map(self) -> Dict[str, str]:
        """lower-cased raw label -> KABCO letter (for case-insensitive matching)."""
        return {str(k).strip().lower(): v for k, v in self.text_to_kabco.items()}

    def normalized_quarantine(self) -> Set[str]:
        return {str(q).strip().lower() for q in self.quarantine_labels}


def load_target_mapping(path: Optional[Path] = None) -> TargetMapping:
    import yaml
    p = Path(path) if path else DATA_DIR / "target_mapping.yml"
    if not p.exists():
        raise ContractViolation(f"target mapping not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    def req(key):
        if key not in raw:
            raise ContractViolation(f"target_mapping.yml missing required key {key!r}")
        return raw[key]

    text_to_kabco = {str(k): str(v).strip().upper() for k, v in (req("text_to_kabco") or {}).items()}
    numeric_code_map = {str(k): str(v).strip().upper() for k, v in (raw.get("numeric_code_map") or {}).items()}
    kabco_to_ordinal = {str(k).strip().upper(): int(v) for k, v in (req("kabco_to_ordinal")).items()}
    kabco_rank = {str(k).strip().upper(): int(v) for k, v in (raw.get("kabco_rank") or {}).items()}
    quarantine = [str(q) for q in (raw.get("quarantine_labels") or [])]
    blank = raw.get("blank_maps_to")
    blank = str(blank).strip().upper() if blank not in (None, "", "null") else None
    class_labels = {int(k): str(v) for k, v in (raw.get("class_labels") or {}).items()}
    n_classes = int(raw.get("n_classes", len(set(kabco_to_ordinal.values()))))
    evidence = raw.get("codebook_evidence")

    # ---- validation (fail-closed, AA2-004) ----
    bad = {v for v in text_to_kabco.values()} - KABCO_LETTERS
    if bad:
        raise ContractViolation(f"text_to_kabco maps to non-KABCO letters: {sorted(bad)}")
    bad = {v for v in numeric_code_map.values()} - KABCO_LETTERS
    if bad:
        raise ContractViolation(f"numeric_code_map maps to non-KABCO letters: {sorted(bad)}")
    missing = KABCO_LETTERS - set(kabco_to_ordinal)
    if missing:
        raise ContractViolation(f"kabco_to_ordinal is incomplete; missing {sorted(missing)}")
    if blank is not None and blank not in KABCO_LETTERS:
        raise ContractViolation(f"blank_maps_to must be a KABCO letter or null, got {blank!r}")
    # ambiguity: a quarantine label must not also be a mappable text label
    overlap = {q.strip().lower() for q in quarantine} & {k.strip().lower() for k in text_to_kabco}
    if overlap:
        raise ContractViolation(f"labels are both mappable and quarantined (ambiguous): {sorted(overlap)}")
    # an override of the fail-closed blank policy MUST cite evidence that exists
    if blank is not None:
        if not evidence:
            raise ContractViolation("blank_maps_to override set without 'codebook_evidence'")
        ev = (DATA_DIR.parent / evidence)
        if not ev.exists():
            raise ContractViolation(f"codebook_evidence file does not exist: {evidence}")
    ords = set(kabco_to_ordinal.values())
    if ords != set(range(len(ords))):
        raise ContractViolation(f"ordinal classes must be contiguous 0..k-1, got {sorted(ords)}")

    return TargetMapping(
        target_column=str(req("target_column")),
        text_to_kabco=text_to_kabco,
        numeric_code_map=numeric_code_map,
        quarantine_labels=quarantine,
        blank_maps_to=blank,
        kabco_to_ordinal=kabco_to_ordinal,
        kabco_rank=kabco_rank,
        class_labels=class_labels,
        n_classes=n_classes,
        codebook_evidence=evidence,
        source_path=str(p),
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Schema:
    raw: dict
    source_path: str

    @property
    def target_col(self) -> str: return self.raw["target_col"]
    @property
    def row_id_col(self) -> str: return self.raw["row_id_col"]
    @property
    def group_col(self) -> str: return self.raw["group_col"]
    @property
    def year_col(self) -> str: return self.raw["year_col"]
    @property
    def datetime_col(self) -> str: return self.raw.get("datetime_col")
    @property
    def required_columns(self) -> List[str]: return list(self.raw["required_columns"])
    @property
    def allowed_feature_columns(self) -> List[str]: return list(self.raw["allowed_feature_columns"])
    @property
    def numeric_sentinels(self) -> Dict[str, dict]: return dict(self.raw.get("numeric_sentinels", {}))
    @property
    def string_missing_tokens(self) -> List[str]: return list(self.raw.get("string_missing_tokens", []))
    @property
    def target_domain(self) -> dict: return dict(self.raw.get("target_domain", {}))
    @property
    def documented_columns(self) -> Set[str]: return set(self.raw.get("columns", {}))


def load_schema(path: Optional[Path] = None) -> Schema:
    p = Path(path) if path else DATA_DIR / "schema.json"
    if not p.exists():
        raise ContractViolation(f"schema not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    for key in ("target_col", "row_id_col", "year_col", "required_columns", "allowed_feature_columns"):
        if key not in raw:
            raise ContractViolation(f"schema.json missing required key {key!r}")
    return Schema(raw=raw, source_path=str(p))


# ---------------------------------------------------------------------------
# Feature-availability ledger
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ledger:
    df: pd.DataFrame
    source_path: str

    def allowed(self) -> List[str]:
        return list(self.df.loc[self.df["allowed_post_crash_triage"] == True, "column"])  # noqa: E712

    def strict_allowed(self) -> List[str]:
        """FEAT-001 (v4): the conservative strict scene tier — allowed fields whose values are
        unambiguous scene observables (excludes contributing-circumstance/sequence/damage/
        test/insurance families and the high-coupling kinematics trio, pending Gate-3 form-level
        recording-time evidence). Fail-closed: a ledger without the column cannot serve a
        strict-tier run."""
        if "strict_scene_tier" not in self.df.columns:
            raise ContractViolation(
                "feature ledger has no 'strict_scene_tier' column; cannot run feature_tier=strict")
        m = (self.df["allowed_post_crash_triage"] == True) & (self.df["strict_scene_tier"] == True)  # noqa: E712
        return list(self.df.loc[m, "column"])

    def prohibited(self) -> List[str]:
        return list(self.df.loc[self.df["availability"] == "post_outcome", "column"])

    def high_coupling(self) -> List[str]:
        return list(self.df.loc[self.df["high_coupling"] == True, "column"])  # noqa: E712

    def tier(self) -> Dict[str, str]:
        return dict(zip(self.df["column"], self.df["availability"]))


def load_ledger(path: Optional[Path] = None) -> Ledger:
    p = Path(path) if path else DATA_DIR / "feature_availability_ledger.csv"
    if not p.exists():
        raise ContractViolation(f"feature ledger not found: {p}")
    df = pd.read_csv(p)
    for col in ("column", "availability", "allowed_post_crash_triage", "high_coupling"):
        if col not in df.columns:
            raise ContractViolation(f"ledger missing column {col!r}")
    # normalise booleans that may load as strings
    bool_cols = ["allowed_post_crash_triage", "high_coupling"]
    if "strict_scene_tier" in df.columns:
        bool_cols.append("strict_scene_tier")
    for b in bool_cols:
        df[b] = df[b].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    return Ledger(df=df, source_path=str(p))


# ---------------------------------------------------------------------------
# The Gate-0 dataframe check
# ---------------------------------------------------------------------------

@dataclass
class ContractResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_rows: int = 0
    observed_target_values: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "errors": self.errors, "warnings": self.warnings,
            "n_rows": self.n_rows, "observed_target_values": self.observed_target_values,
        }


def validate_dataframe(df: pd.DataFrame, schema: Schema, mapping: TargetMapping) -> ContractResult:
    """Fail-closed structural + domain check. Returns a result; caller decides to raise."""
    errors: List[str] = []
    warnings: List[str] = []

    cols = set(df.columns)

    # 1. required columns
    missing = [c for c in schema.required_columns if c not in cols]
    if missing:
        errors.append(f"missing {len(missing)} required column(s): {missing[:12]}"
                      + (" …" if len(missing) > 12 else ""))

    # 2. target column + domain
    tcol = schema.target_col
    observed: Dict[str, int] = {}
    if tcol not in cols:
        errors.append(f"target column {tcol!r} is absent")
    else:
        sev = df[tcol]
        observed = {str(k): int(v) for k, v in sev.value_counts(dropna=False).items()}
        mappable = mapping.normalized_text_map()
        quarantine = mapping.normalized_quarantine()
        numeric = {str(k).strip().lower() for k in mapping.numeric_code_map}
        undocumented = []
        for val, _cnt in sev.value_counts(dropna=True).items():
            s = str(val).strip().lower()
            if s == "" or s in mappable or s in quarantine or s in numeric or s in {"nan"}:
                continue
            undocumented.append(str(val))
        if undocumented:
            errors.append(f"target has undocumented value(s) not in the codebook: {undocumented[:12]}")

    # 3. id / year / datetime keys
    for key in (schema.row_id_col, schema.year_col):
        if key and key not in cols:
            errors.append(f"key column {key!r} is absent")
    if schema.datetime_col and schema.datetime_col not in cols:
        warnings.append(f"datetime column {schema.datetime_col!r} absent; chronological ordering limited")

    # 4. row-id uniqueness (crash-level unit)
    if schema.row_id_col in cols:
        n_dup = int(df[schema.row_id_col].duplicated().sum())
        if n_dup:
            warnings.append(f"{n_dup} duplicate {schema.row_id_col!r} values (expected unique per crash)")

    # 5. undocumented extra columns -> warning (dropped downstream), not a hard failure
    if schema.documented_columns:
        extra = sorted(cols - schema.documented_columns)
        if extra:
            warnings.append(f"{len(extra)} column(s) not in schema (ignored): {extra[:8]}"
                            + (" …" if len(extra) > 8 else ""))

    ok = not errors
    return ContractResult(ok=ok, errors=errors, warnings=warnings,
                          n_rows=int(len(df)), observed_target_values=observed)


def assert_valid(df: pd.DataFrame, schema: Schema, mapping: TargetMapping) -> ContractResult:
    res = validate_dataframe(df, schema, mapping)
    if not res.ok:
        raise ContractViolation("data failed the Gate-0 contract:\n  - " + "\n  - ".join(res.errors))
    return res


# ---------------------------------------------------------------------------
# Provenance hashing
# ---------------------------------------------------------------------------

def _sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hash_contract_inputs(
    schema: Schema, mapping: TargetMapping, ledger: Ledger
) -> Dict[str, str]:
    return {
        "schema_sha256": _sha256_file(schema.source_path),
        "target_mapping_sha256": _sha256_file(mapping.source_path),
        "feature_ledger_sha256": _sha256_file(ledger.source_path),
    }
