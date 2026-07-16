"""
crashsev.cli — the single governed entry point for the crash-severity study
(evidentiary route: Route R, retrospective out-of-time — research/ROUTE_DECISION.md;
the historical "Route A" label named the raw-09-12 DATA route, not the study route)
(AA2-007/008/011/012/014/017).

Phases form a state machine that governs *within-study* access to the held-out, out-of-time 2012
evaluation. Scope of the guarantee, stated precisely (GOV-001, v4): the develop phase
(``prepare_development``) partitions rows BY YEAR before any target interpretation — final-year
outcomes are never validated, mapped, audited, or serialized during development, and development
artifacts cannot yield them by derivation. Precision note: the shared source table IS loaded and
content-hashed WHOLE as an opaque integrity operation before partitioning (final-year target
bytes pass through memory and the hash), so the isolation is semantic/analytic, not a byte-level
or process-boundary non-access claim. The FULL ``prepare()`` path — which does map all years —
is used only by ``evaluate-final``, whose job is to read the final year. It is NOT a prospective
seal — the 2012 data were locally available and the evaluation is retrospective
(see research/ROUTE_DECISION.md):

    validate-data      Gate-0 contract check on a data file (no modelling).
    develop            Map target, hold out the evaluation year, run development CV for model
                       selection + a dev-only calibration fit. Writes a development report.
                       NEVER touches the held-out year.
    freeze-experiment  Write FROZEN.lock pinning config/contract/data/split/development hashes
                       + git commit. Freezes the experiment.
    evaluate-final     Requires FROZEN.lock matching the current inputs and a clean git tree;
                       fits each model on all development data and evaluates ONCE, under the frozen
                       config, on the held-out out-of-time year; writes an immutable run bundle and a
                       FINAL marker. Refuses to run twice for the same frozen experiment.

Design choices that make the governance real:
  * Held-out-year feature/label MATRICES are materialised only inside ``evaluate-final``;
    ``develop`` never fits or selects on them (see the v3.x scope caveat above).
  * ``evaluate-final`` recomputes the data/config/contract/split hashes and refuses if any
    differs from FROZEN.lock, or if the git working tree is dirty (``--allow-dirty`` is a
    development-only escape hatch that is refused once a lock exists).
  * Run bundles are created atomically (temp dir -> content hash -> rename) and refuse to
    overwrite an existing bundle; a STATUS file marks RUNNING/COMPLETE/FAILED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import calibration as Cal
from . import contracts as K
from . import metrics as M
from . import models as Models
from . import preprocessing as pp
from . import splits as S
from . import target as T
from . import uncertainty as U

REPO_ROOT = Path(__file__).resolve().parents[2]      # .../aca
PKG_ROOT = Path(__file__).resolve().parents[1]        # .../remediation


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    return _sha256_bytes(Path(p).read_bytes())


def _git(args) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT)] + args,
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


# Output directories the governed phases WRITE to; changes here are not "source dirty".
_OUTPUT_PREFIXES = ("remediation/experiment/", "remediation/runs/", "remediation/_local_data/")


def source_tree_dirty() -> bool:
    """True iff tracked source / contract / data (anything OUTSIDE the phase-output dirs) has
    uncommitted changes. The final-test governance requires the *code* to be committed; it must
    not be tripped by the report/lock/results the phases themselves emit.

    Parsing note (v4 fix): ``_git`` strips the whole stdout blob, which removes the FIRST
    porcelain line's leading status space (`` M path`` -> ``M path``); the old fixed-offset
    ``ln[3:]`` then ate the path's first character and the output-prefix exemption never
    matched. Parse by splitting off the status token instead."""
    for ln in _git(["status", "--porcelain"]).splitlines():
        if not ln.strip():
            continue
        parts = ln.strip().split(None, 1)
        if len(parts) < 2:
            continue
        path = parts[1].split(" -> ")[-1].strip().strip('"')
        if not any(path.startswith(pre) for pre in _OUTPUT_PREFIXES):
            return True
    return False


def _git_state() -> Dict[str, object]:
    return {"commit": _git(["rev-parse", "HEAD"]),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_git(["status", "--porcelain"])),
            "source_dirty": source_tree_dirty()}


def _lib_versions() -> Dict[str, str]:
    import sklearn, scipy
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
         "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}
    avail = Models.availability()
    if avail["xgboost"]:
        import xgboost; v["xgboost"] = xgboost.__version__
    if avail["ebm"]:
        import interpret; v["interpret"] = getattr(interpret, "__version__", "unknown")
    return v


DEFAULT_CONFIG = {
    "use_case": "post_crash_triage",
    "year_col": "Year",
    "group_col": "Crash Number",
    "row_id_col": "Crash Number",
    "final_test_years": [2012],
    "min_frequency": 0.01,
    "primary_metric": "ordinal_mae",
    "baseline_set": ["majority", "ordinal_median", "prior_probability",
                     "multinomial_logistic", "proportional_odds", "frank_hall_logistic",
                     "shallow_tree"],
    "bootstrap_resamples": 2000,
    "seed": 42,
    "cv_strategy": "rolling",          # "rolling" (origin) or "grouped"
    "cv_seeds": [1, 2, 3],
    "cv_n_splits": 3,
    "cv_models": None,                  # None -> all registry models except EBM
    "include_optional_models": True,
    "categorical_max_cardinality": 100,
    "calibrate": True,
    "calibration_method": "sigmoid",
    "calibration_folds": "grouped",     # "temporal" (v4: rolling-origin year folds) or "grouped"
    "exclude_high_coupling": False,     # ablation toggle (drops ejection/restraint/seat)
    "feature_tier": "broad",            # "strict" (v4 primary: scene-observable tier) or "broad"
    "decision_rule": "argmax",          # "posterior_median" (v4 primary; loss-consistent) or "argmax"
    "n_classes": 3,
}


def select_allowed_features(ledger, cfg: dict, df: pd.DataFrame):
    """Resolve the admissible feature set: tier (FEAT-001 strict scene tier vs broad), the
    high-coupling ablation toggle, an optional explicit exclusion list (used by prespecified
    sensitivities such as the high-missingness ablation, MISS-002), and presence in the data."""
    base = ledger.strict_allowed() if cfg.get("feature_tier", "broad") == "strict" else ledger.allowed()
    allowed = [c for c in base if c in df.columns]
    if cfg.get("exclude_high_coupling"):
        hc = set(ledger.high_coupling())
        allowed = [c for c in allowed if c not in hc]
    excl = set(cfg.get("exclude_features") or [])
    if excl:
        allowed = [c for c in allowed if c not in excl]
    return allowed


def drop_uninformative(df: pd.DataFrame, allowed, missing_tokens=()) -> Tuple[List[str], List[str]]:
    """MISS-001: drop allowed features that are invariant on the DEVELOPMENT data — all-missing,
    single-valued, or consisting ONLY of documented missing tokens (e.g. 'Rural Urban' is 100%
    "Unknown"/"Null value" strings, which are not NaN but carry no information). Retaining such
    fields misstates the effective input inventory. Decision is train-side only and recorded."""
    toks = {str(t).strip().lower() for t in missing_tokens if str(t).strip()}
    dropped = []
    for c in allowed:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            sv = s.astype(str).str.strip()
            s = sv.mask(sv.str.lower().isin(toks) | (sv == "") | (sv.str.lower() == "nan"))
        if s.dropna().nunique() <= 1:
            dropped.append(c)
    return [c for c in allowed if c not in dropped], dropped


def decide_labels(proba: Optional[np.ndarray], y_argmax: np.ndarray, cfg: dict) -> np.ndarray:
    """OBJ-001 / DECISION-LOSS-001: the configured PRIMARY hard-decision rule. v4 prespecifies
    'posterior_median' (Bayes-optimal for the absolute ordinal loss); 'argmax' is the v3 rule,
    retained as a sensitivity."""
    if cfg.get("decision_rule", "argmax") == "posterior_median" and proba is not None:
        return M.posterior_median(proba)
    return np.asarray(y_argmax, dtype=int)


def _dev_assignment(dev_ids: pd.Series, keep_mask: np.ndarray, dev_years: pd.Series):
    """The DEVELOPMENT-side assignment table + hash — the v4 freeze object (outcome-free on the
    final year: it covers development-year rows only). Kept rows -> 'development'; rows whose
    severity failed the fail-closed mapping -> 'excluded_quarantined_dev'."""
    if dev_ids.isna().any():
        raise K.ContractViolation(f"row-id column has {int(dev_ids.isna().sum())} null id(s)")
    if dev_ids.duplicated().any():
        raise K.ContractViolation(f"row-id column has {int(dev_ids.duplicated().sum())} duplicate id(s)")
    partitions = np.where(np.asarray(keep_mask, dtype=bool), "development", "excluded_quarantined_dev")
    assignment_df = pd.DataFrame({
        "row_id": dev_ids.astype(str).to_numpy(),
        "partition": partitions,
        "year": pd.to_numeric(dev_years, errors="coerce").to_numpy(),
    })
    return assignment_df, S.assignment_hash(assignment_df)


def load_config(path: Optional[str]) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        import yaml
        cfg.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return cfg


def config_hash(cfg: dict) -> str:
    return _sha256_bytes(json.dumps(cfg, sort_keys=True, default=str).encode())


# ---------------------------------------------------------------------------
# data loading + shared preparation (up to, but not including, fitting)
# ---------------------------------------------------------------------------

def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, engine="openpyxl")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, low_memory=False)


def clean_numeric_sentinels(df: pd.DataFrame, schema: K.Schema) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Replace documented numeric sentinels / out-of-range values with NaN (AA2 data quality).
    Deterministic and non-learned (uses fixed schema thresholds, not data statistics), so it
    introduces no train/test leakage. Returns (df, n_flagged_per_column)."""
    df = df.copy()
    flagged: Dict[str, int] = {}
    for col, rule in schema.numeric_sentinels.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = pd.Series(False, index=df.index)
        for nv in rule.get("null_values", []):
            bad = bad | (vals == nv)
        if "valid_min" in rule:
            bad = bad | (vals < rule["valid_min"])
        if "valid_max" in rule:
            bad = bad | (vals > rule["valid_max"])
        vals = vals.where(~bad, other=np.nan)
        df[col] = vals
        flagged[col] = int(bad.sum())
    return df, flagged


class Prepared:
    """Everything shared by develop and evaluate-final, computed identically in both."""
    def __init__(self, **kw): self.__dict__.update(kw)


def prepare(data_path: str, cfg: dict, *, seal_final: bool) -> Prepared:
    """FULL preparation — used by ``evaluate-final`` ONLY (it may read final-year outcomes: that
    is its job). The develop phase uses :func:`prepare_development`, which never interprets
    final-year outcome values (GOV-001 v4 structural isolation)."""
    schema = K.load_schema()
    mapping = K.load_target_mapping()
    ledger = K.load_ledger()
    contract_hashes = K.hash_contract_inputs(schema, mapping, ledger)

    raw = load_any(data_path)
    data_hash = _sha256_bytes(pd.util.hash_pandas_object(raw, index=False).values.tobytes())

    # Gate 0
    K.assert_valid(raw, schema, mapping)

    # target mapping (fail-closed, codebook-driven)
    kabco, y_all, audit = T.map_severity_from_mapping(raw[schema.target_col], mapping)
    keep = y_all.notna().to_numpy()

    # v4 freeze object: the DEVELOPMENT-side assignment (recomputed here EXACTLY as the develop
    # phase computes it, so evaluate-final can verify the frozen dev assignment hash).
    years_all = pd.to_numeric(raw[cfg["year_col"]], errors="coerce")
    final_years = [int(v) for v in cfg["final_test_years"]]
    dev_mask_raw = (~years_all.isin(final_years)).to_numpy() & years_all.notna().to_numpy()
    dev_assignment_df, dev_assignment_sha256 = _dev_assignment(
        raw.loc[dev_mask_raw, cfg["row_id_col"]], keep[dev_mask_raw],
        raw.loc[dev_mask_raw, cfg["year_col"]])

    df = raw.loc[keep].reset_index(drop=True)
    y = y_all[keep].astype(int).reset_index(drop=True)

    # numeric sentinel cleaning
    df, sentinel_flagged = clean_numeric_sentinels(df, schema)

    # feature selection: tier (FEAT-001) + high-coupling toggle + presence
    allowed = select_allowed_features(ledger, cfg, df)

    # class coverage on the whole usable set (AA2-011)
    n_classes = int(cfg["n_classes"])
    present = sorted(pd.Series(y).unique().tolist())
    if present != list(range(n_classes)):
        raise K.ContractViolation(
            f"usable data does not cover all {n_classes} classes (present={present}); "
            f"cannot run a {n_classes}-class study")

    # immutable chronological + grouped split
    dev_idx, test_idx, split_manifest, assignment_df = S.chronological_group_split(
        df, y, year_col=cfg["year_col"], group_col=cfg["group_col"],
        final_test_years=cfg["final_test_years"], row_id_col=cfg["row_id_col"])
    if split_manifest.group_overlap_count != 0:
        raise K.ContractViolation(f"split group overlap = {split_manifest.group_overlap_count}")

    # MISS-001: drop invariant/all-missing features, decided on DEVELOPMENT rows only
    allowed, dropped_invariant = drop_uninformative(df.iloc[dev_idx], allowed,
                                                    schema.string_missing_tokens)
    X = df[allowed].copy()

    X_dev, y_dev = X.iloc[dev_idx].reset_index(drop=True), y.iloc[dev_idx].reset_index(drop=True)
    pp.check_sufficient(X_dev, y_dev)
    numeric_cols, categorical_cols, dropped_hc = pp.split_feature_types(
        X_dev, categorical_max_cardinality=cfg["categorical_max_cardinality"])

    X_test = y_test = groups_test = None
    if seal_final:
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        groups_test = df.iloc[test_idx][cfg["group_col"]].astype(str).to_numpy()

    return Prepared(
        schema=schema, mapping=mapping, ledger=ledger, contract_hashes=contract_hashes,
        data_hash=data_hash, audit=audit, df=df, y=y, X=X, allowed=allowed,
        dropped_invariant=dropped_invariant,
        dev_idx=dev_idx, test_idx=test_idx, split_manifest=split_manifest,
        assignment_df=assignment_df, sentinel_flagged=sentinel_flagged,
        dev_assignment_sha256=dev_assignment_sha256,
        X_dev=X_dev, y_dev=y_dev, X_test=X_test, y_test=y_test, groups_test=groups_test,
        numeric_cols=numeric_cols, categorical_cols=categorical_cols, dropped_hc=dropped_hc,
        groups_dev=df.iloc[dev_idx][cfg["group_col"]].astype(str).to_numpy(),
        years_dev=pd.to_numeric(df.iloc[dev_idx][cfg["year_col"]], errors="coerce").to_numpy(),
    )


def prepare_development(data_path: str, cfg: dict) -> Prepared:
    """GOV-001 (v4) — STRUCTURAL outcome isolation for the develop phase.

    The file is loaded once and content-hashed WHOLE (an opaque integrity operation over bytes —
    no value of any final-year field is interpreted). Rows are then partitioned **by year** (the
    split axis — never an outcome), the final-test-year rows are reduced to two outcome-free
    facts — their ROW COUNT and their group-id set (identifiers, needed for the zero-overlap
    guarantee) — and dropped. Everything downstream (Gate-0 validation, target mapping, the
    mapping audit, sentinel cleaning, feature selection, CV) sees development-year rows only, so
    no development artifact can carry, or allow the reconstruction of, any final-year outcome
    (the v3.x subtraction defect is impossible: the audit universe IS the development rows)."""
    schema = K.load_schema()
    mapping = K.load_target_mapping()
    ledger = K.load_ledger()
    contract_hashes = K.hash_contract_inputs(schema, mapping, ledger)

    raw = load_any(data_path)
    data_hash = _sha256_bytes(pd.util.hash_pandas_object(raw, index=False).values.tobytes())

    year_col, group_col, row_id_col = cfg["year_col"], cfg["group_col"], cfg["row_id_col"]
    for c in (year_col, row_id_col):
        if c not in raw.columns:
            raise K.ContractViolation(f"key column {c!r} is absent")

    years_all = pd.to_numeric(raw[year_col], errors="coerce")
    final_years = sorted(int(v) for v in cfg["final_test_years"])
    final_mask = years_all.isin(final_years).to_numpy()
    dev_mask = (~final_mask) & years_all.notna().to_numpy()
    n_final_rows_raw = int(final_mask.sum())
    n_excluded_year = int(len(raw) - final_mask.sum() - dev_mask.sum())
    final_group_ids = (set(raw.loc[final_mask, group_col].astype(str))
                       if group_col in raw.columns else set())

    # SPLIT-001 temporal contract, on YEARS only (outcome-free)
    dev_years = sorted({int(v) for v in years_all[dev_mask].dropna().unique()})
    if not dev_years or n_final_rows_raw == 0:
        raise ValueError(f"Chronological split produced an empty partition "
                         f"(dev years={dev_years}, final rows={n_final_rows_raw})")
    if max(dev_years) >= min(final_years):
        raise ValueError(f"SPLIT-001: development years {dev_years} are not all before the final "
                         f"years {final_years}; the temporal split requires earlier-only development.")
    span = dev_years + final_years
    if span != list(range(span[0], span[-1] + 1)):
        raise ValueError(f"SPLIT-001: development+final years {span} are not contiguous; refusing.")

    dev_raw = raw.loc[dev_mask].reset_index(drop=True)
    del raw  # final-year rows leave the phase here; only their count + group ids survive

    # Gate 0 on DEVELOPMENT rows only (validating the final-year target domain would read outcomes)
    K.assert_valid(dev_raw, schema, mapping)

    # fail-closed target mapping — DEVELOPMENT rows only; the audit universe is n_dev_year_rows
    kabco, y_all, audit = T.map_severity_from_mapping(dev_raw[schema.target_col], mapping)
    keep = y_all.notna().to_numpy()

    assignment_df, dev_assignment_sha256 = _dev_assignment(
        dev_raw[row_id_col], keep, dev_raw[year_col])

    df = dev_raw.loc[keep].reset_index(drop=True)
    y = y_all[keep].astype(int).reset_index(drop=True)
    df, sentinel_flagged = clean_numeric_sentinels(df, schema)

    allowed = select_allowed_features(ledger, cfg, df)
    allowed, dropped_invariant = drop_uninformative(df, allowed, schema.string_missing_tokens)
    X = df[allowed].copy()

    n_classes = int(cfg["n_classes"])
    present = sorted(pd.Series(y).unique().tolist())
    if present != list(range(n_classes)):
        raise K.ContractViolation(
            f"development data does not cover all {n_classes} classes (present={present})")

    overlap = (len(set(df[group_col].astype(str)) & final_group_ids)
               if group_col in df.columns else 0)
    if overlap != 0:
        raise K.ContractViolation(f"split group overlap = {overlap}")

    pp.check_sufficient(X, y)
    numeric_cols, categorical_cols, dropped_hc = pp.split_feature_types(
        X, categorical_max_cardinality=cfg["categorical_max_cardinality"])

    dev_split = {
        "strategy": "chronological_final_test_with_crash_grouping (v4 develop: year-partition "
                    "BEFORE any target interpretation)",
        "development_years": dev_years,
        "final_test_years": final_years,
        "n_dev_year_rows": int(len(assignment_df)),
        "n_development": int(len(df)),
        "n_dev_quarantined": int((~keep).sum()),
        "n_final_test_rows_raw": n_final_rows_raw,   # ROW COUNT only — outcomes never read
        "n_excluded_year": n_excluded_year,
        "group_overlap_count": int(overlap),
        "dev_assignment_sha256": dev_assignment_sha256,
        "class_prevalence": {"development": S._prevalence(y)},
        "reconciles": int(len(df)) + int((~keep).sum()) == int(len(assignment_df)),
        "notes": ("v4 GOV-001: the develop phase partitions by YEAR before any target "
                  "interpretation; final-year outcome values are never validated, mapped, "
                  "audited, or serialized here. The final side contributes only its raw row "
                  "count and its group-id set (identifiers, for the zero-overlap check)."),
    }

    return Prepared(
        schema=schema, mapping=mapping, ledger=ledger, contract_hashes=contract_hashes,
        data_hash=data_hash, audit=audit, allowed=allowed, dropped_invariant=dropped_invariant,
        assignment_df=assignment_df, dev_assignment_sha256=dev_assignment_sha256,
        sentinel_flagged=sentinel_flagged, dev_split=dev_split,
        n_final_rows_raw=n_final_rows_raw,
        X_dev=X, y_dev=y,
        numeric_cols=numeric_cols, categorical_cols=categorical_cols, dropped_hc=dropped_hc,
        groups_dev=df[group_col].astype(str).to_numpy(),
        years_dev=pd.to_numeric(df[year_col], errors="coerce").to_numpy(),
    )


# ---------------------------------------------------------------------------
# fitting + aligned prediction
# ---------------------------------------------------------------------------

def _sample_weight(spec, y_tr) -> Optional[np.ndarray]:
    if spec.class_weight_mode == "sample_weight":
        from sklearn.utils.class_weight import compute_sample_weight
        return compute_sample_weight("balanced", y_tr)
    return None


def fit_model(spec, X_tr, y_tr, numeric_cols, categorical_cols, cfg):
    pipe = Models.make_pipeline(spec, numeric_cols, categorical_cols,
                                min_frequency=cfg["min_frequency"])
    sw = _sample_weight(spec, y_tr)
    fit_kw = {"clf__sample_weight": sw} if sw is not None else {}
    pipe.fit(X_tr, y_tr, **fit_kw)
    return pipe, {"sample_weight_applied": sw is not None}


def predict_aligned(pipe, X, n_classes: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Predict labels and a probability matrix whose columns are aligned to class labels
    0..n_classes-1 (filling absent classes with 0), per the persisted ``classes_`` (AA2-011)."""
    y_pred = np.asarray(pipe.predict(X)).astype(int)
    proba = None
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        raw = np.asarray(pipe.predict_proba(X), dtype="float64")
        classes = np.asarray(clf.classes_).astype(int)
        proba = np.zeros((raw.shape[0], n_classes))
        for j, c in enumerate(classes):
            if 0 <= c < n_classes:
                proba[:, c] = raw[:, j]
        rs = proba.sum(axis=1, keepdims=True)
        proba = np.divide(proba, rs, out=np.full_like(proba, 1.0 / n_classes), where=rs > 0)
    return y_pred, proba


def _ordinal_mae(yt, yp):
    return M.ordinal_mae_from_cm(M.confusion_matrix_from_preds(yt, yp, 3))


def dev_only_calibrate(spec, X_dev, y_dev, X_test, groups_dev, cfg, n_classes, years_dev=None):
    """Fit a probability calibrator on DEVELOPMENT data only and return calibrated probabilities
    on the held-out out-of-time test (AA2-014). CAL-001 (v4): with
    ``calibration_folds: temporal`` the internal folds are rolling-origin YEAR folds (calibrate
    on a later development year than the fold's training years), mirroring the temporal
    generalisation the final test measures — the v3 GroupKFold over unique crash ids was
    effectively a random row split. Skips models that need sample_weight (the calibrator's
    internal refits would drop the weights) and any estimator without predict_proba.
    Returns (proba_aligned, method) or (None, reason)."""
    if not cfg.get("calibrate"):
        return None, "disabled"
    if spec.class_weight_mode == "sample_weight":
        return None, "skipped (sample_weight model)"
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GroupKFold
    if cfg.get("calibration_folds", "grouped") == "temporal" and years_dev is not None:
        yrs = pd.Series(pd.to_numeric(pd.Series(years_dev), errors="coerce"))
        uniq = sorted(int(v) for v in yrs.dropna().unique())
        folds = []
        for i in range(1, len(uniq)):
            tr = np.where(yrs.isin(uniq[:i]).to_numpy())[0]
            va = np.where((yrs == uniq[i]).to_numpy())[0]
            if len(tr) and len(va):
                folds.append((tr, va))
        if not folds:
            return None, "skipped (temporal folds unavailable: <2 development years)"
        method_label = f'{cfg.get("calibration_method", "sigmoid")} (temporal rolling-origin folds)'
    else:
        n_groups = len(np.unique(groups_dev))
        n_splits = min(3, max(2, n_groups))
        folds = list(GroupKFold(n_splits=n_splits).split(X_dev, y_dev, groups=groups_dev))
        method_label = f'{cfg.get("calibration_method", "sigmoid")} (grouped folds)'
    base = Models.make_pipeline(spec, [c for c in X_dev.columns if pd.api.types.is_numeric_dtype(X_dev[c])],
                                [c for c in X_dev.columns if not pd.api.types.is_numeric_dtype(X_dev[c])],
                                min_frequency=cfg["min_frequency"])
    cal = CalibratedClassifierCV(base, method=cfg.get("calibration_method", "sigmoid"), cv=folds)
    cal.fit(X_dev, y_dev)
    raw = np.asarray(cal.predict_proba(X_test), dtype="float64")
    classes = np.asarray(cal.classes_).astype(int)
    P = np.zeros((raw.shape[0], n_classes))
    for j, c in enumerate(classes):
        if 0 <= c < n_classes:
            P[:, c] = raw[:, j]
    rs = P.sum(axis=1, keepdims=True)
    P = np.divide(P, rs, out=np.full_like(P, 1.0 / n_classes), where=rs > 0)
    return P, method_label


# ---------------------------------------------------------------------------
# development-phase CV folds
# ---------------------------------------------------------------------------

def _rolling_folds(prep: Prepared, cfg: dict):
    """Rolling-origin folds over development years: train on years < Y, validate on year Y."""
    years = pd.Series(prep.years_dev)
    uniq = sorted(int(y_) for y_ in years.dropna().unique())
    for i in range(1, len(uniq)):
        tr = np.where(years.isin(uniq[:i]).to_numpy())[0]
        va = np.where((years == uniq[i]).to_numpy())[0]
        if len(tr) and len(va):
            yield tr, va, f"train<={uniq[i-1]}|val={uniq[i]}"


def _grouped_folds(prep: Prepared, cfg: dict):
    groups = prep.groups_dev
    from sklearn.model_selection import GroupKFold
    n_groups = len(np.unique(groups))
    n_splits = min(cfg["cv_n_splits"], max(2, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    for i, (tr, va) in enumerate(gkf.split(prep.X_dev, prep.y_dev, groups=groups)):
        yield tr, va, f"gkf_fold{i}"


def dev_cv(prep: Prepared, cfg: dict, log=print) -> dict:
    """Run development cross-validation for each model over folds x seeds; return per-trial and
    aggregated ordinal-MAE (+ within-one accuracy). Selection uses ONLY development data."""
    reg = Models.build_registry(include_optional=cfg["include_optional_models"], seed=cfg["seed"])
    cv_models = cfg["cv_models"] or [m for m in reg if m != "ebm"]
    folds = list(_grouped_folds(prep, cfg) if cfg["cv_strategy"] == "grouped" else _rolling_folds(prep, cfg))
    trials: List[dict] = []
    for name in cv_models:
        spec = reg[name]
        # stochastic models are repeated across seeds; deterministic ones run once
        seeds = cfg["cv_seeds"] if _is_stochastic(name) else [cfg["seed"]]
        for seed in seeds:
            reg_s = Models.build_registry(include_optional=cfg["include_optional_models"], seed=seed)
            spec_s = reg_s[name]
            for tr, va, tag in folds:
                Xtr, ytr = prep.X_dev.iloc[tr], prep.y_dev.iloc[tr]
                Xva, yva = prep.X_dev.iloc[va], prep.y_dev.iloc[va]
                if pd.Series(ytr).nunique() < cfg["n_classes"]:
                    continue
                try:
                    # PREP-001 (v4): feature typing/cardinality is decided INSIDE the fold, on
                    # the fold's training rows only — a validation-year category can no longer
                    # influence a training-time schema decision.
                    num_f, cat_f, _ = pp.split_feature_types(
                        Xtr, categorical_max_cardinality=cfg["categorical_max_cardinality"])
                    pipe, _ = fit_model(spec_s, Xtr, ytr, num_f, cat_f, cfg)
                    yp_arg, proba = predict_aligned(pipe, Xva, cfg["n_classes"])
                    yp = decide_labels(proba, yp_arg, cfg)   # primary rule (v4: posterior median)
                    trial = {"model": name, "seed": int(seed), "fold": tag,
                             "ordinal_mae": _ordinal_mae(yva.to_numpy(), yp),
                             "within_one": M.within_one_accuracy_from_cm(
                                 M.confusion_matrix_from_preds(yva.to_numpy(), yp, cfg["n_classes"]))}
                    # PO-002: archive per-fold optimizer convergence for iterative models
                    clf_f = pipe.named_steps["clf"]
                    if hasattr(clf_f, "optimizer_success_"):
                        trial["convergence_ok"] = bool(clf_f.optimizer_success_)
                    trials.append(trial)
                except Exception as exc:
                    trials.append({"model": name, "seed": int(seed), "fold": tag, "error": repr(exc)})
        log(f"  dev-CV done: {name}")
    # aggregate
    agg: Dict[str, dict] = {}
    for name in cv_models:
        maes = [t["ordinal_mae"] for t in trials if t["model"] == name and "ordinal_mae" in t]
        if maes:
            agg[name] = {"mean_ordinal_mae": float(np.mean(maes)),
                         "std_ordinal_mae": float(np.std(maes)),
                         "n_trials": len(maes)}
    return {"strategy": cfg["cv_strategy"], "n_folds": len(folds),
            "seeds": cfg["cv_seeds"], "per_trial": trials, "aggregate": agg}


def _is_stochastic(name: str) -> bool:
    return name in {"random_forest", "ordinal_random_forest",
                    "random_forest_unweighted", "ordinal_random_forest_unweighted",
                    "decision_tree", "shallow_tree", "xgboost", "ebm"}


def select_primary_baseline(cv: dict, cfg: dict) -> str:
    """Pick the comparator baseline as the prespecified baseline with the best mean dev-CV
    ordinal MAE (AA2-013) — chosen on development evidence, not hard-coded."""
    agg = cv["aggregate"]
    cand = [(b, agg[b]["mean_ordinal_mae"]) for b in cfg["baseline_set"] if b in agg]
    if not cand:
        return "ordinal_median"
    return min(cand, key=lambda t: t[1])[0]


def develop_split_view(split_dict: dict) -> dict:
    """GOV-001: return a copy of the split manifest that is safe to write into DEVELOPMENT
    artifacts. The held-out cohort's class prevalence is an *outcome distribution*; it is removed
    from this view. Structural counts (row totals, group overlap, assignment hash) and the
    development-side prevalence remain — they are needed for reconciliation and the freeze checks.
    NOTE (v3.x limitation): this redaction is defence-in-depth, not isolation — the development
    report separately carries the whole-extract target audit, from which the held-out outcome
    counts remain derivable by subtraction. The v4 correction removes whole-extract mapping from
    the develop phase entirely."""
    view = dict(split_dict)
    cp = view.get("class_prevalence")
    if isinstance(cp, dict):
        view["class_prevalence"] = {"development": cp.get("development")}
        view["final_test_class_prevalence"] = "withheld_from_development_artifacts (GOV-001)"
    view["notes"] = ("Final test is the latest contiguous year(s), held out from development and "
                     "not read during model fitting or selection; group_overlap_count MUST be 0.")
    return view


def select_best_candidate(cv: dict, reg: dict):
    """Pick the calibration candidate as the registered *candidate* model with the best mean
    dev-CV ordinal MAE (CAL-SELECT-001). Development evidence only — final-test results MUST NOT
    enter this choice. Returns a model name or None."""
    agg = cv.get("aggregate", {})
    cand = [(n, agg[n]["mean_ordinal_mae"]) for n in agg
            if reg.get(n) is not None and getattr(reg[n], "kind", None) == "candidate"
            and agg[n].get("mean_ordinal_mae") is not None]
    if not cand:
        return None
    return min(cand, key=lambda t: t[1])[0]


# ---------------------------------------------------------------------------
# immutable run bundle
# ---------------------------------------------------------------------------

def write_bundle(out_root: Path, run_id: str, files: Dict[str, str], manifest: dict) -> Path:
    """Atomically materialise an immutable run bundle. Refuses to overwrite an existing bundle;
    writes into a temp dir, hashes every artifact into the manifest, marks STATUS, then renames."""
    final_dir = out_root / run_id
    if final_dir.exists():
        raise FileExistsError(f"run bundle already exists (refusing to overwrite): {final_dir}")
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = out_root / f".{run_id}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        (tmp / "STATUS").write_text("RUNNING\n", encoding="utf-8")
        # BUNDLE-CRLF-001 (v4): write the EXACT bytes that are hashed. The v3 writer used
        # write_text (platform CRLF on Windows) while hashing the LF in-memory content, so
        # artifact_sha256 matched the materialised files only after LF-normalisation.
        artifact_hashes = {}
        for fname, content in files.items():
            payload = content.encode("utf-8")
            (tmp / fname).write_bytes(payload)
            artifact_hashes[fname] = _sha256_bytes(payload)
        manifest["artifact_sha256"] = artifact_hashes
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
        (tmp / "STATUS").write_text("COMPLETE\n", encoding="utf-8")
        os.replace(tmp, final_dir)
    except Exception:
        try:
            (tmp / "STATUS").write_text("FAILED\n", encoding="utf-8")
        except Exception:
            pass
        raise
    return final_dir


def _json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def _run_id(prefix: str, *hashes: str) -> str:
    short = _sha256_bytes("".join(hashes).encode())[:12]
    return f"{prefix}_{short}"


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_validate_data(args):
    schema = K.load_schema(); mapping = K.load_target_mapping()
    df = load_any(args.data)
    res = K.validate_dataframe(df, schema, mapping)
    print(json.dumps(res.to_dict(), indent=2)[:4000])
    print("\nGATE 0:", "PASS" if res.ok else "FAIL")
    sys.exit(0 if res.ok else 2)


def cmd_develop(args):
    cfg = load_config(args.config)
    out = Path(args.out)
    # GOV-001 (v4): structural isolation — this phase never validates, maps, audits, or
    # serializes final-year outcome values (see prepare_development). The development report's
    # target audit covers development-year rows ONLY, so no final-year outcome distribution is
    # present or derivable from any development artifact.
    prep = prepare_development(args.data, cfg)
    print(f"[develop] dev-year rows={prep.dev_split['n_dev_year_rows']}  "
          f"usable dev={len(prep.y_dev)}  dev-quarantined={prep.audit.n_quarantined}  "
          f"final-year rows (raw; outcomes never read in this phase)={prep.n_final_rows_raw}")
    cv = dev_cv(prep, cfg, log=print)
    primary = select_primary_baseline(cv, cfg)
    print(f"[develop] selected primary comparator baseline (dev evidence): {primary}")
    report = {
        "phase": "develop",
        "protocol_version": "v4-structural-isolation",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": cfg, "config_sha256": config_hash(cfg),
        "contract_sha256": prep.contract_hashes, "data_sha256": prep.data_hash,
        "target_audit_dev_only": prep.audit.to_dict(),
        "split": prep.dev_split,
        "sentinel_flagged": prep.sentinel_flagged,
        "feature_tier": cfg.get("feature_tier", "broad"),
        "decision_rule": cfg.get("decision_rule", "argmax"),
        "allowed_feature_columns": prep.allowed,
        "dropped_invariant_features": prep.dropped_invariant,
        "numeric_features": prep.numeric_cols, "categorical_features": prep.categorical_cols,
        "dropped_high_cardinality": prep.dropped_hc,
        "dev_cv": cv, "primary_baseline": primary,
        "git": _git_state(), "environment": {"os": platform.platform(), "libraries": _lib_versions()},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "development_report.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    prep.assignment_df.to_csv(out / "split_assignment.csv", index=False)
    print(f"[develop] wrote {out/'development_report.json'} and split_assignment.csv "
          f"(dev-side assignment only)")


def cmd_freeze(args):
    out = Path(args.out)
    rep_path = out / "development_report.json"
    if not rep_path.exists():
        sys.exit("freeze refused: no development_report.json (run `develop` first).")
    report = json.loads(rep_path.read_text(encoding="utf-8"))
    git = _git_state()
    if source_tree_dirty() and not args.allow_dirty:
        sys.exit("freeze refused: source/contract/data has uncommitted changes. "
                 "Commit them, or pass --allow-dirty (dev only).")
    # v4: the frozen split object is the DEVELOPMENT-side assignment hash. (The v3 lock pinned a
    # whole-cohort assignment whose membership depended on quarantining final-year severities —
    # i.e. it was outcome-dependent; the dev-side hash is outcome-free and is recomputed and
    # verified by evaluate-final.)
    split_hash = report["split"].get("dev_assignment_sha256") or report["split"]["assignment_sha256"]
    lock = {
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_version": report.get("protocol_version", "v3"),
        "config_sha256": report["config_sha256"],
        "contract_sha256": report["contract_sha256"],
        "data_sha256": report["data_sha256"],
        "split_sha256": split_hash,
        "development_report_sha256": _sha256_bytes(rep_path.read_bytes()),
        "primary_baseline": report["primary_baseline"],
        "git_commit": git["commit"], "git_branch": git["branch"],
    }
    (out / "FROZEN.lock").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"[freeze] wrote {out/'FROZEN.lock'} (primary baseline: {lock['primary_baseline']})")


def cmd_evaluate_final(args):
    cfg = load_config(args.config)
    out = Path(args.out)
    lock_path = out / "FROZEN.lock"
    if not lock_path.exists():
        sys.exit("evaluate-final refused: experiment is not frozen (run `freeze-experiment`).")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (out / "FINAL.done").exists():
        sys.exit("evaluate-final refused: the frozen experiment was already finalised (FINAL.done exists).")

    git = _git_state()
    if source_tree_dirty():
        sys.exit("evaluate-final refused: source/contract/data has uncommitted changes; "
                 "commit before the final evaluation.")

    prep = prepare(args.data, cfg, seal_final=True)
    # governance: inputs must match the freeze exactly. v4 locks pin the DEVELOPMENT-side
    # assignment hash (outcome-free); v3 locks pinned the whole-cohort assignment — support both.
    if lock.get("protocol_version", "v3").startswith("v4"):
        split_ok = prep.dev_assignment_sha256 == lock["split_sha256"]
    else:
        split_ok = prep.split_manifest.assignment_sha256 == lock["split_sha256"]
    checks = {
        "config": config_hash(cfg) == lock["config_sha256"],
        "contract": prep.contract_hashes == lock["contract_sha256"],
        "data": prep.data_hash == lock["data_sha256"],
        "split": split_ok,
        "git_commit": git["commit"] == lock["git_commit"],
    }
    if not all(checks.values()):
        sys.exit(f"evaluate-final refused: inputs changed since freeze: "
                 f"{[k for k,v in checks.items() if not v]}")

    n_classes = cfg["n_classes"]
    reg = Models.build_registry(include_optional=cfg["include_optional_models"], seed=cfg["seed"])
    primary = lock["primary_baseline"]
    # CAL-SELECT-001: choose the calibration candidate from DEVELOPMENT evidence only. Load the
    # development report that was frozen into the lock, verify it is unchanged, then pick the best
    # dev-CV candidate. Final-test outcomes never enter this choice.
    dev_report_path = out / "development_report.json"
    if not dev_report_path.exists():
        sys.exit("evaluate-final refused: development_report.json missing (run `develop`).")
    if _sha256_bytes(dev_report_path.read_bytes()) != lock["development_report_sha256"]:
        sys.exit("evaluate-final refused: development_report.json changed since freeze "
                 "(development evidence must be frozen before the evaluation).")
    _dev_report = json.loads(dev_report_path.read_text(encoding="utf-8"))
    best_candidate = select_best_candidate(_dev_report["dev_cv"], reg)
    print(f"[evaluate-final] out-of-time evaluation on held-out {cfg['final_test_years']} "
          f"(n_test={len(prep.y_test)}); comparator={primary}; "
          f"calibration candidate (dev-selected)={best_candidate}")

    tracemalloc.start()
    results: Dict[str, dict] = {}
    preds: Dict[str, np.ndarray] = {}
    proba_store: Dict[str, np.ndarray] = {}
    model_cards: Dict[str, dict] = {}
    pred_tables: Dict[str, str] = {}
    encoded_dims = None
    for name, spec in reg.items():
        try:
            t0 = time.perf_counter()
            pipe, meta = fit_model(spec, prep.X_dev, prep.y_dev, prep.numeric_cols, prep.categorical_cols, cfg)
            fit_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            yp_arg, proba = predict_aligned(pipe, prep.X_test, n_classes)
            predict_seconds = time.perf_counter() - t0
        except Exception as exc:
            results[name] = {"error": repr(exc), "kind": spec.kind}
            continue
        # OBJ-001 (v4): the configured PRIMARY decision rule scores the run; argmax is recorded
        # as a sensitivity when it is not the primary rule.
        yp = decide_labels(proba, yp_arg, cfg)
        if encoded_dims is None:
            try:
                encoded_dims = int(pipe.named_steps["pre"].transform(prep.X_dev.iloc[:50]).shape[1])
            except Exception:
                encoded_dims = None
        m = M.compute_all(prep.y_test.to_numpy(), yp, n_classes)
        m["kind"] = spec.kind; m["note"] = spec.note
        m["decision_rule"] = cfg.get("decision_rule", "argmax")
        m["sample_weight_applied"] = meta["sample_weight_applied"]
        if proba is not None and cfg.get("decision_rule", "argmax") != "argmax":
            m["ordinal_mae_argmax"] = _ordinal_mae(prep.y_test.to_numpy(), yp_arg)
            m["accuracy_argmax"] = M.accuracy_from_cm(
                M.confusion_matrix_from_preds(prep.y_test.to_numpy(), yp_arg, n_classes))
        # PO-001 (v4): surface optimizer convergence; a failed fit is flagged, never silent.
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "optimizer_success_"):
            m["convergence"] = {
                "ok": bool(clf.optimizer_success_),
                "n_iter": int(getattr(clf, "optimizer_n_iter_", -1)),
                "grad_norm": float(getattr(clf, "optimizer_grad_norm_", float("nan"))),
            }
            if not clf.optimizer_success_:
                print(f"[evaluate-final] WARNING: {name} optimizer did NOT converge — "
                      f"flagged in results; excluded from any superlative claim")
        if proba is not None:
            m["log_loss"] = M.multiclass_log_loss(prep.y_test.to_numpy(), proba, n_classes)
            m["brier"] = M.multiclass_brier(prep.y_test.to_numpy(), proba, n_classes)
            m["rps"] = M.ranked_probability_score(prep.y_test.to_numpy(), proba, n_classes)
            m["ece"] = Cal.expected_calibration_error(prep.y_test.to_numpy(), proba)
            proba_store[name] = proba
        results[name] = m
        preds[name] = yp
        model_cards[name] = {"class_weight_mode": spec.class_weight_mode,
                             "sample_weight_applied": meta["sample_weight_applied"],
                             "estimator": type(spec.make_estimator()).__name__, "note": spec.note,
                             # RESOURCE-001 (v4): per-model wall time on the run hardware
                             "fit_seconds": round(fit_seconds, 3),
                             "predict_seconds": round(predict_seconds, 3)}
        # per-crash predictions (LOCAL bundle only; carries Crash Number -> kept out of git via runs/.gitignore)
        pt = pd.DataFrame({"group_id": prep.groups_test, "y_true": prep.y_test.to_numpy(), "y_pred": yp})
        if proba is not None:
            pt["y_pred_argmax"] = yp_arg
            for c in range(n_classes):
                pt[f"proba_{c}"] = proba[:, c]
        pred_tables[f"predictions_{name}.csv"] = pt.to_csv(index=False)

    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

    # paired crash-level bootstrap CIs vs the comparator baseline (resampling unit = crash;
    # cross-crash independence approximation, one row per crash — NOT a dependence-aware cluster CI)
    ci_block: Dict[str, dict] = {}
    if primary in preds:
        for name, yp in preds.items():
            results[name]["ordinal_mae_ci"] = U.bootstrap_metric_ci(
                prep.y_test.to_numpy(), yp, _ordinal_mae, groups=prep.groups_test,
                n_resamples=cfg["bootstrap_resamples"], seed=cfg["seed"])
            if name != primary:
                ci_block[name] = U.paired_difference_ci(
                    prep.y_test.to_numpy(), yp, preds[primary], _ordinal_mae,
                    groups=prep.groups_test, n_resamples=cfg["bootstrap_resamples"], seed=cfg["seed"])

    # dev-only calibration for the headline models: comparator baseline + best DEVELOPMENT candidate.
    # CAL-SELECT-001: `best_candidate` was chosen above from the frozen dev-CV report; final-test
    # `results` MUST NOT influence which model is calibrated.
    calibration_block: Dict[str, dict] = {}
    if cfg.get("calibrate"):
        for name in {primary, best_candidate} - {None}:
            spec = reg.get(name)
            if spec is None:
                continue
            try:
                cal_proba, method = dev_only_calibrate(
                    spec, prep.X_dev, prep.y_dev, prep.X_test, prep.groups_dev, cfg, n_classes,
                    years_dev=prep.years_dev)
                if cal_proba is None:
                    calibration_block[name] = {"calibrated": False, "reason": method}
                    continue
                calibration_block[name] = {
                    "calibrated": True, "method": method,
                    "ece_raw": results[name].get("ece"),
                    "ece_calibrated": Cal.expected_calibration_error(prep.y_test.to_numpy(), cal_proba),
                    "log_loss_calibrated": M.multiclass_log_loss(prep.y_test.to_numpy(), cal_proba, n_classes),
                    "reliability_severe": Cal.reliability_points(prep.y_test.to_numpy(), cal_proba, n_classes - 1),
                }
            except Exception as exc:
                calibration_block[name] = {"calibrated": False, "reason": repr(exc)}

    run_id = _run_id("final", prep.data_hash, config_hash(cfg), lock["frozen_utc"])
    manifest = {
        "run_id": run_id, "phase": "evaluate-final",
        "protocol_version": lock.get("protocol_version", "v3"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_source": str(args.data), "data_sha256": prep.data_hash,
        "config": cfg, "config_sha256": config_hash(cfg),
        "frozen_lock": lock, "contract_sha256": prep.contract_hashes,
        "git": git, "environment": {"os": platform.platform(), "libraries": _lib_versions()},
        "target_audit": prep.audit.to_dict(), "split": prep.split_manifest.to_dict(),
        "dev_assignment_sha256": prep.dev_assignment_sha256,
        "sentinel_flagged": prep.sentinel_flagged,
        "feature_tier": cfg.get("feature_tier", "broad"),
        "decision_rule": cfg.get("decision_rule", "argmax"),
        "allowed_feature_columns": prep.allowed,
        "dropped_invariant_features": prep.dropped_invariant,
        "numeric_features": prep.numeric_cols, "categorical_features": prep.categorical_cols,
        "encoded_feature_dims": encoded_dims,
        "peak_memory_mb": round(peak / 1e6, 1),
        "primary_metric": cfg["primary_metric"], "primary_baseline": primary,
        "model_cards": model_cards, "results": results,
        "paired_difference_vs_baseline": ci_block,
        "calibration": calibration_block,
        "model_availability": Models.availability(),
    }
    files = dict(pred_tables)
    files["split_assignment.csv"] = prep.assignment_df.to_csv(index=False)
    files["target_audit.json"] = json.dumps(prep.audit.to_dict(), indent=2)
    final_dir = write_bundle(Path(args.out_runs), run_id, files, manifest)

    # committed aggregate (NO per-crash rows / ids) + FINAL marker
    out.mkdir(parents=True, exist_ok=True)
    agg = {k: manifest[k] for k in ("run_id", "protocol_version", "generated_utc", "data_sha256",
           "config", "config_sha256", "contract_sha256", "git", "environment", "target_audit",
           "split", "dev_assignment_sha256", "sentinel_flagged", "feature_tier", "decision_rule",
           "allowed_feature_columns", "dropped_invariant_features", "numeric_features",
           "categorical_features", "encoded_feature_dims", "peak_memory_mb", "primary_metric",
           "primary_baseline", "model_cards", "results", "paired_difference_vs_baseline",
           "calibration", "model_availability")}
    (out / "final_results.json").write_text(json.dumps(agg, indent=2, default=_json_default), encoding="utf-8")
    (out / "FINAL.done").write_text(f"{run_id}\n{final_dir}\n", encoding="utf-8")
    print(f"[evaluate-final] bundle: {final_dir}")
    print(f"[evaluate-final] committed aggregate: {out/'final_results.json'}")
    print(f"[evaluate-final] comparator={primary}  "
          f"encoded_dims={encoded_dims}  peak_mem={manifest['peak_memory_mb']}MB")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m crashsev.cli",
                                 description="Governed leakage-controlled crash-severity study.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate-data"); p.add_argument("--data", required=True); p.set_defaults(fn=cmd_validate_data)
    p = sub.add_parser("develop")
    p.add_argument("--data", required=True); p.add_argument("--config", default=None)
    p.add_argument("--out", default=str(PKG_ROOT / "experiment")); p.set_defaults(fn=cmd_develop)
    p = sub.add_parser("freeze-experiment")
    p.add_argument("--out", default=str(PKG_ROOT / "experiment"))
    p.add_argument("--allow-dirty", action="store_true"); p.set_defaults(fn=cmd_freeze)
    p = sub.add_parser("evaluate-final")
    p.add_argument("--data", required=True); p.add_argument("--config", default=None)
    p.add_argument("--out", default=str(PKG_ROOT / "experiment"))
    p.add_argument("--out-runs", default=str(PKG_ROOT / "runs")); p.set_defaults(fn=cmd_evaluate_final)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
