"""
crashsev.splits — immutable, leakage-aware train/development/final-test partitioning
(VAL-002, EVAL-003, EVAL-004).

The original project uses a single random stratified split, with a different test fraction
for XGBoost (0.20) than for the other models (0.25), each trainer calling
``train_test_split`` independently. That design (a) leaks temporal regime because crash
records are not IID across years, and (b) evaluates different models on different rows.

This module fixes both:

* One split is created ONCE, before any learned transformation, and shared by every model.
* The default final test is a *chronological* later period (VAL-002); development uses
  earlier years.
* Grouping by crash id keeps all rows of the same crash in the same partition (no
  group leakage across the boundary).
* The split is serialised to an immutable manifest carrying counts, per-split class
  prevalence, and a deterministic SHA-256 hash of the id->partition assignment, so any
  reviewer can verify the exact partition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


@dataclass
class SplitManifest:
    strategy: str
    group_col: Optional[str]
    row_id_col: Optional[str]
    year_col: Optional[str]
    final_test_years: List[int]
    development_years: List[int]
    n_total: int
    n_development: int
    n_final_test: int
    n_excluded: int
    excluded_reason_counts: Dict[str, int]
    class_prevalence: Dict[str, Dict[str, float]]
    group_overlap_count: int
    assignment_sha256: str
    reconciles: bool
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "group_col": self.group_col,
            "row_id_col": self.row_id_col,
            "year_col": self.year_col,
            "final_test_years": self.final_test_years,
            "development_years": self.development_years,
            "n_total": self.n_total,
            "n_development": self.n_development,
            "n_final_test": self.n_final_test,
            "n_excluded": self.n_excluded,
            "excluded_reason_counts": self.excluded_reason_counts,
            "class_prevalence": self.class_prevalence,
            "group_overlap_count": self.group_overlap_count,
            "assignment_sha256": self.assignment_sha256,
            # Reconciliation invariant: development + final_test + excluded == total, exactly.
            "reconciles": self.reconciles,
            "notes": self.notes,
        }


def _assignment_hash(ids: Sequence, partitions: Sequence[str]) -> str:
    """Deterministic hash of the (sorted) id -> partition mapping."""
    pairs = sorted((str(i), str(p)) for i, p in zip(ids, partitions))
    blob = json.dumps(pairs, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def assignment_hash(assignment_df: pd.DataFrame) -> str:
    """Public: recompute the split hash from a saved ``row_id``/``partition`` table.

    A reviewer runs this on the persisted ``split_assignment.csv`` and checks it equals the
    manifest's ``assignment_sha256`` — i.e. the saved assignment reproduces the hash.
    """
    return _assignment_hash(assignment_df["row_id"].tolist(), assignment_df["partition"].tolist())


def _prevalence(y: pd.Series) -> Dict[str, float]:
    vc = y.value_counts(normalize=True).sort_index()
    return {str(int(k)): float(v) for k, v in vc.items()}


def chronological_group_split(
    df: pd.DataFrame,
    y: pd.Series,
    *,
    year_col: str,
    group_col: Optional[str],
    final_test_years: Sequence[int],
    row_id_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, SplitManifest, pd.DataFrame]:
    """
    Split into development (earlier years) and final test (``final_test_years``).

    Returns ``(dev_idx, test_idx, manifest, assignment_df)`` where ``*_idx`` are positional
    indices into ``df`` and ``assignment_df`` is the reconstructible row_id -> partition table.

    Correctness guarantees (AA2-006):

    * **Reconciliation.** Every row is assigned exactly one of {development, final_test,
      excluded}; ``n_development + n_final_test + n_excluded == n_total`` exactly, and the
      assignment hash is computed over *this* three-state assignment (a row that is neither
      development nor final_test is ``excluded``, never silently counted as development).
    * **Missing years are explicit.** Rows whose year is null / not a development year and not
      a final-test year are ``excluded`` with a recorded reason, not dropped invisibly.
    * **Stable unique row id.** ``row_id_col`` must be present, non-null, and unique; duplicate
      or null ids abort (a hash over a non-injective id set is not reconstructible).
    * **Group safety.** Because the boundary is temporal and crashes do not span years, a crash
      id cannot appear in both partitions; zero overlap is *verified* and recorded.
    """
    if year_col not in df.columns:
        raise KeyError(f"year_col {year_col!r} not in dataframe")
    if not row_id_col or row_id_col not in df.columns:
        raise ValueError(f"row_id_col {row_id_col!r} must be a column in df (needed for a reconstructible split)")

    ids = df[row_id_col]
    if ids.isna().any():
        raise ValueError(f"row_id_col {row_id_col!r} has {int(ids.isna().sum())} null id(s); a stable unique id is required")
    if ids.duplicated().any():
        n_dup = int(ids.duplicated().sum())
        raise ValueError(f"row_id_col {row_id_col!r} has {n_dup} duplicate id(s); split requires unique row ids")
    ids = ids.astype(str).to_numpy()

    years = pd.to_numeric(df[year_col], errors="coerce")
    final_test_years = [int(y_) for y_ in final_test_years]

    test_mask = years.isin(final_test_years).to_numpy()
    dev_mask = (~test_mask) & years.notna().to_numpy()
    excluded_mask = ~(test_mask | dev_mask)  # null year, or year not in dev∪test

    dev_idx = np.where(dev_mask)[0]
    test_idx = np.where(test_mask)[0]
    excl_idx = np.where(excluded_mask)[0]

    if len(dev_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Chronological split produced an empty partition "
            f"(dev={len(dev_idx)}, test={len(test_idx)}); check final_test_years."
        )

    # SPLIT-001: enforce the temporal-split *contract*, not just "everything not-final is dev".
    # Development years must all fall strictly BEFORE the final years and, together with them, form a
    # contiguous gap-free run — otherwise a future year or a gap year could silently enter
    # development. (For 2009-2011 dev / 2012 final this passes; a 2013 or a 2009,2011 gap fails.)
    dev_years = sorted({int(y) for y in years.iloc[dev_idx].dropna().unique().tolist()})
    test_years_sorted = sorted(int(y) for y in final_test_years)
    if dev_years and test_years_sorted:
        if max(dev_years) >= min(test_years_sorted):
            raise ValueError(
                f"SPLIT-001: development years {dev_years} are not all before the final years "
                f"{test_years_sorted}; the temporal split requires earlier-only development.")
        span = dev_years + test_years_sorted
        if span != list(range(span[0], span[-1] + 1)):
            raise ValueError(
                f"SPLIT-001: development+final years {span} are not contiguous (a gap would let an "
                f"unmodelled period leak in or out); refusing the split.")

    # Three-state partition array (default excluded -> the AA2-006 fix).
    partitions = np.full(len(df), "excluded", dtype=object)
    partitions[dev_idx] = "development"
    partitions[test_idx] = "final_test"

    # Reconciliation invariant.
    n_total = int(len(df))
    reconciles = (len(dev_idx) + len(test_idx) + len(excl_idx)) == n_total
    if not reconciles:
        raise AssertionError("split does not reconcile: dev+test+excluded != total")

    # Reasons for exclusion (transparency).
    excl_reasons: Dict[str, int] = {}
    if len(excl_idx):
        null_year = int(years.iloc[excl_idx].isna().sum())
        if null_year:
            excl_reasons["null_year"] = null_year
        other = len(excl_idx) - null_year
        if other:
            excl_reasons["year_outside_dev_and_test"] = int(other)

    # Verify group disjointness across the boundary.
    overlap = 0
    if group_col and group_col in df.columns:
        dev_groups = set(df.iloc[dev_idx][group_col].astype(str))
        test_groups = set(df.iloc[test_idx][group_col].astype(str))
        overlap = len(dev_groups & test_groups)

    dev_years = sorted(int(y_) for y_ in years.dropna().unique() if int(y_) not in final_test_years)

    assignment_df = pd.DataFrame({
        "row_id": ids,
        "partition": partitions,
        "year": years.to_numpy(),
    })

    manifest = SplitManifest(
        strategy="chronological_final_test_with_crash_grouping",
        group_col=group_col,
        row_id_col=row_id_col,
        year_col=year_col,
        final_test_years=final_test_years,
        development_years=dev_years,
        n_total=n_total,
        n_development=int(len(dev_idx)),
        n_final_test=int(len(test_idx)),
        n_excluded=int(len(excl_idx)),
        excluded_reason_counts=excl_reasons,
        class_prevalence={
            "development": _prevalence(y.iloc[dev_idx]),
            "final_test": _prevalence(y.iloc[test_idx]),
        },
        group_overlap_count=int(overlap),
        assignment_sha256=_assignment_hash(ids, partitions.tolist()),
        reconciles=bool(reconciles),
        notes=(
            "Final test is the latest contiguous year(s), held out from development (an exposed "
            "retrospective cohort, not a prospective seal). group_overlap_count MUST be 0; "
            "excluded rows are explicit."
        ),
    )
    return dev_idx, test_idx, manifest, assignment_df


def grouped_dev_folds(
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    *,
    group_col: Optional[str],
    n_splits: int = 5,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Development-only cross-validation folds that keep crash groups together (GroupKFold).

    Used for hyperparameter selection and repeated-seed variability on development data.
    The final test is NEVER touched here.
    """
    n = len(df_dev)
    if group_col and group_col in df_dev.columns:
        groups = df_dev[group_col].astype(str).to_numpy()
    else:
        groups = np.arange(n)  # each row its own group -> ordinary KFold behaviour
    n_groups = len(np.unique(groups))
    n_splits = int(min(n_splits, max(2, n_groups)))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, va in gkf.split(df_dev, y_dev, groups=groups):
        yield tr, va


def blocked_temporal_dev_folds(
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    *,
    year_col: str,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Rolling-origin temporal folds on development data: for each year Y (after the first),
    train on years < Y, validate on year Y. This mirrors the temporal generalisation the
    final test measures, and avoids validating on the future.
    """
    years = pd.to_numeric(df_dev[year_col], errors="coerce")
    uniq = sorted(int(y_) for y_ in years.dropna().unique())
    for i in range(1, len(uniq)):
        train_years = uniq[:i]
        val_year = uniq[i]
        tr = np.where(years.isin(train_years).to_numpy())[0]
        va = np.where((years == val_year).to_numpy())[0]
        if len(tr) and len(va):
            yield tr, va
