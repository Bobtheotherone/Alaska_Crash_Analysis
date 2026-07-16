"""
crashsev — leakage-controlled reference pipeline for ordinal crash-severity prediction.

This package is the *corrected* research pipeline produced by the graduate-research
remediation of the Alaska Crash Analysis project. It is deliberately self-contained
(depends only on numpy / pandas / scikit-learn / scipy / matplotlib) so that it can be
audited and executed without the Django/PostGIS application stack.

Design principles (each maps to a reconnaissance issue ID; see research_audit/REMEDIATION_LEDGER.md):

* VAL-001 / METH-001 : every learned transformation is fit on training folds only,
  through an sklearn Pipeline/ColumnTransformer. The split is created *before* any fit.
* VAL-002 / EVAL-003 : the default design is a chronological final-test period with a
  crash-level grouping constraint and an immutable, hashed split manifest.
* DATA-002 / DATA-005 : the target mapping is an explicit, documented, fail-closed
  contract. Unknown / blank / unmapped severity values are NEVER coerced to a class;
  they are quarantined and audited.
* METH-001 (leakage) : outcome-derived predictors are excluded through a pre-specified
  deterministic denylist (a feature-availability ledger), not a supervised full-data
  screen.
* EVAL-002 : the primary metric is ordinal mean absolute error; a full ordinal /
  calibration secondary suite is reported.
* STAT-002 : uncertainty is reported via paired crash-level (case/row) bootstrap confidence
  intervals under a cross-crash independence approximation (not a dependence-aware cluster CI).

NOTHING in this package fabricates empirical results. The only data it can generate on
its own is a clearly-labelled SYNTHETIC STRUCTURAL FIXTURE (crashsev.synth), used solely
to exercise and test the harness. Empirical results require the real, lawful Alaska
extract supplied via --data (DATA-001 resolved 2026-07-09; byte identity recorded in
research/DATA_AUTHORITY_AND_ACCESS.md).
"""

__version__ = "0.1.0"

__all__ = [
    "schema",
    "target",
    "leakage",
    "splits",
    "preprocessing",
    "metrics",
    "calibration",
    "uncertainty",
    "models",
    "synth",
]
