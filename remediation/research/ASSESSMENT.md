# Honest assessment: what is demonstrated, what is not (AA2-025)

Written to replace inflated "ready now / faculty-score" language with a sober separation of
**framework readiness** (the engineering) from **empirical readiness** (the scientific claim),
each tied to executed evidence and stated remaining limits. Scores are avoided; the reader can
judge against the checkable artifacts.

## What is demonstrated (with executed evidence)

* **A leakage-controlled, governed, reproducible framework.** Executable data/target/feature
  contracts reject impostor data and malformed codebooks; a chronological, crash-grouped,
  reconstructible split; a fail-closed KABCO target evidence-checked against the extract (an internal
  check, not an official source-agency codebook validation); strict ordinal metrics; a governed CLI
  with a final-test lock and immutable run bundles. Evidenced by 40 passing tests (CI workflow run
  locally on Windows/Python 3.13; not on hosted CI) and the committed run artifacts.
* **A real, single-shot empirical result** on Alaska 2009-2012 crashes: development-selected
  models evaluated exactly once on a held-out, out-of-time 2012 test under a frozen configuration
  (within-study governance; not a prospective seal — retrospective, as the 2012 outcomes were
  locally available during the work), with ordinal metrics, calibration, and paired crash-level
  bootstrap confidence intervals (`experiment/final_results.json`).
* **A separated leakage factorial** quantifying, one factor at a time, how outcome-derived
  features, pre-split preprocessing, and random splits inflate apparent performance
  (`experiment/leakage_factorial.md`).
* **An honest re-analysis** of the original four confusion matrices under their own contaminated
  protocol, showing why that evaluation is insufficient (`reanalysis/`).
* **Self-reproduced, bit-exactly (same project; not an independent third party).** From a fresh
  `git archive` extract: 54/54 tests pass; every paper figure and re-analysis table regenerates
  byte-identically from committed aggregates; Gate-0 accepts the real extract and rejects an
  impostor; and a full `develop → freeze → evaluate-final` re-run reproduces the held-out 2012-test
  metrics **and** seeded bootstrap CIs to 0.000e+00 (`research/REPRODUCTION_LOG.md`).

## What is NOT demonstrated (stated plainly)

* **No causal or interventional claim.** The models are descriptive classifiers.
* **No generalization beyond Alaska 2009-2012 reported crashes.** A different state, later years,
  or unreported crashes are out of frame. The single 4-year window is a real external-validity
  limit (the only raw extract with an intact temporal axis in the licensed archive).
* **No deployment / clinical validity.** This is a retrospective methodological study.
* **The legacy Django app is not fixed or production-safe.** Its security patches are specified
  but not executed here (they need a live PostGIS/Django harness); see
  `research/APPLICATION_SCOPE.md`.
* **The exact producing scripts of the prior cleaned CSVs are unrecovered;** statements about
  their Year-drop are inference (`research/DATA_PROVENANCE.md`).

## Remaining work a reviewer would reasonably ask for

* Extend to 2013-2017 once a lawful *raw* extract with a temporal axis is obtained (only a
  derived pickle exists in the archive).
* Sensitivity of the headline result to the small-but-real serialization of the modelling table
  across environments (the governed content hash already makes this a non-issue; see the log).
* Sensitivity of the target to the two high-coupling occupant-kinematic features (an ablation
  toggle exists: `exclude_high_coupling`).

The intent is that every positive claim above points to a file a reviewer can run or read, and
every limitation is disclosed rather than discovered.
