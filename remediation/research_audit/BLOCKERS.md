# Blocking-condition report — updated (Route A executed)

The first-round remediation proceeded around several external blockers. This file records each one
and **its current status** after lawful access to the Alaska 2009–2012 extract was granted
(`research/DATA_LICENSE_NOTE.md`, `research/ROUTE_DECISION.md`). The two invalidating blockers are
resolved; the rest are documented and scoped rather than open.

## BLOCKER-1 — Raw dataset absent (DATA-001) · **RESOLVED**

* **Was:** the actual Alaska crash extracts, provenance, and checksums were absent from the repo.
* **Resolved by:** an explicit owner grant to inspect and analyse the local archive. The study now
  runs on `Crash Level 09-12 (1).xlsx` (50,543 rows), de-identified locally to a git-ignored
  modelling table. Provenance and the canonical content hash
  (`data_sha256 = 059559cd…`, from `pd.util.hash_pandas_object`, serialization-independent) are
  recorded in every run manifest and in `research/DATA_PROVENANCE.md`.
* **Result produced:** development on 2009–2011, a single out-of-time 2012 evaluation (evaluated once under a frozen configuration — within-study governance, not a prospective seal)
  (`experiment/final_results.json`; `paper/final_paper.pdf`).

## BLOCKER-2 — Source severity codebook absent (DATA-002) · **RESOLVED (empirically)**

* **Was:** the meaning/ordering of `Crash Severity` codes was unconfirmed, so KABCO→{0,1,2} was an
  assumption and blank-handling was a guess.
* **Resolved by:** an internal codebook evidence check against the extract (an internal check, not an official source-agency codebook validation)
  (`data/codebook_verification_09_12.md`, `codebook_evidence_09_12.json`): blank severity is
  **property-damage-only (O)**, evidence-checked by **0.0%** injury/fatality contamination among blank-severity
  rows. The mapping remains fail-closed — `Unknown` / `Not Reported` / `Null value` are quarantined,
  never coerced (3,699 quarantined on this extract).

## BLOCKER-3 — Data license / publication rights (LIC-001, ETH-001) · **governed, not open**

* **Status:** the owner grant authorizes inspection, analysis, and shipping **code + aggregates**.
  It does **not**, by itself, authorize public redistribution of raw records or coordinates; the
  source agency's terms would govern any release beyond the personal portfolio. Enforced by
  construction: no raw records/coordinates/ids are committed; only aggregates. See
  `research/DATA_LICENSE_NOTE.md` and the `LICENSE` data carve-out.

## BLOCKER-4 — Provenance of the prior study's screenshots (REP-004) · **resolved by inspection**

* **Resolved:** the four confusion-matrix screenshots were produced by `peyton_original/…` scripts
  (OHE `min_frequency=0.01`, 80/20 split, RandomizedSearchCV) — **not** the integrated pipeline —
  evidenced by the `Crash Type_infrequent_sklearn` feature label. Their exact run command is still
  unlogged, so they are used only as **exploratory** results and re-analysed, never treated as a
  valid benchmark (`reanalysis/`, with the report `.docx` hash matching the recorded provenance).

## BLOCKER-5 — Django/PostGIS security fixes not executed here · **out of scope (documented)**

* **Status:** the legacy Django/React/PostGIS app is an out-of-scope demo layer
  (`research/APPLICATION_SCOPE.md`). Its security defects are identified with file:line and the
  corrected code + acceptance tests are specified, but executing them needs a provisioned PostGIS
  test DB and is **not** claimed done here.

## Non-blockers (explicitly)

External validation on non-Alaska jurisdictions and a real-scale performance benchmark are **out of
scope**, not blockers — the study is deliberately limited to the tested Alaska 2009–2012 data, and
those claims are removed rather than forced (`research/ESTIMAND_AND_SCOPE.md`).
