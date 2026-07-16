# Route Decision

**Decision: ROUTE R — a real Alaska crash-severity empirical study evaluated as a *retrospective,
out-of-time* comparison under a leakage-controlled, chronological, crash-grouped protocol, together
with historical evaluation forensics and executable governance.**
Recorded 2026-07-10 for the portfolio finalization (branch `portfolio-research-finalization-v3`).
Supersedes the v2 "Route A" framing, which correctly established that a *real empirical study* is
possible on lawful data, but which over-described the 2012 evaluation as a *sealed* one-shot
holdout. The empirical study is unchanged; its evidentiary status is corrected here.

## Route R vs Route C — why the 2012 evaluation is retrospective, not confirmatory

A *prospective confirmation* (Route C) would require a later or external cohort whose outcomes and
aggregate target distribution had **not** been seen by anyone selecting models or analyses, a
protocol frozen and **externally timestamped before outcome access**, and an independently
witnessable one-time evaluation. **None of these holds here:**

* The raw 2009–2012 extract — including 2012 — was **locally available throughout** the work; 2012
  outcomes were never in third-party custody.
* There is **no external timestamp**: the branch is unpublished, there is no signed public tag, and
  no hosted CI run predates the evaluation.
* The **legacy project already modelled this data/era**, so the outcome distribution was, in effect,
  previously exposed.

Therefore the 2012 result is reported as **retrospective and out-of-time**. What the project *does*
provide is an executable **within-study** leakage control with a precisely stated scope: development
model **fitting and selection** use only development years (the 2012 feature/label matrices are
materialised only inside `evaluate-final`, behind a lock on config/contract/data/split/git hashes;
`crashsev/cli.py`). It is procedural, not a technical air-gap: in the v3.x code path `develop`
loads and target-maps all years in memory before splitting, and its report's whole-extract target
audit makes the 2012 outcome counts derivable by subtraction (GOV-001, third-round finding) —
structural isolation is the v4 correction. The control prevents *this study's* development from
tuning on 2012 — a real and tested property of fitting/selection — but it does not make 2012 a
sealed prospective holdout. Route C is **not** pursued, and no synthetic or relabeled cohort is
substituted for it.

## Why Route R is the right portfolio choice

A rigorous retrospective evaluation-remediation study is a strong, defensible contribution and does
not require a superior model. Its value is the combination of: (1) historical evaluation forensics
(re-analysing the original four confusion matrices under ordinal/class metrics); (2) executable
research-governance controls (contracts, leakage ledger, split lock, dev-only calibration,
provenance); (3) a transparent retrospective chronological comparison with uncertainty and
decision-rule/weighting sensitivity; and (4) reproducible research engineering. Chasing a
prospective Route C on data that does not exist would add nothing but false confidence.

## Prerequisites for a real empirical study — met (evidence, not preference)

| Prerequisite | Status | Evidence |
|---|---|---|
| Complete verifiable repository | ✅ | `C:\aca`, `git fsck` clean, base commit `bb9247a`, branch `portfolio-research-finalization-v3` |
| Lawful modelling data | ✅ | `Crash Level 09-12 (1).xlsx` — 50,543 crash rows, 100 columns, 2009–2012, full `DateTime`; used under an owner grant (`DATA_LICENSE_NOTE.md`, `DATA_AUTHORITY_AND_ACCESS.md`) |
| Evidence-calibrated target mapping | ✅ (internal check, not an official codebook) | blank = property-damage-only supported by 0.0% injury/fatality contamination of 32,046 blank rows (`data/codebook_verification_09_12.md`); this is an internal check on the extract, **not** an official source-agency codebook validation |
| Confirmed analytical unit | ✅ | crash; `Crash Number` unique per row (50,543/50,543) |
| Row / group id | ✅ | `Crash Number` (one row per crash, so row id = group id; there is no within-crash clustering) |
| Use / analysis rights | ✅ | owner grant for analysis (not public redistribution); `DATA_LICENSE_NOTE.md` |
| Temporal + class support for an out-of-time evaluation | ✅ | 4 balanced years; severe (class 2) per year 442/443/416/450; the 2012 held-out year carries 450 severe crashes |

The mapping reproduces the original paper's reported class balance to within 0.85 percentage points
(this extract 0.684 / 0.279 / 0.037 vs. paper 0.677 / 0.287 / 0.035; 46,844 usable crashes after
fail-closed quarantine of 3,699 Unknown / Not Reported / Null-value rows) — corroboration that the
study operates on the same phenomenon under a documented, reversible target definition.

## Research question (estimand-anchored, retrospective)

> For police-reported Alaska motor-vehicle crashes (2009–2012), how accurately can a crash's
> **3-level ordinal injury severity** {none/PDO · minor/possible · serious/fatal} be classified from
> information available **at or before the crash scene** — *excluding every outcome-derived field* —
> and how does that accuracy transfer forward to a later, **held-out (out-of-time, retrospective)**
> year (2012) under proper ordinal, calibration, and uncertainty accounting, relative to trivial,
> statistical, and tree-based baselines?

Secondary (mechanistic, on the same real data): how much do specific evaluation defects —
outcome-derived features, pre-split preprocessing, and random vs. temporal splitting — inflate
apparent performance relative to the leakage-controlled protocol?

## Scope discipline

* **Predictive, not causal.** No coefficient or importance is read as an intervention effect
  (`research/ESTIMAND_AND_SCOPE.md`).
* **This extract, this window.** Alaska 2009–2012 police-reported crashes only; not generalised to
  other states, later years, or unreported crashes.
* **Retrospective evaluation.** The 2012 evaluation is run once under a frozen configuration and
  within-study governance, but 2012 was exposed; it is not a sealed prospective holdout.
* **Prior-group cleaned CSVs are not used for modelling** (they lack a Year axis;
  `research/DATA_PROVENANCE.md`).

## What Route R delivers

Executable data/target/feature contracts; a reconstructible, group-safe split with a
reconciliation invariant; a governed CLI (development CV, seeds, model selection, dev-only
calibration) under an enforced within-study final-evaluation lock; corrected model/metric/probability
semantics; atomic immutable run bundles; a **real, retrospective out-of-time result** with ordinal
metrics, calibration, decision-rule/weighting sensitivity, and paired crash-level bootstrap CIs; a
leakage factorial on the real data; and a minimal cross-platform environment with failure-mode tests.
