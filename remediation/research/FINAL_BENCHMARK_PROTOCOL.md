# Final Benchmark Protocol — Analysis Plan (fixed on development evidence; not an externally witnessed preregistration)

> ## Protocol amendment v4 — 2026-07-11 (third-round remediation) — READ FIRST
>
> Committed **before** the v4 freeze and final evaluation. The third-round reconnaissance
> confirmed protocol defects in the executed v3 study; the v4 corrections below are therefore
> **prespecified here, on development evidence and recon findings only** — no v4 final-test
> outcome existed when this amendment was written. The executed v3 result (weighted ordinal RF
> 0.3310 vs majority 0.3614 under argmax, broad tier) is retained as **historical evidence**; it
> is no longer the governed headline benchmark after these changes.
>
> | v3 (as executed) | v4 (prespecified) | Why (finding) |
> |---|---|---|
> | `develop` loaded and target-mapped ALL years; dev report carried a whole-extract audit from which 2012 outcome counts were derivable by subtraction | **Structural isolation**: `prepare_development` partitions by YEAR before any target interpretation; final-year outcomes are never validated/mapped/audited/serialized in develop; poison-token sentinel test enforces it | GOV-001 |
> | Freeze pinned a whole-cohort split hash whose membership depended on quarantining final-year severities (outcome-dependent) | Freeze pins the outcome-free **development-side assignment hash**; `evaluate-final` recomputes and verifies it | GOV-001 |
> | Hard decisions = argmax everywhere; posterior median only post-hoc | **`decision_rule: posterior_median` is the PRIMARY hard-decision rule** (Bayes rule for the absolute ordinal loss), applied in dev CV and the final run; argmax recorded as a sensitivity | OBJ-001 / DECISION-LOSS-001 |
> | Broad at-event feature tier, on ledger assertions | **`feature_tier: strict`** (scene-observable tier; ledger column `strict_scene_tier`, 50 fields) is the PRIMARY benchmark; the broad tier runs as a separately frozen SENSITIVITY (`configs/route_r_09_12_broad_sensitivity.yml`) | FEAT-001 |
> | `empirical_prior` = stochastic hard-label draws, misdescribed as a probability baseline | **`prior_probability`** = deterministic training-prevalence probability forecast | BASE-001 |
> | Calibration folds = GroupKFold over unique crash ids (≈ random) | **`calibration_folds: temporal`** — rolling-origin YEAR folds, development-only | CAL-001 |
> | Feature typing/cardinality decided on all development years before CV folds | Typing decided **inside each fold** on its training rows | PREP-001 |
> | 100%-missing 'Rural Urban' remained an allowed feature | Invariant/all-missing allowed features **auto-dropped** on development rows and recorded | MISS-001 |
> | PO optimizer status stored, never surfaced | Convergence recorded in results, flagged, excluded from superlatives if failed | PO-001 |
> | Bundle writer hashed LF content but materialised CRLF | Writer writes the **exact bytes it hashes** | BUNDLE-CRLF-001 |
> | No per-model timings | fit/predict wall-times recorded in model cards | RESOURCE-001 |
>
> Also prespecified: the bounded TARGET-001 sensitivity (blank = PDO vs blank = excluded) runs as
> a **development-only** diagnostic (`crashsev/target_sensitivity.py`); the machine-readable v4
> config is `configs/route_r_09_12.yml`. Success criteria are otherwise unchanged (§2, §8): the
> primary comparison remains the development-selected candidate vs the development-selected
> trivial baseline on ordinal MAE with a paired crash-level bootstrap CI, now under the
> posterior-median rule and the strict tier.

> ## Execution amendment — 2026-07 (Route A) — historical
>
> This protocol was written as an **analysis plan** during the earlier "Route B" phase, before data
> access, and its body below is preserved unedited as the analysis plan of record. It fixes the success criteria and analysis plan on development evidence; it is **not** an externally witnessed preregistration (the branch is unpublished and 2012 data were locally available). When lawful access to the
> real Alaska archive was granted (`research/DATA_LICENSE_NOTE.md`, `research/ROUTE_DECISION.md`), the
> study was executed under this protocol with the following **disclosed deviations** — recorded here
> rather than by silently editing the analysis plan:
>
> | Analysis-plan default (below) | As executed (Route A) | Why |
> |---|---|---|
> | Cohort 2009–2017; final test **2016–2017** | Cohort **2009–2012**; final test **2012** (held-out, out-of-time) | Only the `Crash Level 09-12` **raw** extract has an intact temporal axis; `13-17` exists solely as a derived pickle (`research/DATA_PROVENANCE.md`). |
> | Runner `crashsev/run_benchmark.py`; config `configs/final_benchmark.yml` | Governed CLI `crashsev.cli` (validate-data / develop / freeze-experiment / evaluate-final); config `configs/route_a_09_12.yml` | The runner was refactored into a phase-governed, final-test-locked CLI (AA2-007/008). `run_benchmark.py` remains as a deprecation shim. |
> | Ten development seeds | Three CV selection seeds (`cv_seeds: [1,2,3]`), one frozen final config | Selection is over development folds only; the held-out 2012 test is evaluated once under the frozen configuration (within-study governance; not a prospective seal). |
> | Baseline `ordinal_logistic` (Frank–Hall); ablation `crashsev/leakage_demo.py` | Renamed `frank_hall_logistic`; a true `proportional_odds` baseline added; leakage ablation is the one-factor-at-a-time factorial `crashsev/leakage_factorial.py` (AA2-013/020) | Naming/model corrections and a cleaner factorial design. |
>
> The scientific commitments that matter for validity are **unchanged**: predictive (non-causal) RQ;
> a single declared primary metric (ordinal MAE); a chronological, crash-grouped, held-out 2012 final test
> evaluated exactly once under a frozen configuration (within-study governance; not a prospective
> seal — 2012 outcomes were locally available during the work); a prespecified baseline set with the comparator chosen on development
> evidence; paired crash-level bootstrap CIs; and the permitted/​not-permitted claim list. Executed
> result: `experiment/final_results.json`; self-reproduction — same project, not an independent third party (bit-exact re-run confirmed):
> `research/REPRODUCE.md` and `research/REPRODUCTION_LOG.md`.

**Commit this file and the machine-readable config BEFORE any final-test evaluation.**
Committing before final-test access fixes the analysis plan on development evidence; it is **not**
an externally witnessed preregistration, because the branch is unpublished and 2012 data were
locally available (EXP-001).
This protocol is executable: the machine-readable companion is `configs/route_a_09_12.yml`
(originally `configs/final_benchmark.yml`) and the runner is the governed CLI `crashsev.cli`
(originally `crashsev/run_benchmark.py`).

> **Execution status:** **EXECUTED** on the real Alaska 2009–2012 extract under the amendment above.
> Gate-0 fields in `data/DATA_MANIFEST.md` are complete and `data/target_mapping.yml` is
> evidence-checked against the extract (`data/codebook_verification_09_12.md`) — an internal check,
> not an official source-agency codebook validation.

## 1. Research question (RQ-001)

> On a fixed, provenance-documented Alaska police-reported crash dataset, how well do
> leakage-controlled models predict three-level ordinal crash severity for **later time
> periods** relative to trivial, statistical, and tree-based baselines, and what
> calibration, subgroup, and error-profile limitations remain?

## 2. Hypotheses

* **H1** — At least one learned model improves the primary metric (ordinal MAE) over the
  strongest simple baseline on the untouched chronological final test, by a margin whose
  paired 95% CI excludes zero.
* **H2** — Any improvement is not confined to one seed, year, or subgroup, and does not
  depend on outcome-derived features (survives the leakage/temporal ablations).
* **H3** — A selected model's probabilities meet a prespecified calibration standard, or can
  be calibrated on development data without degrading the primary metric.
* **H4 (descriptive)** — Stable out-of-sample feature associations identify predictively
  useful variables; they do **not** establish causes.

## 3. Unit, cohort, target

* Analytical unit: **crash** (group key `Crash Number`); confirm hierarchy before running (⛔).
* Target: `Crash Severity` → 3-level ordinal via `data/target_mapping.yml` (fail-closed).
* Cohort: police-reported Alaska crashes, 2009–2017; rows with unmapped severity are
  quarantined (reported, not modelled).

## 4. Split (VAL-002)

* **Final test:** the latest contiguous period (default `2016–2017`), held out before any fit.
* **Development:** earlier years; model selection uses rolling-origin temporal folds
  (`blocked_temporal_dev_folds`) or crash-grouped folds (`grouped_dev_folds`).
* Grouping: no crash spans partitions; `group_overlap_count` MUST be 0 (asserted at runtime).
* The split is hashed (`split_manifest.assignment_sha256`) and immutable.

## 5. Features (METH-001)

* Permitted: pre-event + at-event + temporal tiers of the feature-availability ledger
  (`use_case: post_crash_triage`).
* Excluded (denylist, deterministic, target-free): all POST_OUTCOME and IDENTIFIER columns
  (injury/fatality counts, EMS/damage/enforcement, ids). Unknown-timing columns fail closed.

## 6. Baselines and candidates (EVAL-001)

Baselines: `majority`, `ordinal_median`, `empirical_prior`, `multinomial_logistic`,
`ordinal_logistic` (Frank-Hall), `shallow_tree`.
Candidates: `decision_tree`, `random_forest`, `ordinal_random_forest` (corrected MLRF
replacement), `xgboost` (if installed), `ebm` (if installed).
Every model uses the **same** split ids, cohort, target, permitted features, and pipeline.

## 7. Metrics (EVAL-002)

* **Primary:** ordinal mean absolute error (adjacent error = 1, two-step = 2).
* **Secondary:** QWK, macro-F1, balanced accuracy, per-class P/R/F1, within-one accuracy,
  two-step error rate, natural-prevalence accuracy (always shown beside the majority
  baseline), severe-class precision/recall, confusion matrix.
* **Probabilistic:** log loss, Brier, ranked probability score (RPS), ECE + reliability.

## 8. Uncertainty & selection (STAT-001/002)

* 95% **paired crash-level (case/row) bootstrap** CIs — resampled at the crash level under a
  cross-crash independence approximation (each crash is one row, so there is no within-crash
  grouping; this is an approximation, not a dependence-aware group bootstrap) — for each metric and
  each model−baseline difference (`crashsev/uncertainty.py`).
* Model selection uses **development data only**: mean development ordinal MAE; deterministic
  tie-break toward lower complexity, then better calibration. "No clear winner" is an
  allowed, reportable outcome.
* Multiplicity: when reporting many pairwise/subgroup comparisons, apply Holm correction or
  label them exploratory.
* Practical significance: report absolute effect (errors / cost per 1,000 crashes), not
  p-values alone.

## 9. Seeds

Ten fixed development seeds for stochastic models to characterise variability; one locked
final configuration is frozen before final-test evaluation.

## 10. Ablations (run only these; A1–A2 are headline diagnostics)

* **A1 leakage pipeline:** contaminated (full-data preprocessing + outcome features + random
  split) vs corrected — reported ONLY as a diagnostic of optimism, never as a benchmark
  (implemented: `crashsev/leakage_demo.py`).
* **A2 temporal validation:** random vs chronological split (optimism gap).
* **A3 leakage features:** denylist-only vs denylist + in-fold statistical screen.
* **A4 temporal/source proxies:** with/without `Year`.
* **A5 class handling:** class weights on/off; threshold/calibration on natural prevalence.
* **A6 cleaning thresholds:** semantic-only vs configured variance/cardinality drops.
* **A7 MLRF design:** corrected ordinal RF vs ordinary RF vs ordered logistic.

## 11. Robustness / error analysis (EXP-003)

Final-test performance by year, region/urban-rural, and missingness burden (with N,
prevalence, CI; small groups suppressed); severe false negatives / false positives;
two-step errors; high-confidence errors; model disagreement. No fairness/national claims
from sparse slices.

## 12. Final-test rule (EVAL-003)

The final test is evaluated **once**, after the configuration is locked. Any post-test
change invalidates it as a final test; a new untouched period or external holdout is
required. The prior published screenshots are treated as exploratory/invalid for inference.

## 13. Claims permitted / not permitted

* **Permitted:** comparative predictive performance on the tested Alaska data and horizon;
  calibration/robustness bounds; predictive (not causal) associations.
* **Not permitted:** causation; national/MMUCC-wide generality; deployment readiness;
  "best model" without the declared objective and CI; operational "best threshold" without
  stakeholder costs (use ordinal MAE + full PR tradeoff instead).

## 14. Resource measurement (SW-006, PERF-001)

Per model: fit+predict wall time, peak RAM where measurable, model size, raw/encoded feature
counts, hardware. Reported as a supported research-execution envelope, not a scalability claim.
