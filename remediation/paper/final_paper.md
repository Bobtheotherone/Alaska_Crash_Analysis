> **SUPERSEDED (2026-07-15).** This markdown draft is retained for lineage only. The authoritative manuscript is the LaTeX build `paper/latex/main.tex` -> `paper/latex/main.pdf` (*Leakage-Controlled Ordinal Classification of a Researcher-Defined Alaska Crash-Severity Outcome: A Governed, Self-Reproduced Retrospective Out-of-Time Study Across Four Project Iterations*). Where the two disagree, the LaTeX manuscript and its verified artifacts govern.

# Leakage-Controlled, Governed Ordinal Classification of Alaska Crash Severity: A Reproducible, Retrospective Out-of-Time Study

**Author.** Naythan Mercado.
**Version / date.** v4 (corrected-protocol benchmark), 2026-07-11.
**Release.** Branch `portfolio-research-finalization-v4`; annotated tag `portfolio-v4` (the v3
release at tag `portfolio-v3` / commit `e06e803` is preserved un-overwritten; corrections are new
commits). The v4 governed result (run `final_8af9d5bc23d8`) was generated at commit `458f784`
under a clean-source freeze with **structural outcome isolation** (GOV-001 v4); the historical v3
result (run `final_f27613102c96`, §7.7) was generated at `e55d6f1`. The full generator →
run-bundle → artifact-commit → release chain with hashes is `research/PROVENANCE_CHAIN.md`; the
handoff archive SHA-256 and full history are recorded in the handoff `provenance/` (git bundle +
`REPO_GIT_STATE.txt`). Tags are convenience pointers only — cite commit hashes for permanent
references.
**Study route.** Route R — retrospective out-of-time evaluation and evaluation-remediation
(`research/ROUTE_DECISION.md`). The 2012 evaluation year is treated as **exposed**, not sealed.
**Data access.** Modelling source used under a data-owner grant for analysis; **not** a public
data-release license. No raw records or identifiers are published (`research/DATA_AUTHORITY_AND_ACCESS.md`).
**AI assistance.** Prepared with material AI assistance (code generation/editing, statistical
review, documentation, packaging) under the author's direction; the author is responsible for
understanding and defending every claim (`research/AUTHORSHIP_AND_AI_ASSISTANCE.md`).

**Study type.** A retrospective, *predictive* (not causal) study of three-level ordinal crash
severity for police-reported Alaska crashes (2009–2012), evaluated out-of-time on a later year
(2012) under a leakage-controlled, crash-grouped, chronological protocol with executable
within-study governance — in v4 with **structural outcome isolation** of the development phase, a
**prespecified loss-consistent decision rule** (posterior median), and a **conservative, author-judged
restricted feature tier** (intended to approximate scene-time availability) as the primary benchmark — together with a controlled re-analysis
of the original project's reported results, a controlled leakage factorial, and a quantified
comparison of how each wave of protocol correction changed the apparent result (§7.7). No causal,
national, or deployment claim is made, and the 2012 evaluation is **not** presented as a sealed,
preregistered, or independently confirmed holdout.

---

## Abstract

Police-reported crash severity is an ordered, severely imbalanced outcome, and crash records are
not independent across time — properties that make a single random train/test split and an
accuracy headline misleading. An earlier version of this project trained four tabular classifiers
on Alaska crash data, reported accuracies and feature-importance "causes," and claimed national
generality and deployment readiness. Re-analysing that project's own four confusion matrices shows
the trivial majority-class predictor scores **0.676 accuracy**, that **three of the four models
score below it**, and that no model dominates across ordinal MAE, quadratic weighted kappa, and
severe-class recall — the strongest-accuracy model reaches its number by predicting the majority
class **86.7%** of the time.

Using the raw 2009–2012 crash-level extract (50,543 crashes; raw byte identity SHA-256
recorded in the restricted reproduction log, withheld from public release) with an **evidence-calibrated** target mapping — blank severity is treated
as property-damage-only, a source-specific coding decision supported because 0.0% of the 32,046
blank-severity crashes carry any injury or fatality count — we build a governed, reproducible
pipeline (`crashsev`): executable data/target/feature contracts, a fail-closed KABCO mapping, a
crash-grouped chronological split that holds out **2012** as an out-of-time evaluation year, a
develop phase with **structural outcome isolation** (final-year outcomes are never validated,
mapped, audited, or serialized during development — enforced by a poison-token sentinel test; the
shared table is loaded and content-hashed whole only as an opaque integrity step, so the
guarantee is semantic/analytic, not byte-level non-access), a
**prespecified loss-consistent decision rule** (posterior median, the Bayes rule for the absolute
ordinal loss), a **conservative, author-judged restricted feature tier** (intended to approximate
scene-time availability; provisional pending form-level evidence) as the primary predictor
set, a deterministic prior-probability baseline, development-only calibration on temporal folds,
paired crash-level bootstrap confidence intervals, and a governance lock with content-addressed,
byte-exact run bundles.

On the corrected (v4) out-of-time 2012 evaluation (n = 11,630), the **prespecified primary
comparison** — the ordinal random forest versus the strongest trivial baseline (majority) on the
primary metric, ordinal MAE — shows a **small but interval-supported improvement: 0.347 vs 0.361
(paired bootstrap Δ = −0.014, 95% CI [−0.023, −0.006], excluding 0; ≈14 fewer class-steps per
1,000 crashes)**. The matched unweighted ablation controls do modestly better still (0.340/0.342),
the class-weighted random forest is now *worse* than the baseline (+0.015 [+0.005, +0.026]), and
the **four random-forest variants** beat the deterministic prior-probability forecast on every
proper score (the remaining learned models do **not** consistently do so).
The v4 protocol corrections themselves are a quantified finding (§7.7): relative to the v3
protocol (broad feature tier, argmax rule, weaker isolation), the honest headline margin shrinks
from −0.030 to −0.014, and severe-crash hard-decision recall collapses from 0.28 to **0.06** —
most of that severe-class assignment in v3 came from at-event fields whose decision-time
availability rests on unevidenced ledger assertions (test-given, restraint/ejection/seat,
contributing circumstances, damage location). Two facts must be held together here: under the
loss-consistent hard rule and the restricted tier, no model both beats the trivial baseline on
the ordinal loss and meaningfully *assigns* the severe class — yet the frozen probabilities
retain **nontrivial severe ranking signal** (one-vs-rest average precision ≈ 0.20 against a
0.039 no-skill prevalence, AUROC ≈ 0.79; §8), so low hard-class recall must not be read as an
absence of predictive information about severe risk, nor as validated severe-crash detection. A controlled 2×2×2 diagnostic quantifies three evaluation defects via
matched contrasts (outcome-derived features, pre-split preprocessing, random splits);
outcome-derived features dominate (−0.349 ordinal-MAE inflation vs −0.021 and −0.033), and the
defects interact rather than add. Because 2012 outcomes were locally available throughout, all
2012 results are reported as **retrospective**, and reproduction to date is **self-reproduction**,
not independent. The contribution is a defensible, reproducible, retrospective out-of-time result,
a reusable evaluation-governance framework, and a measured demonstration of how much each
evaluation defect inflated the apparent result — an engineering-and-methodology contribution — not
a causal safety finding or a general product.

---

## 1. Introduction

Crash severity is recorded on the ordered KABCO scale (K fatal; A serious/incapacitating; B
minor/non-incapacitating; C possible; O none/property-damage-only). Predicting severity from crash
and roadway attributes is attractive for analysis and triage, but three properties make it hard:
the outcome is **ordinal** (confusing "none" with "fatal" is worse than confusing "none" with
"minor"), it is **severely imbalanced** (serious/fatal crashes are a few percent of records), and
crash records are **not IID** across years, so a random split overstates generalisation to future
periods.

An earlier version of this project built a full-stack system around four classifiers and reported
that it identifies what "causes" severe crashes, works on any U.S. dataset, and is deployment
ready. This study does not add models or features; it corrects the **target definition**, the
**evaluation**, and the **governance**, and narrows the **claim** to what the evidence supports.
It differs from a prior "framework-only" draft in one decisive respect: the raw 2009–2012 extract
and an evidence-checkable target mapping are available under an explicit data-owner grant, so a
**real, retrospective out-of-time empirical result** is produced, not deferred.

**Research question.** *For police-reported Alaska crashes (2009–2012), how accurately can
the researcher-defined three-level ordinal severity outcome be classified from a conservative,
author-judged restricted predictor tier — excluding every outcome-derived field and every field
whose decision-time availability the authors judged plausibly post-scene — and how does that accuracy transfer forward to a later year (2012) under
a loss-consistent decision rule and proper ordinal, calibration, and uncertainty accounting,
relative to trivial, statistical, and tree-based baselines?* Because the 2012 outcomes were
accessible during the work, this is answered **retrospectively**: the v4 governance makes the
*development phase* structurally unable to read final-year outcomes (§3.4), but it cannot make
2012 a prospective, externally witnessed holdout.

**Contributions.**
1. An **evidence-calibrated target mapping** for the extract (blank = property-damage-only,
   supported by a 0.0% injury/fatality contamination check), and executable data/target/feature
   **contracts** that reject invalid data and codebooks.
2. A **governed, leakage-controlled pipeline** with a crash-grouped chronological split, a
   within-study final-evaluation lock, **structural outcome isolation of the development phase**
   (v4: final-year outcomes are never validated, mapped, audited, or serialized during
   development; a poison-token sentinel test enforces it; the shared table is loaded and
   content-hashed whole only as an opaque integrity step), a **prespecified posterior-median
   decision rule**, a **conservative author-judged restricted feature tier**, a deterministic
   prior-probability baseline,
   calibration fit on development data only over temporal folds, paired crash-level bootstrap
   uncertainty, and atomic, content-addressed, **byte-exact** run bundles.
3. A **real, retrospective out-of-time empirical result** on the 2012 evaluation under the
   corrected v4 protocol, reported with a prespecified primary comparison, baselines, confidence
   intervals, an argmax decision-rule sensitivity, a **matched weighted/unweighted ablation**,
   calibration, an error breakdown by class and confidence — and a quantified account of how each
   wave of protocol correction changed the apparent result (§7.7), including the collapse of
   severe-class recall once timing-unevidenced fields are excluded.
4. A **leakage/preprocessing diagnostic** on the real data: matched contrasts (same rows, model, and
   seed) that isolate outcome-derived-feature leakage and preprocessing-before-split, plus a
   **validation-optimism** contrast for random vs temporal splitting (a comparison of two different
   estimands, *not* an isolated causal "split effect" — FAC-001), and an **honest re-analysis** of the
   original four confusion matrices.

**Out of scope (deliberately).** Causal inference, national/MMUCC-wide generality, deep learning,
synthetic-data balancing as a result, and production deployment. These would not fix the validity
defects and are rejected as scope inflation.

## 2. Related work and background

**Prediction versus explanation.** Shmueli (2010) distinguishes predictive from explanatory
modelling; a model optimised for out-of-sample prediction does not identify causal effects. Hence
feature associations here are *predictive*, never causal.

**Crash-severity modelling.** Savolainen, Mannering, Lord & Quddus (2011) review injury-severity
methods and emphasise the ordered outcome, unobserved heterogeneity, and reporting/selection
issues — motivating ordinal treatment and temporal caution over generic multiclass classification.

**Ordinal classification.** Frank & Hall (2001) decompose a K-level ordinal target into K−1
cumulative binary problems P(y > k); class probabilities are differences of cumulative
probabilities. We use this for an ordinal random forest with coherent probabilities, and we add a
**true proportional-odds (ordered logit)** model — a single latent linear predictor with ordered
thresholds fit by maximum likelihood — as a distinct baseline. The primary metric is ordinal mean
absolute error; the Bayes-optimal decision for absolute ordinal loss is the posterior median, a
point we test explicitly (§7.6). Quadratic weighted kappa (Cohen 1968) is a chance-corrected
ordinal agreement measure, and the ranked probability score (Epstein 1969) is a proper ordinal
probabilistic score.

**The compared learners.** Random forests (Breiman 2001) average decorrelated trees; XGBoost (Chen
& Guestrin 2016) fits a regularised additive tree sequence; the Explainable Boosting Machine (Nori
et al. 2019) is a glass-box additive model. Native impurity importances are biased and
model-specific, so cross-model scalar importance comparisons and causal readings are unwarranted.

**Leakage, imbalance, calibration, reproducibility.** Standard guidance (scikit-learn "common
pitfalls"; Kaufman et al. 2012) requires every learned transformation and target-aware selection
to be fit on training data only. For imbalanced ordinal outcomes, accuracy is dominated by the
majority class and must be reported beside a trivial baseline; probability quality requires
calibration (Guo et al. 2017) and proper scores. Reproducible research requires versioned data,
environment, code, and run artifacts.

**Positioning against current practice.** Ordinal treatment of crash injury severity with
machine-learning classifiers is an active line of work — including direct comparisons of ordinal
versus nominal formulations on crash data (e.g. the ordinal-classification study of crash injury
severity indexed at PMC8583475) — and this study's matched ordinal-vs-nominal contrast (H2) is
evaluated against exactly that question rather than assumed. The chronological design follows the
structured-validation argument of Roberts et al. (2017): when the intended use is extrapolation
across time, random resampling understates error and validation blocks must mirror the target
structure — which is also why one exposed retrospective year cannot establish temporal
generalisation (§10). Probability evaluation follows the theory of strictly proper scoring rules
(Gneiting & Raftery 2007) — hence log loss, Brier, and RPS beside a deterministic prior forecast —
and the calibration reporting heeds Van Calster et al. (2019): a single aggregate ECE is a weak
summary (here the constant prior *wins* top-label ECE, §7.4), so ECE is descriptive only. The
predictor-timing question (§3.3) is grounded in the Alaska crash-report form itself (form 12-200,
NHTSA-hosted): the form documents which fields belong to the police-reporting process, but not
when each becomes reliable — which is precisely why the tier assignments remain author judgments
pending workflow evidence.

## 3. Data, provenance, and target

### 3.1 Source and access
The modelling source is the raw crash-level extract `Crash Level 09-12 (1).xlsx` (50,543 crashes,
100 columns, 2009–2012, with `Year`, `DateTime`, and a unique `Crash Number` per row; byte identity
SHA-256 recorded 2026-07-11 (retained in the restricted reproduction log; withheld from public
release), and rebuilding the modelling table from that byte-hashed
file reproduces the frozen study input and its governed content hash `059559cd…` exactly —
`research/DATA_AUTHORITY_AND_ACCESS.md` §2), used under
an explicit data-owner grant for analysis (`research/DATA_LICENSE_NOTE.md`,
`research/DATA_AUTHORITY_AND_ACCESS.md`). This grant authorises local analysis; it is **not** a
public data-release license and it is **not** an official source-agency provenance verification.
Precise coordinates, free-text location, and officer/report identifiers are treated as restricted
and are **never committed**; a de-identified local modelling table is derived, and only aggregates
are published (`research/DATA_PROVENANCE.md`). The prior group's cleaned CSVs, which match the
paper's class balance, are **not** used for modelling because their producing scripts are
unrecovered and they lack any temporal column; the raw extract is the fully-provenanced source.

### 3.2 Analytical unit and target (evidence-calibrated mapping)
The unit is the **crash** (`Crash Number`, unique per row). The target is a **researcher-defined
three-level ordinal outcome** constructed from the extract's `Crash Severity` field (old-style
ABC labels; the construction is documented, reversible, and evidence-checked below, but it is
not an official source-system severity construct — §7.8, §10): `blank → O → 0` (none/PDO),
`{Possible=C, Non-Incapacitating=B} → 1` (minor/possible), `{Incapacitating=A, Fatal=K} → 2`
(serious/fatal). Values `Unknown`, `Not Reported`, and `Null value` are **quarantined** — counted,
never coerced to a class. The single consequential inference, **blank = property-damage-only**, is a
source-specific coding decision that is *evidence-checked, not merely assumed*: **0.0% of the 32,046
blank-severity crashes carry any positive fatality or injury count** (`data/codebook_verification_09_12.md`).
This is strong internal evidence for the extract but is not an official codebook validation; §10
records it as a construct-validity limitation. A bounded blank→excluded sensitivity is *specified*
in the protocol but has **not** been separately rerun in this revision — it is scheduled in the
corrected v4 benchmark (`research/RECON3_ISSUE_LEDGER.md`, TARGET-001). Under
this mapping the class balance is **0.684 / 0.279 / 0.037** across 46,844 usable crashes (3,699
quarantined), reproducing the original paper's reported balance (0.677 / 0.287 / 0.035) to within
0.85 percentage points — the same phenomenon under two label vintages (`research/DATA_PROVENANCE.md`).

### 3.3 Predictor availability and the leakage policy
Every one of the extract's 100 columns is classified by *when its value is determined relative to
the outcome* (`data/feature_availability_ledger.csv`): **66 admissible features** (17 pre-event
roadway/geographic context, 44 at-event circumstances, 5 calendar features), **14 outcome-derived
columns** (injury/fatality counts, person injury, EMS transport/extrication, damage extent, tow,
enforcement) that are **prohibited** for prediction, 16 identifiers, and 2 constants. `Year` is
withheld from the features (it is the split axis). Three occupant-kinematic features (ejection,
restraint, seat position) are flagged high-coupling and are ablatable. Real data-quality sentinels
are neutralised from the schema (AADT's int32-minimum on 34% of rows; Temperature's −460/999;
an implausible distance-from-intersection), deterministically and without leakage.

**Conservative restricted tier (v4 primary; FEAT-001).** The at-event assertions above are ledger
rationales, not form-level evidence, so the v4 primary benchmark restricts predictors to a
conservative, **author-judged restricted tier** (`strict_scene_tier` ledger column; 50 fields =
17 pre-event + 28 at-event fields judged plausibly scene-recordable + 5 calendar; the judgments
are the authors', pending form-level evidence — an official form can show a field belongs to the
reporting process without establishing when it becomes reliable): it excludes the high-coupling kinematics trio
(ejection/restraint/seat), test-given, insurance coverage, all contributing-circumstance and
sequence-of-events fields, damage-location fields, haz-mat release, and alcohol/drug-suspected —
every allowed field whose value could plausibly be finalized at investigation or
report-completion stage rather than at the scene. The broad tier is retained as a separately
frozen sensitivity (§7.8). One strict-tier field (`Rural Urban`) is 100% missing-token in the
extract and is auto-dropped as uninformative (recorded in the run artifacts), leaving 49 primary
input fields (376 encoded dimensions). Tier assignments remain provisional pending form-level
recording-time evidence (`research/PROVENANCE_ACQUISITION_PLAN.md`).

### 3.4 Within-study governance
Executable Gate-0 contracts reject an impostor file, an undocumented target value, and a
malformed/unevidenced codebook before any fitting. In the v4 protocol the develop phase runs
under **structural outcome isolation** (GOV-001 v4): rows are partitioned **by year before any
target interpretation**, final-year rows are reduced to two outcome-free facts (their raw row
count, and their group-id set for the zero-overlap check) and dropped, and Gate-0 validation, the
target mapping, the mapping audit, and every serialized development artifact cover
development-year rows **only** — so no development artifact carries, or allows the reconstruction
of, any final-year outcome. The precise boundary of this guarantee: the shared source table *is*
loaded once and content-hashed **whole** before partitioning (an opaque integrity operation —
final-year target bytes pass through memory and the hash without being parsed as outcomes), so
the isolation is **semantic and analytic** — never validated, mapped, audited, serialized, fitted
on, or selected on — not a byte-level or process-boundary non-access claim. A sentinel test enforces this structurally: development succeeds on a
file whose final-year severities are poisoned with an undocumented token, while the full
evaluate-final path must reject the same file at Gate 0. The freeze pins the outcome-free
**development-side assignment hash** (the v3 whole-cohort split hash was itself outcome-dependent
through final-year quarantining), and `evaluate-final` — the only phase permitted to read
final-year outcomes — recomputes and verifies it, alongside the config/contract/data/git hashes,
refusing a dirty source tree or a second evaluation (`research/REPRODUCE.md`). History, disclosed:
the v3.x revisions did **not** have this property — `develop` mapped all years in memory and its
report's whole-extract target audit made the 2012 outcome counts exactly recoverable by
subtraction (third-round reconnaissance finding; the v3 result in §7.7 carries that caveat). The
year is in any case not externally sealed: the raw 2012 data were locally available throughout,
and no external party witnessed the freeze before outcome access. The evaluation is therefore
out-of-time and governed, but **retrospective and exposed**, not a sealed prospective holdout
(`research/ROUTE_DECISION.md`).

## 4. Problem formulation

A crash is described by features `x` from the conservative restricted tier (pre-event +
judged-scene-recordable at-event + calendar; §3.3). The target `y ∈ {0,1,2}` is the ordered collapsed
severity. We seek a probability vector `p(x)` and a hard class `f(x)`. The **primary metric** is
ordinal MAE, `E|f(x) − y|` in class steps, and the v4 protocol **prespecifies the hard-decision
rule that this loss implies**: the posterior median, the smallest class `k` whose cumulative
probability reaches 0.5 — the Bayes-optimal decision under absolute ordinal loss (argmax, which
minimises 0–1 loss instead, is recorded as a sensitivity, §7.6). With a validated stakeholder cost
matrix, expected cost would replace ordinal MAE — no such cost matrix is available, so no
practical decision threshold is invented. Features determined after the outcome are forbidden;
evaluation is on the natural prevalence of the later year. Success criteria were fixed on
development evidence before the final evaluation (`research/ESTIMAND_AND_SCOPE.md` +
`research/FINAL_BENCHMARK_PROTOCOL.md` v4 amendment, committed before the v4 freeze): a model
**shows a conditional difference from the trivial baseline** only if its ordinal-MAE improvement
has a paired bootstrap CI excluding 0 — a statistical-distinguishability criterion, *not* a
practical-utility claim (no smallest effect of practical interest or stakeholder cost function
exists, so no usefulness threshold is invented); minority-class recall is reported prominently;
calibration is disclosed.

## 5. Methods

The pipeline (`crashsev`) is self-contained (numpy/pandas/scikit-learn/scipy/PyYAML/openpyxl) and
is the sole execution layer for every claim (`research/APPLICATION_SCOPE.md`).

**5.1 Contracts (Gate 0).** `crashsev/contracts.py` loads and validates `data/schema.json`,
`data/target_mapping.yml`, and the feature ledger; it rejects files missing required columns,
undocumented target values, and codebooks that are malformed, incomplete, ambiguous, or that
override the fail-closed blank policy without cited evidence. Contract file hashes are recorded in
every run.

**5.2 Fail-closed target.** `crashsev/target.py` maps severity through the codebook; unmapped/blank
(absent an evidenced override) values become `<NA>`, are quarantined and counted, and are never
coerced. A mapping audit reconciles raw and encoded counts exactly.

**5.3 Split before learning.** `crashsev/splits.py` holds out the evaluation year(s) before any fit,
requires a unique row id, assigns every row to exactly one of {development, final_test, excluded}
(missing-year rows are explicit, never silently counted as development), verifies zero crash-group
overlap, saves the id→partition table, and records a hash that the saved table reproduces.
Development selection uses rolling-origin folds (train past years → validate the next).

**5.4 Train-only preprocessing.** `crashsev/preprocessing.py` places median/most-frequent
imputation and sparse one-hot encoding (infrequent bucket, unknown-category handling) inside a
`ColumnTransformer`/`Pipeline` fit on training rows only.

**5.5 Models and roles.** `crashsev/models.py` provides a fixed registry with declared roles:
*trivial baselines* (majority, ordinal-median, and the **deterministic prior-probability
forecast** — the training-prevalence probability vector, the honest trivial probabilistic floor
that v3's stochastic "empirical prior" draws were not), *statistical baselines* (multinomial
logistic, a **true proportional-odds ordered logit** with an analytically gradient-checked
likelihood and **recorded optimizer convergence** (a non-converged fit is flagged and excluded
from superlatives), a Frank–Hall cumulative-logit decomposition, a shallow tree), *candidates*
(decision tree, random forest, a **Frank–Hall ordinal random forest with coherent
`predict_proba`**, XGBoost), the two matched **unweighted ablation controls** (WGT-001; excluded
from candidacy by a guard test), and one *exploratory* model, the EBM, which was **not** run in
the development phase and is therefore reported as final-only exploratory, not a primary peer
(`experiment/development_report.json`; §7.3). Hard predictions use the **prespecified
posterior-median rule** (§4); argmax is recorded per model as a sensitivity. Feature typing and
cardinality decisions are made **inside each CV fold** on its training rows. Seeds propagate from
the config into every stochastic estimator; class imbalance is handled by built-in class weights
or, for models without them, by **balanced `sample_weight` actually applied at fit** and recorded.

**5.6 Metrics, calibration, uncertainty.** `crashsev/metrics.py` enforces strict input contracts
(equal length, label domain, probability shape and row-sums) and computes ordinal MAE (primary),
QWK, macro-F1, balanced accuracy, per-class P/R/F1, within-one/two-step rates, and proper scores
(log loss, Brier, RPS); undefined statistics return an explicit `NaN`/status, never an arbitrary
value. `crashsev/calibration.py` computes ECE and reliability; a calibrator is fit on **development
data only** using **temporal rolling-origin year folds** (calibrate on a later development year
than the fold's training years — the v3 grouped folds over unique crash ids were effectively a
random row split, CAL-001) and evaluated on the later year. `crashsev/uncertainty.py`
computes paired bootstrap CIs **resampled at the crash level**. Because the analytical unit is the
crash and each crash contributes exactly one row, this is a case/row bootstrap under an
**independence approximation across crashes**; it does not model residual dependence between crashes
on the same corridor, day, or reporting batch, which would require cluster identifiers this extract
does not expose. Every interval is furthermore **conditional on the one fitted model, the frozen
protocol, the observed cohort, and the target construction** — it quantifies case-sampling
uncertainty only, not training-seed/refit variability (measured separately in the seed-robustness
addendum, §7.3), model-selection uncertainty, or construct uncertainty. The approximation and
conditionality are stated wherever intervals are reported and are limitations (§10), not a
"dependence-aware" guarantee.

**5.7 Governance and provenance.** `crashsev/cli.py` runs the phases `validate-data → develop →
freeze-experiment → evaluate-final`, writes atomic, content-addressed, overwrite-refusing run bundles
(temp → per-artifact hash → rename; overwrite refused; STATUS marker; not externally immutable), and records git state, library/OS versions, encoded
dimensions, and peak memory. The method text above matches the code; the legacy web application is
explicitly out of scope.

## 6. Experimental design

Configuration is machine-readable (`configs/route_r_09_12.yml` — the v4 canonical config; the v3
config `configs/route_a_09_12.yml` is retained for the historical chain). Development is
2009–2011; the out-of-time evaluation year is **2012**. The protocol corrections were prespecified
and committed **before** the v4 freeze (`research/FINAL_BENCHMARK_PROTOCOL.md`, v4 amendment):
strict feature tier as primary, posterior-median decision rule, deterministic prior-probability
baseline, temporal calibration folds, and structural development isolation. **The prespecified
primary comparison** is the ordinal random forest — the top *candidate* by development rolling-CV
ordinal MAE (the matched unweighted ablation controls score lower but are excluded from candidacy
by design; §7.2) — versus the majority baseline (the strongest trivial baseline on development),
on the primary metric ordinal MAE under the posterior-median rule.

**Prespecified hypotheses** (fixed with the v4 amendment; adjudicated in §7.8):

* **H1** — Under the restricted tier and posterior-median rule, the development-selected ordinal
  RF has lower 2012 ordinal MAE than the majority baseline (paired case-bootstrap CI excluding 0).
* **H2** — The Frank–Hall ordinal decomposition improves ordinal MAE over its matched *nominal*
  RF comparator (same weighting, seed, features, folds).
* **H3** — Balanced class weighting raises severe-class recall at a cost in severe precision
  and/or ordinal MAE.
* **H4** — Removing the timing-questionable broad-tier fields reduces severe-class assignment
  relative to the broad tier.

Hypotheses can fail; failures are reported as results, not adjusted away. All other
model-versus-baseline comparisons are **secondary/exploratory**: they are reported with intervals
but are not treated as confirmatory, and no multiplicity correction is claimed for them; the single
primary comparison carries the headline inference. The comparator baseline is chosen on development
evidence, not hard-coded. Stochastic models use three development seeds. The 2012 evaluation is run
once after freezing; 2,000 paired crash-level bootstraps quantify uncertainty. A 2×2×2 leakage
factorial (§7.5) quantifies three evaluation defects on development data only, reported as matched
simple effects against the leakage-controlled reference with interactions disclosed.

## 7. Results

Results are reported in tiers: (A) the honest re-analysis of the *original* contaminated protocol;
(B) development-phase model selection; (C) the corrected (v4) retrospective out-of-time 2012
result; (D) probabilistic quality and calibration; (E) the leakage factorial; (F) decision-rule
and weighting sensitivity; (G) the historical v3 benchmark and what the corrections changed; and
(H) the prespecified feature-tier and target-mapping sensitivities. Selection evidence
(development) is kept separate from evaluation evidence (2012).

### 7.1 Re-analysis of the original results (exploratory; contaminated protocol)

The four confusion matrices transcribed from the final report's result screenshots reproduce a
recomputation exactly (regression-tested); the source images are packaged with verifiable hashes
(`reanalysis/source_crops/`, traced to the report `.docx` whose SHA-256 matches the recorded
provenance). They were produced by standalone scripts (random 80/20 split,
`OneHotEncoder(min_frequency=0.01)`, no baselines, no uncertainty).

**Table 1 — Ordinal re-analysis of the original four confusion matrices (N = 10,936).**
(`oMAE`/`2-step` lower is better; `pred-0` = fraction predicted class 0. "—" = **undefined**
(METRIC-001): the majority predictor never predicts classes 1–2, so their precision/F1 — and hence
macro-F1 — are undefined, not zero; an earlier revision zero-filled these cells.)

| Model | Acc | oMAE | QWK | macroF1 | balAcc | ≤1 acc | 2-step | sevP | sevR | sevF1 | pred-0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Majority baseline** | 0.676 | 0.359 | 0.000 | — | 0.333 | 0.964 | 0.036 | — | 0.000 | — | 1.000 |
| Decision Tree | 0.580 | 0.451 | 0.221 | 0.452 | 0.472 | 0.969 | 0.031 | 0.234 | 0.339 | 0.277 | 0.610 |
| XGBoost | 0.627 | 0.401 | 0.367 | 0.514 | 0.578 | 0.971 | 0.029 | 0.252 | 0.542 | 0.344 | 0.574 |
| MLRF (RandomForest) | 0.710 | 0.309 | 0.365 | 0.506 | 0.509 | 0.981 | 0.019 | 0.403 | 0.383 | 0.393 | 0.867 |
| EBM | 0.578 | 0.490 | 0.311 | 0.460 | 0.583 | 0.931 | 0.069 | 0.158 | 0.681 | 0.257 | 0.534 |

**Three of four models fall below the 0.676 majority-accuracy baseline**; MLRF exceeds it while
predicting class 0 for **86.7%** of cases. **No model is uniformly best** (MLRF leads accuracy/oMAE;
EBM leads severe recall at 0.158 precision; XGBoost leads macro-F1). Within-one-level accuracy is
93–98% for every model *and* for the majority baseline, confirming it is uninformative here. These
numbers are exploratory (random split, full-data choices, code differing from the delivered
system) and cannot support generalisation. See Figures 5–7.

### 7.2 Development-phase model selection

Rolling-origin development CV (train 2009 → validate 2010; train 2009–2010 → validate 2011; three
seeds; strict tier; posterior-median rule) ranked models by ordinal MAE
(`experiment/development_report.json`). Across all thirteen development models, the two matched
**unweighted ablation controls** led (ordinal RF unweighted 0.3233; RF unweighted 0.3246),
followed by the class-weighted ordinal random forest (0.3358); the three constant-class-0
baselines tied at 0.3465 (majority, ordinal median, and the prior-probability forecast, whose
posterior median is class 0); the class-weighted random forest (0.3744) fell **below the trivial
baselines** on the development objective; XGBoost scored 0.4314 and the class-weighted linear
models trailed badly under the strict tier (0.53–0.65). The ablation controls exist to isolate
the effect of class weighting with everything else held fixed (§7.6) and are excluded from
candidate selection by design (`kind="ablation"`; a governance test enforces that they can never
be chosen), so the ordinal random forest was the top **candidate** — a role distinction, not an
empirical ranking. Among the prespecified baselines the
**majority** baseline had the lowest development ordinal MAE and was selected as the comparator —
the hardest trivial target to beat. The ordinal random forest was thereby designated, **on
development evidence and before the 2012 evaluation**, as the primary candidate and as one of the
two models carried into the calibration analysis (§7.4). The EBM was not included in development.

### 7.3 The corrected retrospective out-of-time 2012 result (v4 headline)

The v4 experiment was frozen and evaluated **once** on the 2012 year (run `final_8af9d5bc23d8`;
n = 11,630; prevalence 0.677 / 0.284 / 0.039; zero group overlap; 49 strict-tier input fields →
376 encoded features; 624 MB peak). "Once" refers to a single governed run against a frozen
configuration; it is not a claim of prospective sealing. Hard predictions use the prespecified
posterior-median rule (§4); argmax appears as a per-model sensitivity in §7.6. Table 2 reports the
result; paired crash-level bootstrap 95% CIs are for the ordinal-MAE difference vs the majority
baseline under the independence approximation of §5.6.

**Table 2 — Corrected (v4) retrospective out-of-time result on the 2012 evaluation (n = 11,630):
all fourteen evaluated models,** sorted by ordinal MAE (posterior-median rule; strict tier). The
**primary** comparison is Ordinal RF vs Majority; *ablation control* rows are the matched WGT-001
unweighted controls (excluded from candidacy by design); EBM is final-only exploratory. A severe
precision of "—" is **undefined** (the model never predicts the severe class; METRIC-001). † the
proportional-odds optimizer did **not** converge on the strict-tier design (recorded and flagged;
excluded from any superlative). The unweighted ordinal RF's severe precision of 1.000 rests on a
severe-prediction count of ~1 and is not meaningful.

| Model | oMAE | Δ vs maj. (95% CI) | Acc | balAcc | QWK | sev. recall | sev. prec. | pred-0 | role |
|---|---|---|---|---|---|---|---|---|---|
| Random forest (unweighted) | 0.3401 | −0.0213 [−0.0261, −0.0166] | 0.686 | 0.359 | 0.143 | 0.000 | — | 0.934 | ablation control |
| Ordinal RF (unweighted) | 0.3423 | −0.0191 [−0.0241, −0.0144] | 0.684 | 0.359 | 0.141 | 0.002 | 1.000 | 0.929 | ablation control |
| **Ordinal RF** | **0.3469** | **−0.0144 [−0.0231, −0.0063]** | 0.670 | 0.408 | 0.249 | 0.058 | 0.520 | 0.790 | primary candidate |
| Majority / Ord. median | 0.3614 | 0.0000 (reference) | 0.677 | 0.333 | 0.000 | 0.000 | — | 1.000 | baseline |
| Prior probability | 0.3614 | +0.0000 [+0.0000, +0.0000] | 0.677 | 0.333 | 0.000 | 0.000 | — | 1.000 | baseline (deterministic prior forecast) |
| Random forest | 0.3764 | +0.0150 [+0.0046, +0.0258] | 0.634 | 0.409 | 0.257 | 0.018 | 0.800 | 0.660 | secondary |
| XGBoost | 0.4442 | +0.0828 [+0.0690, +0.0967] | 0.574 | 0.490 | 0.285 | 0.267 | 0.265 | 0.475 | secondary |
| EBM | 0.4800 | +0.1186 [+0.1047, +0.1324] | 0.555 | 0.488 | 0.264 | 0.313 | 0.175 | 0.461 | exploratory (final-only) |
| Proportional-odds logit † | 0.5118 | +0.1504 [+0.1361, +0.1650] | 0.523 | 0.509 | 0.263 | 0.429 | 0.201 | 0.416 | baseline (non-converged) |
| Frank–Hall logit | 0.5194 | +0.1580 [+0.1440, +0.1721] | 0.576 | 0.535 | 0.268 | 0.624 | 0.130 | 0.607 | baseline |
| Multinomial logit | 0.5208 | +0.1594 [+0.1446, +0.1743] | 0.526 | 0.519 | 0.263 | 0.451 | 0.174 | 0.407 | baseline |
| Decision tree | 0.6040 | +0.2426 [+0.2267, +0.2585] | 0.455 | 0.458 | 0.172 | 0.336 | 0.122 | 0.313 | secondary |
| Shallow tree | 0.6272 | +0.2658 [+0.2499, +0.2819] | 0.386 | 0.416 | 0.129 | 0.164 | 0.197 | 0.139 | baseline |

Findings (Figures 1–3):
* **Primary comparison.** The ordinal random forest beats the majority baseline on ordinal MAE by
  a small margin: 0.347 vs 0.361, an improvement of 0.014 class-steps whose paired bootstrap CI
  [−0.023, −0.006] excludes 0. In intuitive units that is about **14 fewer class-steps of ordinal
  error per 1,000 crashes** (≈4% relative) — half the historical v3 margin (§7.7). It achieves
  this by predicting "no injury" for **79%** of crashes and recovering only **5.8%** of severe
  crashes. This is the single prespecified **primary retrospective comparison** (§6); no
  comparison in this study is labelled confirmatory — the evaluation year was historically
  exposed (§3.4).
* **The matched unweighted controls top the table.** The two WGT-001 ablation controls attain the
  lowest ordinal MAE of all fourteen models (unweighted RF 0.3401; unweighted ordinal RF 0.3423,
  vs the primary candidate's 0.3469) — by predicting "no injury" ≈93% of the time and recovering
  essentially **zero** severe crashes. They are excluded from candidate selection by design (a
  governance test enforces this), so they do not displace the prespecified primary comparison —
  but they show that the candidate's balanced class weighting does **not** help the ordinal
  objective, and under the corrected protocol the class-weighted plain random forest is actually
  **worse than the trivial baseline** (+0.015 [+0.005, +0.026]).
* **Severe-crash hard-decision assignment collapses under the restricted tier — but ranking
  signal remains (the FEAT-001/INTERP result).** With the timing-questionable fields removed, no
  model both beats the trivial baseline on the ordinal loss *and* meaningfully **assigns** the
  severe class under the loss-consistent rule: the tree ensembles that win on oMAE recover 0–6%
  of the 450 severe crashes, while the models that recover more (XGBoost 27%, EBM 31%, the
  class-weighted linears 43–62%) do so at ≤27% precision and ordinal error far worse than doing
  nothing. The frozen probabilities nevertheless retain **nontrivial severe ranking signal**
  across every model family (one-vs-rest AP 0.17–0.23 vs the 0.039 no-skill prevalence; AUROC
  0.75–0.80; `experiment/severe_ranking.md`, §8) — the collapse is a property of the hard
  decision under the absolute-ordinal-loss rule, not proof that severe risk carries no
  information. In v3, the same primary candidate *assigned* 28% of severe crashes; that
  assignment behaviour came predominantly from the excluded fields (restraint/ejection/seat,
  test-given, contributing circumstances, damage location; §7.7–§7.8), whose availability at
  decision time is exactly what remains unevidenced.
* **Seed/refit robustness (retrospective addendum).** Refitting the four forest variants under
  five seeds (nothing else varied; `experiment/seed_robustness.md`) leaves every refit on the
  same side of the majority baseline as the published result, quantifying the training-noise
  component the case-bootstrap interval conditions away.
* **Different metrics still crown different winners** (ordinal RF on ordinal MAE among
  candidates; XGBoost on QWK, 0.285; Frank–Hall logit on balanced accuracy, 0.535, and severe
  recall, 0.624). "Best model" remains undefined without a declared objective — precisely the
  gap in the original study.

### 7.4 Probabilistic quality and calibration

The v4 registry contains the honest trivial probabilistic floor explicitly: the deterministic
**prior-probability forecast** (log loss 0.7474, Brier 0.4592, RPS 0.2558). Exactly the **four
random-forest variants** beat it on all three proper scores — best overall are the matched
unweighted controls (unweighted ordinal RF log loss 0.6878; unweighted RF Brier 0.4232 and RPS
0.2313), with the primary candidate at 0.7075 / 0.4322 / 0.2384. **The remaining learned models
do not consistently beat the prior forecast**: XGBoost (log loss 0.7958), EBM (0.8372), the
class-weighted linears (0.86–2.02), and the single trees all lose to it on at least one proper
score (an earlier draft of this revision claimed "every learned model" beat the prior — that was
false and is now machine-checked against the artifact at packaging). The forest models'
probabilistic improvement over trivial prior knowledge is therefore real but, like the point
improvement, modest — and family-specific. The
hard one-hot **majority baseline remains a probabilistic disaster** (log loss 11.15): a
classifier that never assigns probability to the minority classes is catastrophically penalised
the moment such a crash occurs — a fact accuracy hides entirely. Top-label ECE illustrates its
own limitation here: the **constant prior forecast attains the best ECE of all models (0.009)**
while carrying zero discriminative information — which is why ECE is reported as a descriptive
diagnostic beside the proper scores, never as a ranking criterion. Development-only calibration
now uses **temporal rolling-origin folds** (§5.6): it cuts the primary candidate's ECE from 0.059
to 0.032 and its log loss from 0.708 to 0.692, and calibrating the constant majority classifier
recovers, as expected, approximately the prior forecast (calibrated log loss 0.748, ECE 0.013) —
all without fitting on the evaluation year (Figure 4). The two models carried into the
calibration analysis — the **majority comparator** and the **ordinal random forest** — were
selected **on development evidence before the 2012 evaluation** (the comparator baseline and the
top development candidate, §7.2), not because of any 2012 outcome; a governance test enforces
that final-result objects cannot enter this selection (`tests/`).

### 7.5 Leakage factorial (matched defect contrasts and interactions)

The original project's "leakage diagnostic" changed three things at once and attributed the whole
gap to leakage. We instead run a 2×2×2 factorial over five seeds on development data only (a fixed
balanced random forest; broad feature tier by construction, since its point is to toggle the
prohibited outcome-derived columns; `crashsev/leakage_factorial.py`) and report **matched simple
effects against the leakage-controlled reference cell** (ordinal MAE 0.363). Two factors are matched
contrasts — same rows, model, and seed, one factor toggled: adding the 14 outcome-derived columns,
and fitting preprocessing before the split. The third, random vs temporal splitting, changes the
evaluation population itself, so its row is a **validation-optimism** contrast between two
estimands, not an isolated causal "split effect" (FAC-001). These are methodological
perturbations, not causal effects.

**Table 3 — Matched simple effects vs the leakage-controlled reference (negative Δ ordinal MAE =
apparent improvement, i.e. inflation), and the same contrast when leakage is already present.**

| Defect (contrast vs reference) | Δ oMAE | Δ acc | Δ sev. recall | Δ oMAE given leakage ON |
|---|---|---|---|---|
| Add outcome-derived (leakage) features | **−0.349** | **+0.339** | **+0.403** | — |
| Fit preprocessing before the split | −0.021 | +0.021 | −0.001 | −0.000 |
| Random (non-temporal) split — validation-optimism | −0.033 | +0.034 | +0.019 | +0.002 |

**The defects interact; they are not additive.** Outcome-derived features dominate: they alone
collapse ordinal MAE from 0.363 to 0.014 (accuracy 0.986, severe recall 0.61) — the model
"succeeds" only because it is reading the injury outcome — and once they are present, the other two
defects have essentially no room left to matter: their conditional effects shrink to ≈0.000 and
+0.002 (interaction terms +0.020 and +0.035 on ordinal MAE, i.e. their effects vanish under
leakage). Under the leakage-controlled protocol the two smaller defects are genuine and an order of
magnitude smaller than leakage (−0.021 and −0.033 vs −0.349). An earlier revision of this section
described *marginal* factorial averages (−0.327 / −0.005 / −0.010) as effects varied "one at a
time" and called them "cleanly separated"; that mislabelled the estimand — the marginals average
matched and leakage-saturated cells together and understate the two smaller defects roughly
three- to four-fold. They remain available in `experiment/leakage_factorial.json` under an
explicit `marginal_main_effects_NOT_one_at_a_time` label; Table 3 reports the matched contrasts.
The substantive lesson is unchanged and now correctly stated: the single most consequential defect
is including outcome-derived predictors, which the feature-availability ledger forbids.

### 7.6 Decision-rule and weighting sensitivity

In v4 the loss-consistent **posterior-median** rule is the prespecified primary decision (§4), and
**argmax is the recorded sensitivity** — the reverse of v3, where argmax was primary and the
median only a post-hoc check. Per model, both rules are recomputed from the frozen 2012
probability vectors (`experiment/decision_rule_sensitivity.md`; per-model `ordinal_mae_argmax`
in `experiment/final_results.json`). The rule behaves exactly as decision theory predicts for the
near-symmetric models: the median improves the primary candidate (0.3469 vs 0.3501 under argmax)
and both unweighted controls (0.3401 vs 0.3434; 0.3423 vs 0.3438). For the **class-weighted plain
RF the median rule hurts** (0.3764 vs 0.3582 under argmax): balanced weighting inflates the
predicted mass on classes 1–2, so the cumulative rule crosses 0.5 above class 0 for many crashes
that argmax still assigns to class 0 — under the loss-consistent rule, that model loses to the
trivial baseline outright. **Class weighting is assessed with the matched single-model ablation
(WGT-001):** each of `random_forest` and `ordinal_random_forest` is run with
`class_weight="balanced"` and with `class_weight=None`, holding everything else fixed. Under the
corrected protocol the weighting is now **clearly counter-productive for the ordinal objective**:
it costs the plain RF +0.036 ordinal MAE (0.3401 → 0.3764) to raise severe recall only from
0.000 to 0.018, and costs the ordinal RF +0.005 (0.3423 → 0.3469) to raise severe recall from
0.002 to 0.058. The severe-recall gains that weighting bought in v3 (up to 0.28 for the primary
candidate) required the timing-questionable broad-tier fields (§7.7–§7.8); under the restricted tier,
weighting mostly buys ordinal error.

### 7.7 The historical v3 benchmark, and what the corrections changed

The v3 governed run (`final_f27613102c96`, generated at `e55d6f1`; broad feature tier, 483
encoded dimensions, argmax rule, fitting/selection-level isolation only) reported: ordinal RF
**0.3310** vs majority **0.3614** (paired Δ −0.0304 [−0.0369, −0.0236]), plain RF 0.3436,
unweighted controls 0.3278/0.3306, severe recall 0.278 (ordinal RF) / 0.198 (RF). Those numbers
were internally reproducible (bit-exact self-reproduction) — but the third-round reconnaissance
established that the protocol behind them had material defects: the development artifacts allowed
the 2012 outcome distribution to be reconstructed by subtraction, the split hash was itself
outcome-dependent, the decision rule contradicted the declared loss, and the feature tier
included fields whose decision-time availability rests on unevidenced assertions. The v3 result
is therefore retained as **historical evidence with those caveats**, not as the study's benchmark.

**What the corrections changed is itself a finding of the remediation study:**

| Quantity | v3 protocol | v4 corrected protocol |
|---|---|---|
| Primary Δ vs majority (ordinal MAE) | −0.0304 [−0.0369, −0.0236] | **−0.0144 [−0.0231, −0.0063]** |
| Primary candidate severe recall | 0.278 | **0.058** |
| Class-weighted RF vs baseline | better (−0.0178) | **worse (+0.0150)** |
| Best-oMAE model (ablation control) | unweighted RF 0.3278 | unweighted RF 0.3401 |
| Encoded dimensions | 483 (broad tier) | 376 (strict tier, 49 fields) |

The v3→v4 shift bundles three simultaneous changes (feature tier, decision rule, isolation
mechanics), so this pair of runs alone does not decompose it — but the recorded per-model argmax
sensitivities (§7.6) and the separately frozen broad-tier v4 sensitivity (§7.8) do: under the v4
protocol the rule change moves the primary candidate only ≈0.003 (0.3501 argmax → 0.3469 median),
so the bulk of the headline shrinkage and essentially all of the severe-recall collapse trace to
the **feature tier** — that is, to removing exactly the fields whose availability at decision
time is unevidenced. An evaluation-remediation study should expect corrections to shrink its
headline; measuring *how much* each defect had inflated it is the point.

### 7.8 Prespecified sensitivities: feature tier and target mapping

**Feature tier (FEAT-001; separately frozen broad-tier run `final_10517423399d`).** The broad
tier reruns the complete v4 protocol with one toggle — `feature_tier: broad` (66 allowed fields,
481 encoded dims) — under its own freeze, so tier is the *only* difference from the primary
benchmark. With the rule and protocol held fixed, the tier alone accounts for most of the
v3→v4 headline shift:

| Quantity (posterior-median rule, v4 protocol) | Strict tier (primary) | Broad tier (sensitivity) |
|---|---|---|
| Ordinal RF Δ vs majority | −0.0144 [−0.0231, −0.0063] | −0.0287 [−0.0373, −0.0204] |
| Ordinal RF severe recall | 0.058 | 0.171 |
| Unweighted controls (oMAE) | 0.3401 / 0.3423 | 0.3258 / 0.3253 |
| Class-weighted RF Δ vs majority | +0.0150 [+0.0046, +0.0258] (worse) | −0.0062 [−0.0169, +0.0051] (CI includes 0) |

Combining this with §7.7's rule sensitivity gives a clean two-step decomposition of the v3
headline: moving from argmax to the loss-consistent median rule (broad tier fixed) lowered the
candidate's severe recall from 0.278 to 0.171 with the oMAE nearly unchanged (0.3310 → 0.3327);
tightening the tier (rule fixed) then halved the headline margin and collapsed severe recall to
0.058. The broad tier's stronger numbers do **not** justify promoting it back to primary: its
additional fields (restraint/ejection/seat, test-given, contributing circumstances, sequence,
damage location, insurance) are precisely those whose decision-time availability is unevidenced,
so its result is an upper bound conditional on winning the Gate-3 timing evidence — and the
proportional-odds optimizer failed to converge under both tiers (flagged in both artifacts).

**Target mapping (TARGET-001; development-only; `experiment/target_sensitivity.md`).** The
prespecified bounded sensitivity — blank = PDO (primary) vs blank = excluded (the fail-closed
alternative) — produced a finding stronger than a robustness pass or fail: under blank→excluded
the usable development cohort retains only 11,045 rows with classes {1, 2} present (prevalence
0 / 0.88 / 0.12). **Class 0 exists in this extract almost entirely through the blank→PDO
inference**, so the alternative mapping does not test the same estimand — it defines a different
(2-class) problem. The blank = PDO decision is therefore not a marginal coding choice but the
*constitution of the majority class*; the internal evidence for it remains strong (0.0%
injury/fatality contamination of 32,046 blanks, §3.2), and obtaining the official Alaska
codebook (Gate 3) is now demonstrably material to construct validity rather than a formality.

**High-missingness sensitivity (MISS-002; separately frozen run `final_d612d030bc51`).** The
prespecified rule (fields with > 50% development missingness; `experiment/missingness_summary.md`)
flags exactly two restricted-tier fields — `Unit 1 Person 1 Residence State` (~74%) and
`Direction` (~64%) — and the complete protocol reruns without them (365 encoded dims). The
result is honestly mixed: the primary contrast **survives in direction but attenuates** — the
candidate's Δ vs majority weakens from −0.0144 to **−0.0091 [−0.0181, −0.0001]**, a CI that only
barely excludes zero, so part of the primary candidate's margin **does lean on high-missingness
administrative fields** whose absence patterns could encode reporting practice; the unweighted
controls' margin is robust (−0.0226 [−0.0279, −0.0175]); the class-weighted RF remains worse
than the baseline (+0.0203); severe hard-assignment is unchanged (0.056). The proportional-odds
optimizer again failed to converge (flagged).

**Hypothesis adjudication (H1–H4, §6).**

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** candidate beats majority on oMAE (CI excl. 0) | **Supported** | Δ −0.0144 [−0.0231, −0.0063]; direction persists across all five refit seeds (§7.3) and, attenuated, under the high-missingness removal (−0.0091 [−0.0181, −0.0001]) |
| **H2** ordinal decomposition beats matched nominal RF | **Not supported** | Final: unweighted ordinal RF 0.3423 vs unweighted RF 0.3401 (nominal marginally better); across five refit seeds both average 0.3397 — indistinguishable within seed noise. The ordinal decomposition provides no measurable oMAE advantage over its matched nominal comparator here, and this failure is reported as a result |
| **H3** weighting buys severe recall at oMAE/precision cost | **Supported** | Recall 0.000→0.018 (RF) and 0.002→0.058 (ordinal RF) at +0.036 / +0.005 oMAE and large precision costs (§7.6) |
| **H4** broad tier assigns more severe crashes | **Supported** | Hard-assignment recall 0.171 (broad) vs 0.058 (strict), protocol and rule fixed |

## 8. Error analysis

The confusion structure localises the failure modes. Across all models, the dominant confusion is
**class 1 (minor) mistaken for class 0 (none)**: minor injuries are the hardest to separate from
no-injury under the restricted tier (the primary candidate assigns 2,256 of 3,303 minor
crashes to class 0). Severe (class 2) crashes are now essentially unresolved by the models that
win on the ordinal loss (ordinal RF recovers 5.8%; the unweighted controls 0–0.2%), and only
recovered at steep cost by the models that lose on it (XGBoost 27% at 27% precision; the
class-weighted linears 43–62% recall at 13–20% precision and ordinal error far worse than the
trivial baseline). Two-step errors — predicting "none" for a serious/fatal crash or vice-versa —
are the costliest an ordinal metric weights; the aggressive-recall models incur the most of them,
which is exactly why they lose on ordinal MAE despite winning on balanced accuracy. The corrected
protocol's substantive severe-class finding therefore has two parts that must not be conflated.
**Hard decisions:** under the loss-consistent posterior-median rule and the restricted tier, the
oMAE-leading models rarely assign the severe class (0–5.8% recall) — and the assignment behaviour
the v3 protocol showed came predominantly from the excluded, timing-questionable fields (§7.7).
**Ranking:** the same frozen probabilities carry real severe-risk ordering information — severe
one-vs-rest average precision 0.17–0.23 against a no-skill prevalence of 0.039 (4.4–5.9×) and
AUROC 0.75–0.80 across every model family, with the class-weighted linears ranking marginally
*best* (`experiment/severe_ranking.md`; Figure 8). Low hard-class recall under an
ordinal-loss-optimal rule must not be read as an absence of predictive information about severe
risk; equally, this ranking signal is **not** validated severe-crash detection — turning it into
an operating point would require a real stakeholder cost function and evaluation on data not
used here, and no threshold is selected on the 2012 year (that would be post hoc). The result is
descriptive of this cohort, not a statement about the task in general; any operating point is a
deliberate precision/recall choice, not a free lunch.

Concretely, on the 2012 cohort the primary candidate's confusion structure (regenerated via
`python -m crashsev.error_analysis`; `experiment/error_analysis.md`) shows that of **450
serious/fatal crashes it recovers 26**, misses **424** (of which **193 are the worst, off-by-two
`2→0` errors**), and raises **24** false serious/fatal alarms; its high-confidence predictions
(max probability ≥ 0.8, 1,279 cases) are wrong 9.8% of the time, and the per-case error rate
falls monotonically with predicted confidence (0.57 in the [0.0,0.4) bin, 0.43 in [0.4,0.6),
0.24 in [0.6,0.8), 0.10 in [0.8,1.0)). This is the honest severe-class picture behind the
aggregate ordinal-MAE improvement: the headline gain is driven entirely by common-class ordering,
not by catching rare severe crashes.

## 9. Discussion

The evidence supports a narrow, useful conclusion and refutes the original broad ones. Under the
fully corrected v4 protocol — outcome-derived fields removed, timing-unevidenced fields removed
from the primary tier, structural development isolation, a loss-consistent decision rule,
preprocessing fit train-only, a chronological crash-grouped out-of-time evaluation, ordinal
metrics, temporal-fold calibration, and paired uncertainty — the honest picture is: a small,
interval-supported ordinal-MAE improvement over the trivial baseline (−0.014 class-steps, ≈4%
relative) driven entirely by common-class ordering; probabilistic forecasts that beat the
deterministic prior on every proper score for the four forest variants only, by similarly modest
margins; and severe crashes that the loss-consistent hard rule rarely assigns, while the
probabilities retain modest, unvalidated ranking signal (§8). Each successive wave of protocol
correction *shrank* the apparent result (§7.7): the original study's dramatic numbers were mostly
leakage (−0.349 apparent inflation from outcome-derived fields, §7.5), and the v3 remediation's
own moderate numbers were partly decision-time-questionable features (severe recall 0.28 → 0.06)
plus a loss-inconsistent decision rule. The result does **not** support that any model is "best,"
that the system identifies crash *causes*, that it generalises to arbitrary U.S. datasets, or
that it is deployment-ready. That the honest answer keeps getting smaller as the protocol gets
stricter is itself the informative result — and measuring that shrinkage, defect by defect, is
what an evaluation-remediation study is for. The contribution is best read as evaluation
forensics plus a reusable, executable governance framework, demonstrated on a real retrospective
comparison, rather than as a new predictive capability.

## 10. Limitations and threats to validity

* **Exposed, retrospective evaluation.** The 2012 outcomes were locally available throughout the
  work. The executable lock keeps development *fitting and selection* on 2009–2011, but 2012 is not
  a prospective, preregistered, or externally witnessed holdout; all 2012 results are retrospective,
  and any protocol refinement made while 2012 existed on disk is an amendment, not a preregistration.
* **Development-artifact outcome isolation (GOV-001) — corrected in v4, caveated for v3.** The
  v4 develop phase is structurally outcome-isolated (year-partition before any target
  interpretation; sentinel-tested), so v4 development artifacts carry no derivable final-year
  outcome. The historical v3 result (§7.7) does not have this property: its development report
  allowed the 2012 outcome counts to be reconstructed by subtraction.
* **Predictor timing rests on ledger assertions (FEAT-001) — mitigated, not resolved.** The
  conservative restricted tier is the v4 primary predictor set, but its tier assignments are
  themselves the authors' judgments, not form-level evidence; several excluded fields might in
  fact be scene-available (costing recall unnecessarily), and some retained fields could still be
  revised post-scene. Form-level recording-time evidence is an open external item
  (`research/PROVENANCE_ACQUISITION_PLAN.md`).
* **Proportional-odds non-convergence.** On the strict-tier design the ordered-logit optimizer
  hit its iteration cap without converging; the fit is reported with an explicit non-convergence
  flag, excluded from any superlative, and not tuned post hoc (retuning after seeing final
  results would breach the freeze).
* **Data authority.** The extract is used under a data-owner grant, not verified against an official
  source-agency provenance record; a reviewer without the licensed file cannot recompute from raw.
  The raw file's cryptographic identity is recorded where available (`research/DATA_AUTHORITY_AND_ACCESS.md`).
* **External validity.** Findings describe Alaska 2009–2012 *reported* crashes only; the single
  4-year window (the only raw extract with an intact temporal axis in the licensed archive) is a
  real limit. Unreported crashes are out of frame.
* **Construct validity.** The KABCO ordering and 3-level collapse are documented and reversible;
  blank = PDO is evidence-checked for this extract but is a source-specific coding decision, not an
  official codebook validation.
* **Statistical dependence.** Intervals use a crash-level bootstrap under a cross-crash independence
  approximation (§5.6); residual spatial/temporal dependence is unmodelled. Severe crashes are rare
  (≈450 in the 2012 evaluation), so severe-class and subgroup estimates are correspondingly uncertain.
* **Measurement/selection.** "Alcohol suspected" and similar are scene judgments, not adjudicated
  facts; high-missingness fields (e.g. contributing circumstance) carry informative missingness.
* **Reproduction status.** The bit-for-bit re-run to date is **self-reproduction** by the same
  project, not independent reproduction by a third party.
* **No causal identification.** No coefficient or importance is read as an intervention effect.

## 11. Reproducibility and ethics

All code, contracts, tests, re-analysis, and the committed aggregate results regenerate from the
repository (`research/REPRODUCE.md`); the environment is pinned at the direct-dependency level
(`requirements-lock.txt` — a minimal direct-pin list, **not** a fully resolved transitive lock; a
documented limitation). The v4 governed artifact (run `final_8af9d5bc23d8`) was **generated at
commit `458f784` under a clean-source freeze with structural outcome isolation**
(`source_dirty = False` recorded inside the artifact; the freeze pins the outcome-free
development-side assignment hash), and its bundle files hash **byte-exactly** to the manifest's
`artifact_sha256` (the v3 writer needed LF-normalisation — BUNDLE-CRLF-001, disclosed and fixed).
The historical v3 artifact (run `final_f27613102c96`, generated at `e55d6f1`, committed at
`e17c263`) is preserved with its own committed skeleton. Generator commit, run bundle,
artifact commit, and release commit are distinct objects; the full chain, with hashes, is
`research/PROVENANCE_CHAIN.md` — a release commit necessarily postdates the artifact it packages,
and no revision of this paper claims otherwise. A CI
workflow (`.github/workflows/crashsev-ci.yml`) is defined and has been run **locally** on Windows
(Python 3.13); it has **not** been executed on hosted CI because the branch is unpublished, so no
hosted "green" status is claimed. Reproduction performed within this project is labelled
**self-reproduction** in `research/REPRODUCTION_LOG.md`; no independent third-party reproduction has
occurred. Raw records, coordinates, and identifiers are never committed; the complete per-crash
run bundles (`final_8af9d5bc23d8` for v4, `final_f27613102c96` for the historical v3) are
retained **locally only** because their rows are keyed by `Crash Number` (an identifier). What
*is* committed, per run, is the bundle's cryptographic skeleton under `evidence_release/<run>/`:
the run manifest with per-artifact SHA-256s, the freeze markers (`FROZEN.lock`, `FINAL.done`),
and a de-identification transform witness that ties each original prediction file's hash to its
published de-identified counterpart. A reviewer with a licensed copy of the extract (byte
identity SHA-256 per the restricted reproduction log, §3.1) reproduces the result by re-running
`develop → freeze-experiment → evaluate-final`, which yields an identical development-assignment
hash and identical aggregates. A reviewer **without** the data can recompute the reported results
from the shipped de-identified 2012 predictions, with the reproduction scope stated precisely:
hard-label metrics reproduce **exactly** treating the shipped `y_pred` as authoritative, and
probability-derived metrics and the paired bootstrap intervals reproduced **exactly in our own
text-round-trip verification** (the de-identification witness checks per-model metric equality
file by file); any residual text-serialization discrepancy would be bounded by float round-trip
precision and should be reported against the witness. Responsible-use
boundary: false-severe predictions and false reassurance both carry costs, geographic reporting can
stigmatise, and no deployment is endorsed.

## 12. Conclusion

For police-reported Alaska crashes (2009–2012), the researcher-defined three-level ordinal
severity outcome is only weakly predictable from the conservative restricted feature tier under
the corrected v4 protocol — structural development isolation, a loss-consistent posterior-median
rule, and an author-judged decision-time tier: the ordinal random forest beats a trivial
majority baseline on ordinal MAE by a small, interval-supported (conditional case-bootstrap)
margin (0.347 vs 0.361; ≈14 class-steps per 1,000 crashes) that persists across five refit
seeds, the matched unweighted controls do marginally better still, and the four random-forest
variants beat the deterministic prior forecast on every proper score (the other learned models
do not consistently). Under the loss-consistent hard rule the oMAE-leading models rarely
**assign** the severe class (0–6% recall; models that assign more pay with precision ≤27% and
ordinal error worse than doing nothing), while the frozen probabilities retain modest severe
**ranking** signal (AP ≈5× prevalence; AUROC ≈0.79) that is real but unvalidated for any
operational use. Each wave of protocol correction shrank the apparent result (§7.7), which is
the remediation study's central measured finding: the original project's strength was mostly
leakage, and part of the remainder was decision-time-questionable features and a
loss-inconsistent decision rule. No model dominates across metrics, so a declared objective is
indispensable. The contribution is a governed, reproducible, leakage-controlled *retrospective*
study and framework, an evidence-calibrated target mapping, an honest re-analysis, and a controlled
leakage factorial — not a causal safety finding or a general product.

## 13. Recommendations and future work

Deliberately restrained to what the evidence motivates — no model families, dashboards, or
deployment work belong here:

1. **Source semantics (highest value).** Obtain the applicable Alaska codebook or a custodian
   attestation for the blank `Crash Severity` semantics: §7.8 shows the blank = PDO decision
   *constitutes* class 0, so this single document determines the outcome's construct validity
   (`research/PROVENANCE_ACQUISITION_PLAN.md`).
2. **Field-timing evidence.** Per-field recording-stage evidence (form section, recorder,
   revisability) for the at-event ledger, replacing the author-judged tier with documented tiers;
   rerun the benchmark on the evidenced tier.
3. **Later temporal validation.** A genuinely later Alaska cohort (or an external jurisdiction
   with compatible semantics) evaluated once under the frozen protocol — the only way to turn the
   single exposed retrospective year into temporal-transfer evidence.
4. **Decision thresholds only after a utility function exists.** The severe ranking signal (§8)
   becomes operationally meaningful only with a real stakeholder cost matrix and data not used
   here; until then no operating point should be published.
5. **Independent reproduction and external witnessing.** A third party executing
   `research/REPRODUCE.md` end-to-end, plus a hosted CI run and an external timestamp once push
   authorization exists.

## References

1. Shmueli, G. (2010). To Explain or to Predict? *Statistical Science*, 25(3), 289–310. doi:10.1214/10-STS330.
2. Savolainen, P., Mannering, F., Lord, D., & Quddus, M. (2011). The statistical analysis of highway crash-injury severities: A review and assessment of methodological alternatives. *Accident Analysis & Prevention*, 43(5), 1666–1676. doi:10.1016/j.aap.2011.03.025.
3. Frank, E., & Hall, M. (2001). A Simple Approach to Ordinal Classification. *ECML 2001*, LNCS 2167, 145–156. doi:10.1007/3-540-44795-4_13.
4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. doi:10.1023/A:1010933404324.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*, 785–794. doi:10.1145/2939672.2939785.
6. Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML: A Unified Framework for Machine Learning Interpretability. arXiv:1909.09223.
7. Cohen, J. (1968). Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial credit. *Psychological Bulletin*, 70(4), 213–220. doi:10.1037/h0026256.
8. Epstein, E. S. (1969). A Scoring System for Probability Forecasts of Ranked Categories. *Journal of Applied Meteorology*, 8(6), 985–987. doi:10.1175/1520-0450(1969)008<0985:ASSFPF>2.0.CO;2.
9. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*, 1321–1330.
10. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM TKDD*, 6(4), 15. doi:10.1145/2382577.2382579.
11. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
12. NHTSA. *Model Minimum Uniform Crash Criteria (MMUCC), 6th Edition.* (Cited to establish MMUCC as a minimum guideline, not proof of cross-state semantic interchangeability.)
13. Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. doi:10.1111/ecog.02881. (Structured validation must mirror the intended extrapolation.)
14. Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *JASA*, 102(477), 359–378. doi:10.1198/016214506000001437.
15. Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Medicine*, 17, 230. doi:10.1186/s12916-019-1466-7.
16. An ordinal-classification study of crash injury severity comparing ordered and nominal machine-learning formulations. PMC8583475, https://pmc.ncbi.nlm.nih.gov/articles/PMC8583475/. (Cited for the ordinal-vs-nominal question this study's H2 tests; identified by index to avoid misattribution.)
17. State of Alaska. *Motor Vehicle Crash/Accident Report, form 12-200* (rev. 2001), NHTSA-hosted: https://www.nhtsa.gov/sites/nhtsa.gov/files/documents/ak_12-200_par_rev9_12_2001.pdf. (Documents the police-report field inventory; does not by itself establish per-field recording time — §3.3.)

## Appendices

* **A. Data/target/feature contracts** — `data/schema.json`, `data/target_mapping.yml`, `data/feature_availability_ledger.csv`, `data/codebook_verification_09_12.md`.
* **B. Estimand, scope, provenance, authority** — `research/ESTIMAND_AND_SCOPE.md`, `research/DATA_PROVENANCE.md`, `research/DATA_LICENSE_NOTE.md`, `research/DATA_AUTHORITY_AND_ACCESS.md`, `research/ROUTE_DECISION.md`.
* **C. Run artifacts** — `experiment/development_report.json`, `experiment/final_results.json` (v4); `experiment/broad_sensitivity/` and `experiment/lowmiss_sensitivity/` (the separately frozen tier and high-missingness sensitivities); the content-addressed run bundles (`final_8af9d5bc23d8` v4 primary; `final_10517423399d` broad; `final_d612d030bc51` low-missingness; `final_f27613102c96` historical v3) are retained **locally only** (per-crash rows keyed by `Crash Number`) — each bundle's manifest, freeze markers, and de-identification witness are committed under `evidence_release/<run>/`, and the de-identified per-crash predictions ship in the handoff `evidence/`; `experiment/leakage_factorial.md`, `experiment/decision_rule_sensitivity.md`, `experiment/error_analysis.md`, `experiment/target_sensitivity.md`, `experiment/severe_ranking.md`, `experiment/missingness_summary.md`, `experiment/seed_robustness.md`; the machine-generated release record `research_audit/RELEASE_STATE.md`.
* **D. Re-analysis** — `reanalysis/reanalysis_table.md`, hashed source crops in `reanalysis/source_crops/`.
* **E. Tests & CI** — `tests/` (governance and contract tests), `.github/workflows/crashsev-ci.yml` (defined; run locally, not on hosted CI).
* **F. Conformance, disposition, authorship** — `research/PROTOCOL_CONFORMANCE.md`, `research/ISSUE_DISPOSITION_CURRENT.md`, `research/AUTHORSHIP_AND_AI_ASSISTANCE.md`, `research_audit/AA2_REMEDIATION_LEDGER.md`.

### Figures
* **Figure 1** — Out-of-time ordinal MAE with 95% crash-level bootstrap CIs (`fig_final_ordinal_mae_ci.png`).
* **Figure 2** — The core tension: severe-class recall vs ordinal MAE (`fig_final_tradeoff.png`).
* **Figure 3** — Severe-class precision–recall on the 2012 evaluation (`fig_final_severe_pr.png`).
* **Figure 4** — Probabilistic quality (log loss, all 14 models, log scale) and calibration ECE, raw vs development-fit (`fig_final_probabilistic.png`).
* **Figure 5** — Original re-analysis: accuracy and ordinal MAE vs the majority baseline (`fig_reanalysis_accuracy_and_mae.png`).
* **Figure 6** — Original re-analysis: severe-class precision–recall (`fig_reanalysis_severe_pr.png`).
* **Figure 7** — Original re-analysis: predicted vs true class distribution (`fig_reanalysis_predicted_dist.png`).
* **Figure 8** — Severe-class one-vs-rest precision–recall RANKING curves vs the no-skill prevalence (retrospective, exploratory; no operating threshold selected) (`fig_severe_ranking_pr.png`).
