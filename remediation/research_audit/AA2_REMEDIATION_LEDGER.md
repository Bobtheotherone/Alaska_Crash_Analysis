# AA2 remediation ledger

Tracks every issue in `Alaska_Crash_Analysis_Upgrade_Adversarial_Audit.md` (AA2-001…026)
against the v2 remediation on branch `graduate-research-remediation-v2`. Status is honest and
current; a claim of RESOLVED requires a named, checkable artifact meeting the audit's acceptance
criterion. This file is updated as work lands and is itself a deliverable.

Status legend: **DONE** (artifact meets acceptance criterion) · **WIP** (in progress) ·
**PLANNED** (design fixed, not yet built) · **SCOPED-OUT** (deliberately out of scope, with reason).

Route note: the v2 study is **Route A** (real Alaska 2009–2012 empirical study). See
`research/ROUTE_DECISION.md`. Several audit items framed around "no data" (AA2-002) are resolved
by the lawful data + an evidence-checked codebook (an internal check, not an official source-agency codebook validation) rather than deferred.

## P0 — blocking

| ID | Sev | Title (abbrev.) | Status | Resolving artifact / note |
|---|---|---|---|---|
| AA2-001 | S1 | Deliverable can't substantiate repo/commits/clean state | **DONE** | `C:\aca` real repo, `git fsck` clean, working tree clean; a **git bundle** (full verifiable history, `git bundle verify`) and a **clean release archive** (`git archive HEAD` → no `__pycache__`/bytecode/`_local_data`) are produced in packaging with SHA-256s. A self-reproduction (same project; not an independent third party) reproduces the out-of-time 2012 evaluation result **bit-exactly** (`research/REPRODUCTION_LOG.md`). |
| AA2-002 | S0 | Data / codebook / unit / rights absent | **DONE** | `research/DATA_LICENSE_NOTE.md` (owner grant); `data/codebook_verification_09_12.md` + `codebook_evidence_09_12.json` (blank=O **evidence-checked** — internal check, not an official source-agency codebook validation; 0.0% contamination); `Crash Number` unique per row; raw-file SHA recorded per run. |
| AA2-003 | S0 | Gate 0 is prose; runner accepts impostor CSV | **DONE** | `crashsev/contracts.py` loads `data/schema.json`, validates required columns + target domain; 3-col impostor rejected (missing 67 cols), undocumented target value rejected, real 50,543-row xlsx accepted. Wired into CLI Gate-0 (task 5). |
| AA2-004 | S0 | `target_mapping.yml` never loaded | **DONE** | `contracts.load_target_mapping` loads+validates YAML and **aborts** on malformed/incomplete/ambiguous/unevidenced-override mappings (all four tested); `target.map_severity` consumes its exact mapping incl. `blank_maps_to: O`; contract files hashed into the run manifest. |
| AA2-005 | S1 | Estimand / prediction moment unclear | **DONE** | `research/ESTIMAND_AND_SCOPE.md` freezes unit, target, prediction moment (at/before scene), use case, no-claims list; ledger reviewed against the **real 100 columns** (`data/feature_availability_ledger.csv`). |
| AA2-006 | S0 | Split manifest mishandles missing years / dup ids / reconciliation | **DONE** | `splits.py` rewritten: default `excluded` (missing-year row no longer hashes as development — tested), unique `Crash Number` row id enforced (dup/null abort), `assignment_df` saved + `assignment_hash()` reproduces manifest hash, reconciliation invariant (dev+test+excl==total) asserted, excluded reasons recorded. `final_test_years=[2012]` set in CLI config. |
| AA2-007 | S0 | Protocol/runner diverge (no dev CV, selection, seeds, calibration) | **DONE** | `crashsev/cli.py develop` runs rolling-origin (or grouped) CV over dev years for **selection**, per-seed repeats recorded in `development_report.json`; smoke-tested end to end. |
| AA2-008 | S1 | Final-test access not governed/locked | **DONE** | `cli.py` phases `validate-data/develop/freeze-experiment/evaluate-final` + `FROZEN.lock` (config/contract/data/split/git hashes) + dirty-tree refusal (verified: evaluate-final refused a dirty tree) + `FINAL.done` guard (one evaluation under a frozen configuration; within-study governance, not a prospective seal). |
| AA2-009 | S1 | `requirements-lock.txt` invalid/nonportable | **DONE** | Replaced the whole-env freeze (had an unrelated local path) with a minimal fully-pinned lock (only packages used); `pyproject.toml` extras separate `models`/`paper`/`test`; a CI workflow is defined to install the lock + import-check + run tests on Windows + Linux × py3.11-3.13; these steps were run locally only (Windows/Py 3.13), not on hosted CI (branch unpublished). |

## P1 — required for a credible study

| ID | Sev | Title (abbrev.) | Status | Resolving artifact / note |
|---|---|---|---|---|
| AA2-010 | S1 | Metric code truncates/accepts bad labels/no prob checks | **DONE** | `metrics._validate_labels`/`_validate_proba`: equal length, integer + `[0,k-1]` domain (blocks negative-index wrap via `np.add.at`), finiteness, nonneg, row-sum≈1, shape — all four rejections tested. |
| AA2-011 | S1 | 3-class study accepts 2-class dev data | **DONE** | CLI `prepare` aborts if usable data lacks any class; per-fold coverage checked in dev CV; `predict_aligned` reindexes probability columns to persisted `classes_` (fills absent classes). |
| AA2-012 | S1 | `dev_cv` and seed don't affect models | **DONE** | `build_registry(seed)` threads seed everywhere; CLI rebuilds the registry per seed, invokes rolling/grouped CV, records per-trial `{model,seed,fold,ordinal_mae}` in the dev report. |
| AA2-013 | S1 | `ordinal_logistic` hard-coded as "strongest" but weak | **DONE** | Prespecified `baseline_set` incl. true `proportional_odds` + renamed `frank_hall_logistic`; `select_primary_baseline` picks the comparator by best **dev-CV** ordinal MAE, not hard-coded; all baselines reported. |
| AA2-014 | S2 | Calibration/PR/uncertainty/error-analysis are promises | **DONE** | `dev_only_calibrate` fits a calibrator via internal grouped CV on **development data only** and reports raw-vs-calibrated ECE/log-loss for the headline models; crash-level (case/row) bootstrap CIs (under a cross-crash independence approximation) + per-class error slices regenerate from the bundle. |
| AA2-015 | S2 | Model descriptions inaccurate (Frank–Hall, weighting, seeds) | **DONE** | Frank-Hall renamed + note corrected ("NOT proportional odds"); true ordered logit added; `class_weight_mode="sample_weight"` for xgb/ebm applied via `clf__sample_weight` in CLI + recorded; seeds from config. |
| AA2-016 | S2 | Dense OHE memory growth | **DONE** | `build_preprocessor(sparse=True)` keeps OHE sparse (centre-free scaling); `dense_required` only for POC/EBM; CLI records encoded dims + peak memory. |
| AA2-017 | S1 | Run bundles overwriteable/partial | **DONE** | `write_bundle` creates atomically (temp dir → per-artifact SHA-256 in manifest → `os.replace`), refuses to overwrite an existing bundle, and writes a `STATUS` marker (RUNNING/COMPLETE/FAILED); run id is a content hash. |
| AA2-018 | S1 | Security/Django fixes undelivered; app may use bad pipeline | **DONE** | `research/APPLICATION_SCOPE.md` designates `crashsev` as the canonical research layer and the Django app as an explicitly out-of-scope legacy demo; security patches are documented, **not** claimed executed (need a PostGIS/Django harness). |
| AA2-019 | S1 | Tests don't support "proves correctness" | **DONE** | 40 tests across contracts (impostor/malformed-mapping aborts), split edges (excluded/dup-id/reconciliation/hash), metric contracts, ordered-logit gradient check, seed propagation, sparse OHE, immutable-bundle overwrite refusal, sentinel cleaning; the CI workflow is defined to run them on 2 OS × 3 Pythons; they pass locally only (Windows/Py 3.13: 40 passed), not on hosted CI (branch unpublished), and also pass under self-reproduction (same project; not an independent third party). Claims are "tested behaviours," not "proof." |
| AA2-020 | S1 | Leakage diagnostic conflates 3 factors | **DONE** | `crashsev/leakage_factorial.py`: 2×2×2 factorial × 5 seeds (dev-only), one factor at a time. Separated main effects on ordinal MAE: leakage features **−0.327** (dominant), pre-split preprocessing −0.005, random split −0.010 — no composite-gap attribution. |
| AA2-021 | S2 | Re-analysis assumes ordering; screenshots not packaged | **DONE** | `reanalysis/source_crops/` holds the 4 confusion-matrix images extracted from the report `.docx` (its SHA-256 **matches** the recorded provenance) with per-image hashes; every matrix cell is traceable docx→image→JSON; order-dependent claims qualified. |
| AA2-022 | S1 | Paper has false method–code equivalence / overbroad claims | **DONE** | `paper/final_paper.md` rewritten as a Route A empirical study driven by `experiment/final_results.json`; every headline number traces to a committed artifact; method text matches the code (e.g., factorial model corrected to "random forest"); no causal/national/deployment claims. |

## P2 — polish / delivery hygiene

| ID | Sev | Title (abbrev.) | Status | Resolving artifact / note |
|---|---|---|---|---|
| AA2-023 | S2 | PDF not reproducible; clipped captions; missing Figure 3 | **DONE** | `paper/render_paper.py` renders all 7 figures (incl. Figure 3), title/author/date + `<meta>` metadata, aspect-preserved frame-capped images (no overflow/clipping), and patches an xhtml2pdf 0.2.17 join bug; `err=0`, 12 pages, valid PDF (regenerates identically under self-reproduction — same project, not an independent third party). |
| AA2-024 | S2 | Citations unverified; literature thin | **DONE** | 12 references with DOIs/venues covering ordinal classification (Frank–Hall), severity review (Savolainen), leakage (Kaufman), calibration (Guo), proper scores (Cohen/Epstein), and the learners (Breiman/Chen–Guestrin/Nori); each supports a specific claim. |
| AA2-025 | S3 | Self-scores inflated | **DONE** | `research/ASSESSMENT.md` replaces scores with a demonstrated-vs-not-demonstrated split tied to executed evidence + a remaining-work list; no admissions-oriented language. |
| AA2-026 | S2 | No license, no CI evidence, caches included | **DONE** | `LICENSE` added (MIT for code, with an explicit **data carve-out**: the crash extract and per-crash derivatives are excluded and governed by `DATA_LICENSE_NOTE.md`). a CI workflow (`.github/workflows/crashsev-ci.yml`) is defined for the 2 OS × 3 Python matrix, and its steps (install lock → import → contract self-check → 40 tests) pass locally on Windows/Py 3.13 (not yet executed on Actions — this branch is unpushed). The clean release archive is built with `git archive`, so it contains **only tracked files** — no `__pycache__`, bytecode, `_local_data/`, or run bundles (verified). |

## Cross-cutting acceptance test (definition of done for the whole ledger)

A reviewer, given only the committed package and a licensed copy of the raw extract, can:
1. `pip install` the lock on a clean OS and import `crashsev`;
2. run `python -m crashsev.cli validate-data --data <xlsx>` and see an impostor rejected / the real file accepted;
3. run `develop` → `freeze-experiment` → `evaluate-final` and obtain **one** immutable bundle whose hashes match;
4. regenerate every number and figure in the paper from that bundle;
5. confirm no raw records, coordinates, or ids are committed anywhere.
