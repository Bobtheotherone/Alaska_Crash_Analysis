# Issue Disposition — current HEAD (portfolio finalization)

> **Update (v4 corrected-protocol execution, 2026-07-12).** The protocol items deferred from the
> third round are **executed** (see `RECON3_ISSUE_LEDGER.md`, v4 execution update): structural
> develop isolation, posterior-median primary rule, deterministic `prior_probability` baseline
> (closes **BASE-PROB-001** ✅), temporal calibration folds, fold-local typing, strict feature
> tier + separately frozen broad-tier sensitivity, token-aware invariant drop (**MISSING-001**
> now ✅-bounded: invariant fields dropped and the tier ablation run; a per-field missingness
> ablation beyond the tier family remains future work), per-model timings (closes
> **PERF-MEASURE-001** ✅), byte-exact bundles, PO convergence flagging, and the executed
> TARGET-001 sensitivity (**TARGET-SEM-001** residual now evidenced: blank=PDO *constitutes*
> class 0 — official codebook is construct-critical, Gate 3). Governed v4 result:
> Δ −0.0144 [−0.0231, −0.0063] vs majority; severe recall 0.058 (paper §7.3/§7.7/§7.8).

> **Update (third remediation round, 2026-07-11 — v3.1 truth reconciliation).** A repository-wide
> reconnaissance of the delivered `e06e803` handoff (SHA-verified — the genuine artifact) confirmed
> five S1 findings; every material claim was re-verified against source. This round corrects
> claims and artifacts only (no governed rerun; numbers unchanged): paper model-ranking and
> proper-score statements, the incomplete Table 2, NaN-vs-zero reanalysis regeneration, the
> generator/artifact/release provenance chain, factorial estimand + interactions, stale
> "sealed/provably/grouped-bootstrap" language including renderer-injected PDF captions, stale
> counts/config/CI triggers, and the missing archived empirical-run evidence (now committed as a
> cryptographic skeleton + de-identification witness). Protocol corrections (structural outcome
> isolation, decision rule, prior baseline, temporal calibration, strict feature tier) are the
> **v4 rerun**, tracked in **`RECON3_ISSUE_LEDGER.md`**. Several ✅ rows below were found stale or
> wrong in the third round and are corrected in place with change notes.

> **Update (second remediation round, 2026-07-10).** An independent reconnaissance of the `5be0fbf`
> handoff raised four contradicted major claims (development access to 2012, calibration selected
> from an old artifact, factorial attribution, weighted-vs-unweighted comparison). All were verified
> in source and corrected; the authoritative artifact was **regenerated from current code under a
> clean-tree freeze** (generator commit `e55d6f1` recorded *inside the artifact*; committed at
> `e17c263`; released at `e06e803` — an earlier version of this note said "generator == release",
> which wrongly collapsed that chain; see `PROVENANCE_CHAIN.md`. The prior `final_results.json` had
> been produced by an older commit `bd8d0428` predating the calibration fix). The primary comparison
> reproduced to the digit (ordinal RF 0.3310 vs majority 0.3614, interval unchanged). Per-issue
> detail in **`RECON_ISSUE_LEDGER.md`**; realigned claims in `paper/final_paper.md` §2–§8.


Dispositions the issue program of the finalization master prompt (§7) against the working tree
(current branch `portfolio-research-finalization-v4`; the machine-generated authoritative release
record is `research_audit/RELEASE_STATE.md`). An issue is **closed** only when its acceptance
criterion is evidenced, not because text changed.

**Note on the reconnaissance report.** The prompt names an input
`Alaska_Crash_Analysis_Current_HEAD_Reconnaissance_2026-07-10.md` (expected SHA-256
`ace8f1e8…d904`). **That file is not present** anywhere in the workspace, repo, or backup (searched
by name and by SHA). Its issues could not be mined or its hash verified. This ledger therefore uses
the prompt's own §7 issue program as the authoritative issue list. **Blocker recorded; non-fatal**
because §7 is self-contained.

Status key: ✅ resolved · 🟡 partial/residual · ⚪ out-of-scope or blocked (documented) · ⛔ external blocker.

## P0 — truth first

| ID | Status | Evidence & action | Residual |
|---|---|---|---|
| **EVAL-LOCK-001** | ✅ | 2012 relabeled **exposed, retrospective out-of-time** across paper, README, research docs, configs, and handoff; `ROUTE_DECISION.md` states Route R and why not Route C; governance kept but reframed as within-study. | Resolved v3.1: remaining "sealed"/"provably" code docstrings and the `__init__` "grouped bootstrap" line relabelled; the GOV-001 artifact-derivability caveat (2012 counts reconstructible by subtraction from the dev report) is now stated in code, paper §3.4/§10, `ROUTE_DECISION.md`, and `REPRODUCE.md` — structural isolation is the v4 fix. |
| **CAL-SELECT-001** | ✅ | Calibrator fit **development-only** (`dev_only_calibrate`, AA2-014). The two calibrated models (majority comparator + ordinal RF, the top **development** model, §7.2) were selected on development evidence before the 2012 evaluation; paper §7.4 states the rule; a guard test asserts final-result objects cannot select calibration. | — |
| **DATA-AUTH-001** | ✅ (bounded) | `DATA_AUTHORITY_AND_ACCESS.md` uses neutral "restricted project copy of unverified source-custodian provenance"; separates source-agency authority (unverified) / owner permission (granted) / possession / reproducibility access; governed content hash recorded. | Updated 2026-07-11: raw byte-hash **recorded** (value withheld from public release — restricted reproduction log; 25,391,525 B — the file was in the backup's `2025\` subdirectory; the earlier "absent" note scanned only the root and is corrected, not deleted). Raw→table→frozen-hash chain verified. Source-agency provenance still unverified (Gate 3, `PROVENANCE_ACQUISITION_PLAN.md`). |
| **PROV-EXT-001** | 🟡 | Local branch + annotated tag `portfolio-v3` created at finalization; full history shipped as a git bundle; provenance is content-hash immutable. No prospective-preregistration claim is made. | **Not pushed** (no authorization) → no external timestamp/witness; stated everywhere. |

## P1 — academic defensibility

| ID | Status | Evidence & action | Residual |
|---|---|---|---|
| **STAT-MULT-001** | ✅ | Paper §6 declares **one primary comparison** (ordinal RF vs majority, chosen on development); all other model-vs-baseline CIs are secondary/exploratory with no multiplicity claim; "significant/significantly" removed; effect given in intuitive units (~30 class-steps per 1,000 crashes); retrospective caution applied. | — |
| **REPRO-INDEP-001** | ✅ | `REPRODUCTION_LOG.md` and paper §11 relabel the re-run as **self-reproduction** (same project); no independent third-party reproduction claimed. | Independent reproduction remains to be done by a third party. |
| **AUTHOR-001** | ✅ | Author = **Naythan Mercado** on the paper title page; `AUTHORSHIP_AND_AI_ASSISTANCE.md` gives contribution + a broad AI-assistance disclosure + a responsibility checklist. | Applicant must complete the self-attestation checklist (external human action). |
| **PROTO-CONF-001** | ✅ | `PROTOCOL_CONFORMANCE.md` maps every protocol item to an execution status; unexecuted/unsupported items are removed or marked exploratory (e.g., EBM final-only; no H4 feature-stability claim). | — |
| **DECISION-LOSS-001** | ✅ | `experiment/decision_rule_sensitivity.md` compares argmax / posterior-median / expected-round on the frozen 2012 predictions; paper §7.6: **primary comparison robust** (ordinal RF 0.331/0.330), **secondary fragile** (RF ties baseline under median); probability-alignment/missing-class unit tests exist; decision-rule test added. | — |
| **UNC-DEP-001** | ✅ | Relabeled **crash-level (case/row) bootstrap under a cross-crash independence approximation**; effective cluster count = one row per crash (no within-crash clustering, so no cluster structure to exploit); stated at every interval and in §10; `uncertainty.py` docstring corrected. | Residual spatial/temporal dependence between crashes is unmodelled (no cluster ids in the extract) — disclosed. |
| **TARGET-SEM-001** | ✅ | "empirically verified codebook" → **evidence-calibrated/evidence-checked mapping**; raw target preserved and quarantined; row-flow by raw code documented; source-controlled hashed `target_mapping.yml`; blank=PDO framed as a source-specific coding decision, not official validation. | A bounded blank→PDO sensitivity is acknowledged but not separately rerun; v3.1 corrected the paper line that said it "is exercised" to "specified, not rerun"; the rerun is scheduled in the v4 benchmark (TARGET-001). |
| **MODEL-FAMILY-001** | ✅ | Fixed registry with declared roles (`models.py`); **EBM marked exploratory** (absent from development; `development_report.json`); availability frozen by protocol, not env-dependent. | — |
| **CLASS-WEIGHT-001** | ✅ | **Matched same-model ablation now run** (WGT-001): `random_forest` and `ordinal_random_forest` each run weighted and unweighted with identical seed/features/folds/2012 evaluation. Balanced weighting raises severe recall but lowers precision and *mildly worsens* ordinal MAE (plain RF 0.328→0.344; ordinal RF 0.331≈0.331); the unweighted plain RF has the lowest 2012 oMAE (0.328). See `experiment/final_results.json` (`*_unweighted`) and paper §7. | Mechanism now evidenced by matched controls, not cross-family inference. |
| **TEST-GOV-001** | ✅ | Existing failure-mode tests (leakage aliases, invalid target map, future-period contamination, group overlap, probability misalignment, seed propagation) plus **new** guard tests: calibration-selection cannot see final results; decision-rule alignment; exploratory-status metadata present. | Some governance checks remain non-automatable audit items (documented). |
| **PAPER-OVER-001** | ✅ (re-fixed v3.1) | The v3 scan covered `final_paper.md` only and **missed the rendered outputs**: `render_paper.py`'s figure-caption list injected "sealed 2012 … grouped-bootstrap" into the shipped HTML/PDF (PDF-001, third round). v3.1 corrects the captions and extends the prohibited-claim scan to the rendered HTML (`research_audit/claim_scan.py`, run at packaging). Three §7 superlative/status claims (dev ranking, proper scores, "confirmatory weight") were also corrected against the artifacts (RESULT-001). | Scan must always cover rendered outputs, not just sources. |
| **LICENSE-001** | ✅ | `remediation/LICENSE` (MIT, remediation code) + `DATA_LICENSE_NOTE.md` (data carve-out) + `DATA_AUTHORITY_AND_ACCESS.md`; legacy code rights attributed to the original group (`PRIOR_WORK_LINEAGE.md`); no data license implied by the code license. | Legacy third-party notices are minimal (legacy app out of scope). |

## P2 / P3

| ID | Status | Evidence & action | Residual |
|---|---|---|---|
| **BASE-PROB-001** | 🟡 | **Corrected (third round; BASE-001):** the earlier ✅ text here was wrong — `empirical_prior` is `DummyClassifier(strategy="stratified")`, i.e. *stochastic hard-label draws* from the training prior (final log loss 15.55 — one-hot draws), **not** a deterministic probability baseline. It is now labelled "baseline (random draws)" in Table 2; §7.4 notes the calibrated majority (log loss 0.747) approximates the honest prior forecast. | v4 adds a deterministic `strategy="prior"` probability baseline (`prior_probability`) and retires the misleading name. |
| **FACTORIAL-DOC-001** | ✅ (re-fixed v3.1) | The third round found the earlier ✅ premature (FAC-001): the v3 paper described *marginal* factorial means as effects varied "one at a time" and "cleanly separated", and the module docstring named the wrong fixed model. v3.1 rebuilds report+paper around **matched simple effects vs the reference** (−0.349 / −0.021 / −0.033) with **interactions** (+0.020 / +0.035 — the smaller effects vanish under leakage); marginals retained under `marginal_main_effects_NOT_one_at_a_time`; docstring corrected; the rerun reproduced all 8×3 cell means bit-identically. | — |
| **MISSING-001** | 🟡 | Numeric sentinels neutralised; 2 constant columns excluded; high-missingness fields flagged with informative-missingness note; `modeling_table_audit.md`. | A full missingness-by-year/class table and a high-missingness-feature ablation are bounded/partial. |
| **INTERP-H4-001** | ✅ | No stable-feature (H4) hypothesis is asserted; importances are explicitly predictive/non-causal and are not featured as findings. | — |
| **SPLIT-CONTRACT-001** | ✅ | `splits.py` now **actively enforces** earlier-only + contiguous development years — a future year or a gap year aborts the split (SPLIT-001) — plus complete row accounting and zero group overlap; negative tests (future-year, gap-year) in `test_governance.py` and `test_split_and_pipeline.py`. | — |
| **PERF-MEASURE-001** | 🟡 | **Corrected (third round; RESOURCE-001):** peak memory (`peak_memory_mb`) and encoded design-matrix dimensions ARE recorded, but the earlier claim that `model_cards` carry per-model fit/inference measures was wrong — they hold roles/notes only. | v4 rerun records per-model fit wall-time and batch-inference time on named hardware. |
| **CI-PUBLIC-001** | ⛔ | CI workflow defined; run **locally** (Windows/Py 3.13). Not run on hosted CI — **no push authorization**; no "green" claimed. | Requires push authorization to close. |
| **GITSTATE-STALE-001** | ✅ | `REPO_GIT_STATE.txt` and provenance records regenerated at finalization to agree on branch/commit/tag/dirty state. | — |
| **SEC-LEGACY-001** | 🟡 | **Fixed at source:** local OS usernames, a third-party contributor handle, and the exact filesystem path to the *restricted raw extract* were redacted from shipped code and docs (`%USERPROFILE%`, repo-relative `data/raw/`, standard `C:\OSGeo4W\bin`) — removing PII and a restricted-location leak. **Still disclosed (not removed):** legacy local-dev DB defaults (`alaska`/`postgres`) in the isolated legacy web app are trivial non-production values, out of scope for the science. | Legacy DB defaults left in place (parameterising the legacy app to env vars would destabilise unrelated code); disclosed instead. |
| **PDF-PUB-001** | ✅ | Professional PDF rendered from `final_paper.md` with correct author metadata, no placeholders, embedded figures; build log recorded. | Rendering stack is the project's `render_paper.py`. |

## Summary
All P0 and every substantive P1/P2 item is resolved or bounded with an explicit, disclosed
residual, **after the third-round corrections above** (four rows previously self-certified ✅ on
claims their artifacts contradicted — BASE-PROB-001, FACTORIAL-DOC-001, PAPER-OVER-001,
PERF-MEASURE-001 — are corrected in place with change notes; this ledger is itself evidence that
self-certification must be re-audited against artifacts, not prose). Protocol-level items
(structural outcome isolation, decision rule, prior baseline, temporal calibration, strict feature
tier) are open by design until the **v4 rerun** — see `RECON3_ISSUE_LEDGER.md` for the complete
third-round disposition. Hard external blockers: **PROV-EXT-001 / CI-PUBLIC-001** (push
authorization) and the Gate-3 evidence items in `PROVENANCE_ACQUISITION_PLAN.md`.
