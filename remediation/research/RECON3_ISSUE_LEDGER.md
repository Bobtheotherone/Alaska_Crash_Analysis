# Third- and Fourth-Round Reconnaissance — Verified Disposition Ledger

## Fourth round (v4.1, 2026-07-12): "Independent v4 Readiness Assessment" of `Handoff_4aa0751.zip`

Every material finding was re-verified against source/artifacts before any change (verification
notes inline). Verdict on the review: **substantially correct** — it caught two central
result-interpretation errors introduced in the v4 paper and a packaging regression; one of its
measurements (an ECE round-trip delta of ~8.6e-5) did **not** reproduce here (our text round-trip
gives 0.000e+00), but its underlying wording point was accepted anyway.

| Finding | Verified? | v4.1 action |
|---|---|---|
| **P0-RESULT-002** universal "every learned model beats the prior on every proper score" is false | ✅ (artifact: exactly the four RF variants qualify; XGBoost/EBM/linears/trees do not) | Corrected in abstract/§7.4/§9/§12/matrix/handoff docs with an explicit was-false note; `claim_scan` now **computes the beats-prior set from the artifact** and forbids the universal form |
| **P0-INTERP-001** "severe crashes essentially unpredictable" conflates hard decisions with ranking | ✅ (verified: severe one-vs-rest AP 0.17–0.23 vs 0.039 prevalence; AUROC 0.75–0.80 — reviewer's numbers reproduced) | New `crashsev/severe_ranking.py` + `experiment/severe_ranking.md` + Figure 8 (PR ranking curves, no threshold selected); abstract/§7.3/§8/§12 rewritten to hold hard-decision and ranking findings apart; "unpredictable" is now a forbidden scan term |
| **P0-GOV-002** "never loaded/read" still literally false (whole-table load + hash precedes partitioning) | ✅ | Minimal-resolution wording everywhere (paper §3.4/abstract/contributions, cli docstring, REPRODUCE): isolation is **semantic/analytic** (never validated/mapped/audited/serialized/fitted/selected), explicitly *not* byte-level non-access; "never loaded" forbidden by the scan |
| **P1-FEAT-002** "strict scene-observable" overstates author judgments | ✅ | Tier renamed throughout to **conservative, author-judged restricted tier** (ledger column name kept for artifact stability); §2/§3.3 note that the 12-200 form shows field membership, not recording time |
| **P1-TARGET-002** outcome is researcher-defined absent the codebook | ✅ | §3.2 now opens with "researcher-defined three-level ordinal outcome"; conclusion matches |
| **P1-STAT-002** CI conditional; "useful" language; seed/refit robustness missing | ✅ | §4 criterion relabelled "conditional difference" (explicitly not utility); §5.6 states full conditionality; **five-seed refit robustness executed** (`crashsev/seed_robustness.py` → `experiment/seed_robustness.md`, labelled a retrospective exploratory addendum) |
| **P1-MISS-002** high-missingness robustness untested | ✅ | `crashsev/missingness_summary.py` (missingness overall/by-year/by-class, dev rows only) + prespecified >50% rule (flags exactly Residence State ~74%, Direction ~64%) + **separately frozen low-missingness benchmark** (`configs/route_r_09_12_lowmiss_sensitivity.yml` → `experiment/lowmiss_sensitivity/`) |
| **P1-LIT-002** literature/hypotheses/recommendations below standard | ✅ | §2 "Positioning against current practice" (Roberts 2017; Gneiting & Raftery 2007; Van Calster 2019; the PMC8583475 ordinal-crash study; AK form 12-200 — refs 13–17); prespecified H1–H4 in §6 with adjudication in §7.8 (**H2 fails at final and is reported as failing**); new §13 Recommendations |
| **P1-DOC-002** control-doc drift (Route A refs, v3 branch, 54 tests, "not executed" ablation, "one at a time") | ✅ (each instance verified) | `PROTOCOL_CONFORMANCE.md` rewritten to v4.1; disposition/authorship branch refs fixed; REPRODUCE v3-era notes marked; cli "Route A" docstring relabelled; **generated `research_audit/RELEASE_STATE.md`** (commit/tags/config/live test count/runs/hashes/gate results) is now the authoritative record |
| **P1-PDF-002** Figure 2 labels overlap; Figure 4 labels small; no page numbers | ✅ (visually confirmed) | Fig 2: numbered markers + side key + deterministic collision offsets; Fig 4: horizontal log-scale bars with full-size labels + values; page-number footer added to the renderer; verified by direct image inspection |
| **EVID-001** "every metric and interval" too broad | ◐ (their 8.6e-5 ECE delta did not reproduce — our round-trip is exact — but the scoping point is right) | §11 states reproduction scope precisely (y_pred authoritative for hard metrics; probability metrics exact in our text-round-trip verification, witness-checked) |
| **RUN-001** synthetic run still shipped | ✅ (root cause found: a worktree-pathspec commit silently re-added the on-disk files over the staged `rm --cached`) | Properly untracked with a dedicated no-pathspec commit; `git ls-files remediation/runs/` is empty; cause documented in the commit message |
| **PO-002** dev-fold convergence not archived | ✅ | dev-CV trials now record per-fold `convergence_ok` for iterative models |
| **PRIV-001** sorted-order surrogates leak source ordering | ✅ (precautionary) | Witness transform now assigns surrogates under a **seeded permutation** (seed documented for bundle-holders); prior scheme noted in the witness |
| **PROV-002** REPO_GIT_STATE "DIRTY"; bundle named portfolio-v3 | ✅ (cause: gitignore matched only top-level experiment/ outputs) | gitignore broadened to `experiment/**/`; state file + bundle naming regenerated at v4.1 packaging |
| **AUTHOR-001** responsibility checklist unchecked | ✅ (human-only) | Remains the applicant's action; restated in the release notes and reviewer brief |
| "All three gates complete" overstatement | ✅ (fair hit — Gate 3 was only initiated) | Release materials now say Gate 3 is **initiated** (raw byte identity + chain verified) with the external asks explicitly open |

---

# Third-Round Reconnaissance — Verified Disposition Ledger (v3.1, 2026-07-11)

**Input.** "Repository-Wide Graduate Research Reconnaissance Report" (2026-07-11) auditing the
delivered `Alaska_Crash_Analysis_Portfolio_Handoff_e06e803.zip` (SHA-256 `7c3684c7…` — verified:
the genuine released artifact, unlike the 2026-07-10 stale-file review incident). The report found
**no S0** and **five S1** findings. Every material finding was independently re-verified against
source at `C:\aca` before any change. Verdict shorthand: **CONFIRMED** (reproduced against
source/artifacts), **CONFIRMED+** (confirmed and sharpened by our verification), **NUANCED**
(confirmed with a correction to the report's framing).

**Response tracks.** *T1 (v3.1)*: truth reconciliation — claims/artifacts corrected, no governed
rerun, all v3 numbers unchanged. *T2 (v4)*: protocol corrections requiring a re-freeze and rerun.
*G3*: external evidence only the applicant/custodian can supply
(`PROVENANCE_ACQUISITION_PLAN.md`).

> **v4 execution update (2026-07-12).** Every "T2" action below is **EXECUTED**: structural
> outcome isolation + dev-side-hash freeze (GOV-001; sentinel-tested), posterior-median primary
> rule (OBJ-001), deterministic `prior_probability` baseline (BASE-001), temporal calibration
> folds (CAL-001), fold-local typing (PREP-001), strict scene tier as primary + separately frozen
> broad-tier sensitivity (FEAT-001), token-aware invariant drop (MISS-001), PO convergence
> recorded/flagged (PO-001 — and it did NOT converge on the strict design, disclosed), byte-exact
> bundles (BUNDLE-CRLF-001), per-model timings (RESOURCE-001), and the TARGET-001 sensitivity
> (finding: blank→excluded collapses the estimand — blank=PDO *constitutes* class 0). Governed
> results: primary `final_8af9d5bc23d8` (Δ −0.0144 [−0.0231, −0.0063]; severe recall 0.058);
> broad sensitivity `final_10517423399d` (Δ −0.0287 [−0.0373, −0.0204]; severe recall 0.171).
> The corrections *shrank* the result, as §7.7 of the paper reports — that measured shrinkage is
> the remediation study's central finding. Two new self-found defects were fixed en route:
> the missing-token invariant rule and a `source_tree_dirty` first-porcelain-line parser bug in
> the governance gate (both regression-tested). Remaining open items are **G3-external only**.

## S1 findings

| ID | Verdict | Verification evidence | v3.1 action | Remaining (track) |
|---|---|---|---|---|
| **GOV-001** | **CONFIRMED+** | `prepare()` maps all years before splitting; `splits.py` computes final-test prevalence; and — sharper than the report — the dev report's whole-extract `target_audit.mapped_counts` {32046, 13047, 1751} minus dev-side counts recovers the 2012 outcome counts **exactly** {7877, 3303, 450}: the round-2 redaction is arithmetically vacuous. Fitting/selection matrices verified dev-only. | Claim accuracy: precise scope stated in `cli.py` docstring/comments, paper §3.4 + new §10 bullet, `ROUTE_DECISION.md`, `REPRODUCE.md`; "provably cannot read 2012" removed everywhere. | **T2:** structural isolation — develop loads/maps development years only; dev-only audit; sentinel + derivability tests. |
| **FEAT-001** | **CONFIRMED** | Ledger grants at-event status on one-line rationales (`Test Given` "administered at scene", `Insurance Coverage` "scene fact", sequence-of-events, damage-location, contributing circumstances); no form/workflow evidence anywhere in the repo. | Paper §10 bullet added; conclusion softened ("ledger's pre-event/at-event attributes … pending form-level evidence"). | **T2:** conservative strict tier as primary benchmark. **G3:** form-level recording-time evidence (potential S0 until resolved). |
| **DATA-PROV-001** | **CONFIRMED+** | Report right that no raw byte hash/custodian record existed; sharper: the repo itself recorded the raw file as *absent at finalization*. Our re-scan found it in the backup's `2025\` subdirectory. | Byte identity recorded (value withheld from public release — restricted reproduction log; 25,391,525 B); raw→modelling-table rebuild reproduces the committed CSV byte-identically and the frozen `data_sha256 = 059559cd…` exactly; docs updated (`DATA_AUTHORITY_AND_ACCESS.md` §2, `DATA_PROVENANCE.md`); acquisition plan written. | **G3:** custodian/source-agency attestation, extraction record, official codebook. |
| **ART-001** | **NUANCED** | True for the archive: zip `runs/` held only a git-tracked **synthetic** bundle; the real `final_f27613102c96` (STATUS=COMPLETE) exists locally with `FROZEN.lock`/`FINAL.done` (originally under `dist/regen/exp3`), and the committed aggregates are byte-identical to its freeze-time outputs. Not fabricated/lost — a packaging + claim-accuracy failure. | Committed the bundle's cryptographic skeleton at `evidence_release/final_f27613102c96/` (manifest, STATUS, FROZEN.lock, FINAL.done, de-id witness); untracked the stale synthetic bundle; paper §11/Appendix C now state the local-only privacy design and point at the committed skeleton. | **T2:** v4 rerun produces its bundle + skeleton in one pass. |
| **RESULT-001** | **CONFIRMED** | §7.2 "ordinal RF (0.315) and RF (0.339) led" false (dev leaders: unweighted controls 0.3108/0.3122); §7.4 "best proper scores" false across 14 models (unweighted ordinal RF log loss 0.668; unweighted RF Brier 0.409 / RPS 0.221); §11 "generated by this tagged commit itself" false (generator `e55d6f1` ≠ release `e06e803`); Table 2 "sorted by oMAE" omitted the two lowest-oMAE models and printed majority sev-precision 0.000 where the artifact says NaN; claim-matrix row 7 rubber-stamped the false superlative. Presentation inconsistency, **not concealment** — §7.6 disclosed the unweighted results. | All corrected in place with explicit "an earlier revision said X; that was wrong" notes: §6/§7.2/§7.3/§7.4/§7.6/§11/conclusion; Table 2 now carries all 14 models (4 d.p., "—" for undefined); claim-matrix rows 7/7b/21 rewritten. | Automated claim scan (`research_audit/claim_scan.py`) wired into packaging. |

## S2 findings

| ID | Verdict | v3.1 action | Remaining (track) |
|---|---|---|---|
| TARGET-001 | CONFIRMED (mapping evidence internal, not authoritative; the "exercised" sensitivity claim contradicted our own disposition ledger) | Paper line corrected to "specified, **not** rerun". | **T2:** blank→excluded sensitivity rerun. **G3:** official codebook. |
| MODEL-001 | CONFIRMED (presentation; ablation-role design intent verified, guard-tested) | Full 14-model table; role semantics explained in §7.2/§7.3. | — |
| OBJ-001 | CONFIRMED (argmax primary everywhere; median post-hoc only) | Already disclosed in §7.6; no v3.1 change (numbers frozen). | **T2:** posterior-median prespecified primary rule. |
| STAT-001 | CONFIRMED as disclosed limitation; residual defect was the "confirmatory weight" language | "Confirmatory" removed (§7.3, §7.6); interval scope language already accurate (§5.6/§10). | **T2:** seed/refit variability reporting in v4 rerun. |
| TEMP-001 | CONFIRMED (two rolling transitions; one exposed test year) | Already scoped in §10/limitations. | **G3:** later cohort for any transfer claim. |
| FAC-001 | CONFIRMED (marginals mislabelled "one at a time"/"cleanly separated"; docstring named wrong model; interactions material: preproc −0.021→−0.000, split −0.033→+0.002 under leak OFF→ON) | Report + paper rebuilt around matched simple effects (−0.349/−0.021/−0.033) + interactions (+0.020/+0.035); marginals kept under explicit label; docstring fixed; rerun reproduced all cells bit-identically. | — |
| REAN-001 | CONFIRMED by execution (current code → NaN; committed table had 0.2689/0.0000) | `reanalysis/` regenerated under METRIC-001 semantics; paper Table 1 majority row "—" with convention note; md renders "—", CSV keeps `nan`. | — |
| CAL-001 | CONFIRMED (GroupKFold over unique crash ids ≈ random folds; 10-bin top-label ECE dominant) | No v3.1 change (results frozen); classwise severe reliability already recorded. | **T2:** temporal calibration folds; calibrated proper scores. |
| BASE-001 | CONFIRMED (`empirical_prior` = stochastic `strategy="stratified"`; disposition + readiness docs called it "deterministic" — both corrected) | Table 2 role "baseline (random draws)"; §7.4 notes calibrated majority ≈ prior forecast (0.747); ledgers corrected. | **T2:** deterministic `strategy="prior"` baseline. |
| MISS-001 | CONFIRMED (Rural/Urban 100%-missing yet allowed; audit table matches report) | No v3.1 change (frozen inputs). | **T2:** auto-drop invariant/all-missing allowed fields; strict-tier + missingness ablation. |
| TRACE-001 | CONFIRMED+ (generator/artifact/release conflation; deid mapping discarded with no witness; **new sub-finding BUNDLE-CRLF-001**: v3.x `write_bundle` hashed LF content but materialised CRLF, so `artifact_sha256` matches files only after LF-normalisation) | `PROVENANCE_CHAIN.md` (full chain, both quirks disclosed); de-id transform witness generated and committed (original↔deid↔published hashes close, metric-equality checks pass, published copies byte-identical). | **T2:** `write_bundle` writes the exact bytes it hashes. |
| TEST-001 | CONFIRMED (docs said 40 / 46 / 54; suite itself is green — 54/54 verified locally; the recon's own run also 54) | `REPRODUCE.md` count fixed + marked packaging-generated; `tests_and_ci/` regenerated at packaging. | **T2:** governance property tests (outcome-isolation sentinel, artifact regeneration). |
| LOCK-001 | CONFIRMED (28 direct pins; "fully-pinned" overstated) | Paper §11 now says "direct-dependency level … not a fully resolved transitive lock — a documented limitation". | **T2/G3:** resolved lock or CI-validated constraints. |
| LIT-001 | CONFIRMED (12 references; missing ordinal-crash/proper-scoring/calibration/structured-validation anchors) | Deferred — literature additions belong with the v4 paper rewrite, not a truth pass. | **T2 (paper):** minimum additions per the recon §8. |
| PDF-001 | CONFIRMED (renderer `FIG_FILES` injected "sealed 2012 … grouped-bootstrap" captions into HTML/PDF — present ×5 in the shipped HTML — contradicting the clean PNG titles; clipping report accepted, layout audit added) | Captions synced to the paper's figure list; figure+caption kept together; PDF rebuilt; automated layout audit + rendered-output claim scan added. | — |

## S3/S4 findings

| ID | Verdict | v3.1 action / status |
|---|---|---|
| PREP-001 | CONFIRMED (typing/cardinality decided on all dev years pre-fold) | Disclosed; **T2** makes typing fold-local. |
| PO-001 | CONFIRMED (`optimizer_success_` stored, never surfaced) | **T2** records + flags convergence, excludes failed fits from superlatives. |
| CFG-001 | CONFIRMED (stale synthetic 2016-17 "final" config; Route A/R naming) | Config retired to `configs/archive/` with README; canonical config header carries the naming note. |
| DOC-001 | CONFIRMED (four ✅ ledger rows contradicted by artifacts; claim-matrix row 7) | All corrected in place with change notes; summary now warns against prose-based self-certification. |
| LEGACY-001 | CONFIRMED (root README foregrounds the Django/React app) | Root README gains a research-first pointer at packaging; deeper restructuring deferred (out of scientific scope). |
| ERR-001 | CONFIRMED (aggregate error analysis, no uncertainty/strata) | **T2:** prespecified strata + intervals in v4 rerun. |
| RESOURCE-001 | CONFIRMED (no per-model timings; a ledger row overclaimed them) | Ledger corrected; **T2** records timings. |
| CI-001 | CONFIRMED (push filter listed only the old v2 branch) | Workflow triggers now include the v3/v4 release branches; hosted run still needs push authorization (**G3**). |
| TERM-001 | CONFIRMED (cli "provably", `__init__` "grouped bootstrap", REPRODUCE "cannot see it", route naming) | All relabelled; terminology enforced by the claim scan. |
| PDF-A11Y-001 | ACCEPTED (untagged PDF, base-14 fonts not embedded — consistent with the xhtml2pdf stack) | Deferred to the v4 paper build (P3). |

## New findings from this round's own verification (not in the recon)

| ID | Finding | Action |
|---|---|---|
| **BUNDLE-CRLF-001** | `write_bundle` hashes in-memory LF content but `write_text` materialises CRLF on Windows → manifest `artifact_sha256` never matched the on-disk bundle files byte-for-byte (matches after LF-normalisation). | Disclosed in `DEID_TRANSFORM.md` + `PROVENANCE_CHAIN.md`; **T2** fixes the writer to emit the exact bytes it hashes. |
| **RAW-LOCATION-001** | The finalization-time "raw file absent" record was a scan-depth error (file present in `OLD_BACKUP\2025\`). | Corrected in `DATA_AUTHORITY_AND_ACCESS.md` §2 with the original statement preserved as history. |

## Where the reconnaissance needed correction (for the record)

1. **ART-001 framing**: the empirical run and freeze chain exist and verify locally — the defect is
   archive packaging + claim accuracy, not a missing/fabricated run (the recon's own open question
   "does a complete real bundle exist outside the handoff?" — answer: yes, verified).
2. **RESULT-001 framing**: the unweighted controls were deliberately non-competing ablations with a
   guard test, and §7.6 disclosed their superiority — internally inconsistent presentation and
   false ranking language, not concealment.
3. **FAC-001 model naming**: the *paper* named the fixed model correctly (balanced random forest);
   the wrong name was in the module docstring only.
4. Everything else checked out as cited; the recon's line references were accurate against the
   delivered artifact.
