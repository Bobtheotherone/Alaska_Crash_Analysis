# Revision Summary — r3 submission-readiness revision (2026-07-15, late evening)

Baseline: the locked correction release (PDF `294b8cd7…`, 43 pp, repo `7fd8584` on
`portfolio-final-correction`, tag `portfolio-final-r2`). This pass executes the
eleven-phase external review directive after verifying every reviewer claim against the
source and artifacts (`DISCREPANCY_LEDGER_R3.md`). **No frozen empirical result
changed**: run `final_8af9d5bc23d8`, its predictions, bootstrap witness, configurations,
and every governed artifact are byte-identical; nothing was retrained, retuned,
relabeled, or recalibrated. Every edit carries an ID in `CORRECTION_LEDGER.json`
(section `submission_readiness_2026_07_15_r3`, 25 entries, L-/R3-INF- numbering keyed
to the discrepancy ledger).

## r3.1 Technical corrections (Phase 2)

- **F1 convention (L-01):** the Appendix B claim that a never-predicting model's
  macro-F1 / severe F1 are "undefined" was wrong under the standard count-based
  definition (F1 = 2TP/(2TP+FP+FN) = 0 when TP=0, FN>0). §6.4 now states the
  convention (precision undefined iff TP+FP=0; recall/F1 measured zeros; macro-F1
  averages including zeros); Table 6 majority macro-F1 **0.269**, Table 7 majority
  severe F1 **0.000** (regenerated from the transcribed matrices); `metrics.py` gains a
  `zero_division` parameter whose frozen `'nan'` STORAGE default keeps pipeline reruns
  byte-identical; regression tests cover a majority-only classifier under both
  conventions. This does **not** alter the governed 2012 oMAE, the primary model, the
  primary interval, or the headline conclusion.
- **Complete tables (L-03):** Appendix A is now generated from
  `experiment/final_results.json` and actually contains every §6.4 hard-label metric
  (macro-F1, within-one, two-step, per-class precision/recall, severe F1 added); every
  value recomputed from stored confusion matrices and asserted against stored fields.
- **Frank–Hall construction (L-04):** new §6.2.1 documents the exceedance
  decomposition, cumulative-minimum monotonicity repair, differencing, 1e-9 floor, row
  renormalization, and degenerate-fold handling; validity machine-verified over all
  162,820 released rows. The reviewer's "no repair exists" contingency does not apply.
- **Bootstrap specification (L-05):** §6.5 now states paired 95% *percentile*
  intervals, 2.5/97.5 endpoints with linear interpolation, seed 42, same resampled
  crash sets for both models, **no refitting**, conditioning, and the impossibility of
  degenerate resamples. F1/bootstrap/Frank–Hall blocks were added to the
  machine-extracted `experiment/metric_conventions.json` with pin tests.

## r3.2 Selection transparency (Phase 3)

New §7.1 answers eligibility, exclusions-by-role, metric, fold/seed aggregation
(unweighted trial mean; folds equal-weighted, never pooled), tie-break (stable min in
frozen registry order; the three trivial baselines tied at 0.3465 and majority won by
frozen list order), the all-development refit at seed 42, and no-search. New generated
**Appendix F** publishes the complete per-fold/per-seed development record with
eligibility and outcomes. Results §8.3 and Discussion §10.1 state plainly: *the
unweighted random forest achieved the lowest 2012 oMAE; the weighted ordinal forest
retained the primary designation because roles were frozen before the final execution.*

## r3.3 Standalone auditability and governance (Phases 4-5)

- New §12.2 **Code, data, and evidence availability** with the 10-row reproducibility
  manifest, verification commands, license (MIT; holder placeholder in `LICENSE`
  completed), raw-data status, and visible `[AUTHOR ACTION REQUIRED]` markers for the
  public repository URL, DOI, raw-hash publication permission, and reproduction
  contact — none invented.
- New generated **Appendix G**: all 49 retained fields (representation,
  per-development-year missingness, thoroughness flags), the 14 prohibited
  outcome-derived fields, and the disposition of every remaining column.
- New generated **year-by-year cohort table** (`tab:cohort-year`), reconciled against
  the frozen target audits and q6 audit shares.
- §4.2 coarsening sentence corrected (L-09): the raw field **does** distinguish A/K and
  B/C; the coarsening is analytical, not a claim of absent finer labels.

## r3.4 Attribution, disclosure, abstract, terminology (Phases 6-8)

Neutral contribution statement replaces "the role ordering above is intentional"; a
single front-matter **Claim boundary** paragraph now bounds the paper once. The AI
disclosure separates ownership, assisted activities, verification, and responsibility,
adds the machine-checked release-side privacy fact, and routes the external-AI privacy
attestation to the author via `paper/AI_USE_AND_PRIVACY.md` (marker in text). The
abstract is 343 words (within the 275--350 target) with every required number and boundary. "severe-risk ranking"
→ "severe-class score ranking" everywhere (incl. Figure 10); defensive phrases
neutralized; §5's duplicate leakage-controlled definition now cross-references the
introduction and claim boundary.

## r3.5 Structure, figures, literature (Phases 9-10)

Recommendations now precede the Conclusion; the closing paragraph adds the
methodological-judgment-as-readiness framing without weakening any limitation. New
**Figure 8** severe-class reliability diagram (calibrated points from the frozen run
artifact; raw curve recomputed from released probabilities under the same 10-bin rule;
counts annotated); ECE figure retained. Figures 6/7/10 (r2 numbering; r3 Figures
6/7/11 — the new reliability diagram is Figure 8) regenerated with larger
labels/legends — numerical content unchanged. Title **retained** (matches delivered r2
handoff and application correspondence). Focused-review sentence replaces the
"no systematic literature review" disclaimer; **no citation added** (the one material
candidate could not be full-text inspected through accessible channels).

## r3.6 Verification results (this build)

pytest **96/96** · manuscript verifier **893/893** (tex+PDF; input-expanded) ·
frozen/locked **34/34** · audit_paper **42 pass / 0 warn / 0 fail** (11 figures, 17
LoT entries, 56 pp, margins/blank/fonts checked) · all **8** `gen_*.py --check`
regeneration gates PASS · build 0 errors / 0 overfull / 0 infinite-glue / 0 undefined
references · **independent recompute 11/11 PASS** (own-implementation recomputation of
every headline metric from `predictions_lossless.parquet` alone; two 1-ulp
float-accumulation artifacts documented, no substantive discrepancy) · redline
`redline_r2_to_r3.pdf` compiled from `latexdiff --flatten`.

## r3.7 Authorized numeric-token changes (whole-PDF multiset diff vs `294b8cd7`)

From the captured token diff (43 → 56 pages). **Net-removed (7 distinct):** `2025`
(one occurrence — the deleted roles-page sentence "credited for the 2025 capstone
work"; the Iteration III year still appears 5 times) and `21, 25, 30, 31, 32, 34`
(front-matter/ToC page-number tokens under the new pagination). **Added (325
distinct),** all accounted for by: (a) the generated development record (fold/seed
means and SDs, e.g. `0.3564, 0.3366, 0.3781, 0.4844, 0.6652…`, seeds `1,2,3`, `42`);
(b) the complete hard-label detail table (macro-F1 incl. the corrected `0.269` and
`0.330`, within-one/two-step values, per-class precision/recall, severe F1 `0.104`,
`0.004`); (c) the year-by-year cohort table (per-year source/mapped/quarantined/class
counts: `12,862, 12,459, 12,682, 12,540, 11,920, 11,590, 11,704, 942, 869, 978, 910,
8,086, 7,902, 8,181, 24,169, 9,744, 1,301, 2,789, 38,003…` and shares `3.6, 3.9`);
(d) the retained-field missingness columns and exclusion counts; (e) the reliability
diagram (bin counts `11,148, 295, 86, 42, 28, 22, 8,706, 1,958, 635, 214, 67, 33, 15,
5, 4, 2` — `8,7061,958` in the raw diff is a text-extraction concatenation of the two
adjacent figure annotations `8,706` and `1,958`); (f) the bootstrap specification
(`95, 2.5, 97.5`); (g) new subsection cross-references (`6.2, 6.4, 6.5, 7.1, 7.2,
12.2, 12.4`) and ToC/LoF/LoT page numbers up to `56`. **No governed value changed; no
unexplained token remains.**

---

# Revision Summary — locked post-audit correction release (2026-07-15)

Baseline: the audited final-submission build (PDF `c94ba2b6…`, 39 pp, repo `6ef7906` on
`portfolio-final-submission`, tag `portfolio-final`; AXIOM-LOCAL verdict **accept**,
with residual minor findings). This pass applies the smallest complete set of
manuscript, package, and verifier corrections identified by the two independent audits
**without changing the frozen empirical result**: run `final_8af9d5bc23d8`, its
predictions, bootstrap witness, configurations, and every governed artifact are
byte-identical; nothing was retrained, retuned, relabeled, or recalibrated. Every edit
carries an ID in `CORRECTION_LEDGER.json` (section `locked_revision_2026_07_15`,
49 entries).

## 1. Numerical corrections

**None.** The single authorized numerical manuscript correction (historical
decision-tree macro-F1 0.452 → 0.451) had already been applied by the completion pass
(CORR-1) and is re-verified in this release: 0.451 appears in the structured cell
artifact, the generated table, the PDF, and the verifier expectation; the stale 0.452
survives only inside correction-history text.

## 2. Wording and scope changes (manuscript; ledger IDs E-001…E-030)

- **Title** names the researcher-defined outcome (title page, PDF metadata, README,
  claim matrix): *…Ordinal Classification of a Researcher-Defined Alaska
  Crash-Severity Outcome…* (E-001).
- **"Leakage-controlled" defined** with the custodian clause in the abstract,
  introduction, and methods (E-002); research question rewritten around excluded
  outcome descendants and author-judged timing (E-003).
- **Confirmatory language removed** — the §8.3 "statistically supported" residue (an
  audit finding) replaced with the observed/descriptive framing; H1 verdict is now "Met
  the protocol's numerical criterion (descriptive)" (E-004–E-006, E-023).
- **Target semantics**: all 32,046 class-0 records are blank-derived (63.4% of source
  rows, 68.4% of mapped rows); the blank→PDO evidence is a source-specific internal
  consistency check, not independent semantic validation; the quarantined
  Unknown/Not Reported/Null value zero-count signature, the 93.4% person-level
  missingness, and the duplicated/mislabelled count field are stated with
  machine-verified values (E-007–E-009).
- **Timing**: no operational prediction timestamp validated; retained fields author
  judged, not verified simultaneously available and unrevised (E-011).
- **Model role**: "protocol-designated weighted candidate" used consistently; the
  designation is a frozen-registry role, not oMAE leadership (development oMAE leader:
  unweighted ordinal forest 0.3233; 2012 leader: unweighted random forest 0.3401);
  weighting stated as a protocol role, not an optimal compromise (E-013–E-014).
- **Posterior median** qualified (Bayes optimal for meaningful posteriors; loss-matched
  otherwise); headline-label reconstruction quantified: 162,820 labels, zero mismatches
  (E-015). Calibrated-label counterfactual quantified from the artifact: 1,894 of
  11,630 labels, oMAE 0.3469→0.3442, severe recall 4.2% vs frozen 5.8%; post hoc and
  unused (E-016).
- **Metric conventions stated**: unnormalized RPS with the formula and the ×2
  relationship to the /(K−1) convention; log-loss clip ε = 1e-15 (machine-extracted)
  with row renormalization; top-label ECE with 10 equal-width bins (E-017–E-019).
- **Leakage factorial** reworded as conditional diagnostic contrasts, not causal
  effects (E-020); "one governed run" scoped to the frozen primary execution with every
  later analysis pointed at the status ledger (E-021).
- **Dependence/rare-class analyses** carry resampling units, cluster counts
  (366/53/4/18/9), positive counts (450), unchanged-verdict statements, and the
  no-semantics/no-transfer scope sentence (E-024); ranking language tightened (E-025).
- **Conclusion** adds the not-established list (independent replication, validated
  physical injury severity, operational feature availability, transportability,
  deployment utility) (E-026). **Reproduction tiers** defined as controlled terms
  (E-027); release-content wording per spec with the parquet named authoritative
  (E-028). A **positioning subsection** contrasts the governed protocol with the
  project's own prior iterations using only locally verifiable propositions (E-029).
- **New Appendix E**: the complete 2012 analysis-status ledger (24 analyses; statuses
  restricted to protocol-primary / protocol-secondary / exploratory-final-only /
  post-hoc), generated from the same machine-readable CSV (E-022).

## 3. Package repairs (R-002, R-011, R-012, PKG-1…3)

- **Hard-coded machine paths are gone, and the claim is now test-backed** (this
  statement was previously made while `verify_frozen_and_locked.py` still pinned
  `C:\aca`): the acceptance verifier resolves its root from `--root` /
  `CRASHSEV_REMEDIATION` / its own location, confines every inspected path to the
  target package, git-checks only the target package (explicit SKIP when it is not a
  checkout, or when the nearest checkout is an enclosing repository), and reports
  explicit SKIPs for the local-only run bundle and licensed table.
  `tests/test_verifier_paths.py` adds a static drive-letter scan over all shipped
  scripts and a decoy-`aca`-checkout regression test proving the clean-room verifier
  ignores unrelated checkouts. `make_figures.py` and `audit_paper.py` defaults are now
  location-derived.
- The repository README was stale at the superseded v3 headline (0.331, Δ−0.030,
  "prespecified primary comparison", 54 tests) — modernized to the frozen v4 governed
  result and the corrected title; the markdown draft paper carries a SUPERSEDED banner
  (PKG-1, PKG-2). The unresolved-dependency list now includes the prediction-moment
  phrase and departmental template approval (PKG-3).
- The handoff additionally ships the `c94ba2b6` audit report, the captured-output
  evidence (`gate_outputs.json`, `cleanroom_results.json`, `rebuild_identity.json`,
  `pdf_token_diff.json`, `metric_conventions.json`), and extracts/verifies at a short
  path; Windows `--basetemp C:\tmp\aca-tests` guidance documents how MAX_PATH/ACL
  failures differ from scientific failures (R-011, R-012).

## 4. Verifier and report repairs (R-001, R-003…R-010, D-001…D-004)

- Scientific-language gate extended (case-insensitive; `statistically supported`,
  `statistically significant`, role-language, agency misstatements; context-qualified
  check for unqualified "independent replication" over LaTeX **and** extracted PDF
  text; failures name line/page); the gate prints the exact enforced list and
  `FINAL_VERIFICATION_REPORT.md` quotes it — the report is now **generated from
  captured gate output only** (`tools/gen_final_verification_report.py`) and marks
  uncaptured sections as such instead of asserting them (R-001, D-001).
- Metric-convention sidecar `experiment/metric_conventions.json` (machine-extracted:
  unnormalized RPS, alternative divisor 2, log-loss ε, ECE binning) with regression
  tests that recompute governed values from the lossless parquet and fail on any silent
  convention switch. Recorded as a sidecar because the frozen probability artifacts are
  preserved byte-for-byte (R-007, R-008).
- Bootstrap evidence tiers stated verbatim in the claim matrix, READMEs, and report
  (R-003); parquet-authoritative + default-parser regression tests (R-004);
  environment-specific counts reported separately, never merged (R-005); release-copy
  identity and source-rebuild identity reported as two different checks with the
  toolchain named (R-006); positional table verification retained and extended to the
  new appendix table — manuscript verification now 548 checks (R-009); the 0.451 chain
  re-verified (R-010). Claim matrix revised per D-003; this summary and the ledger
  follow D-002/D-004.

## 5. Post-hoc analyses

**None added.** The only new artifact is documentation (the metric-convention sidecar
and the regenerated status ledger); no new interrogation of the 2012 cohort occurred.

## 6. Authorized numeric-token changes (whole-PDF multiset diff vs `c94ba2b6`)

From the captured `pdf_token_diff.json` (39 → 43 pages). **Removed** (6 distinct):
`17, 20, 26, 28, 29` (front-matter page-number tokens shifted by the relayout) and one
occurrence of `95` (the abstract now states the descriptive interval per the E-005
wording without the "95% CI" macro; the 95% level remains defined in §8.3). **Added**
(84 distinct), all accounted for by: (a) the spec-mandated new values — 63.4, 68.4,
93.4, 4.2, 0.3442, 1,894, 162,820, 1,511, 366, 53, and the RPS/ECE convention tokens
(0.5, 2, 10, 15); (b) additional occurrences of existing governed values quoted in the
new sentences (0.3233, 0.3401, 0.3469, 5.8, 32,046, 46,844, 50,543, 49, 450, 2,000,
11,630, 42, 3.7, 1,512); (c) navigation artifacts — table-of-contents/list page
numbers under the new pagination, section cross-references (§4.x/6.x/8.x/12.1) and
run-identifier digit fragments (`10517423…`, `…612…`, `…030…`, `…399`) printed by the
new Appendix E table, and one extra occurrence of six citation years (1968–2021) from
the positioning subsection. **No governed value changed; no unexplained token remains.**

## 7. Unresolved external evidence (no local action possible)

Unchanged in kind and explicitly not faked: official blank-severity semantics;
custodian-verified per-field recording times and a real prediction moment; a genuinely
later unexposed cohort; a stakeholder utility/cost function and prospective threshold;
end-to-end third-party replication with the licensed source; departmental template
approval if required. See `unresolved_external_evidence.md` and
`custodian_semantics_request.md` (ready to send).
