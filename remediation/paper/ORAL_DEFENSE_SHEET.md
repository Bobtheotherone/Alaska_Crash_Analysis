# Oral-defense sheet — likely faculty questions and defensible answers (r4, 2026-07-16; content unchanged in substance from r3 — the r4 pass changed availability/disclosure text and typesetting only, no governed result)

Each answer is grounded in a specific artifact or manuscript location; nothing here
goes beyond what the paper claims.

**1. Target semantics — "What exactly are you predicting?"**
A researcher-defined recorded-severity outcome: classes 0/1/2 mapped from the raw
`Crash Severity` field (blank→0, C/B→1, A/K→2), with undocumented labels quarantined,
never coerced. Blank constitutes *all* of class 0. The blank→PDO rule survives an
internal consistency check (explicit zero injury/fatality counts on all 32,046 blanks)
but that is not official semantics — the codebook request is drafted and the
limitation is stated wherever the target is used (§4.2; `target_reporting_process_audit.json`).
I predict what the report *records*, not clinically validated injury.

**2. Outcome leakage — "How do you know your predictors aren't the outcome?"**
Three structural controls: an explicit exclusion ledger (14 outcome-derived fields
prohibited everywhere — Appendix G lists each with its mechanism); train-only fitting
of every learned transformation inside scikit-learn pipelines; and structural isolation
of 2012 outcomes from the development command, enforced by a poison-token sentinel
test. What leakage-control does NOT mean is also stated: recording times of retained
fields are author-judged, not custodian-verified, and reporting-process signal remains
possible — a missingness-indicators-only probe beats the majority baseline slightly
(0.3583 vs 0.3614), which the paper reports against itself (§8.8, §11.5).

**3. Temporal validation — "Why rolling-origin and one held-out year?"**
The estimand is transfer to a later period, so folds respect time (train ≤2009 → val
2010; train ≤2010 → val 2011) and the final evaluation is the latest year. One year is
not stability evidence — that is limitation §11.2 and roadmap Phase 2.

**4. Why is 2012 "exposed" rather than prospective?**
The project possessed 2012 before the protocol existed. Software prevents the
development pipeline from *using* it, but no process can turn historically held data
into a sealed prospective holdout. Hence "retrospective, historically exposed," no
preregistration claim, and the H1 verdict is "met the protocol's numerical criterion
(descriptive)" rather than confirmatory (§5, §7.2, claim boundary).

**5. Why isn't the protocol-primary model the overall oMAE leader?**
Because eligibility was frozen by role before the final run: the unweighted forests
are matched ablations that exist to measure the cost of class weighting, so they are
excluded from the designation they explain. The unweighted RF achieved the lowest 2012
oMAE (0.3401 vs 0.3469) — the paper says so plainly (§8.3), and Appendix F publishes
every score, eligibility flag, and outcome. Choosing the ablation *after* seeing 2012
would be post-hoc selection — exactly what the protocol exists to prevent (§7.1).

**6. Effects of class weighting?**
Matched-pair evidence (H3): weighting buys severe recall 0.002→0.058 in the ordinal
forest at +0.005 oMAE and reduced severe precision; 0.000→0.018 at +0.036 in the
nominal forest. Weighting is a protocol role, not a stakeholder-validated utility
choice — no cost matrix exists (§8.6).

**7. Frank–Hall probability validity?**
K−1 binary exceedance models; monotonicity is enforced by construction
(cumulative-minimum repair, clipping), classes recovered by differencing, floored at
1e-9, rows renormalized; degenerate folds are skipped by the CV loop with a
constant-exceedance fallback in the estimator. Tests verify every released probability
vector is a valid distribution and that the posterior-median rule reproduces all
162,820 stored labels exactly; an independent recompute confirmed it (§6.2.1;
FH-CONV-001).

**8. Calibration?**
Fit on development data only over temporal folds; applied to 2012 for probability
quality only — never for headline labels or selection. Raw ECE 0.059 → 0.032
calibrated; the severe-class reliability diagram (Fig. 8) shows raw probabilities
running above empirical frequency in sparse mid-range bins while calibrated
probabilities track the low-probability bins holding nearly all crashes. The
calibrated-label counterfactual (0.3442, 4.2% recall) is reported and deliberately
unused (§8.4).

**9. Bootstrap assumptions?**
Paired percentile intervals over 2,000 crash-level resamples of fixed predictions
(seed 42; no refitting) — case-sampling variation conditional on the fitted models,
protocol, cohort, and target, under cross-crash independence. Post-hoc proxy-cluster
resampling (day/week/region/borough/maintenance) widens but never overturns the
interval; agency clustering is unquantified because agency fields were removed at
de-identification (§6.5, §8.8, §11.7).

**10. Severe-class performance — "5.8% recall is terrible, isn't it?"**
For detection, yes — the paper says it is not a severe-crash detector, in the abstract
and the claim boundary. The severe-class *score ranking* is nontrivial within-cohort
(AP 0.204 [0.169, 0.243] vs prevalence 0.039; AUROC 0.793), which motivates
prospective evaluation but selects no operating threshold — that requires a
stakeholder cost matrix (§9).

**11. Reproducibility vs replication?**
Four tiers, never conflated: metric recomputation from de-identified evidence (anyone;
demonstrated independently, 11/11); self-reproduction from licensed source (done by
the project, bit-exact); clean-room package verification (ships with its own
verifier); independent third-party replication (has NOT occurred — stated wherever
reproduction is mentioned) (§12.1).

**12. Your individual contribution?**
Sole author of Iteration IV: the target and feature-tier contracts, the frozen
protocol and role registry, the governed pipeline and its tests, the analyses, and the
manuscript. Iteration III (with Peyton Ratzer) is credited historically as the
platform this work evaluates; client and mentor roles are context, not endorsement
(roles page; §2).

**13. Role of AI assistance?**
Disclosed precisely: generative-AI tools assisted, under my direction, with code
scaffolding/editing, debugging, statistical and editorial critique, documentation,
typesetting, and package review. The research questions, contracts, protocol,
decisions, and responsibility are mine; every reported number is pinned to a frozen
artifact by machine verification (893 manuscript checks, 96 tests) rather than by
trust in any assistant. No AI system is an author; the external-AI privacy attestation
is an explicit author action (§12.4; `AI_USE_AND_PRIVACY.md`).

**Bonus — "Why did the result get smaller across iterations, and why is that good?"**
Because the question got harder and better specified: outcome-derived fields accounted
for by far the largest apparent inflation (−0.349 in the matched diagnostic);
correcting rule and tier roughly halved the margin (−0.0304 → −0.0144) and collapsed
severe assignment. Each protocol tightening made the result smaller and more
defensible; that judgment — not the number — is the contribution (§8.7, §10.2,
Conclusion).
