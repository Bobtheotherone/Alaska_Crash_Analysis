# Protocol Deviations (retrospective amendments)

Every change made to the analysis or its description **after** the earlier v2 run, recorded honestly
because the 2012 outcomes were already exposed. None of these is a preregistration; all are
retrospective amendments to how the existing evidence is framed and analysed.

| # | Deviation | Nature | Why it is not result-shopping |
|---|---|---|---|
| 1 | 2012 relabeled from "sealed one-shot final test" to **exposed, retrospective out-of-time evaluation** | Framing / evidentiary status | Downgrades the claim; the numbers are unchanged. It removes overclaim, it does not manufacture a result (`ROUTE_DECISION.md`, EVAL-LOCK-001). |
| 2 | Decision-rule sensitivity (argmax vs posterior-median) added | New post-hoc analysis | Deterministic re-derivation from the **already-frozen** 2012 probability vectors; no retraining, no re-selection. It could only have weakened the headline; it happens to confirm the primary comparison and reveal the secondary as fragile (§7.6). |
| 3 | EBM demoted from peer to **exploratory (final-only)** | Scope correction | EBM was absent from development; treating it as a confirmatory peer would have been the error. Demoting it is more conservative. |
| 4 | Calibration-selection rule made explicit (development-only; comparator + top-dev model) | Documentation + guard test | The fit was already development-only; this documents the selection and adds a test forbidding final-result-driven selection (CAL-SELECT-001). |
| 5 | "grouped/dependence-aware bootstrap" relabeled **crash-level bootstrap under an independence approximation** | Framing correction | The unit is the crash (one row each); there was never within-crash grouping to exploit. The relabel is more accurate and more modest (UNC-DEP-001). |
| 6 | "empirically verified codebook" relabeled **evidence-calibrated mapping** | Framing correction | The 0.0%-contamination evidence is unchanged; the word "verified" overstated it relative to an official codebook (TARGET-SEM-001). |
| 7 | One **primary comparison** declared; others secondary/exploratory; "significant" removed | Inferential discipline | Reduces multiplicity and removes casual significance language; narrows, not widens, the confirmatory claim (STAT-MULT-001). |
| 8 | Reproduction relabeled **self-reproduction** | Framing correction | No third party re-ran it; the category was wrong (REPRO-INDEP-001). |
| 9 | Author set to Naythan Mercado; AI assistance disclosed | Authorship | Truthful attribution and disclosure (AUTHOR-001). |

**Rule applied.** Because these are retrospective, no deviation is described as preregistered, and any
analysis touched after 2012 was on disk is treated as an amendment. The corrected 2012 numbers (if a
future rerun is performed) must be written to a new `retrospective_corrected_<ts>_<commit>` namespace
and must **not** overwrite the historical `runs/final_17f99a4c8f94/` bundle.
