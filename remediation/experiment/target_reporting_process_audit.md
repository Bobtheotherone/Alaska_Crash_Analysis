# Target & reporting-process audit (TARGET-REP-001; post hoc, v4.1 revision)

POST HOC (v4.1 revision addendum): retrospective analysis of the historically exposed cohort; NOT prespecified; NOT part of the frozen one-shot benchmark; no frozen artifact is modified and no model role changes.

| question | finding |
|---|---|
| 1. class-0 provenance | ALL 32,046 class-0 records arise from the blank->PDO mapping; the raw field has no explicit PDO value |
| 2. count fields among blanks | explicit zeros for 32,046/32,046 rows on all three count fields; zero positives; zero missing |
| 3. person-level injury among blanks | missing/unknown-token for ~93%+ of blank rows (co-missing) |
| 4. quarantined rows | same all-zero count signature as blanks (evidence is not specific to blanks) |
| 5. count-field consistency | 'Injuries with Fatalities' positive for 1,511/1,512 zero-fatality incapacitating crashes; column duplicated/mislabelled at source |
| 6. blank share by year | stable: 2009: 0.629, 2010: 0.634, 2011: 0.645, 2012: 0.628 |
| 8. retained-field missingness | outcome-correlated for several retained fields (top contrast: Unit 1 Person 1 Gender +0.231) |
| 9. missingness-only probe | logistic oMAE 0.3583, RF oMAE 0.3641 vs majority 0.3614 (deltas and CIs in the JSON) |

**Interpretation rule.** Explicit zero counts weaken the simple shared-missingness explanation for those count fields; they do not establish official blank semantics. The same zero signature in quarantined labels weakens the specificity of the evidence. Reporting-process associations remain possible even if the missingness-only probe fails. Blank severity is NOT called officially equivalent to PDO anywhere in this study; the custodian codebook (Gate 3) remains the outstanding construct evidence.

*Generated 2026-07-15T19:55:01Z by `crashsev/target_reporting_audit.py` (commit 42194a61); aggregates only (probe predictions are surrogate-keyed).*
