# Target-mapping sensitivity — blank = PDO vs blank = excluded (TARGET-001, v4)

Development years ONLY (the held-out final year is never touched). Rolling-origin folds,
prespecified decision rule `posterior_median`, feature tier `strict`, seed 42, compact model set.

| mapping | n usable (dev) | classes present | prevalence 0/1/2 | majority | ordinal_median | prior_probability | random_forest_unweighted | ordinal_random_forest |
|---|---|---|---|---|---|---|---|---|
| primary_blank_is_PDO | 35,214 | [0, 1, 2] | 0.6863/0.2767/0.0369 | 0.3465 | 0.3465 | 0.3465 | 0.3233 | 0.3374 |
| alternative_blank_excluded | 11,045 | [1, 2] | 0.0/0.8822/0.1178 | — | — | — | — | — |

**Finding — the blank policy CONSTITUTES class 0 in this extract.** Under the
fail-closed alternative (blank → excluded), the usable development cohort keeps only
11,045 rows with classes [1, 2] present (prevalence 0.0/0.8822/0.1178):
**class 0 (none/PDO) exists in this extract almost entirely through the blank→PDO
inference**, so the alternative mapping does not yield a comparable 3-class study —
it changes the estimand itself (a 2-class minor-vs-severe problem on a third of the
rows). The prespecified 'direction check' is therefore **not computable**, and the
correct conclusion is stronger than a robustness pass or fail: the blank=PDO decision
is not a marginal coding choice but the definition of the majority class, which makes
obtaining the official Alaska codebook (Gate 3, `PROVENANCE_ACQUISITION_PLAN.md`)
material to the study's construct validity — exactly as the internal evidence check
(0.0% injury/fatality contamination of blanks) already suggested, now with the
consequence quantified.

*Generated 2026-07-12T01:16:03Z by `crashsev/target_sensitivity.py`; development data only; no per-crash rows.*
