# Severe-class (class 2) one-vs-rest ranking analysis (v4.1; P0-INTERP-001)

Retrospective, exploratory ranking diagnostics computed deterministically from the frozen
2012 probability vectors — no retraining, no threshold selection (choosing an operating
threshold on the evaluation year would be post hoc, and no stakeholder cost function
exists). These numbers answer a DIFFERENT question than the hard-decision recall in the
main tables: *does the probability output order severe crashes above non-severe ones*,
not *does the ordinal-loss-optimal decision rule assign the severe class*.

No-skill reference (severe prevalence): **AP = 0.03869**. AUROC no-skill = 0.5.

| model | severe one-vs-rest AP | AP / prevalence | AUROC |
|---|---|---|---|
| proportional_odds | 0.22789 | 5.89× | 0.80368 |
| frank_hall_logistic | 0.21829 | 5.64× | 0.80162 |
| xgboost | 0.21584 | 5.58× | 0.79904 |
| multinomial_logistic | 0.21469 | 5.55× | 0.79451 |
| ordinal_random_forest_unweighted | 0.20639 | 5.33× | 0.79248 |
| ordinal_random_forest | 0.20377 | 5.27× | 0.79307 |
| random_forest_unweighted | 0.20125 | 5.20× | 0.78627 |
| random_forest | 0.19579 | 5.06× | 0.78926 |
| ebm | 0.16860 | 4.36× | 0.75235 |
| shallow_tree | 0.12616 | 3.26× | 0.71666 |
| decision_tree | 0.10851 | 2.80× | 0.67340 |
| majority | — (constant forecast) | — | — |
| ordinal_median | — (constant forecast) | — | — |
| prior_probability | — (constant forecast) | — | — |

**Reading.** The oMAE-leading forest models carry **nontrivial severe ranking signal**
(AP ≈ 5× the no-skill prevalence; AUROC ≈ 0.79) even though the posterior-median rule
almost never assigns class 2. Low hard-class recall under the ordinal-loss-optimal rule
therefore must NOT be read as an absence of predictive information about severe risk;
equally, this ranking signal must NOT be read as validated severe-crash detection — any
operating point would require a real cost function and evaluation on data not used here.

*Generated 2026-07-12T04:47:40Z by `crashsev/severe_ranking.py` from `final_8af9d5bc23d8`; aggregate only.*
