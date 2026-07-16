# Final-year seed/refit robustness — forest variants, five seeds (v4.1 addendum)

Retrospective, exploratory robustness analysis (reads the final year; refits models; is
NOT part of the frozen one-shot benchmark and modifies no frozen artifact). Same cohort,
tier (`strict`), preprocessing, and decision rule (`posterior_median`) as the primary run; only the estimator seed varies (seeds [1, 2, 3, 4, 5]). Majority-baseline ordinal MAE = 0.3614 (deterministic).

| model | per-seed oMAE | mean | sd | range | Δ vs majority (all seeds) | direction persists |
|---|---|---|---|---|---|---|
| random_forest_unweighted | 0.3405, 0.3393, 0.3408, 0.3390, 0.3391 | 0.3397 | 0.0008 | 0.0019 | -0.0224 … -0.0206 | yes (better than baseline) |
| ordinal_random_forest_unweighted | 0.3409, 0.3390, 0.3394, 0.3392, 0.3400 | 0.3397 | 0.0007 | 0.0019 | -0.0224 … -0.0205 | yes (better than baseline) |
| ordinal_random_forest | 0.3470, 0.3468, 0.3446, 0.3495, 0.3506 | 0.3477 | 0.0021 | 0.0060 | -0.0168 … -0.0107 | yes (better than baseline) |
| random_forest | 0.3789, 0.3732, 0.3745, 0.3776, 0.3765 | 0.3762 | 0.0021 | 0.0058 | +0.0118 … +0.0175 | no (worse than baseline) |

**Reading.** The frozen primary run used seed 42; these five refits show how much of the
reported margins is training-noise. A 'yes' in the last column means every refit lands on
the same side of the majority baseline as the published result. Seed variability adds to,
and is not captured by, the case-bootstrap interval — which remains conditional on one
fitted model.

*Generated 2026-07-12T05:23:49Z by `crashsev/seed_robustness.py`; aggregates only.*
