# Severe-class metric uncertainty — frozen primary model (SEV-UNC-001; post hoc, v4.1 revision)

POST HOC (v4.1 revision addendum): descriptive case-bootstrap intervals for the frozen primary model's severe-class metrics on the historically exposed 2012 cohort. NOT prespecified; no threshold is selected; no operational validation is implied; no frozen artifact is modified.

| metric | point | 95% case-bootstrap CI | undefined replicates |
|---|---|---|---|
| severe recall | 0.0578 | [0.0381, 0.0795] | 0 |
| severe precision | 0.5200 | [0.3778, 0.6579] | 0 |
| severe average precision | 0.2038 | [0.1688, 0.2426] | 0 |
| severe auroc | 0.7931 | [0.7717, 0.8137] | 0 |

**Reading.** These intervals quantify case-sampling variability only, under the same cross-crash independence approximation as the headline interval; they are conditional on the one fitted model, the frozen protocol, the researcher-defined target, and the exposed cohort. They do not validate any operating point.

*Generated 2026-07-15T19:54:40Z by `crashsev/severe_metric_uncertainty.py` (commit 42194a61); seed 42, 2000 replicates; aggregates only.*
