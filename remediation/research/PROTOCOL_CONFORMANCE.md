# Protocol Conformance (v4.1)

Maps each commitment in `research/FINAL_BENCHMARK_PROTOCOL.md` (the analysis plan **plus its v4
amendment**) to what the code actually executed. Status: **executed** · **partial** ·
**not executed** · **removed from claim** · **exploratory**. Evidence paths are relative to
`remediation/`. The authoritative machine-generated release record is
`research_audit/RELEASE_STATE.md`; where prose and that file disagree, the generated file wins.

| Protocol item | Status | Evidence |
|---|---|---|
| Predictive (non-causal) research question, estimand-anchored, with prespecified hypotheses H1–H4 | executed | `research/ESTIMAND_AND_SCOPE.md`; paper §3–§6 |
| Analytical unit = crash; group id = `Crash Number` (unique) | executed | `crashsev/splits.py`; `data/DATA_MANIFEST.md` |
| Inference-time feature contract; outcome-derived fields forbidden; **conservative restricted tier as v4 primary** (author-judged, pending form evidence) | executed | `data/feature_availability_ledger.csv` (`strict_scene_tier`); paper §3.3 |
| Chronological split: develop 2009–2011, hold out 2012 (out-of-time) | executed | `crashsev/splits.py`; `configs/route_r_09_12.yml` (v4 canonical; `route_a_09_12.yml` = historical v3) |
| **Structural outcome isolation of develop** (year-partition before target interpretation; dev-only audit; poison sentinel) | executed (v4) | `crashsev/cli.py:prepare_development`; `tests/test_governance.py` |
| Freeze pins the **outcome-free development-side assignment hash** | executed (v4) | `evidence_release/final_8af9d5bc23d8/FROZEN.lock` |
| Rolling-origin development CV, 3 seeds, **fold-local feature typing** | executed | `experiment/development_report.json`; paper §7.2 |
| Fold-local learned preprocessing | executed | `crashsev/preprocessing.py` |
| Fixed model registry with roles (trivial/statistical/candidate/ablation/exploratory) | executed | `crashsev/models.py`; paper §5.5, Table 2 |
| Deterministic **prior-probability** baseline (honest probabilistic floor) | executed (v4) | `crashsev/models.py`; paper §7.4 |
| Seeds propagate to every stochastic estimator | executed | `crashsev/models.py:build_registry(seed)`; tests |
| Baseline selected on development evidence (majority) | executed | paper §7.2 |
| **Posterior-median prespecified as the primary hard-decision rule**; argmax recorded as sensitivity | executed (v4) | `crashsev/cli.py:decide_labels`; paper §4, §7.6 |
| Same-model weighted/unweighted ablation (WGT-001) | **executed** (v3 second round onward; earlier "not executed" note is historical) | `final_results.json` (`*_unweighted`); paper §7.6 |
| Calibration fit development-only on **temporal rolling-origin folds**; reliability + proper scores | executed (v4) | `crashsev/cli.py:dev_only_calibrate`; paper §7.4 |
| Primary metric = ordinal MAE; secondary metrics reported; undefined stats = NaN/— | executed | `crashsev/metrics.py`; paper Tables 1–2 |
| One declared primary comparison; others secondary/exploratory; **CI = conditional case-bootstrap, not practical utility** | executed | paper §4, §6–§7.3 |
| Paired crash-level bootstrap CIs (independence approximation, conditional on the fitted model) | executed | `crashsev/uncertainty.py`; paper §5.6 |
| **Seed/refit robustness of the forest results** (five seeds, retrospective addendum) | executed (v4.1) | `experiment/seed_robustness.md`; paper §7.3 |
| Leakage factorial (2×2×2, dev-only) reported as **matched simple effects + interactions** (marginals retained under an explicit not-one-at-a-time label) | executed | `crashsev/leakage_factorial.py`; paper §7.5 |
| **Severe-class one-vs-rest ranking analysis** (AP vs prevalence, AUROC, PR curves; no threshold selected) | executed (v4.1) | `crashsev/severe_ranking.py`; `experiment/severe_ranking.md`; paper §8 |
| **Feature-tier sensitivity** (broad tier, separately frozen) | executed (v4) | `experiment/broad_sensitivity/`; paper §7.8 |
| **Target-mapping sensitivity** (blank=PDO vs blank=excluded, dev-only) | executed (v4) | `experiment/target_sensitivity.md`; paper §7.8 |
| **High-missingness sensitivity** (prespecified >50% rule, separately frozen) | executed (v4.1) | `experiment/missingness_summary.md`; `experiment/lowmiss_sensitivity/`; paper §7.8 |
| Resource measurement (peak memory, per-model fit/predict wall time, dims) | executed (v4) | run `manifest.json` (`model_cards.fit_seconds/predict_seconds`, `peak_memory_mb`) |
| Retrospective out-of-time evaluation, run once under a frozen config | executed | `crashsev/cli.py`; paper §7.3 |
| PO optimizer convergence recorded and flagged (final + dev folds) | executed (v4/v4.1); the strict-design fit did **not** converge — disclosed, excluded from superlatives, not retuned | `final_results.json → results.proportional_odds.convergence`; paper §10 |
| Prospective sealed/preregistered final test | **removed from claim** | reframed as retrospective (`ROUTE_DECISION.md`; EVAL-LOCK-001) |
| EBM as a primary peer | **exploratory** | EBM absent from development; final-only (MODEL-FAMILY-001) |
| Feature-stability hypothesis | **removed from claim** | no stable-feature claim asserted (INTERP-H4-001) |
| Subgroup/fairness/national claims | **removed from claim** | out of scope; paper §1, §10 |
| Independent third-party reproduction | **not executed** | self-reproduction only (REPRO-INDEP-001; Gate 3) |
| Hosted CI | **not executed** | defined + local only; branch unpublished (CI-PUBLIC-001; Gate 3) |

**Conformance summary.** Every item the paper relies on is executed and evidenced under the v4/v4.1
protocol. Items that were planned but not executed are either **removed from the claim** or
explicitly **marked exploratory / external (Gate 3)** here and in the paper — no paper claim rests
on an unexecuted item.
