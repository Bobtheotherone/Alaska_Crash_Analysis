# Numerical verification report — r3 (2026-07-15)

Every manuscript table and figure, its authoritative source artifact(s) with
SHA-256 identity (first 16 hex; full digests in `SHA256SUMS.txt` at packaging),
and the regeneration/verification gate that passed in this build. Gate totals:
pytest 96/96 · manuscript verifier 893/893 · frozen/locked 34/34 · audit_paper
42/0/0 · 8/8 generator checks · independent recompute 11/11.

| Manuscript object | Source artifact (sha256-16) | Verification gate (status: PASS) |
|---|---|---|
| Table 1 (tab:cohort) — cohort accounting | `experiment/final_results.json` (6578e7e1ef967040)<br>`data/codebook_evidence_09_12.json` (082ce407aac534fc) | verify_manuscript_numbers positional spec (cohort_spec) + quarantine-decomposition cross-check |
| Table 2 (tab:cohort-year) — year-by-year cohort | `experiment/cohort_year_table.json` (678fbc39678f7b30)<br>`paper/latex/cohort_year_table.tex` (209120b49f32e12b) | gen_cohort_year_table --check (reconciles vs frozen audits + q6) + verifier positional spec + render identity |
| Table 3 (tab:key-results) — selected 2012 results | `experiment/final_results.json` (6578e7e1ef967040) | verify_manuscript_numbers positional spec (key_spec), 8 models x 6 checked columns |
| Table 4 (tab:hypotheses) — H1–H4 adjudication | `experiment/final_results.json` (6578e7e1ef967040)<br>`experiment/broad_sensitivity/final_results.json` (b813698f17114a67) | verify_manuscript_numbers Sec-3 value checks (H-values re-derived from artifacts) |
| Table 5 (tab:repro-manifest) — reproducibility manifest | `paper/README_REPRODUCE.md` (65e5e3ad7785a7f0) | documentary table; commands verified by executing them in this build (all PASS) |
| Tables 6–8 (tab:full-results-a/-detail/-b) — complete 2012 results | `experiment/full_results_table_cells.json` (abdf93e0fa053329)<br>`paper/latex/full_results_tables.tex` (66037c2726452727)<br>`experiment/final_results.json` (6578e7e1ef967040) | gen_full_results_tables --check (recompute-from-confusion-matrices asserted vs stored) + verifier identity + 3 positional specs |
| Tables 9–10 (tab:prior-performance/-severe) — Iteration III re-analysis | `reanalysis/reanalysis_table_cells.json` (dfc43ab77e88c4af)<br>`data/confusion_matrices_from_paper.json` (9f4dcf0cdc2f83cb) | gen_reanalysis_tables --check (F1-CONV-001) + verifier in-memory regeneration + positional specs |
| Tables 11–12 (tab:artifact-map/-b) — artifact map | `paper/latex/main.tex` (0d51e6c37786ab21) | verify_manuscript_numbers artifact-existence scan over every \artifact{} reference (expanded manuscript; parquets resolve in the evidence tier by policy) |
| Table 13 (tab:hyperparameters) — fixed configurations | `experiment/hyperparameter_ledger.json` (04d384a4543658bc) | gen_hyperparameter_ledger --check + verifier per-cell fragment checks |
| Tables 14–15 (tab:analysis-status/-b) — 2012 analysis-status ledger | `paper/ANALYSIS_STATUS_LEDGER.csv` (134322517823e218)<br>`paper/latex/analysis_status_table.tex` (c0040aead08fc086) | gen_analysis_status_ledger --check (CSV + tex identity; run/commit ids pulled from artifacts) |
| Table 16 (tab:dev-results) — complete development record | `experiment/dev_results_table_cells.json` (a524cdd1d3ffec5e)<br>`paper/latex/dev_results_table.tex` (d4709900fb1a0115)<br>`experiment/development_report.json` (941170ed03d34d56) | gen_dev_results_table --check (selection semantics re-executed + asserted) + verifier identity, positional spec, aggregate-equality check |
| Tables 17–19 (tab:retained-fields/-b, tab:prohibited-fields) + Table 20 (tab:exclusion-summary) | `experiment/feature_governance_cells.json` (b6989fd5a91723b4)<br>`paper/latex/feature_governance_tables.tex` (addde673bcab7fcb)<br>`data/feature_availability_ledger.csv` (a6293c81f4b0c9ae) | gen_feature_governance_tables --check (structural reconciliation vs frozen retained list; full recompute with licensed table) + verifier checks |
| Figures 3–7 and 9–11 (artifact-derived figures) | `experiment/final_results.json` (6578e7e1ef967040)<br>`experiment/leakage_factorial.json` (f3512fad8c8c80de)<br>`reanalysis/reanalysis_metrics.csv` (d915ec1b4fb7f62d) | make_figures.py regenerates from frozen artifacts with internal cross-check assertions (e.g., the Fig. 11 AP tolerance 5e-4); audit_paper page checks |
| Figure 8 (fig06c) — severe-class reliability diagram | `experiment/final_results.json` (6578e7e1ef967040)<br>`experiment/predictions_lossless.parquet` (494e94cde2e1b870) | calibrated points read from the frozen artifact; raw curve recomputed under the frozen binning rule; per-series count totals asserted = 11,630 |
| Abstract + Conclusion numeric statements | `experiment/final_results.json` (6578e7e1ef967040) | verify_manuscript_numbers locked-value list + REQUIRED concepts (tex + extracted PDF text) |

**Frozen-identity anchor:** `experiment/final_results.json` sha256-16 = `6578e7e1ef967040`; run `final_8af9d5bc23d8`; frozen/locked gate 34/34 confirms every governed bundle byte-identical to its pinned hash.

**Independent recomputation (second checking process):** an isolated agent with its own numpy/pandas implementations recomputed, from `experiment/predictions_lossless.parquet` alone: both headline oMAE values, the paired difference, severe recall/precision (26/450, 26/50), the primary confusion matrix, posterior-median label reconstruction (0 mismatches / 162,820), probability validity, count-based majority macro-F1 (0.2692… → 0.269), unnormalized RPS and clipped log loss, and a Monte-Carlo-consistent bootstrap interval — 11/11 PASS, bit-exact except two documented 1-ulp float-accumulation artifacts (RPS accumulation order; alternate paired-mean path).
