# Alaska Crash-Severity Prediction — a leakage-controlled, governed, self-reproduced study

**Author:** Naythan Mercado · **Route:** R (retrospective out-of-time evaluation, `research/ROUTE_DECISION.md`)
· **Prepared with material AI assistance** under the author's direction (`research/AUTHORSHIP_AND_AI_ASSISTANCE.md`).

**Manuscript (authoritative):** *Leakage-Controlled Ordinal Classification of a
Researcher-Defined Alaska Crash-Severity Outcome: A Governed, Self-Reproduced
Retrospective Out-of-Time Study Across Four Project Iterations* — LaTeX source at
`paper/latex/main.tex`, built PDF at `paper/latex/main.pdf`. The earlier
markdown-rendered paper (`paper/final_paper.md` / `.pdf`) is a superseded historical
draft retained for lineage; where the two disagree, the LaTeX manuscript governs.

A predictive study of a **researcher-defined three-level ordinal recorded-severity
outcome** (0 = none/property-damage-only · 1 = possible/non-incapacitating injury ·
2 = incapacitating/fatal; all working mapped classes, not clinically validated
categories) on police-reported Alaska crashes, 2009–2012. It is the graduate-research
remediation of the *Alaska Crash Analysis* capstone: it rebuilds the evaluation to be
leakage-controlled and governed, narrows the claim to what the data support, and uses
the held-out later year (2012) **once** for the frozen primary evaluation execution —
an **exposed, retrospective** cohort, not a sealed prospective holdout — with every
reported number reproducible from a committed artifact. Here “leakage-controlled”
means structural final-outcome isolation, train-only learned transformations, and an
explicit exclusion ledger for outcome-derived fields; it does **not** mean that the
recording time of every retained field has been custodian-verified. “Reproducible”
means computational self-reproduction by the project, not independent replication.

This is a **predictive, non-causal, retrospective** methodological study. It is **not**
a causal safety finding, a nationally general model, a deployable product, or a
prospectively sealed benchmark. See `research/ESTIMAND_AND_SCOPE.md` and
`research/ROUTE_DECISION.md`.

## Research question
> Using a conservative predictor set that excludes identified outcome descendants, but
> whose retained-field recording times remain author judged, how well do models
> developed on 2009–2011 Alaska crashes predict the researcher-defined ordered outcome
> for 2012 crashes, and how much do specified protocol perturbations change the
> apparent result?

## Headline result (v4 frozen protocol; honest)
Development on 2009–2011 (n = 35,214); one frozen primary evaluation execution on 2012
(n = 11,630); restricted author-judged feature tier (49 fields → 376 encoded
dimensions); posterior-median decision rule matched to the ordinal loss; primary metric
**ordinal MAE** (lower better). The primary comparison — the **protocol-designated
weighted candidate** (class-weighted ordinal random forest, a frozen-registry role, not
the oMAE leader) versus the majority baseline — was frozen before the governed
execution; all other model-vs-baseline contrasts are secondary.

* **Primary comparison (run `final_8af9d5bc23d8`).** Ordinal MAE **0.3469** vs
  **0.3614** for the majority baseline: observed paired difference **−0.0144**,
  descriptive crash-level case-bootstrap interval **[−0.0231, −0.0063]** under the
  stated independence approximation — about 14 fewer class-step errors per 1,000
  crashes. Because the cohort was historically exposed, this met the protocol's
  numerical H1 criterion **descriptively**, not confirmatorily.
* **The oMAE leaders almost never predict injury.** The unweighted ablations record
  lower ordinal error (unweighted RF 0.3401; unweighted ordinal RF 0.3423) by
  predicting class 0 for ~93% of crashes; the protocol-designated candidate recovers
  only **5.8%** of severe crashes (26 of 450). It is not a severe-crash detector.
  Models with more severe recall (XGBoost 0.267, Frank–Hall 0.624) do worse than the
  trivial baseline on ordinal error. **No model is best on every metric.**
* **Probabilities:** exactly the four random-forest variants beat the deterministic
  prior forecast on all three proper scores (log loss / Brier / unnormalized RPS);
  development-only calibration improves the candidate's ECE 0.059 → 0.032 without
  touching any hard label. Metric conventions (unnormalized RPS; log-loss clip 1e-15;
  top-label ECE, 10 bins) are machine-recorded in `experiment/metric_conventions.json`.
* **Sensitivities:** removing two high-missingness administrative fields attenuates the
  primary difference to −0.0091 [−0.0181, −0.0001]; a missingness-indicator-only probe
  scores 0.3583 vs 0.3614, so reporting-completeness patterns carry part of the signal;
  five-seed refits keep every forest on the published side; H2 (ordinal vs nominal
  decomposition) is **not supported** and is reported as failing.
* **Leakage factorial (development-only diagnostic):** adding outcome-derived features
  produced the largest matched apparent reduction in oMAE (−0.349), dwarfing
  pre-split preprocessing (−0.021) and random validation (−0.033); conditional
  diagnostic contrasts, not general causal effects.

Every headline number traces to `experiment/final_results.json` and its companion
artifacts; the status of **every** 2012 analysis (protocol-primary /
protocol-secondary / exploratory-final-only / post-hoc) is machine-generated in
`paper/ANALYSIS_STATUS_LEDGER.csv` and rendered as a manuscript appendix.

## Row-level evidence: the parquet is authoritative
Use the lossless parquet (`experiment/predictions_lossless.parquet`; shipped in the
handoff `evidence/` tier) as the **authoritative row-level evidence**. If the frozen
prediction CSVs are used with pandas, pass `float_precision="round_trip"`: the default
parser can perturb exact cumulative-0.5 ties and alter three of the primary model's
posterior-median labels (four labels across all 14 models) —
`tests/test_prediction_evidence.py` pins both facts.

**Bootstrap evidence tiers.** The frozen original-order evidence and recorded bootstrap
witness replay the published interval bit-exactly. The released lossless parquet
reproduces all point metrics exactly. Because de-identification replaces original
identifiers/order, independently resampling the released rows is expected to produce a
Monte-Carlo-equivalent interval rather than the identical resample sequence.

## Reproduce
```bash
cd remediation
python -m pip install -r requirements-lock.txt   # minimal, fully pinned (Py 3.11–3.13)
python -m pip install -e .

# (A) no raw data needed — tests, verifiers, generated ledgers, figures, manuscript
python -m pytest tests -q --basetemp C:\tmp\aca-tests   # short path: avoids Windows MAX_PATH/ACL failures
python tools/gen_reanalysis_tables.py --check
python tools/gen_hyperparameter_ledger.py --check
python tools/gen_analysis_status_ledger.py --check
python tools/gen_metric_conventions.py --check
python tools/verify_manuscript_numbers.py paper/latex paper/latex/main.pdf
python tools/verify_frozen_and_locked.py        # explicit SKIPs outside a full checkout

# (B) full empirical result — needs a licensed copy of the raw extract
python -m crashsev.cli validate-data   --data "<path>/Crash Level 09-12 (1).xlsx"
python -m crashsev.build_modeling_table --data "<path>/Crash Level 09-12 (1).xlsx"
python -m crashsev.cli develop          --data _local_data/modeling_table_09_12.csv --config configs/route_r_09_12.yml
python -m crashsev.cli freeze-experiment
python -m crashsev.cli evaluate-final   --data _local_data/modeling_table_09_12.csv --config configs/route_r_09_12.yml
```
Verification counts are **environment-specific** and are reported separately (live git
checkout vs clean package extraction) in the release's `FINAL_VERIFICATION_REPORT.md`;
never quote one count as universal. Reproduction tiers are controlled terms: metric
recomputation < self-reproduction < clean-room package verification < independent
replication (which has **not** yet occurred). Step-by-step guide:
`research/REPRODUCE.md`; handoff-package instructions: `paper/README_REPRODUCE.md`.

## Data, privacy, and rights
The raw Alaska extract is used under an explicit owner grant for **inspection and
analysis** (`research/DATA_LICENSE_NOTE.md`); it is **not** redistributed. Privacy is
enforced by construction: no raw records, coordinates, identifiers (including the
agency fields present in the licensed source, removed at de-identification), or
free-text location ever enter a committed file — the de-identified modelling table is
written to a git-ignored `_local_data/` path, and only aggregates or surrogate-keyed
de-identified evidence are shared. Provenance: `research/DATA_PROVENANCE.md`.

## What is / isn't demonstrated
An honest, evidence-anchored self-assessment is in `research/ASSESSMENT.md`; scope and
the explicit no-claims list are in `research/ESTIMAND_AND_SCOPE.md` and
`research/APPLICATION_SCOPE.md`. Unresolved external dependencies (official
blank-severity semantics; custodian-verified field recording times; a genuinely later
unexposed cohort; a stakeholder utility function; independent end-to-end replication;
departmental template approval if required) are tracked in
`paper/unresolved_external_evidence.md` and are **not** claimed to be resolved.

## Layout
```
crashsev/            governed pipeline: contracts, target, splits, preprocessing, models,
                     metrics, cli (validate-data/develop/freeze/evaluate-final),
                     build_modeling_table, leakage_factorial, reanalysis
tests/               failure-mode + governance + regression tests (incl. verifier-path,
                     metric-convention, and prediction-evidence guards)
data/                executable contracts (schema, target mapping, feature ledger)
                     + codebook verification + committed aggregate audits (no per-crash rows)
configs/             route_r_09_12.yml + broad/lowmiss sensitivity configs (frozen)
research/            estimand/scope, provenance, licensing, reproduction, assessments
research_audit/      remediation ledger, prior-work lineage, claim scan
experiment/          committed aggregates: development report, final results, factorial,
                     post-hoc addenda, metric_conventions.json (frozen files byte-frozen)
paper/               LaTeX manuscript (paper/latex/), ledgers, claim matrix, revision
                     summary, custodian request; final_paper.* = superseded draft
tools/               verifiers, generators, packager, handoff builder, token diff,
                     rebuild-identity check
reanalysis/          re-analysis of the prior study's confusion matrices
runs/                immutable, hashed run bundles (per-crash predictions kept local)
```

## Scope discipline
No causal claims, no national/MMUCC generalization, no deployment or clinical validity,
no deep learning, no synthetic-data results. The legacy Django/React/PostGIS
application is an out-of-scope demo layer, not the scientific claim
(`research/APPLICATION_SCOPE.md`).

## License
Source code: MIT (`LICENSE`). The crash data and any per-crash derivative are **not**
covered by that license and are governed by the data owner's terms
(`research/DATA_LICENSE_NOTE.md`).
