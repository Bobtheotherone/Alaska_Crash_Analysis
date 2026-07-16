# Estimand, target, and scope

This document fixes *what is being estimated* before any model is fit, so that every downstream
metric has an unambiguous referent and no result can be quietly re-scoped after the fact.

## 1. Population and unit

* **Unit of analysis:** a single police-reported motor-vehicle **crash** (`Crash Number`,
  unique per row in the source extract).
* **Population (frame):** crashes recorded in the Alaska crash-records system for calendar
  years **2009–2012** and present in the `Crash Level 09-12` extract. This is a *reported-crash*
  frame: crashes never reported to police are out of frame, and any reporting/geographic
  coverage bias in the source system is inherited (a stated threat to external validity).

## 2. Target (outcome) — a documented, reversible, fail-closed definition

Raw `Crash Severity` uses old-style KABCO injury labels. The study models a **3-level ordinal**
collapse; the raw KABCO letter is retained alongside so the collapse is reversible.

| raw label(s) in extract | KABCO | ordinal class | meaning |
|---|---|---|---|
| *(blank)* — **evidence-checked** property-damage-only | O | **0** | no apparent injury / PDO |
| Possible | C | **1** | minor / possible injury |
| Non-Incapacitating | B | **1** | minor / possible injury |
| Incapacitating | A | **2** | serious / fatal |
| Fatal | K | **2** | serious / fatal |
| Unknown, Not Reported, Null value | — | *quarantined* | never assigned a class (fail-closed) |

The single consequential inference — **blank severity = O** — is *evidence-checked, not assumed*:
0.0% of the 32,046 blank-severity crashes carry any positive fatality or injury count
(`data/codebook_verification_09_12.md`). This is an internal evidence check of the target mapping,
not an official source-agency codebook validation. The collapse reproduces the original paper's class
balance to <0.85 pp. The ordering O < {C,B} < {A,K} is the standard national KABCO ranking.

**Class imbalance is intrinsic and is the point, not a nuisance:** ≈0.68 / 0.28 / 0.04. The
severe class (2) is ~3.7% of crashes. Metrics are chosen accordingly (ordinal MAE, QWK,
per-class recall, minority-focused), and a majority-class baseline is reported so that any
model claiming skill must beat "always predict class 0."

## 3. Feature set — information available at or before the crash scene

The estimand is a **post-crash triage / data-completion** classifier: given a crash record's
*scene-time and pre-existing* attributes, infer its injury-severity class. Admissible features
are exactly the `pre_event`, `at_event`, and `temporal` (calendar, excluding Year) tiers of the
feature-availability ledger (`data/feature_availability_ledger.csv`). **Every outcome-derived
field is prohibited** — injury/fatality counts, person injury, EMS transport/extrication,
vehicle damage assessment, tow, and post-crash enforcement — because those are known only
*because of* the outcome and constitute label leakage. `Year` is withheld as a feature (it is
the split axis and a source-version proxy); `Latitude`/`Longitude`/ids/free-text are withheld
for privacy and identifier reasons.

## 4. Evaluation target — forward-in-time generalization

The quantity of scientific interest is **generalization to a later period**: train/develop on
2009–2011, evaluate **once** on a held-out **2012** evaluation year (frozen configuration;
within-study governance, NOT a prospective seal), with crash-grouped, leakage-controlled
preprocessing fit on development data only. This is a *retrospective*, out-of-time evaluation —
the 2012 outcomes were locally available during the work — not a prospective forecast. Development-phase model selection
uses rolling-origin (train past → validate next year) and grouped CV; the final test is touched
exactly once, under the governance lock (`crashsev/cli.py`).

## 5. What is explicitly **out of scope** (no-claims list)

* **No causal claim.** No coefficient, split, or importance is interpreted as the effect of an
  intervention. "Alcohol suspected is associated with higher predicted severity" is descriptive
  of the classifier, never "reducing alcohol would reduce severity by X."
* **No national / cross-state generalization.** Results describe Alaska 2009–2012 only.
* **No claim about unreported crashes** or about years outside 2009–2012.
* **No clinical or triage-deployment claim.** This is a retrospective methodological study for a
  research portfolio, not a validated decision system.
* **No estimate of the *original project's* real-world bias.** The leakage factorial quantifies
  defect effects *within this dataset and protocol*; it is not represented as the exact error the
  original authors would have made on their (differently cleaned) data.

## 6. Success criteria (analysis plan fixed on development evidence before the 2012 evaluation; not an externally witnessed preregistration — direction only)

> **v4 amendment (2026-07-11, prespecified before the v4 freeze/final evaluation):** the primary
> hard-decision rule is the **posterior median** (Bayes rule for the absolute ordinal loss);
> argmax is a recorded sensitivity. The primary feature set is the **strict scene-observable
> tier** (`strict_scene_tier` ledger column); the broad tier is a separately frozen sensitivity.
> The trivial probabilistic floor is the deterministic **`prior_probability`** forecast.
> Dev-only calibration uses **temporal rolling-origin folds**. The develop phase is structurally
> outcome-isolated (GOV-001 v4). See `FINAL_BENCHMARK_PROTOCOL.md` (v4 amendment) for the full
> table; the executed v3 result is retained as historical evidence.

Fixed on development evidence before the 2012 evaluation:
1. The leakage-controlled models are compared to the majority and ordinal baselines on **ordinal
   MAE (primary)**; a model is "useful" only if its ordinal-MAE improvement over the best trivial
   baseline has a paired crash-level bootstrap CI (case/row resampling under a cross-crash
   independence approximation) excluding 0.
2. **Minority-class recall** (class 2) is reported prominently; a high overall accuracy that
   collapses the severe class is reported as a *failure mode*, not a success.
3. Probabilistic predictions are assessed for **calibration** (ECE + reliability); miscalibration
   is disclosed, not hidden behind point accuracy.
4. The **leakage factorial** must show the direction and rough magnitude by which outcome-derived
   features and split-before-fit inflate apparent performance.
