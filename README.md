# Alaska Crash Analysis

Alaska Crash Analysis combines a secure Django/React crash-analysis platform with a
governed, leakage-controlled, temporally evaluated ordinal machine-learning study of
police-reported Alaska crash severity (2009–2012).

**No raw Alaska DMV records are distributed anywhere in this repository or its
releases.** The licensed source extract is restricted under an NDA/data-use
agreement; everything published here is code, documentation, aggregate evidence, or
de-identified verification artifacts that passed release-side privacy controls.

## 1. Research portfolio quick links

| Deliverable | Location |
|---|---|
| Final manuscript (PDF) | Asset of the tagged release [`portfolio-final-r4`](https://github.com/Bobtheotherone/Alaska_Crash_Analysis/releases/tag/portfolio-final-r4) (published July 20, 2026); buildable from source at [`remediation/paper/latex/`](remediation/paper/latex/) |
| Tagged release | [`portfolio-final-r4`](https://github.com/Bobtheotherone/Alaska_Crash_Analysis/releases/tag/portfolio-final-r4) — exact commit and per-file identities in the release `MANIFEST.json` / `SHA256SUMS.txt` |
| Verification handoff (self-contained ZIP) | Asset of the same release; verify offline with the packaged `VERIFY_HANDOFF.py` |
| Reproducibility guide | [`remediation/research/REPRODUCE.md`](remediation/research/REPRODUCE.md) (authoritative; two tiers, see §9) |
| Data availability | [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) |
| Citation | [`CITATION.cff`](CITATION.cff) (§14 below) |

## 2. Research question

Using a conservative predictor set that excludes identified outcome descendants —
but whose retained-field recording times remain author-judged, not
custodian-verified — how well do models developed on 2009–2011 Alaska crashes
predict a researcher-defined three-level ordinal recorded-severity outcome for 2012
crashes, and how much do specified protocol perturbations change the apparent
result?

## 3. Main result

One frozen, governed evaluation execution (run `final_8af9d5bc23d8`; development
2009–2011, n = 35,214; single exposure-disclosed test year 2012, n = 11,630):

- Weighted ordinal random forest (protocol-designated primary) oMAE: **0.3469**
- Majority-class baseline oMAE: **0.3614**
- Paired difference: **−0.0144**
- Descriptive 95% crash-level bootstrap interval: **[−0.0231, −0.0063]**
- Severe-class (class 2) recall of the primary model: **5.8%**
- Overall 2012 oMAE leader: **unweighted random forest, 0.3401** — it achieves this
  by predicting "no injury" for ~93% of crashes
- Interpretation: **methodological evaluation infrastructure, not a deployable
  severe-crash detector**

The weighted ordinal forest retained the primary role because the candidate
registry was frozen before the governed execution; role designation was not
re-litigated after seeing 2012 results, and the unweighted forest's better oMAE is
reported alongside it rather than silently promoted.

## 4. Why the result matters

The contribution is the **evaluation protocol, not the effect size**. The project
demonstrates auditing an inherited AI system, detecting leakage and evaluation
optimism (the original pipeline's apparent performance collapsed once
outcome-derived fields were removed and evaluation was made temporal), constructing
a governed temporal/ordinal protocol with structural outcome isolation and frozen
decision rules, preserving reproducible evidence for every reported number, and
narrowing claims to what corrected evidence supports: a modest, descriptively
supported improvement over a trivial baseline — with the severe-crash detection
failure reported as prominently as the headline number.

## 5. Critical limitations

Read these before quoting any number:

- **Unresolved blank-value semantics in the target.** Blank severity is mapped to
  property-damage-only (class 0) on documented evidence, but the source coding
  practice is not custodian-confirmed; class 0 is partly a construction of this
  mapping.
- **Retained-feature recording times are not custodian-verified.** "Leakage-
  controlled" means structural isolation plus an author-judged field tier, not a
  verified as-of-scene data dictionary.
- **The evaluation is retrospectively exposed.** 2012 was historically visible
  during the project's lifetime; the single frozen execution is exposure-disclosed,
  not a sealed prospective holdout. Results are descriptive, not confirmatory.
- **The primary model is not a severe-crash detector** (5.8% severe recall;
  ranking ability exists — severe-class AP ≈ 0.17–0.23, AUROC ≈ 0.75–0.80 at
  prevalence 0.0387 — but hard-rule detection does not).
- **Not preregistered, not causal, not independently replicated, and no
  deployment is endorsed.**

## 6. Project lineage

| Iteration | Scope | Where |
|---|---|---|
| I–II | Data cleaning and exploratory modeling of the licensed extracts (original contributors' capstone work) | `peyton_original/`, `analysis/`, `Data Cleaning` history |
| III | Secure Django/React crash-analysis platform: authenticated upload gateway (MIME sniffing, ClamAV hook, MMUCC schema validation), PostGIS crash store, model-job API, React map UI | `alaska_project/`, `ingestion/`, `crashdata/`, `alaska_ui/`, `frontend/` — see [`docs/APPLICATION_PLATFORM.md`](docs/APPLICATION_PLATFORM.md) |
| IV | Governed research remediation: leakage audit and re-analysis of the inherited pipeline, then the frozen ordinal study summarized above, with verification tooling and releases r1–r4 | [`remediation/`](remediation/README.md) |

The platform work is preserved, not diminished: it is the system context that
motivated the governance questions Iteration IV answers.

## 7. Repository structure

```
├── remediation/           # Iteration IV research portfolio (MIT-licensed code)
│   ├── crashsev/          #   pipeline: cohort build, contracts, models, governance
│   ├── experiment/        #   frozen aggregate artifacts of the governed runs
│   ├── evidence_release/  #   de-identification manifests + frozen-run skeletons
│   ├── reanalysis/        #   leakage/optimism re-analysis of the inherited model
│   ├── research/          #   protocol, provenance, scope, reproduction guide
│   ├── paper/             #   LaTeX manuscript source + release reports
│   ├── tests/             #   failure-mode & governance test suite
│   └── tools/             #   verifiers and packaging (VERIFY_HANDOFF, gates)
├── alaska_project/ ingestion/ crashdata/ alaska_ui/ frontend/   # Iteration III app
├── peyton_original/ analysis/ ml_partner_adapters/              # earlier iterations
├── docs/                  # platform documentation
├── DATA_AVAILABILITY.md   # what is / is not distributed, and why
├── LICENSES/              # component-level license scope
└── CITATION.cff           # citation metadata for the tagged release
```

## 8. Reproduction tiers

| Tier | Needs | Reproduces |
|---|---|---|
| A — no licensed data | Python 3.11–3.13, `pip` | Test suite, generator/verifier gates, re-analysis matrices, every figure, and **exact recomputation of all reported metrics from the released de-identified predictions** (handoff ZIP) |
| B — licensed data | A lawful copy of `Crash Level 09-12 (1).xlsx` | The full from-raw pipeline: byte-identical modelling table, identical frozen hashes, identical governed results |

## 9. No-license verification command

```bash
cd remediation
python -m pip install -r requirements-lock.txt && python -m pip install -e .
python -m pytest tests/ -q                 # governance + failure-mode suite
```

Then, against the downloaded release handoff ZIP (self-contained, offline):

```bash
python VERIFY_HANDOFF.py                   # from the extracted ZIP root
```

Full tier-A instructions: [`remediation/research/REPRODUCE.md`](remediation/research/REPRODUCE.md).

## 10. Licensed-data reproduction requirements

Tier B requires a lawful copy of the licensed extract from the data owner —
this repository cannot grant access (see `remediation/research/DATA_LICENSE_NOTE.md`).
The raw workbook's byte identity is retained in a restricted reproduction log and is
not published pending custodian permission; a licensed holder can still verify
end-to-end because rebuilding from their lawful copy must reproduce the published
governed content hash (`059559cd…`) exactly.

## 11. Privacy and NDA statement

The Alaska DMV crash data were supplied under an NDA and data-use restrictions,
which were respected. Sensitive source records were processed **only in local
computing environments**; no raw crash rows, real crash identifiers, exact
coordinates, free-text locations, or officer/agency identifiers were transmitted to
nonlocal language models or external generative-AI services, and none are
distributed here. External AI assistance operated on code, manuscript text,
synthetic examples, aggregate results, and de-identified release artifacts only.
Details: [`remediation/paper/AI_USE_AND_PRIVACY.md`](remediation/paper/AI_USE_AND_PRIVACY.md)
and [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

## 12. Software / application setup

The Iteration III platform (Django + PostGIS backend, Vite/React frontend, secure
ingestion pipeline) is documented in
[`docs/APPLICATION_PLATFORM.md`](docs/APPLICATION_PLATFORM.md) and
`docs/deployment.md`. Short version: start Postgres/PostGIS, `python manage.py
migrate`, `python manage.py runserver`, and `npm run dev` inside `alaska_ui/`. The
platform is archived scope and not production-hardened.

## 13. License scope

Component-level licensing is documented in [`LICENSES/README.md`](LICENSES/README.md):
the Iteration IV analysis and verification code (`remediation/`) is MIT-licensed
with an explicit **no-data-rights** clause; earlier application/platform code
remains under its contributors' rights; the manuscript text is not MIT-licensed.
Nothing in this repository is a data-release license.

## 14. Citation

See [`CITATION.cff`](CITATION.cff). Preferred citation: the manuscript
*Leakage-Controlled Ordinal Classification of a Researcher-Defined Alaska
Crash-Severity Outcome: A Governed, Self-Reproduced Retrospective Out-of-Time Study
Across Four Project Iterations*, Radames Naythan Mercado-Barbosa, release
`portfolio-final-r4`, July 20, 2026.

## 15. Contact

Questions concerning the released manuscript, code, or verification package:
**Radames Naythan Mercado-Barbosa** — <rnmercado@alaska.edu>.
Access to the licensed source records remains controlled by the data owner; the
author cannot grant source-data access independently.
