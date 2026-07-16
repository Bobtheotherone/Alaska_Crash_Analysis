# Data Manifest — Alaska Crash Analysis (Gate 0)

**Status: COMPLETE for the 2009–2012 extract.** The study was executed on a lawfully accessed raw
extract; this manifest records what was established from the data itself. The authoritative
narratives are `research/DATA_PROVENANCE.md` (lineage), `research/DATA_LICENSE_NOTE.md` (rights and
privacy), and the per-run manifests in `runs/<id>/`. No raw records, coordinates, or ids are
committed anywhere — only the aggregates below.

## 1. Provenance

| Field | Value |
|---|---|
| Dataset | Alaska police-reported crash records (crash level) |
| Extract used | `Crash Level 09-12 (1).xlsx` — 2009–2012, **50,543** rows |
| Rights | Owner grant for inspection/analysis; not a public data-release license (`DATA_LICENSE_NOTE.md`) |
| Custodian (source) | Alaska DOT&PF / crash-records custodian (governs any public release) |
| Raw in repository? | **No** — the workbook stays at its archive path; nothing raw is committed |
| Governed content hash | `data_sha256 = 059559cd…` — `sha256(pd.util.hash_pandas_object(df))`, **serialization-independent**; recorded in every run manifest and `FROZEN.lock` |
| Second extract (`13-17`) | Referenced in prior code but **only a derived pickle exists**, no raw temporal axis → excluded (`DATA_PROVENANCE.md`, `PRIOR_WORK_LINEAGE.md`) |

## 2. Analytical unit & keys

| Field | Value |
|---|---|
| Analytical unit | **crash** (one row per crash) |
| Group / row-id key | `Crash Number` (unique per row — enforced; duplicate/null aborts the split) |
| Time axis | `Year` ∈ {2009, 2010, 2011, 2012} |
| Split | chronological, crash-grouped: **develop 2009–2011, hold out 2012**; `group_overlap_count = 0` (asserted) |

## 3. Target (evidence-checked — internal check, not an official source-agency codebook validation; see `codebook_verification_09_12.md`)

* Column **`Crash Severity`**, source codes **KABCO {K, A, B, C, O}**.
* 3-level collapse: `{O}→0`, `{C, B}→1`, `{A, K}→2`.
* **Blank = property-damage-only (O)** — evidence-checked target mapping, an internal check (NOT an official source-agency codebook validation): 0.0% injury/fatality contamination
  among blank-severity rows (`codebook_evidence_09_12.json`).
* **Fail-closed:** `Unknown` / `Not Reported` / `Null value` are **quarantined, never coerced**.

## 4. Row-flow accounting (real run)

```
raw rows                                   : 50,543
- quarantined (unmapped severity, NOT coerced) : 3,699   (target_audit.n_quarantined)
= rows with a valid target                 : 46,844
-> development (2009–2011)                  : 35,214   (split.n_development)
-> held-out 2012 evaluation                : 11,630   (split.n_final_test)
-> excluded                                :      0
```
Class prevalence — development 0/1/2 = 68.6% / 27.7% / 3.7%; final test = 67.7% / 28.4% / 3.9%
(serious-or-fatal is ~3.8% — a hard, imbalanced minority).

## 5. Features (see `feature_availability_ledger.csv`, `schema.json`)

100 columns classified by when they become available relative to the outcome. **66 allowed**
predictors (→ 483 encoded dims); the rest are excluded as **outcome-derived** (injury/fatality/EMS/
damage/enforcement), **identifiers/privacy** (`Crash Number`, `CDS Number`, `Report ID`, `Officer ID`,
`Latitude`, `Longitude`, `Street`, `Intersecting Street`, `Route`, `Milepoint`), **temporal ids**, or
**constants**. `Year` is retained but flagged as a source-version proxy (ablation toggle exists).

## 6. Privacy classification

`Latitude`/`Longitude` are exact coordinates (re-identification risk in sparse Alaska communities);
these and all identifiers/free-text location are dropped by the ledger and **never** enter the
modelling table or any committed artifact. The de-identified modelling table is written to a
git-ignored `_local_data/` path.

## 7. What is safe to distribute

The contracts (`schema.json`, `target_mapping.yml`, `feature_availability_ledger.csv`), the codebook
evidence check, the code, and the committed **aggregates** (`data/*.md`, `experiment/*.json`,
`reanalysis/*`). **Not** the raw data, coordinates, ids, or any per-crash artifact.
