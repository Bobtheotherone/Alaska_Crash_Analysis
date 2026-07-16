# Data Availability

## What is distributed

Publicly available, in this repository and in the assets of the tagged release
(`portfolio-final-r4`, published July 20, 2026):

- All Iteration IV analysis, governance, and verification **source code** and the
  full test suite (`remediation/`).
- The **manuscript LaTeX source** and every figure input
  (`remediation/paper/latex/`), plus the built PDF as a release asset.
- **Aggregate evidence** for every reported number: frozen aggregate JSON artifacts
  (`remediation/experiment/*.json`), re-analysis matrices, ledgers, and manifests.
- **De-identified row-level verification evidence** inside the self-contained
  verification handoff ZIP (release asset): the lossless predictions parquet and
  prediction CSVs keyed by salted-hash surrogate ids, with the de-identification
  transform documented in `remediation/evidence_release/*/DEID_TRANSFORM.md` and
  verified by the packaged `VERIFY_HANDOFF.py`.
- Column-level **metadata** about the licensed extract (dtype, null rates, unique
  counts) with identifier-tier sample values redacted
  (`remediation/data/column_inventory_09_12.csv`).

## What is not distributed, and why

- **The licensed raw extract** (`Crash Level 09-12 (1).xlsx`) and any per-crash
  derivative that retains identifiers: supplied under an NDA/data-use agreement
  with the data owner; not redistributable. This includes the local modelling
  table (`_local_data/`, never committed) and local run bundles keyed by real
  crash numbers (`remediation/runs/`, never committed).
- **Real crash identifiers, exact coordinates, free-text locations, and
  officer/agency identifiers**: identifier-tier fields, dropped before the
  released evidence was built.
- **The raw workbook's byte-identity hash**: retained in a restricted reproduction
  log but not published, because permission to disclose that source identifier has
  not been confirmed under the NDA/data-use agreement. (A licensed holder can
  still verify provenance end-to-end: rebuilding the modelling table from a lawful
  copy must reproduce the published governed content hash `059559cd…` exactly.)

## How to obtain the restricted data

Access to the licensed source records is controlled by the data owner and,
ultimately, the source agency's release terms (see
`remediation/research/DATA_LICENSE_NOTE.md`). The author cannot grant source-data
access independently. Questions: Radames Naythan Mercado-Barbosa,
<rnmercado@alaska.edu>.

## Verification without the restricted data

Every reported metric is exactly recomputable from the released de-identified
predictions (verification handoff ZIP, release asset), and the full no-license
verification tier is described in `remediation/research/REPRODUCE.md` (tier A).
