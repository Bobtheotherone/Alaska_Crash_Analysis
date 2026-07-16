# Data Authority and Access (DATA-AUTH-001)

This record states, in deliberately neutral language, exactly what is and is not established about
the provenance and lawful use of the modelling data. It distinguishes four separate things that are
often conflated.

> **Redaction note.** For privacy, local OS usernames and third-party contributor handles have been
> replaced by neutral placeholders (`%USERPROFILE%`, "S23 contributor") throughout the shipped docs.
> Only identifiers are redacted; the provenance facts (a personal licensed backup under
> `…\Downloads\BACKUP\OLD_BACKUP`, two prior contributor eras, the raw extract predating both) are
> unchanged. The declared author of this study is Naythan Mercado (see `AUTHORSHIP_AND_AI_ASSISTANCE.md`).

## 1. The four provenance layers (kept distinct)

| Layer | Status | Notes |
|---|---|---|
| **Source-agency authority** | **NOT independently verified** | The extract is best described as a *restricted project copy of unverified source-custodian provenance.* It has **not** been verified against an official Alaska DOT&PF / agency record, and no official codebook was obtained. |
| **Project-owner permission** | **Granted (for analysis)** | The repository/data owner granted permission to inspect and analyse the archived extracts (`DATA_LICENSE_NOTE.md`). This is a private grant for analysis, **not** a public data-release license and **not** a statement of source-agency authority. |
| **Repository possession** | **Yes (derived, de-identified)** | A de-identified local modelling table is derived from the extract and kept **local** (git-ignored). Only aggregates and de-identified predictions are published. |
| **Lawful reproducibility access** | **Reviewer-provided** | An external reviewer needs their own lawful copy of the raw extract to reproduce the from-raw pipeline (Route B). Without it, they can still reproduce every metric from the shipped de-identified predictions (Route A). |

## 2. Cryptographic identity

* **Governed content hash (modelling dataframe):** `data_sha256 = 059559cd7a9c40a37f7f60e12e6a82137dc8c70d12817246c2a19d59ea3e097b` — computed on dataframe *content*
  (`pd.util.hash_pandas_object`), so it is stable across file serialisations. Recorded in every run
  manifest and in `FROZEN.lock`.
* **Raw file byte-hash (recorded 2026-07-11):**
  SHA-256 **withheld from public release** — retained in the restricted reproduction log
  (`private_reproduction_log/RAW_WORKBOOK_IDENTITY.md`, local-only) because custodian permission
  to publish this source identifier has not been confirmed under the NDA and data-use agreement;
  size **25,391,525 bytes**, located at `%USERPROFILE%\Downloads\BACKUP\OLD_BACKUP\2025\Crash Level 09-12 (1).xlsx`.
  (An earlier finalization-time scan looked only at the `OLD_BACKUP` root and wrongly recorded the
  file as absent; it sits in the `2025\` subdirectory. That earlier statement is corrected here,
  not deleted from history — see `PROVENANCE_CHAIN.md`.)
* **Verified raw → study-input chain (2026-07-11):** loading this byte-hashed workbook passes
  Gate-0 (`validate-data`), and rebuilding the modelling table from it with
  `crashsev.build_modeling_table` reproduces the existing `_local_data/modeling_table_09_12.csv`
  **byte-identically** and reproduces the governed content hash `059559cd…` **exactly** — the same
  `data_sha256` pinned in `FROZEN.lock` and in every run manifest of the final study
  (`final_f27613102c96`). The frozen study input is therefore cryptographically anchored to the
  byte-hashed raw file; what remains unverified is the layer above the file (source-agency
  authority, extraction record, official codebook — see `PROVENANCE_ACQUISITION_PLAN.md`).

## 3. What a reviewer can and cannot verify

* **Can verify (no raw data):** the modelling-table content hash as used by the pipeline; the split
  hash; the target-map and contract hashes; every reported metric and interval from the shipped
  de-identified 2012 predictions; all code, tests, and re-analysis.
* **Can verify (with a lawful copy of the raw file):** the raw byte identity
  (hash withheld from public release — restricted reproduction log; 25,391,525 bytes) and the from-raw cohort construction — rebuilding the modelling
  table reproduces the frozen `data_sha256 = 059559cd…` (verified 2026-07-11).
* **Cannot verify (pending Gate-3 external evidence):** that the extract faithfully represents the
  official Alaska source record (source-agency authority, extraction query/date, official codebook).

## 4. Redistribution and privacy boundary

The raw extract and any per-crash records, coordinates, free-text location, and officer/report
identifiers are **never** published (`DATA_PROVENANCE.md`). `Crash Number` is treated as an
identifier and is excluded from committed artifacts; the sealed-year predictions shipped for audit
are de-identified to opaque surrogates. Nothing in this package authorises public redistribution of
the source data.

## 5. Consequence for claims

Because source-agency authority is unverified, every empirical conclusion is bounded to "this
restricted extract of Alaska 2009–2012 reported crashes" and no claim of official data provenance is
made. This limitation does not block the study: the numerical and methodological evidence remains
auditable from the committed artifacts and the de-identified predictions.
