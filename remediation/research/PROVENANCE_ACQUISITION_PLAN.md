# Provenance Acquisition Plan (Gate 3 — external evidence)

Actions only the applicant (or an external party) can complete. Each item names the audit finding
it closes, the evidence that counts, and what to record in the repository when obtained.
Local prerequisites are already done: the raw workbook byte hash is recorded and the
raw → modelling-table → frozen-`data_sha256` chain is verified (`DATA_AUTHORITY_AND_ACCESS.md` §2).

| # | Ask | Closes | Status |
|---|---|---|---|
| 1 | Source-agency / custodian confirmation of the extract | DATA-PROV-001 | ⛔ external |
| 2 | Official Alaska severity codebook in force 2009–2012 | TARGET-001 | ⛔ external |
| 3 | Field-level recording-time evidence for the at-event ledger fields | FEAT-001 | ⛔ external |
| 4 | Push authorization → hosted CI run + externally timestamped release | CI-PUBLIC-001, PROV-EXT-001 | ⛔ needs authorization |
| 5 | Independent clean-room reproduction by a third party | REPRO-INDEP-001 | ⛔ external |

## 1. Source-agency / custodian confirmation

**Goal.** Upgrade the extract from *"restricted project copy of unverified source-custodian
provenance"* to a documented lineage: who produced `Crash Level 09-12 (1).xlsx`, from what system,
with what query/filters, on what date, and under what terms.

**Who to ask.** The project data owner who granted analysis access (see `DATA_LICENSE_NOTE.md`),
then upstream: the original capstone sponsor/instructor, and if reachable the issuing agency
(Alaska DOT&PF / the state crash-records custodian).

**Evidence that counts.** Any of: the original transmittal email/portal record; an extraction
description (system name, query or export settings, extract date); a written statement from the
custodian that the file (identified by the SHA-256 recorded in the restricted reproduction log; withheld from public release) is a faithful export.

**Record as** `research/SOURCE_CUSTODIAN_RECORD.md` quoting the evidence verbatim (redact personal
identifiers), plus the raw byte hash it attests to. Until then, all claims stay bounded to
"this restricted extract".

## 2. Official Alaska severity codebook (2009–2012 era)

**Goal.** Replace the *evidence-checked* blank=PDO inference and the KABCO text-synonym mapping
with the authoritative coding manual for the crash-report form in force 2009–2012.

**Where to look.** Alaska DOT&PF highway-safety publications; the crash report form and its
officer instruction manual for those years; NHTSA/FHWA state-data documentation referencing
Alaska's KABCO usage. A university librarian or the agency records office can locate the exact
manual revision.

**Evidence that counts.** The manual pages defining: each `Crash Severity` value, what a blank
means, how *Unknown / Not Reported / Null value* are produced, and when injury status is finalized
(scene vs report completion vs 30-day fatality update).

**Record as** a citation + scanned/quoted pages in `research/CODEBOOK_AUTHORITY.md`; then run the
prespecified target-mapping sensitivity (blank→0 vs blank→excluded vs manual-directed) as part of
the corrected benchmark and report both.

## 3. Field-level timing evidence (decision-time availability)

**Goal.** For every `at_event` field in `data/feature_availability_ledger.csv`, evidence of *when*
its value is first recorded and whether it is revised after investigation — not a one-line
rationale. Priority fields (currently allowed on assertion only): `Unit 1 Person 1 Test Given`,
`… Insurance Coverage`, `… Restraint`, `… Ejected`, `… Seat Location`, primary/secondary
contributing circumstances, sequence-of-events fields, damage-location fields,
`… Alcohol or Drug Use Suspected`.

**Evidence that counts.** The crash-report form layout (which page/section the officer fills at
scene vs at report completion), the officer instruction manual, or a custodian statement about
post-submission amendment practice.

**Record as** new ledger columns (`recording_stage`, `evidence_source`) with one row per field.
Until this arrives, the corrected benchmark treats a conservative **strict tier** as primary and
the broad tier as an ablation (see `FINAL_BENCHMARK_PROTOCOL.md` v4).

## 4. Push authorization → hosted CI + external timestamp

**Goal.** A hosted (GitHub Actions) matrix run of the test suite tied to the release commit, and
an external timestamp for the freeze/release so "frozen before X" is witnessable.

**Steps once authorized.** Push the branch + tags to a (private is fine) GitHub repository;
the defined workflow (`.github/workflows/crashsev-ci.yml`, triggers updated for the
`portfolio-research-finalization-*` branches) runs on push; archive the run URL/log in
`research/CI_RECORD.md`. The push itself gives the tag a provider-side timestamp; optionally also
timestamp the release SHA-256 with any independent service or a dated email to a third party.

## 5. Independent clean-room reproduction

**Goal.** Reproduction by someone other than this project: (a) no-data route — verify the handoff,
recompute all metrics/CIs from the de-identified predictions, rebuild figures/paper; (b) licensed
route — obtain the raw file lawfully, confirm the byte hash against the restricted reproduction log, and rerun
`build_modeling_table → develop → freeze-experiment → evaluate-final` per `REPRODUCE.md`,
confirming the split hash and aggregates.

**Record as** the reproducer's signed log (environment, commands, hashes, discrepancies) in
`research/INDEPENDENT_REPRODUCTION_LOG.md`. Until then every reproduction claim stays labelled
**self-reproduction**.
