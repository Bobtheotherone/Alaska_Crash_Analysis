# README_REPRODUCE — Alaska Crash Analysis, correction-release handoff

This package is the self-contained verification bundle for the graduate-portfolio
manuscript *Leakage-Controlled Ordinal Classification of a Researcher-Defined Alaska
Crash-Severity Outcome: A Governed, Self-Reproduced Retrospective Out-of-Time Study
Across Four Project Iterations*. Every path in this package is relative; nothing
depends on the author's machine layout, and no verifier consults any directory outside
the extracted package.

**Row-level evidence:** use the lossless parquet (`evidence/predictions_lossless.parquet`)
as the authoritative row-level evidence. If the prediction CSVs are used with pandas,
pass `float_precision="round_trip"`; the default parser can perturb exact ties and
alter three of the primary model's posterior-median labels (four across all models).

## Package layout

| Path | Contents |
|---|---|
| `paper/` | Final PDF, LaTeX source archive, page-render contact sheet, audit and test reports |
| `canonical/` | The project repository tree at the release commit (code, tests, configs, contracts, manuscript source, verification tools, ledgers, claim matrix, de-identified bundle skeletons under `remediation/evidence_release/`) |
| `evidence/` | De-identified prediction evidence: per-tier surrogate-keyed CSVs and the two lossless parquets (hash-pinned by committed artifacts) |
| `provenance/` | Git state, commit log, and a full `git bundle` of the repository history |
| `CORRECTION_LEDGER.json` | Every correction of the 2026-07-15 completion pass AND the locked correction release (old value, new value, exact source, tests, impact) |
| `ANALYSIS_STATUS_LEDGER.csv` | Status of every analysis involving the 2012 cohort (protocol-primary / protocol-secondary / exploratory-final-only / post-hoc) |
| `REVISION_SUMMARY.md`, `FINAL_VERIFICATION_REPORT.md` | What changed and what was verified, with exact totals |
| `unresolved_external_evidence.md` | The external dependencies no local work can resolve |
| `MANIFEST.json`, `SHA256SUMS.txt`, `VERIFY_HANDOFF.py` | Integrity manifest and stand-alone verifier |

## Verification tiers

**Tier 0 — package integrity (no dependencies beyond Python 3.9+).**

    python VERIFY_HANDOFF.py

Hashes every file against `SHA256SUMS.txt`, rejects unexpected/forbidden content
(virtualenvs, caches, raw spreadsheets, secrets, non-surrogate crash identifiers),
recomputes headline ordinal-MAE values from the de-identified CSVs for all four evidence
tiers, and checks the lossless parquets against their committed hash pins.

**Tier 1 — full no-license verification (Python environment per
`canonical/remediation/requirements-lock.txt`).**

    cd canonical/remediation
    python -m pytest tests -q --basetemp C:\tmp\aca-tests
    python tools/gen_reanalysis_tables.py --check
    python tools/gen_hyperparameter_ledger.py --check
    python tools/gen_analysis_status_ledger.py --check
    python tools/gen_metric_conventions.py --check
    python tools/gen_full_results_tables.py --check
    python tools/gen_dev_results_table.py --check
    python tools/gen_feature_governance_tables.py --check
    python tools/gen_cohort_year_table.py --check
    python tools/verify_manuscript_numbers.py paper/latex ../../paper/Mercado-Barbosa_UAA_Student_Paper_Final_Submission.pdf
    python tools/verify_frozen_and_locked.py

**Windows note (short base path).** Run pytest with a short `--basetemp` such as
`C:\tmp\aca-tests`. At a long default `%TEMP%` path, Windows MAX_PATH limits or
temp-folder ACLs can fail tests with `OSError`/`PermissionError` **before any assertion
runs** — those are environment failures, not scientific test failures, and must not be
silently ignored: shorten the path (or enable long paths) and rerun rather than
dismissing them.

The manuscript verifier checks every locked identity, every result-table cell
positionally per (table, row, column) against the frozen artifacts, forbidden and
required language (including the correction-release scientific-language gate), and
referenced-artifact existence. Verification counts are environment-specific and are
reported separately in `FINAL_VERIFICATION_REPORT.md` — a live git checkout with the
local run bundles runs everything, while this clean extraction reports three named
sections as explicitly skipped: the git frozen-path gate (not a git checkout; history
is verifiable via `git clone provenance/portfolio-final.bundle`), the local run-bundle
hash/metric/bootstrap replay (per-crash CSVs stay local by policy; the parquet supports
metric recomputation instead), and the licensed modeling-table hash (raw-derived table
absent by policy). Never present one environment's count as universal.

To rebuild the PDF from source (TeX distribution with `newtx`, `siunitx`,
`threeparttable`): `cd canonical/remediation/paper/latex && bash build.sh`. The shipped
PDF's identity is recorded in `paper/` and in `MANIFEST.json`; byte-identical rebuilds
are not promised across TeX distributions, so verification of the shipped PDF is
hash-based plus the text-level checks above.

**Tier 2 — licensed-data reproduction.** A holder of the licensed extract
(`Crash Level 09-12 (1).xlsx`) can rebuild the modeling table
(`python -m crashsev.build_modeling_table`), re-run the governed benchmark, and compare
against the frozen run identities recorded in
`canonical/remediation/experiment/final_results.json` (`final_8af9d5bc23d8`) and the
sensitivity bundles. The de-identified `evidence_release/` skeletons pin every artifact
hash of the governed bundles.

## What this package cannot do

- It contains **no raw crash rows, no coordinates, no free-text locations, no
  officer/report/agency identifiers, and no real crash numbers** (predictions are keyed
  by `T#####` surrogates; the surrogate map was destroyed).
- **Bootstrap evidence tiers.** The frozen original-order evidence and recorded
  bootstrap witness replay the published interval bit-exactly. The released lossless
  parquet reproduces all point metrics exactly. Because de-identification replaces
  original identifiers/order, independently resampling the released rows is expected to
  produce a Monte-Carlo-equivalent interval rather than the identical resample sequence.
- Blank-severity semantics, per-field recording times, and agency-cluster dependence
  require the custodian evidence listed in `unresolved_external_evidence.md`; the
  ready-to-send request is `canonical/remediation/paper/custodian_semantics_request.md`.
- **Reproduction tiers are controlled terms** and are never conflated: (1) *metric
  recomputation* — what Tier 0/1 above perform from the released evidence; (2)
  *self-reproduction* — the project's own licensed-source rerun (performed);
  (3) *clean-room package verification* — Tier 0/1 executed from this extracted
  package; (4) *independent replication* — an end-to-end third-party rerun with
  licensed source. Metric recomputation is **not** independent replication; tier (4)
  is exactly what this package is built to invite, and has not yet occurred.
- **Accessibility status (documented honestly).** The shipped PDF has embedded fonts,
  numbered bookmarks, document-title metadata, and machine-extractable reading-order
  text (the language gates run on that text). It is *not* a tagged PDF: tagged-PDF
  generation remains incompatible with this legacy-font (newtx) pdflatex toolchain, so
  screen-reader table semantics are limited to the extraction order.
