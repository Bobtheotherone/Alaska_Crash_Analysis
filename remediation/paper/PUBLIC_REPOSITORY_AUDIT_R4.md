# PUBLIC REPOSITORY AUDIT — r4

Scope: `https://github.com/Bobtheotherone/Alaska_Crash_Analysis` (local clone `C:\aca`),
audited 2026-07-16 before any r4 publication action. The audit covers (a) the
already-public remote content, and (b) the local portfolio lineage that r4 would
publish for the first time, held to the same standard because pushing makes it public.

Method summary (commands recorded in `RELEASE_WORKLOG_R4.md`):

- Tracked-tree inventory: `git ls-files` (382 files at baseline `111f6f9`), extension
  triage for xlsx/xls/csv/parquet/db/sqlite/zip/tar/7z/pem/key/env/ipynb/pkl/joblib/
  feather/bak/dump.
- Ignored-but-present inventory: `git status --ignored --porcelain`.
- Full-history object census: `git rev-list --objects --all` (1,146 named objects),
  filename triage over every object ever committed on any ref.
- Full-history added-path census: `git log --all --diff-filter=A --name-only`
  (419 unique paths ever added).
- Content-level secret scan over **every blob on every ref** (custom scanner via
  `git cat-file --batch`; gitleaks unavailable on this host): AWS keys, GitHub
  tokens (`ghp_/gho_/ghu_/ghs_/ghr_`), fine-grained PATs, Google API keys, Slack
  tokens, private-key blocks, Django `SECRET_KEY` literals, password assignments,
  DB connection strings, Anthropic/OpenAI keys.
- Content-level NDA scan over the tracked tree: Alaska-range decimal coordinates,
  coordinate pairs, officer/badge identifiers, trooper/detachment references,
  milepost values, crash-number-like values.
- Raw-workbook-identity scan over **every blob on every ref** for the licensed
  workbook's SHA-256 (full 64-char value and 8-char prefix).
- Remote-side checks: `git ls-remote --heads/--tags origin`, per-branch tree scans
  of `origin/integrate-peyton-ml`, `origin/integrate-peyton-ml-v2`, `origin/main`,
  GitHub Releases API (`repos/.../releases`).
- LFS: `git lfs ls-files` (none); `.gitattributes` reviewed (binary/EOL protection only).

## 1. Current-tree audit (baseline `111f6f9`)

| Check | Result |
|---|---|
| Raw source workbooks (xlsx/xls) tracked | **None.** No xlsx/xls blob exists anywhere in history either. |
| Raw CSV exports / row-level source data | **None.** The five tracked CSVs are metadata/aggregates: `column_inventory_09_12.csv` (per-column dtype/null/unique stats with `sample_value` **already `[redacted]`** for every identifier-tier column: Crash Number, Officer ID/Agency, Latitude, Longitude, Street, Route, Intersecting Street, Date, DateTime), `feature_availability_ledger.csv`, `ANALYSIS_STATUS_LEDGER.csv`, `audit_resolution_matrix.csv`, `reanalysis_metrics.csv` (model-level metrics). |
| Real crash identifiers | **None found.** Group identifiers in released evidence are salted-hash surrogates (see `evidence_release/*/DEID_TRANSFORM.md`); the de-identified parquet is not tracked in git (ships only inside the verified handoff ZIP). |
| Exact latitude/longitude values | **None.** The only Alaska-range decimals are the state map-viewport constants in `alaska_ui/src/components/CrashMap.tsx` (71.38957 / −179.14734 and the built JS copies) — geographic bounding-box constants, not crash records. |
| Officer / agency / detachment identifiers | **Column names only** (schema docs, feature ledgers, drop-lists). `experiment/agency_leave_one_out.json` documents that agency fields were removed at the identifier tier and reports region-proxy aggregates only. |
| Free-text crash locations | **None** (inventory samples `[redacted]`). |
| `.env`, tokens, credentials, keys, cloud secrets | **None** in tree or history (see §3). Django `settings.py` reads `SECRET_KEY` from the environment; the only literal is the self-describing dev default `"dev-secret-key-not-for-production"`, and `DJANGO_SECRET_KEY` is mandatory when `DEBUG=False`. |
| Unredacted screenshots | The single tracked screenshot (`Screenshot 2025-11-21 175650.png`, from the original public initial commit) shows GDAL/PROJ/GEOS package versions — no data. |
| Binary databases, model caches, notebooks with outputs | None tracked. `peyton_original/DataCleaning/Clean Combined Crash Data.ipynb` (already public) is a single code cell with **no outputs**; it references workbook filenames only. |
| Accidental archives / parquet / temp files | None tracked. Local `dist/`, `remediation/runs/`, `remediation/_local_data/`, experiment parquets and `FROZEN.lock`s are ignored and stay local. |
| **Raw workbook SHA-256** | **FINDING F-R4-01 — present at baseline.** See §5. |

## 2. Full-history audit (all local refs — the lineage r4 would publish)

- No data file (xlsx/csv-of-rows/parquet/db/zip/env/key/pem) was **ever** committed,
  with two explainable exceptions:
  1. **Synthetic run bundle** (`remediation/runs/run_20260710T034946Z_b79e27bf/…`,
     added `f4fe6ff`, untracked again in `afcaabe`): eleven per-model prediction CSVs
     (2,901 rows each) plus manifests. The bundle's own committed `manifest.json`
     declares `"data_source": "SYNTHETIC_STRUCTURAL_FIXTURE (TEST-ONLY; not real
     Alaska data)"`, `"is_synthetic": true`, `"synthetic_n": 12000`; group ids are
     synthetic sequence ids (`AK2009…`). **Disposition: no NDA content; no action.**
  2. `repo_flattened.txt` (text flatten of the application repo, in the already-public
     initial commits): passed the same secret/NDA scans as the tree it flattens.
     **Disposition: no action.**
- **Raw workbook SHA-256 in history**: 5 historical blobs carry the full 64-char
  value and 23 carry the 8-char prefix, all reachable from `111f6f9`
  (see FINDING F-R4-01).

## 3. Secrets scan (all blobs, all refs)

Two hits total, both the literal test fixture `password="password"` in
`ingestion/tests/test_ingestion_gateway.py` (and its copy inside the historical
`repo_flattened.txt`). Not credentials. **No real secret, token, key, or connection
string anywhere in history.** No credential rotation required.

## 4. Remote / release-asset audit (already-public surface)

- Remote heads: `integrate-peyton-ml` (default) @ `bb9247a`, `integrate-peyton-ml-v2`
  @ `10e0691`, `main` @ `91071b9`. Remote tags: none. GitHub Releases: **none** (API
  returns 0), so no release assets exist to audit.
- Per-branch tree scans of all three public branches: no env/db/credential/secret
  filenames; no data files; the raw-workbook hash does **not** appear in any blob
  reachable from the public branches (all 28 hash-bearing blobs are local-only).
- Git LFS: no LFS objects on any ref.

**The currently public repository is clean.** No `PUBLIC_HISTORY_INCIDENT_R4.md` is
required: nothing restricted is in *public* history.

## 5. FINDING F-R4-01 — raw workbook SHA-256 in the local (unpublished) lineage

- **What**: the licensed raw workbook's byte-identity SHA-256 (full value) appears at
  baseline `111f6f9` in three tracked provenance documents
  (`remediation/research/DATA_AUTHORITY_AND_ACCESS.md`, `…/DATA_PROVENANCE.md`,
  `…/PROVENANCE_CHAIN.md`) and as an 8-char prefix in eight further files
  (`ISSUE_DISPOSITION_CURRENT.md`, `PROVENANCE_ACQUISITION_PLAN.md`,
  `RECON3_ISSUE_LEDGER.md`, `RELEASE_READINESS.md`, `REPRODUCTION_LOG.md`,
  `research_audit/claim_scan.py`, and the superseded pre-LaTeX renders
  `paper/final_paper.md` / `final_paper.html`). Across the full local history,
  5 blobs carry the full value and 23 the prefix — **all reachable from `111f6f9`**,
  i.e. from any push of the portfolio lineage.
- **Exposure category**: contractually conservative disclosure control (manuscript
  item 4.3): permission to publish the source identifier has not been confirmed
  under the NDA/data-use agreement. A hash is not raw data (preimage-resistant), and
  the value was recorded deliberately as chain-of-custody evidence for the
  controlled-recipient handoff; the finding is about *public* disclosure only.
- **Currently reachable publicly?** **No.** Not on any remote ref; no releases exist.
  The r3 handoff ZIP containing these documents (and a full `--all` git bundle under
  `provenance/`) was delivered to the university recipient, not published.
- **Containment applied in r4 (working tree)**:
  1. All eleven current-tree occurrences redacted; wording now matches the
     manuscript's 4.3 statement (hash retained in the restricted reproduction log;
     not published pending custodian permission).
  2. The full value and its provenance context moved to
     `private_reproduction_log/RAW_WORKBOOK_IDENTITY.md` — **local-only, gitignored**
     (Phase 2 rules) — so the "retained in the restricted reproduction log"
     statement is concretely true.
  3. `research_audit/claim_scan.py` now asserts the redaction language instead of
     the hash prefix.
  4. The r4 handoff/packaging embeds provenance for **published refs only** (no
     `--all` bundle in any public asset), and the package forbidden-content scan
     fails closed on the hash value and prefix.
- **Addendum (commit messages)**: one commit body reachable from `111f6f9`
  (`c5a513e`, "gate3(provenance): record raw workbook byte hash…") quotes the
  hash's truncated 8-char prefix (`…` -elided). Commit *subjects* are clean, so
  the handoff's `COMMIT_LOG.txt` (`git log --oneline`) carries nothing; but a
  push of the lineage would publish this message body as well.
- **Residual risk & required decision (push gate)**: the *historical* blobs remain
  reachable from `111f6f9` ancestry. Publishing the full lineage therefore still
  publishes the historical hash-bearing blobs even though the r4 tree is clean.
  Because rewriting the frozen r3 lineage is out of scope without explicit author
  approval, the push is **gated on an author decision** among:
  (a) obtain custodian permission to disclose the source identifier, then push the
  full lineage as-is; (b) publish a **snapshot branch** (r4 tree committed atop the
  public default branch; full lineage stays local/controlled) — no historical blobs
  published, PR-compatible, recommended default; (c) approve a pre-publication
  local history scrub (never-published history, so not a public rewrite, but it
  would re-identify ~51 commits including the frozen r3 baseline — not recommended).
  Recorded in `AUTHOR_ACTIONS_R4.md`.
- **History rewriting necessary?** Not for any published content. Only option (c)
  above would involve rewriting, and it is neither required nor recommended.

## 6. Dispositions

| Finding | Severity | Disposition |
|---|---|---|
| F-R4-01 raw-workbook hash in unpublished lineage + baseline tree | Material for publication decision only (no public exposure occurred) | Tree redacted in r4; private log created; packaging fails closed; **push route = author decision at the Phase 10 gate** |
| F-R4-02 synthetic run bundle in history | Informational | Self-declared synthetic fixture; no action |
| F-R4-03 `password="password"` test fixture (public history) | Informational | Placeholder in a test; not a credential; no action |
| Remote branches / releases / LFS | — | Clean |
| Secrets (all refs) | — | Clean |
| NDA content (rows, identifiers, coordinates, locations, officers/agencies) | — | Clean; identifier-tier samples already `[redacted]` at source |

Package-level verification of the rebuilt r4 assets (forbidden-content scan of the
final ZIP, clean-room extraction, manifest validation) is executed and recorded in
Phase 8 (`NUMERICAL_VERIFICATION_R4.md` and `RELEASE_WORKLOG_R4.md`).
