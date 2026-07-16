# RELEASE WORKLOG — r4 submission-readiness pass

Author of record: Radames Naythan Mercado-Barbosa <rnmercado@alaska.edu>
Operator: Claude Code (release-engineering assistant), directed session of 2026-07-16.
Baseline: r3 (`portfolio-final-r3`, commit `111f6f9a8cb928a83c30da1aca77f823f91e5aed`).
Working branch: `portfolio-final-r4` (created from the verified r3 baseline).
Target annotated tag: `portfolio-final-r4` (created only after final verification).
Target archival publication date: 2026-07-20.

Rules in force for this pass: additive editorial / documentation / packaging /
repository-polish / verification / release work ONLY. The frozen governed run
`final_8af9d5bc23d8` and every governed number are unchanged. No history rewrite,
no force-push, no fabricated DOI/URL/hash/permission.

---

## Phase 0 — Protect the workspace (2026-07-16)

State recorded before any edit:

- `git status` (on `portfolio-final-r3`): working tree clean.
- HEAD at start: `111f6f9a8cb928a83c30da1aca77f823f91e5aed`
  (= branch `portfolio-final-r3` = annotated tag `portfolio-final-r3`,
  tag object `0b92b642697cc245f984af821c13726096509f9d`).
- Local branches: graduate-research-remediation, graduate-research-remediation-v2,
  integrate-peyton-ml, portfolio-final-correction, portfolio-final-r3,
  portfolio-final-submission, portfolio-research-finalization-v3,
  portfolio-research-finalization-v4, portfolio-revision-final.
- Local tags: portfolio-final, portfolio-final-r2, portfolio-final-r3,
  portfolio-v3, portfolio-v4, portfolio-v4.1.
- Remote: `origin = https://github.com/Bobtheotherone/Alaska_Crash_Analysis.git`.
- Remote heads (via `git ls-remote --heads origin` after
  `git fetch --all --tags --prune`):
  - `integrate-peyton-ml` @ `bb9247a7a7787a141854bb768b58e0f4d17c8e5c` (default branch)
  - `integrate-peyton-ml-v2` @ `10e069128b4b0fea5aee7da458a0cc25418e7983`
  - `main` @ `91071b9b578bd0422649dd7a089f4510892cad6f`
- Remote tags: **none**. Commit `111f6f9` is **not** on any remote branch.
  The entire portfolio lineage (51 commits ahead of `origin/integrate-peyton-ml`,
  0 behind, fast-forward compatible) is local-only at the start of this pass.
- Submodules: none.
- GitHub CLI auth: logged in as `Bobtheotherone`, scopes `gist, read:org, repo`,
  HTTPS protocol. Remote writes are technically possible; policy gates below.

Actions taken:

1. `git fetch --all --tags --prune` — no changes received.
2. `git bundle create pre-r4-complete-backup.bundle --all` — complete-history
   backup written at repo root (7,920,695 bytes) and verified with
   `git bundle verify` → "okay / records a complete history". This bundle is
   local-only and MUST NEVER be committed or pushed (gitignore pattern added in
   Phase 2).
3. `git switch -c portfolio-final-r4 111f6f9` — working branch created.
4. This worklog created at `remediation/paper/RELEASE_WORKLOG_R4.md`.

Ancestry determination for Phase 10: `git merge-base --is-ancestor
origin/integrate-peyton-ml 111f6f9` → **yes**; merge-base is exactly
`bb9247a` (the default-branch tip), so `portfolio-final-r4` is a normal
descendant of the current default branch. A regular push + pull request is
the correct publication path; no `--allow-unrelated-histories`, no force.

Protection invariants confirmed:
- Branch `portfolio-final-r3` and tag `portfolio-final-r3` are left untouched
  for the remainder of this pass (verified again before tagging in Phase 9).
- The r3 release directory `release/portfolio_final_r3/` is preserved
  byte-identical; r4 artifacts go to `release/portfolio_final_r4/`.

---

## Log of subsequent phases

### Phase 1 — Full public-repository privacy audit (2026-07-16)

Executed the audit described in `PUBLIC_REPOSITORY_AUDIT_R4.md` (method + results
there). Highlights:

- Already-public remote surface (3 branches, 0 tags, 0 releases, no LFS): **clean**.
  No `PUBLIC_HISTORY_INCIDENT_R4.md` needed.
- Full local history (1,146 objects, all refs): no data files, no secrets
  (only the `password="password"` test fixture), one **synthetic** run bundle
  (self-declared `SYNTHETIC_STRUCTURAL_FIXTURE`, `is_synthetic: true`).
- **F-R4-01**: the licensed raw workbook's SHA-256 (full value in 3 tracked docs,
  8-char prefix in 8 more; 28 historical blobs, all reachable from `111f6f9`).
  Containment executed in the working tree:
  - created local-only `private_reproduction_log/RAW_WORKBOOK_IDENTITY.md`
    (full value + provenance + restoration procedure);
  - redacted all 11 tracked files (`research/DATA_AUTHORITY_AND_ACCESS.md`,
    `DATA_PROVENANCE.md`, `PROVENANCE_CHAIN.md`, `ISSUE_DISPOSITION_CURRENT.md`,
    `PROVENANCE_ACQUISITION_PLAN.md`, `RECON3_ISSUE_LEDGER.md`,
    `RELEASE_READINESS.md`, `REPRODUCTION_LOG.md`, `research_audit/claim_scan.py`,
    `paper/final_paper.md`, `paper/final_paper.html`) to the manuscript-4.3
    withholding language; `claim_scan.py`'s expectation switched from the hash
    prefix to the withholding phrase;
  - residual working-tree scan: zero occurrences outside the private log
    (one stale ignored `.pyc` cache noted; never tracked).
  - Push-route decision (full lineage vs snapshot branch vs custodian permission)
    is deferred to the Phase 10 gate → `AUTHOR_ACTIONS_R4.md`.
- Governed artifacts untouched: no file under `remediation/evidence_release/` or
  `remediation/experiment/` modified.

### Phase 2 — Exclusion rules strengthened (2026-07-16)

- Root `.gitignore` extended with commented, NDA-reasoned categories:
  licensed workbook filenames + fail-closed `*.xlsx/*.xls/*.xlsm` (no spreadsheet
  in this project is intentionally public; deliberate exception requires
  `git add -f` + privacy review), raw/restricted data dirs, restricted
  reproduction records (`private_reproduction_log/`, `restricted_hashes/`,
  `*.private-manifest.json`), secrets (`.env*` except `.env.example`, `*.pem`,
  `*.key`, `credentials*.json`, `secrets*.json`), local backups and staging
  (`*.bundle` — bundles embed FULL local history — plus `release/`,
  `model_cache/`, `.llm_cache/`, `local_ai_exports/`).
- Verified: `pre-r4-complete-backup.bundle` and `private_reproduction_log/` are
  ignored; `git ls-files -i -c --exclude-standard` shows **zero** tracked files
  caught by the new rules (nothing intentionally public is excluded; the
  de-identified evidence parquets ship via the verified package, unaffected).
- `remediation/.gitignore` already encodes the privacy rationale for
  `_local_data/`, `runs/`, split assignments, and lossless parquets — unchanged.
- Fail-closed packaging enforcement is implemented with the r4 packaging changes
  (Phase 7): package-level forbidden-content scan extended with prohibited paths
  and sensitive tokens (raw-hash tokens sourced from the local-only restricted
  log so the scanner itself never embeds them), and the public handoff's
  provenance bundle restricted to published refs only (no `--all`).

### Phases 3–6 — repository polish + manuscript edits (2026-07-16)

- Phase 3: root README replaced with the 15-section research-forward page
  (application docs preserved at `docs/APPLICATION_PLATFORM.md` via `git mv`);
  new `CITATION.cff` (version portfolio-final-r4, date-released 2026-07-20,
  NO doi field), `DATA_AVAILABILITY.md`, `LICENSES/README.md` (component-scoped
  licensing; MIT stays scoped to `remediation/`), `SECURITY.md`, root
  `REPRODUCIBILITY.md` pointer. All referenced paths existence-checked.
- Phase 4: all five markers resolved in `main.tex` (§12.2 ×4, §12.4);
  `AI_USE_AND_PRIVACY.md` attestation completed; r3-tag self-reference → r4;
  `unresolved_external_evidence.md` item 10 → RESOLVED;
  zero-marker token sweep over the expanded source (8/8 tokens absent) after
  "placeholder"→"sentinel"/"stand-in box" rewording (main.tex line ~347,
  preamble color names, feature-governance generator tabnotes).
- Phase 5: `\ContinuedFloat` on the three continuation parts (main.tex
  artifact-map; generators for analysis-status and retained-fields; files
  regenerated; gates PASS). Build: LoT contiguous 1–17; PDF table numbers
  contiguous; the three continued captions reuse numbers 11/13/15; no
  undefined refs; 0 overfull; the single benign underfull box is identical to
  the r3 build (build4.out).
- Phase 6: availability wording → release-asset model (§12.2 intro, Table 5
  tabnote); CI-claim sentence future-proofed.
- Verifier upgraded for r4 (926 checks: r4 REQUIRED ×9, FORBIDDEN marker
  tokens ×8, locked tag r4, §9a raw-hash withholding scan sourcing the value
  from the restricted log only, §9b continuation/LoT checks; graceful skip of
  the git gate when the governed baseline commit is absent).
- Gates re-run at this state: pytest 96/96 · manuscript verifier 926/926 ·
  frozen/locked 34/34 · audit_paper 42/0/0 · 8/8 generator checks · claim scan
  clean.
- Redline r3→r4 generated with `latexdiff-so --flatten` (both sides flattened
  against the identical `main.bbl`; first attempt without the old-side bbl
  mangled bibliography URLs and was discarded); compiled to 56 pp.
- CI: `verify-portfolio.yml` added (no-license tier only); `crashsev-ci.yml`
  branch filter extended to the release branches and the default branch.
- Packaging tools updated for r4 (see REVISION_MEMO_R4 rows 16–17): r4 docs
  staged, redline r3→r4 shipped, release-metadata block (URL/tag/date/no-DOI),
  fail-closed forbidden-content scans in both packagers, handoff provenance
  carries NO git bundle (public repository is the history provenance).
- r4 reports written: REVISION_MEMO_R4, PUBLIC_REPOSITORY_AUDIT_R4,
  PRIVACY_AND_NDA_ATTESTATION_R4, NUMERICAL_VERIFICATION_R4,
  ADMISSIONS_REVIEW_CHECKLIST_R4, AUTHOR_ACTIONS_R4, GITHUB_PUBLICATION_R4.

### Phases 7–8 — packaging + full verification (2026-07-16)

All executed from the committed r4 content (first content commit, then the
deterministic package/handoff rebuilt from the final commit):

- `package_release.py ../release/portfolio_final_r4`: 12/12 gates re-run inside
  the packager (pytest 96/96; 8/8 generator checks; manuscript verifier
  926/926; frozen/locked 34/34; audit_paper 42/0/0); release assembled;
  **forbidden-content scan clean**; final PDF sha256
  `aafe14106567b82899982e289a165abf71920c69bbdc0b2ebcf804858b5c6c6e`
  (56 pages, 1,020,157 B); source zip sha256
  `92f746e3693541fffde6da731c0751f76b54676b141ca39483ebb2c5b0f7bc4e`.
- `rebuild_identity_check.py`: clean rebuild from the shipped source —
  extracted text identical on all 56 pages (binary differs only by embedded
  build timestamps/ID, the established criterion; rebuild_identity.json).
- `pdf_token_diff.py` vs the r3 PDF (`ad417e67…`): tiny numeric-token diff,
  exactly the enumerated expectation — removed old continuation table numbers
  (18/19/20 + LoT shifts), added duplicated continuation numbers (11/13/15…),
  "20, 2026" (release date), "256" (SHA-256 wording in §12.2); **every
  governed numeric token unchanged** (pdf_token_diff.json).
- `build_handoff.py … --tier1`: deterministic handoff built; **no git bundle
  in the public package** (provenance = public repository at the release
  tag); staged-package forbidden-content scan clean; clean-room extraction
  `VERIFY_HANDOFF.py` **PASS** (integrity, hygiene, privacy, headline-metric
  recompute, pins); clean-room tier-1: pytest 92 passed + 4 skipped
  (extraction-skips by design), 8/8 generator checks, manuscript verifier
  920/920 (6 checks skip in extraction by design: raw-hash log local-only,
  .lot build artifact, git gate), frozen/locked all-skip in extraction by
  design (local-only lock files; gate PASSes at repo packaging time 34/34).
- Release directory finalized: PDF, source zip, stable-named verification ZIP
  `Alaska_Crash_Analysis_UAA_Portfolio_Final_R4.zip`, MANIFEST.json,
  SHA256SUMS.txt over every release file (the ZIP's own hash is recorded
  there, deliberately not repeated here to avoid a self-referential hash
  chain), README_RELEASE.md, CITATION.cff, LICENSE_SCOPE.md, redlines, all
  r4/r3 reports, captured verification JSONs, page renders + contact sheet.
- Note on the second content commit: it adds only this worklog section (a
  record of outcomes, not an input to any verified artifact); the handoff and
  release manifests are rebuilt from the final commit so `release_commit` and
  `canonical/` match it exactly. The PDF is not rebuilt (its source is
  unchanged; hash above remains the build of record).

### Phase 10 — publication (2026-07-16)

- Author chose publication route (a): **snapshot commit** `3d95b7cb` created
  with `git commit-tree` (tree byte-identical to the frozen finalization
  commit `c1ad9bd`; parent = public default tip `bb9247a`); branch and
  annotated tag `portfolio-final-r4` re-pointed to it before any push; the
  full lineage preserved locally at `portfolio-final-r4-local`; handoff
  rebuilt at the snapshot commit (clean-room PASS; MANIFEST `release_commit`
  = `3d95b7cb`); release SHA256SUMS surgically refreshed for the two files
  that change with the rebuild (MANIFEST.json, the stable-named ZIP, plus the
  re-captured cleanroom_results.json) and every one of the 48 lines
  re-verified.
- GitHub auth: the CLI token initially lacked the `workflow` scope (push
  rejected); resolved with an author-supplied PAT consumed from a local file
  (deleted immediately; never in chat or repo).
- Pushed `refs/heads/portfolio-final-r4` and `refs/tags/portfolio-final-r4`
  (no force). PR #2 opened against `integrate-peyton-ml`. Repository
  description + 11 topics set; no homepage.
- **First hosted CI executions ever for this project.** Results on the
  snapshot commit: 3.12/3.13 green on BOTH OSes (8 matrix-cell/step successes
  overall) with two defect classes, both CI-infrastructure, zero scientific
  impact:
  1. `xgboost==3.3.0` (the frozen environment's pin) requires Python >=3.12,
     so the 3.11 matrix cells cannot install the lock → 3.11 removed from the
     matrix; `research/REPRODUCE.md` environment note corrected to
     Py 3.12–3.13 (its "not run on hosted CI / branch unpublished" sentence
     was also stale after publication and was updated).
  2. `verify_manuscript_numbers.py` artifact-existence check for the two
     policy-gitignored parquets knew only two resolutions (working tree,
     package evidence tier); in a bare hosted checkout neither exists → added
     the third resolution: verify the committed pin sidecar and emit a
     SKIP-NOTE (the parquets ship in the release evidence tier and are
     hash-checked at packaging). Local verifier remains 926/926.
- These fixes are follow-up commits on the public branch; the release tag
  remains on the verified snapshot commit unless the author approves
  re-pointing before the release is drafted (recorded in
  `GITHUB_PUBLICATION_R4.md`).
