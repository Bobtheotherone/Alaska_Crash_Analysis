# GitHub publication record — r4

Completed 2026-07-16 after publication of the repository content. The one
remaining scheduled action is publishing the drafted release on 2026-07-20.

| Field | Value |
|---|---|
| Repository | https://github.com/Bobtheotherone/Alaska_Crash_Analysis (public) |
| Default branch (unchanged) | `integrate-peyton-ml`; after PR #2 its tip is merge commit `a47fe3f` and it contains the full r4 portfolio |
| Publication route | **Snapshot publication** (route (a), author-approved): snapshot commit `3d95b7cb` (tree byte-identical to the frozen local finalization commit `c1ad9bd`) + hosted-CI fix commit `8227e6e` (workflow matrix, verifier CI-path resolution, REPRODUCE.md note — no manuscript/artifact change). The 51-commit development lineage remains local (`portfolio-final-r4-local`) and in the controlled r3 delivery; rationale: `PUBLIC_REPOSITORY_AUDIT_R4.md` F-R4-01 |
| Branch | `portfolio-final-r4` @ `8227e6e5b55d5f513a56dbec4a56c13e38962a8f` (pushed, no force) |
| Pull request | https://github.com/Bobtheotherone/Alaska_Crash_Analysis/pull/2 — "Finalize UAA graduate research portfolio r4" — **MERGED** 2026-07-16T12:30:09Z (merge commit, branch preserved) |
| CI | First hosted runs on `3d95b7cb` surfaced two infrastructure defects (Python 3.11 vs the `xgboost==3.3.0` pin; bare-checkout resolution of the policy-gitignored parquets) — fixed in `8227e6e`. On `8227e6e`: **all green** — `verify-portfolio` and `crashsev-ci` on push + pull_request events (runs 29497940674, 29497943068; Linux+Windows, Py 3.12/3.13) |
| Final tag | annotated `portfolio-final-r4` → commit `8227e6e` (tag object `fd3a28c0`). Initially pushed at `3d95b7cb`; re-pointed once, **pre-release and with explicit author approval**, onto the all-checks-green commit; never moved after the release exists |
| Release | "UAA Graduate Research Portfolio — Final Submission r4", **DRAFT** (id 355073567) on tag `portfolio-final-r4`; on publication the URL is https://github.com/Bobtheotherone/Alaska_Crash_Analysis/releases/tag/portfolio-final-r4 |
| Publication date | **2026-07-20** (draft published on that date; earlier only with explicit author approval) |
| Release assets (9, uploaded; API digests verified == local files) | `Mercado-Barbosa_UAA_Student_Paper_Final_Submission.pdf` `sha256:aafe14106567b82899982e289a165abf71920c69bbdc0b2ebcf804858b5c6c6e` (56 pp) · `Alaska_Crash_Analysis_UAA_Portfolio_Final_R4.zip` `sha256:b0ddb9806cf851434751f9dbf75a72c90bd8b66310a54c0a313f0c82f2f185d1` (518 entries; MANIFEST `release_commit` = `8227e6e`) · `Mercado-Barbosa_UAA_Student_Paper_Final_Submission_Source.zip` `sha256:92f746e3693541fffde6da731c0751f76b54676b141ca39483ebb2c5b0f7bc4e` · `SHA256SUMS.txt` · `MANIFEST.json` · `CITATION.cff` · `REVISION_MEMO_R4.md` · `NUMERICAL_VERIFICATION_R4.md` · `redline_r3_to_r4.pdf` |
| DOI | **None issued.** Manuscript statement: "No archival DOI had been issued at the time this submission package was finalized. The tagged public release is dated July 20, 2026." Optional Zenodo registration remains an author action (`AUTHOR_ACTIONS_R4.md` §3); no DOI was fabricated anywhere |
| Repository metadata | description "Secure Alaska crash-analysis platform and governed, reproducible ordinal crash-severity evaluation."; topics: alaska, crash-analysis, data-leakage, django, machine-learning, ordinal-classification, react, reproducible-research, temporal-validation, traffic-safety, uaa; no homepage |
| Pre-r4 remote state (verified before any write) | 3 branches, 0 tags, 0 releases, no LFS |
