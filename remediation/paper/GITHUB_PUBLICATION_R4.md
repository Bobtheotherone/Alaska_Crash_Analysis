# GitHub publication record — r4

Status as of 2026-07-16 (finalization). This record is updated at each remote
milestone; fields marked *awaiting* are completed at the corresponding gate.

| Field | Value |
|---|---|
| Repository | https://github.com/Bobtheotherone/Alaska_Crash_Analysis (public) |
| Default branch (unchanged) | `integrate-peyton-ml` @ `bb9247a` |
| Working branch | `portfolio-final-r4` (local), created from r3 baseline `111f6f9` |
| Ancestry | `111f6f9` is a fast-forward descendant of the default branch (51 ahead / 0 behind) — normal PR possible on route (b); route (a) publishes a snapshot commit on the default branch tip |
| Push | *awaiting the author's route decision* (`AUTHOR_ACTIONS_R4.md` §1); pre-push privacy scan complete (`PUBLIC_REPOSITORY_AUDIT_R4.md`); no force-push under any route; backup bundle excluded by `.gitignore` |
| Pull request | *awaiting push* — title: "Finalize UAA graduate research portfolio r4" |
| CI | `.github/workflows/verify-portfolio.yml` (no-license tier only; no restricted data; history gates skip gracefully on shallow/snapshot checkouts) + `crashsev-ci.yml` (branch filter extended). Hosted status *awaiting push*; no badge until green |
| Final tag | annotated `portfolio-final-r4` — created locally at finalization; under route (a) it is re-created on the public snapshot commit before anything is pushed (the tag is never moved after publication) |
| Release | title "UAA Graduate Research Portfolio — Final Submission r4"; DRAFT after CI green; **publication date 2026-07-20** (earlier only with explicit author approval) |
| Release assets (planned) | final PDF · verification handoff ZIP · manuscript source ZIP · SHA256SUMS.txt · MANIFEST.json · CITATION.cff · REVISION_MEMO_R4.md · NUMERICAL_VERIFICATION_R4.md · redline_r3_to_r4.pdf |
| Asset hashes | recorded in `release/portfolio_final_r4/SHA256SUMS.txt` at packaging and re-verified against the uploaded assets before publication |
| DOI | **none issued** — manuscript states: "No archival DOI had been issued at the time this submission package was finalized. The tagged public release is dated July 20, 2026." Zenodo requires author login (`AUTHOR_ACTIONS_R4.md` §3); never fabricated |
| Repository metadata (after push) | description: "Secure Alaska crash-analysis platform and governed, reproducible ordinal crash-severity evaluation."; topics: machine-learning, ordinal-classification, temporal-validation, data-leakage, reproducible-research, traffic-safety, crash-analysis, django, react, alaska, uaa; no homepage set |
| GitHub releases before r4 | none (API-verified 0) |
| Remote tags before r4 | none (ls-remote verified) |
