# Author actions — r4 (2026-07-16)

**No unresolved author actions remain in the manuscript.** All five r3 markers
are resolved in the source and absent from the PDF (machine-checked). The items
below are the only remaining human actions, and all are *external publication*
steps, not manuscript work.

## 1. REQUIRED — choose the publication route for the git lineage (blocks the push)

The r4 privacy audit (`PUBLIC_REPOSITORY_AUDIT_R4.md`, finding F-R4-01) shows
the licensed raw workbook's SHA-256 in the *unpublished local history* that
leads to the r3/r4 baseline (5 full-value blobs, 23 prefix blobs, and one
commit-body prefix at `c5a513e` — all reachable from `111f6f9`). The r4
**working tree** is fully redacted, and no release asset carries the value; but
pushing the full lineage would publish those historical objects, which
contradicts the manuscript's own §12.2 statement that the identifier is
withheld pending custodian permission. Pick one:

- **(a) Snapshot publication — recommended default.** Publish the r4 tree as a
  fresh commit on top of the public default branch (`integrate-peyton-ml`);
  tag THAT commit `portfolio-final-r4`; open the PR from it. Publishes zero
  historical hash-bearing objects; a normal PR (no unrelated histories, no
  force-push); the full local lineage stays intact locally and in the r3
  controlled-delivery handoff. Cost: the public repo shows the portfolio as
  one release commit rather than 51 development commits (the development
  record is preserved in the manuscript's Appendix F, the worklogs, and the
  packaged provenance).
- **(b) Obtain custodian permission first.** If the data owner confirms the
  hash may be published, push the full lineage as-is (fast-forward from the
  default branch) and enjoy the complete public development history. Requires
  a documented permission (record it in the restricted reproduction log).
- **(c) Local history scrub before first publication.** Rewrite the ~51
  never-pushed commits to strip the hash, then push the rewritten lineage.
  NOT recommended: it re-identifies every commit including the frozen r3
  baseline `111f6f9` referenced throughout the governance record, for little
  gain over (a).

*How to proceed:* reply with (a), (b), or (c). Everything else is staged so the
chosen route is a few commands, executed on your confirmation.

## 2. REQUIRED — publish the GitHub release on July 20, 2026

The draft release (title "UAA Graduate Research Portfolio — Final Submission
r4", tag `portfolio-final-r4`, all assets + SHA256SUMS) is prepared after the
push. Publication before 2026-07-20 needs your explicit approval; on/after
that date, confirm and it goes live.

## 3. OPTIONAL — archival DOI (Zenodo)

No Zenodo credential exists on this workstation, so no DOI was reserved and
the manuscript truthfully states none had been issued. If you want a DOI:
log in at zenodo.org (GitHub sign-in works), either enable the repository
under "GitHub" in Zenodo settings (a DOI is minted when the GitHub release is
published) or create a deposition and reserve a DOI, then provide the exact
DOI string. It can be added to CITATION.cff and the release notes immediately;
the PDF's availability wording would only change in a later tagged revision
(the frozen r4 artifacts are never edited in place).

## 4. Standing items carried from r3 (external evidence, unchanged)

Gate-3 source-agency provenance (custodian attestation, official codebook,
field-timing documentation) and any departmental template sign-off remain
external items documented in `unresolved_external_evidence.md`; they are
disclosed in the manuscript and are not release blockers.
