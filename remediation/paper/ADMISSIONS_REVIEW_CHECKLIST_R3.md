# Final admissions-review checklist — r3 (2026-07-15)

Verdict: **PASS WITH AUTHOR ACTION** — every technical acceptance criterion is met and
machine-verified; the five visible `[AUTHOR ACTION REQUIRED]` markers (repository URL,
DOI, raw-hash permission, reproduction contact, external-AI privacy attestation) plus
the human sign-off items in `AUTHOR_ACTIONS_R3.md` remain the author's to resolve
before archival submission.

| # | Acceptance criterion | Status | Evidence |
|---|---|---|---|
| 1 | Appendix B F1 error corrected | PASS | Table 6 majority macro-F1 0.269; Table 7 severe F1 0.000; §6.4 convention; regression tests; verifier positional cells |
| 2 | Every table title accurately describes its contents | PASS | "Complete" tables generated with every §6.4-declared metric; two-part tables titled "part 1/2 of 2"; verifier 893/893 |
| 3 | Selection rule fully specified | PASS | §7.1 (eligibility, aggregation, tie-break, refit seed 42, no search); Appendix F complete record; generation-time semantic assertions |
| 4 | Frank–Hall probability construction documented and validated | PASS | §6.2.1; FH-CONV-001 sidecar block; validity tests over all 162,820 released rows; independent recompute check 7 |
| 5 | Bootstrap interval method fully specified | PASS | §6.5 (paired percentile 95%, seed 42, no refit, interpolation); BOOT-CONV-001 sidecar block + behavioural pin test |
| 6 | Every artifact path anchored to a discoverable release or bundle | PASS | §12.2 anchoring sentence (release tag + `canonical/remediation/`); verifier artifact-existence scan over the expanded manuscript |
| 7 | Retained and prohibited feature lists inspectable | PASS | Appendix G (49 retained with per-year missingness; 14 prohibited; disposition summary), generated + reconciled |
| 8 | Attribution neutral and precise | PASS | Rewritten roles page; no-endorsement sentence; defensive wording gated FORBIDDEN |
| 9 | AI assistance and data privacy disclosed accurately | PASS with author action | Precise disclosure + machine-checked release privacy; external-AI attestation is marker 5 (cannot be auto-verified) |
| 10 | Abstract clear and substantially tighter | PASS | 343 words, within the 275–350 target (was ~640); all six required elements; every governed number retained |
| 11 | Terminology does not overstate prospective risk or deployment validity | PASS | "severe-class score ranking"; claim-boundary paragraph; FORBIDDEN list extended and green (tex+PDF) |
| 12 | All frozen values preserved unless documented correction required | PASS | Frozen gate 34/34; token diff fully enumerated (REVISION_SUMMARY §r3.7); zero governed-number changes; F1 correction documented as reporting-convention change |
| 13 | PDF builds cleanly | PASS | 56 pp; 0 errors, 0 overfull, 0 infinite-glue, 0 undefined refs/citations; fonts embedded; bookmarks/links checked (audit_paper 42/0/0) |
| 14 | Paper understandable and auditable without private explanation | PASS with author action | Availability section + manifest + verification commands; markers 1–4 complete the public-discoverability chain |
| 15 | Headline metrics independently recomputed from released evidence | PASS | Independent own-implementation recompute: 11/11 (bit-exact except two documented 1-ulp float-path artifacts) |

Machine gate totals for this build: pytest 96/96 · manuscript verifier 893/893 ·
frozen/locked 34/34 · audit_paper 42 pass/0 warn/0 fail · 8/8 generator checks ·
independent recompute 11/11.
