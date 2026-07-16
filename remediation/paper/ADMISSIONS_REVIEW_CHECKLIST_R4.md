# Admissions review checklist — r4 final-submission release (2026-07-16)

Verdict at r4 finalization: **PASS WITH EXTERNAL PUBLICATION ACTION** — the
manuscript itself is submission-ready with zero editorial markers; the only
remaining items are external publication actions (push-route decision, release
publication on 2026-07-20, optional DOI), enumerated in `AUTHOR_ACTIONS_R4.md`.

## Manuscript readiness

- [x] All five r3 `[AUTHOR ACTION REQUIRED]` markers resolved in the LaTeX
      source and absent from the built PDF (verifier FORBIDDEN, 8 tokens).
- [x] Real public repository URL and r4 release URL in §12.2.
- [x] Contact stated: rnmercado@alaska.edu (no implied DMV access).
- [x] July 20, 2026 treated as the release date; truthful no-DOI statement;
      **no fabricated DOI anywhere** (CITATION.cff carries no DOI field).
- [x] Raw workbook SHA-256 withheld (restricted reproduction log); tex+PDF
      machine-scanned for both the value and its 8-char prefix (verifier §9a).
- [x] AI/privacy attestation in §12.4 + `AI_USE_AND_PRIVACY.md` §3: sensitive
      DMV data processed locally only; not supplied to nonlocal LLMs; external
      AI limited to code/docs/synthetic/aggregates/manuscript/de-identified.
- [x] List of Tables contiguous (1–17); continuation pages reuse their original
      numbers via `\ContinuedFloat` ("part 2 of 2 (continued)"); machine-checked
      (verifier §9b).
- [x] Handoff-package language matches actual availability (release asset).
- [x] Abstract 343 words (within 275–350); claim-boundary language intact.
- [x] 56 pages; searchable text; fonts embedded; no overfull boxes; the single
      benign underfull alignment box is unchanged from r3; no undefined
      references or citations; PDF metadata correct.

## Scientific invariants (unchanged, machine-anchored)

- [x] Primary oMAE 0.3469 · majority 0.3614 · Δ −0.0144 · [−0.0231, −0.0063]
      · severe recall 5.8% · run `final_8af9d5bc23d8` — locked-value gate.
- [x] Frozen governed artifacts byte-identical (frozen/locked 34/34).
- [x] 2012 oMAE leader (unweighted RF 0.3401) stated plainly; primary-role
      designation explained (frozen registry, not post-hoc selection).
- [x] Not prospective / preregistered / causal / independently replicated /
      deployment-ready / outcome-blind — all enforced by FORBIDDEN + REQUIRED
      language gates.

## Verification gates (this build)

- [x] pytest 96/96 (`--basetemp` note documented)
- [x] verify_manuscript_numbers 926/926 (r3: 893 + 33 r4 release-state checks)
- [x] verify_frozen_and_locked 34/34
- [x] audit_paper 42 pass / 0 warn / 0 fail
- [x] 8/8 generator `--check` gates
- [x] claim scan clean
- [x] packaging-time: forbidden-content scans (release dir + handoff stage),
      clean-room extraction + `VERIFY_HANDOFF.py`, rebuild-identity and
      numeric-token-diff gates — captured in `gate_outputs.json` /
      `FINAL_VERIFICATION_REPORT.md` / `pdf_token_diff.json`.

## Public repository readiness

- [x] Full-history privacy audit complete (`PUBLIC_REPOSITORY_AUDIT_R4.md`):
      remote surface clean; no data files or secrets on any ref; F-R4-01
      (raw-workbook hash in the local lineage) contained — tree redacted,
      restricted log created, packaging fails closed, push route gated on the
      author's decision.
- [x] 15-section faculty-facing README; CITATION.cff (version
      portfolio-final-r4, date-released 2026-07-20); DATA_AVAILABILITY.md;
      SECURITY.md; component-scoped licensing (`LICENSES/README.md` — MIT
      remains scoped to `remediation/`; manuscript text not MIT; earlier
      contributors' rights retained).
- [x] Public CI = no-license tier only (`verify-portfolio.yml`); no badge
      added before a green hosted run exists.

## External actions (not manuscript defects)

- [ ] Author decision on the publication route for the git lineage
      (F-R4-01; options in `AUTHOR_ACTIONS_R4.md`) → push → PR → CI green.
- [ ] GitHub release published July 20, 2026 (draft prepared earlier; assets +
      SHA256SUMS attached; publication before that date only with explicit
      author approval).
- [ ] Optional: Zenodo deposition + DOI (requires author login); if minted,
      update §12.2/CITATION.cff wording in a subsequent tagged revision — do
      not edit the frozen r4 artifacts in place.
