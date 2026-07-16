# Privacy and NDA attestation of record — r4 (2026-07-16)

Author of record: **Radames Naythan Mercado-Barbosa** (rnmercado@alaska.edu),
University of Alaska Anchorage.

This document records, as the attestation of record for the r4 final-submission
release, the privacy facts supplied by the author for this pass. The same
attestation appears in the manuscript (§12.4 "Authorship and AI assistance")
and in `paper/AI_USE_AND_PRIVACY.md` §3.

## Author-supplied facts (recorded verbatim in substance)

1. The Alaska DMV crash data were supplied under an NDA and associated
   data-use restrictions.
2. Those restrictions were respected.
3. Sensitive source data were processed only in local computing environments.
4. No NDA-protected source records were supplied to nonlocal LLMs,
   cloud-hosted generative-AI systems, or other external AI services.
5. External AI assistance may have operated on code, manuscript text,
   synthetic examples, aggregate results, and properly de-identified release
   artifacts.
6. Only local models and locally executed analytical code processed sensitive
   source records.

## Scope discipline

This attestation is deliberately bounded. It does **not** claim that no
internet-connected software was ever used on the workstation, that AI
assistance was absent (it was material, and is disclosed), or that the
release-side controls substitute for it. It distinguishes:

- **restricted source records** (raw crash rows, real crash identifiers, exact
  coordinates, free-text locations, officer/agency identifiers): local-only,
  never supplied to external AI services, never distributed;
- **non-restricted working material** (source code, documentation excerpts,
  synthetic examples, aggregate statistics, manuscript text, de-identified
  artifacts that passed the release-side privacy controls): external AI
  assistance operated on this class, under the author's direction.

## Machine-checked complements (repository evidence, not attestation)

- Release-side: the handoff verifier's forbidden-content scan and the r4
  packaging scans confirm no raw rows, identifiers, coordinates, free-text
  locations, or agency identifiers in any released artifact
  (`PUBLIC_REPOSITORY_AUDIT_R4.md`; `gate_outputs.json` at packaging).
- Repository-side: the r4 full-history audit found no data files or secrets on
  any ref; the licensed workbook's byte identity is withheld from all public
  artifacts and retained in a local-only restricted reproduction log
  (F-R4-01 containment).
- Handling policy: the licensed extract and every per-crash derivative live
  only in git-ignored local paths (`_local_data/`, `runs/`,
  `private_reproduction_log/`).

## Responsibility

The author designed the study, made all methodological decisions, executed and
verified the analyses, and accepts responsibility for the complete manuscript.
No AI system is an author.
