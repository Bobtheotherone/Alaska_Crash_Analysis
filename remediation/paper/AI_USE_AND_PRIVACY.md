# AI-use and privacy record (r4 final-submission pass, 2026-07-16)

This record accompanies the manuscript section "Authorship and AI assistance." It
separates (1) what is verifiable from the repository, (2) what the author
discloses, and (3) what the author attests. The attestation in §3 was supplied by
the author for the r4 submission pass and is recorded here verbatim in substance;
no repository artifact can prove it, which is why it is an attestation.

## 1. Verifiable from the repository and release (machine-checked)

- Every released artifact is de-identified: the handoff verifier's forbidden-content
  scan (`VERIFY_HANDOFF.py`) rejects raw spreadsheets, secrets, coordinates, and
  non-surrogate crash identifiers; prediction evidence is surrogate-keyed.
- Raw crash rows, exact coordinates, free-text locations, officer/report/agency
  identifiers, and real crash numbers appear in NO committed or shipped artifact
  (`runs/` bundles and the licensed modeling table are local-only by policy).
- Every reported numerical claim is pinned to a frozen artifact by the manuscript
  verifier; the verification suites and their totals are captured at packaging time.
- The r4 public-repository audit (`PUBLIC_REPOSITORY_AUDIT_R4.md`) additionally
  scanned every blob on every ref for secrets and restricted content.

## 2. Disclosed AI-supported activities (author-directed)

Generative-AI tools were used under the author's direction for:

- code scaffolding and editing (analysis pipeline, verification tools, tests);
- debugging suggestions;
- statistical and editorial critique of drafts;
- documentation assistance (READMEs, ledgers, summaries);
- figure and LaTeX typesetting support;
- reproducibility-package review and release packaging.

Tools: Anthropic Claude models operated through the Claude Code assistant, as
recorded in the repository's commit trailers (e.g., "Claude Opus 4.8", "Claude
Fable 5"), over the development period reflected in the repository history
(application platform work, late 2025; Iteration IV remediation and release
passes, July 2026). Locally executed models and locally executed analytical code
were the only systems that processed sensitive source records (§3).

The author set the research questions, target and feature-tier contracts, and the
frozen protocol; made the methodological and interpretive decisions; directed and
executed the governed analyses; inspected generated artifacts; and ran the
machine-verification suites. The author accepts responsibility for the complete
manuscript. No AI system is an author.

## 3. Author attestation — privacy of project material relative to external AI services (completed r4)

The author attests:

> All licensed Alaska DMV crash records were handled under the applicable NDA and
> data-use restrictions, and those restrictions were respected. Sensitive source
> records were processed only in local computing environments by locally executed
> models and analysis code. No raw crash rows, real crash identifiers, exact
> coordinates, free-text locations, officer or agency identifiers, or other
> NDA-protected source data were transmitted to or processed by nonlocal language
> models, cloud-hosted generative-AI systems, or other external AI services.
> External AI assistance was limited to source code and documentation excerpts,
> synthetic examples, aggregate statistics, manuscript text, and de-identified
> artifacts that had passed the release-side privacy controls.

Basis recorded with the attestation: local-only handling policy for the licensed
extract and its per-crash derivatives (git-ignored `_local_data/` and `runs/`
paths), de-identification and forbidden-content gates on everything released, and
the author's review of their own tool usage. This attestation is deliberately
scoped: it does not claim that no internet-connected software was ever used on the
workstation, and it distinguishes restricted source records (never supplied to
external AI services) from code, manuscript text, synthetic examples, aggregates,
and de-identified artifacts (which external AI assistance did operate on).

The corresponding privacy statement appears in the manuscript's "Authorship and
AI assistance" subsection (§12.4). The former resolution instruction for that
subsection's marker is closed as of r4: the marker has been replaced by the
attestation above.
