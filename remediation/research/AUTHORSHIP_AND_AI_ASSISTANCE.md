# Authorship, Contribution, and AI-Assistance Statement

**Applicant / submitting author:** Naythan Mercado.
**Artifact:** the `crashsev` remediation study and paper (`paper/final_paper.md`), branch
`portfolio-research-finalization-v4` (originally drafted on the v3 branch; the release record
is `research_audit/RELEASE_STATE.md`).
**Date:** 2026-07-10.

This statement is a condition of submitting the work as portfolio material. It is written to be
truthful about who did what and about the role of AI assistance. It does **not** claim that the
applicant personally hand-wrote AI-generated code or prose.

## 1. Provenance of the two layers

The repository contains two distinct layers:

1. **Legacy 2025 capstone** (Django app, `alaska_ui/`, `analysis/`, `peyton_original/`, the four
   original classifiers and the original report/poster). This was **group coursework** and is
   included only as historical context and as the object of the re-analysis. Its authorship is the
   original student group; it is **not** claimed as the applicant's individual work. See
   `research_audit/PRIOR_WORK_LINEAGE.md`.
2. **Remediation study** (`crashsev/` pipeline, `paper/final_paper.md`, the governance/contract/test
   layer, and the research control documents). This is the portfolio contribution.

## 2. Material AI assistance (disclosed)

The remediation layer was produced with **material assistance from an AI coding/research agent**
(Anthropic Claude, "Fable-5 / Claude Code" harness), under the applicant's direction. AI assistance
materially contributed to the following categories:

* **Code generation and editing** — the `crashsev` package, tests, and configuration.
* **Statistical and methodological review** — leakage controls, ordinal metrics, decision-rule and
  uncertainty analysis, and the Route-R reframing.
* **Documentation and technical writing** — drafting of the paper and the research control documents.
* **Literature-search support** — locating and formatting references (each citation must still be
  independently verified by the applicant; see the reference list in `paper/final_paper.md`).
* **Packaging and reproducibility engineering** — the handoff, manifest, and verifier.

This disclosure is deliberately broad: a reviewer should assume AI assistance touched every part of
the remediation layer. What AI assistance did **not** do is relieve the applicant of responsibility
for understanding and defending the work.

## 3. Applicant responsibility attestation

The applicant, Naythan Mercado, is responsible for understanding and being able to explain and
defend every claim in the paper. Before submission the applicant should complete the checklist
below (self-attestation; not a claim of sole authorship):

- [ ] I can state the research question, the estimand, and what is explicitly **not** claimed.
- [ ] I can explain the target mapping (KABCO → 3 classes, blank = PDO) and why it is
      evidence-checked, not officially validated.
- [ ] I can explain the crash-grouped chronological split and why 2012 is a **retrospective,
      exposed** out-of-time evaluation, not a sealed holdout.
- [ ] I can explain the leakage controls (feature-availability ledger; fold-local preprocessing) and
      the leakage factorial result (outcome-derived features dominate).
- [ ] I can define ordinal MAE and explain the argmax vs posterior-median decision-rule finding.
- [ ] I can interpret the primary comparison, its confidence interval, and why "significant" is
      avoided and multiplicity is handled by declaring one primary comparison.
- [ ] I can explain the calibration selection rule (development-only) and the reproducibility status
      (self-reproduction, not independent).
- [ ] I can walk through the main code path (`crashsev/cli.py` phases) and the governance lock.
- [ ] I can state the limitations (exposure, data authority, dependence approximation, external
      validity).

## 4. Integrity boundary

No result, interval, citation, hash, or run log in this project is fabricated. Where the applicant
cannot yet defend a specific element, that element should be removed or explicitly marked as not
independently verified rather than presented as understood. The AI assistance disclosed here is a
statement of fact, not a disclaimer of responsibility.
