# Reproducibility

The authoritative reproduction guide for the Iteration IV research portfolio is
**[`remediation/research/REPRODUCE.md`](remediation/research/REPRODUCE.md)**.

Two tiers:

- **Tier A — no licensed data (anyone).** Installs from
  `remediation/requirements-lock.txt`, runs the governance/failure-mode test
  suite, regenerates the re-analysis matrices and every figure, and exactly
  recomputes all reported metrics from the released de-identified predictions in
  the verification handoff ZIP (a `portfolio-final-r4` release asset; verify with
  the packaged `VERIFY_HANDOFF.py`, fully offline).
- **Tier B — licensed data.** With a lawful copy of the restricted extract,
  rebuilds the modelling table byte-identically, reproduces the frozen governed
  hashes, and re-executes the governed pipeline; see
  `remediation/research/REPRODUCE.md` §B and
  [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

Historical note: `remediation/REPRODUCIBILITY.md` is a superseded tombstone from
the pre-data ("Route B synthetic fixture") phase and is retained only so old
links do not dangle.
