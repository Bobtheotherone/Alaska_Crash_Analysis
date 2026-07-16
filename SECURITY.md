# Security Policy

## Scope

This repository is a **research archive and portfolio**, not an operated service.
The Iteration III Django/React platform is archived scope: it is not deployed by
this project, is not production-hardened, and **no deployment is endorsed**
(known local-dev caveats are disclosed in
`remediation/research/ISSUE_DISPOSITION_CURRENT.md`, item SEC-LEGACY-001).

## Reporting a vulnerability

If you find a security issue in the code, or — more importantly — anything in the
repository, its history, or its release assets that looks like restricted data
(raw crash rows, real crash identifiers, exact coordinates, officer/agency
identifiers) or a live credential, please report it privately:

- Email: **rnmercado@alaska.edu** (subject line starting with `SECURITY:`)
- Please do not open a public issue for suspected data exposure.

You should receive an acknowledgment within a few days. Suspected restricted-data
exposure will be treated as the highest priority (containment first, then
disclosure notes in the release documentation).

## Data handling

The licensed Alaska DMV source data are restricted under an NDA/data-use
agreement and are never distributed here; release packages pass a
forbidden-content scan and a de-identification witness check before publication.
See `DATA_AVAILABILITY.md`.
