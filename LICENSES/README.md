# License scope by component

This repository accumulates work from multiple project iterations and
contributors. Licensing is therefore **component-scoped**; there is deliberately
no root LICENSE file claiming the whole tree.

| Component | Paths | License / rights |
|---|---|---|
| Iteration IV research, analysis, and verification code and documentation | `remediation/` | **MIT** — [`remediation/LICENSE`](../remediation/LICENSE), copyright (c) 2026 Radames Naythan Mercado-Barbosa. The license text carries an explicit scope note: it grants **no rights to the restricted Alaska crash data** or any per-crash derivative. |
| Manuscript text (the research paper itself) | `remediation/paper/latex/`, built PDF release asset | **Not MIT.** The manuscript is the author's scholarly work, distributed for review and citation; all other rights reserved by the author. (The LaTeX *build tooling* in `remediation/tools/` is MIT as part of `remediation/`.) |
| Iteration III application platform | `alaska_project/`, `ingestion/`, `crashdata/`, `alaska_ui/`, `frontend/`, `docs/`, root app files (`manage.py`, `docker-compose.yml`, …) | **No explicit license granted.** Multi-contributor capstone platform code, preserved as project history; contributors' rights are retained. Do not reuse outside this repository without permission from the contributors. |
| Earlier iteration materials | `peyton_original/`, `analysis/`, `ml_partner_adapters/` | **No explicit license granted.** Original contributors' work, preserved unmodified for lineage; contributors' rights are retained. |
| Restricted source data | *never in this repository* | Licensed from the data owner under NDA/data-use restrictions; see [`../DATA_AVAILABILITY.md`](../DATA_AVAILABILITY.md) and `remediation/research/DATA_LICENSE_NOTE.md`. Nothing here is a data-release license. |

Why this shape: the paper states that the *Iteration IV analysis and verification
code* is released under MIT terms — that statement is scoped to `remediation/` and
is implemented by `remediation/LICENSE`. Placing a root MIT license would silently
relicense earlier contributors' platform work, which the current rights record
does not support; scoping the grant keeps the public claim exactly as broad as the
permissions behind it.
