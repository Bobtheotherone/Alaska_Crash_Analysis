# Provenance record

## Repository under remediation

| Item | Value |
|---|---|
| Remote | `https://github.com/Bobtheotherone/Alaska_Crash_Analysis.git` |
| Base branch | `integrate-peyton-ml` (repo default) |
| Base commit | `bb9247a7a7787a141854bb768b58e0f4d17c8e5c` ("Delete AGENTS.md", 2025-12-11) |
| Working branch | `graduate-research-remediation` |
| Other branches present | `integrate-peyton-ml-v2`, `main` |
| Tags | none |
| Total commits | 12 |
| Dataset in repo | none (confirmed) |

The base commit matches the exact commit the reconnaissance report examined.

## Input artifacts (user-supplied documentation bundle) — SHA-256

| File | SHA-256 |
|---|---|
| `Alaska_Crash_Analysis_Graduate_Reconnaissance.md` | `6d2cb794833a6e278a659168b0e5b56ed35c53aceb4ba08a163f2a788d1ce6fe` |
| `Alaska Car Crash Analysis Final Report.docx` | `c590e46d313fef71df7c1e0ded16e1aabdb8536314c1ecdc69b86618470ff7c3` |
| `Car Crash Analysis Machine Learning Documentation.docx` | `8ed5de7c68dcd35837153d055271a7e72585362297eda72f8e3ea0434c46a828` |
| `Repository_Component_Breakdown.docx` | `ffc250f746bc3f145cdef93b08e4d113340f86dd837ed834adeb01675139b4ca` |

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 (win32) |
| Python | 3.13.13 |
| Key libraries | numpy 2.4.4, pandas 3.0.3, scikit-learn 1.9.0, scipy 1.17.1, matplotlib 3.10.9, xgboost 3.3.0, interpret 0.7.8, PyYAML 6.0.3, pytest 8.4.2 |
| Pinned lock | `remediation/requirements-lock.txt` — 13 direct packages, fully pinned (replaced the earlier whole-env freeze; AA2-009) |

## Re-analysis input provenance

The four confusion matrices in `data/confusion_matrices_from_paper.json` were transcribed
from the embedded result screenshots of the final report
(`word/media/image9.png` = Decision Tree, `image11.png` = XGBoost, `image13.png` = MLRF/RF,
`image15.png` = EBM). Each matrix's row sums equal the reported class supports
(7395 / 3152 / 389), and the recomputed metrics reproduce the reconnaissance report's table exactly (this project, not an independent third-party reproduction; validated by `tests/test_split_and_pipeline.py::test_metrics_reproduce_published_confusion_matrices`).

## Reproducibility of this remediation

Every governed run (`crashsev.cli evaluate-final`) writes an immutable `runs/<id>/` bundle and a
committed `experiment/final_results.json` capturing the git commit + dirty flag, canonical data
content SHA-256, config/contract SHA-256s, seeds, library/OS versions, split hash, target audit,
metrics, and crash-level (case/row) bootstrap CIs (under a cross-crash independence approximation). The **real** Alaska result and the re-analysis are fully
regenerable via `research/REPRODUCE.md`; a **bit-exact** self-reproduction re-run (same project, not an independent third party; all metrics
and seeded bootstrap CIs identical to 0.000e+00) is recorded in `research/REPRODUCTION_LOG.md`.
