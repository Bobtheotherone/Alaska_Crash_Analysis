# Figure Replacement Guide

The revised manuscript deliberately uses placeholders by default. To insert a local figure, export it using the exact filename below and place it in `figures/user/`. The LaTeX source detects it automatically and replaces the placeholder without any other edits.

Preferred format: vector PDF. High-resolution PNG is also accepted at 300 dpi or higher. Do not put a large headline inside the image; the manuscript caption supplies the title. Use a white background, black/gray axes and labels, and one restrained accent color only when it carries meaning. Keep all text at least 8 pt after final scaling.

| Figure | Required filename | Maximum displayed frame | Intended content |
|---|---|---:|---|
| 1 | `fig01_project_lineage.pdf` | 6.5 x 3.10 in | Four project iterations and contribution progression |
| 2 | `fig02_governed_workflow.pdf` | 6.5 x 3.10 in | Governed workflow and outcome-isolation boundary |
| 3 | `fig09_original_reanalysis.pdf` | 6.5 x 3.20 in | Re-analysis of prior confusion-matrix results |
| 4 | `fig03_final_ordinal_mae.pdf` | 6.5 x 3.40 in | Corrected ordinal-error comparison with intervals |
| 5 | `fig04_tradeoff.pdf` | 6.5 x 3.55 in | Severe recall versus ordinal-error tradeoff |
| 6 | `fig06a_probability_scores.pdf` | 6.5 x 3.05 in | Proper-score comparison against the prior forecast |
| 7 | `fig06b_calibration.pdf` | 6.5 x 2.60 in | Development-only temporal calibration |
| 8 | `fig07_leakage_factorial.pdf` | 6.5 x 2.95 in | Matched evaluation-defect contrasts |
| 9 | `fig05_protocol_corrections.pdf` | 6.5 x 3.30 in | Effect of protocol correction |
| 10 | `fig08_severe_ranking.pdf` | 6.5 x 3.65 in | Severe-risk precision-recall ranking curves |

The source package intentionally omits the previous generated graphics so they cannot be loaded accidentally.
