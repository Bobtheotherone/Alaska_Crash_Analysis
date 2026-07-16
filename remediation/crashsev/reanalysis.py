"""
crashsev.reanalysis — honest re-analysis of the paper's published confusion matrices.

This is legitimate SECONDARY analysis of results that already exist in the final report
(the four result screenshots). It recomputes the full ordinal metric suite from the
transcribed confusion matrices (data/confusion_matrices_from_paper.json) and generates the
tables and figures the paper should have shown (baseline comparison, severe-class
precision/recall tradeoff, predicted-distribution collapse).

It is clearly labelled as the ORIGINAL, contaminated-protocol evaluation (random 80/20
split, no baselines, no uncertainty), and is used to demonstrate *why that evaluation is
insufficient* — not to validate it. No number here is fabricated: every value derives
deterministically from the transcribed matrices, which reproduce the reconnaissance's
independent recomputation exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from . import metrics as M

ROOT = Path(__file__).resolve().parents[1]
CM_PATH = ROOT / "data" / "confusion_matrices_from_paper.json"
OUT = ROOT / "reanalysis"
FIGS = ROOT / "paper" / "generated"

MODEL_ORDER = ["decision_tree", "xgboost", "mlrf_random_forest", "ebm"]
PRETTY = {
    "decision_tree": "Decision Tree",
    "xgboost": "XGBoost",
    "mlrf_random_forest": "MLRF (RandomForest)",
    "ebm": "EBM",
    "majority": "Majority baseline",
}


def compute() -> Dict:
    """Recompute the metric suite under the manuscript's reporting convention
    (F1-CONV-001, ``zero_division="zero"``): for a class that is never predicted,
    precision is undefined (em dash) but recall and F1 are measured zeros under the
    count-based definition F1 = 2TP/(2TP+FP+FN), and macro-F1 averages the class F1
    values including those zeros. The frozen pipeline's *storage* convention
    (METRIC-001 NaN) is unchanged; this module is a reporting layer."""
    data = json.loads(CM_PATH.read_text())
    counts = data["provenance"]["test_set_class_counts"]
    maj_cm = [[counts["0"], 0, 0], [counts["1"], 0, 0], [counts["2"], 0, 0]]
    table = {"majority": M.metrics_from_cm(maj_cm, zero_division="zero")}
    for name in MODEL_ORDER:
        table[name] = M.metrics_from_cm(data["reported"][name]["confusion_matrix"],
                                        zero_division="zero")
    return {"class_counts": counts, "metrics": table, "provenance": data["provenance"]}


def _fmt_md(v: float) -> str:
    """Genuinely undefined statistics (e.g. precision with no predicted positives) render
    as an em-dash in human tables; measured zeros print as 0. (The CSV keeps the literal
    ``nan`` so pandas round-trips undefined values as NaN.)"""
    return "—" if isinstance(v, float) and np.isnan(v) else f"{v:.4f}"


def write_tables(res: Dict) -> None:
    OUT.mkdir(exist_ok=True)
    metrics = res["metrics"]
    cols = ["accuracy", "ordinal_mae", "qwk", "macro_f1", "balanced_accuracy",
            "within_one_accuracy", "two_step_error_rate",
            "severe_precision", "severe_recall", "severe_f1", "predicted_class0_share"]
    # CSV (undefined -> literal nan, which pandas reads back as NaN)
    lines = ["model," + ",".join(cols)]
    for name in ["majority"] + MODEL_ORDER:
        r = metrics[name]
        lines.append(name + "," + ",".join(f"{r[c]:.4f}" for c in cols))
    (OUT / "reanalysis_metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Markdown
    hdr = "| Model | Acc | oMAE↓ | QWK | macroF1 | balAcc | ≤1acc | 2-step↓ | sevP | sevR | sevF1 | pred-0 share |"
    sep = "|" + "---|" * 12
    md = [
        "# Re-analysis of the paper's four confusion matrices (ORIGINAL contaminated protocol)",
        "",
        f"Test set: N = {res['class_counts']['total']}  "
        f"(class 0 = {res['class_counts']['0']}, class 1 = {res['class_counts']['1']}, class 2 = {res['class_counts']['2']}).",
        "",
        "Source: result screenshots in the final report (random 80/20 split, `OneHotEncoder(min_frequency=0.01)`,",
        "no baselines, no uncertainty). `oMAE`/`2-step` lower is better. `pred-0 share` = fraction predicted class 0.",
        "Convention (F1-CONV-001): for a class that is never predicted, precision is *undefined* (shown as **—**),",
        "while recall and F1 are measured zeros under the count-based definition F1 = 2TP/(2TP+FP+FN);",
        "macro-F1 averages the class F1 values including those zeros.",
        "",
        hdr, sep,
    ]
    for name in ["majority"] + MODEL_ORDER:
        r = metrics[name]
        md.append(
            f"| {PRETTY[name]} | {_fmt_md(r['accuracy'])} | {_fmt_md(r['ordinal_mae'])} | {_fmt_md(r['qwk'])} | "
            f"{_fmt_md(r['macro_f1'])} | {_fmt_md(r['balanced_accuracy'])} | {_fmt_md(r['within_one_accuracy'])} | "
            f"{_fmt_md(r['two_step_error_rate'])} | {_fmt_md(r['severe_precision'])} | {_fmt_md(r['severe_recall'])} | "
            f"{_fmt_md(r['severe_f1'])} | {_fmt_md(r['predicted_class0_share'])} |"
        )
    maj_acc = metrics["majority"]["accuracy"]
    below = [PRETTY[n] for n in MODEL_ORDER if metrics[n]["accuracy"] < maj_acc]
    md += [
        "",
        "## Key observations",
        f"* The trivial majority-class predictor scores **{maj_acc:.4f} accuracy**. "
        f"Below-majority models: **{', '.join(below)}** ({len(below)} of 4).",
        "* No model is uniformly dominant: MLRF leads accuracy and ordinal MAE (by heavily "
        "predicting class 0 — a "
        f"{metrics['mlrf_random_forest']['predicted_class0_share']:.1%} class-0 share); EBM leads severe recall "
        f"({metrics['ebm']['severe_recall']:.3f}) but at "
        f"{metrics['ebm']['severe_precision']:.3f} severe precision; XGBoost leads macro-F1.",
        "* A declared primary objective is therefore indispensable; 'best model' is undefined here.",
    ]
    (OUT / "reanalysis_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def make_figures(res: Dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    metrics = res["metrics"]
    names = MODEL_ORDER
    maj_acc = metrics["majority"]["accuracy"]

    # Fig 1: accuracy vs majority baseline + ordinal MAE
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    accs = [metrics[n]["accuracy"] for n in names]
    maes = [metrics[n]["ordinal_mae"] for n in names]
    labels = [PRETTY[n] for n in names]
    bars = ax[0].bar(labels, accs, color="#4C78A8")
    ax[0].axhline(maj_acc, color="#E45756", ls="--", lw=2, label=f"majority baseline ({maj_acc:.3f})")
    ax[0].set_ylabel("Accuracy"); ax[0].set_title("Accuracy vs. trivial majority baseline")
    ax[0].set_ylim(0, 0.8); ax[0].legend(); ax[0].tick_params(axis="x", rotation=20)
    maj_mae = metrics["majority"]["ordinal_mae"]
    ax[1].bar(labels, maes, color="#54A24B")
    ax[1].axhline(maj_mae, color="#E45756", ls="--", lw=2, label=f"majority baseline ({maj_mae:.3f})")
    ax[1].set_ylabel("Ordinal MAE (lower is better)"); ax[1].set_title("Ordinal MAE vs. majority baseline")
    ax[1].legend(); ax[1].tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(FIGS / "fig_reanalysis_accuracy_and_mae.png", dpi=140); plt.close(fig)

    # Fig 2: severe-class precision vs recall
    fig, ax = plt.subplots(figsize=(6, 5))
    for n in names:
        ax.scatter(metrics[n]["severe_recall"], metrics[n]["severe_precision"], s=90)
        ax.annotate(PRETTY[n], (metrics[n]["severe_recall"], metrics[n]["severe_precision"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Severe-class (2) recall"); ax.set_ylabel("Severe-class (2) precision")
    ax.set_title("Severe-class precision–recall tradeoff\n(no declared cost => no 'best')")
    ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.5); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "fig_reanalysis_severe_pr.png", dpi=140); plt.close(fig)

    # Fig 3: predicted class distribution vs truth
    counts = res["class_counts"]
    true_dist = [counts["0"] / counts["total"], counts["1"] / counts["total"], counts["2"] / counts["total"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3); w = 0.15
    ax.bar(x - 2 * w, true_dist, w, label="TRUE", color="#333333")
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for i, n in enumerate(names):
        pd_ = metrics[n]["predicted_distribution"]
        ax.bar(x + (i - 1) * w, [pd_[0], pd_[1], pd_[2]], w, label=PRETTY[n], color=palette[i])
    ax.set_xticks(x); ax.set_xticklabels(["0 none/PDO", "1 minor", "2 serious/fatal"])
    ax.set_ylabel("Share of predictions"); ax.set_title("Predicted vs. true class distribution")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIGS / "fig_reanalysis_predicted_dist.png", dpi=140); plt.close(fig)


def main():
    res = compute()
    write_tables(res)
    make_figures(res)
    print("[reanalysis] wrote reanalysis/reanalysis_metrics.csv, reanalysis_table.md, and 3 figures.")


if __name__ == "__main__":
    main()
