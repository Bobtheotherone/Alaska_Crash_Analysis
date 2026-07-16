"""Generate the ten publication figures for the UAA student paper
(Mercado-Barbosa, Iteration IV) as vector PDFs sized to the frames in
FIGURE_REPLACEMENT_GUIDE.md.

Every numeric value is read from frozen, committed artifacts of the
remediation package (no per-crash data is embedded in any output; the
severe-ranking curves are aggregate precision-recall paths derived from
the frozen governed-run predictions and are cross-checked against the
committed `experiment/severe_ranking.md` summary numbers).

Inputs (read-only):
  <REPO>/remediation/experiment/final_results.json            (run final_8af9d5bc23d8)
  <REPO>/remediation/experiment/broad_sensitivity/final_results.json
  <REPO>/remediation/experiment/leakage_factorial.json
  <REPO>/remediation/evidence_release/final_f27613102c96/manifest.json  (historical v3)
  <REPO>/remediation/reanalysis/reanalysis_metrics.csv
  <REPO>/remediation/runs/final_8af9d5bc23d8/predictions_*.csv (frozen 2012 probabilities)

Usage:
  python make_figures.py [--repo <repo_root>] [--out <figures/user dir>]
  (defaults are derived from this file's location; no machine-specific path is assumed)

Deterministic: no randomness, no timestamps in figure content.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

# ----------------------------------------------------------------------
# Global restrained-academic style
# ----------------------------------------------------------------------
ACCENT = "#31597F"          # single restrained accent (muted slate blue)
GRAY_D = "#333333"
GRAY_M = "#7A7A7A"
GRAY_L = "#B8B8B8"
BASELINE = "#8A8A8A"        # baseline reference lines: neutral gray, dashed

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.edgecolor": GRAY_D,
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": GRAY_D,
    "ytick.color": GRAY_D,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,          # embed TrueType (subset) so text stays text
    "axes.unicode_minus": True,  # true minus sign (U+2212); present in Times New Roman
})

PRETTY = {
    "majority": "Majority",
    "ordinal_median": "Ordinal median",
    "prior_probability": "Prior probability",
    "multinomial_logistic": "Multinomial logit",
    "proportional_odds": "Proportional-odds logit",
    "frank_hall_logistic": "Frank-Hall logit",
    "shallow_tree": "Shallow tree",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest (weighted)",
    "ordinal_random_forest": "Ordinal RF (primary)",
    "random_forest_unweighted": "Random forest (unweighted)",
    "ordinal_random_forest_unweighted": "Ordinal RF (unweighted)",
    "xgboost": "XGBoost",
    "ebm": "EBM (exploratory)",
}

ROLE_MARKER = {  # grayscale-safe role coding by marker shape
    "candidate": "o",
    "ablation": "s",
    "baseline": "D",
    "exploratory": "^",
}


def model_role(key: str, res: dict) -> str:
    kind = res[key].get("kind", "")
    if key == "ebm":
        return "exploratory"
    if kind == "candidate":
        return "candidate"
    if kind == "ablation":
        return "ablation"
    return "baseline"


def new_fig(height: float, width: float = 6.5):
    return plt.figure(figsize=(width, height))


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------
# Figure 1 - project lineage (diagram; content = manuscript Table 1)
# ----------------------------------------------------------------------
def fig01_lineage(out: Path):
    fig = new_fig(3.10)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    boxes = [
        ("Iteration I", "poster 2022; archive 2023",
         "L. Stark, C. Cornichuck",
         "Exploratory crash-factor\nanalysis; several classifiers;\npublic-facing web tool",
         "Exploration"),
        ("Iteration II", "2024",
         "M. Bamber, Y. Seto,\nN. Aleman",
         "Regional classification;\nexpanded data model;\ndatabase-backed analysis",
         "Data enrichment"),
        ("Iteration III", "2025",
         "R. N. Mercado-Barbosa,\nP. Ratzer",
         "Secure platform rebuild\n(Django, PostGIS, React);\ndynamic ingestion, cleaning,\nmodel orchestration",
         "Systems engineering"),
        ("Iteration IV", "2026",
         "R. N. Mercado-Barbosa\n(independent)",
         "Target audit;\noutcome isolation;\nchronological validation;\nordinal decision rule;\ngoverned artifacts",
         "Evaluation governance"),
    ]
    n = len(boxes)
    bw, gap = 21.5, (100 - 4 * 21.5) / 5.0
    y0, bh = 34, 52
    for i, (t, yr, who, what, tag) in enumerate(boxes):
        x = gap + i * (bw + gap)
        emph = i == n - 1
        pad = 0.6  # same outward margin the previous rounded boxes had
        box = Rectangle(
            (x - pad, y0 - pad), bw + 2 * pad, bh + 2 * pad,
            linewidth=1.2 if emph else 0.8,
            edgecolor=GRAY_D if emph else GRAY_M,
            facecolor="white",
        )
        ax.add_patch(box)
        cx = x + bw / 2
        ax.text(cx, y0 + bh - 6, t, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="black")
        ax.text(cx, y0 + bh - 13.5, yr, ha="center", va="center",
                fontsize=8, color=GRAY_M)
        ax.text(cx, y0 + bh - 22.5, who, ha="center", va="center", fontsize=8)
        ax.text(cx, y0 + 13, what, ha="center", va="center", fontsize=8)
        ax.text(cx, y0 - 5.5, tag, ha="center", va="center",
                fontsize=8, color=GRAY_M)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + bw + 0.7, y0 + bh / 2), (x + bw + gap - 0.7, y0 + bh / 2),
                arrowstyle="-|>", mutation_scale=9,
                linewidth=0.9, color=GRAY_M))

    # lower annotation band: capability grows through Iteration III;
    # the evidentiary standard keeps rising through Iteration IV.
    # Each caption sits flush above the start of its own arrow so the
    # pairing is unambiguous; the arrows differ in span and weight.
    iii_right = gap + 2 * (bw + gap) + bw
    x0 = gap + 2
    ya, yb = 17, 8
    ax.text(x0, ya + 2.3, "system capability increases",
            ha="left", va="bottom", fontsize=8, color=GRAY_M)
    ax.add_patch(FancyArrowPatch((x0, ya), (iii_right, ya),
                                 arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.8, color=GRAY_M))
    ax.text(x0, yb + 2.3, "evidentiary standard rises",
            ha="left", va="bottom", fontsize=8, color=GRAY_D)
    ax.add_patch(FancyArrowPatch((x0, yb), (100 - gap - 2, yb),
                                 arrowstyle="-|>", mutation_scale=8,
                                 linewidth=1.1, color=GRAY_D))
    save(fig, out, "fig01_project_lineage.pdf")


# ----------------------------------------------------------------------
# Figure 2 - governed workflow / outcome-isolation boundary (diagram)
# ----------------------------------------------------------------------
def fig02_workflow(out: Path):
    fig = new_fig(3.10)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # vertical layout is computed from line counts so no box can clip
    # its own text: 1 pt = 100/(3.10*72) y-units at this figure height
    PT = 100.0 / (3.10 * 72)
    TOP_PAD, GAP, BOT_PAD = 1.5, 0.8, 1.7
    TITLE_H = 8.5 * 1.2 * PT
    LINE_H = 8 * 1.15 * PT
    ARROW_GAP = 3.2

    def box(x, w, top, title, lines, emph=False):
        h = TOP_PAD + TITLE_H + GAP + len(lines) * LINE_H + BOT_PAD
        ax.add_patch(Rectangle(
            (x, top - h), w, h,
            linewidth=1.2 if emph else 0.8,
            edgecolor=GRAY_D if emph else GRAY_M,
            facecolor="white"))
        ax.text(x + w / 2, top - TOP_PAD - TITLE_H / 2, title, ha="center",
                va="center", fontsize=8.5, fontweight="bold", color="black")
        ax.text(x + w / 2, top - TOP_PAD - TITLE_H - GAP, "\n".join(lines),
                ha="center", va="top", fontsize=8, linespacing=1.15)
        return top - h

    def arrow(p, q, color=GRAY_M, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.9, color=color, linestyle=ls))

    # headers
    ax.text(18, 95.5, "Development phase (2009–2011)", ha="center",
            fontsize=9, fontweight="bold")
    ax.text(78, 95.5, "Final evaluation (2012)", ha="center",
            fontsize=9, fontweight="bold")

    # left column (each box top follows the previous bottom + arrow gap)
    b1 = box(3, 30, 93.2, "Licensed raw extract",
             ["50,543 crashes, 2009–2012;", "whole-file integrity hash"])
    b2 = box(3, 30, b1 - ARROW_GAP, "Chronological partition",
             ["by year before target reading;", "zero crash-group overlap"])
    b3 = box(3, 30, b2 - ARROW_GAP, "Development only",
             ["fail-closed target audit;", "train-only preprocessing;",
              "rolling-origin folds",
              "(09$\\rightarrow$10, 09–10$\\rightarrow$11)"])
    b4 = box(3, 30, b3 - ARROW_GAP, "Freeze",
             ["config + selected candidate +", "dev-side assignment hash"])

    arrow((18, b1 - 0.3), (18, b1 - ARROW_GAP + 0.3))
    arrow((18, b2 - 0.3), (18, b2 - ARROW_GAP + 0.3))
    arrow((18, b3 - 0.3), (18, b3 - ARROW_GAP + 0.3))

    # boundary: dashed line named horizontally at the top, explained in a
    # horizontal strip along the bottom (no rotated text anywhere)
    ax.plot([50, 50], [7.5, 88.3], ls=(0, (5, 4)), color=ACCENT, lw=1.3)
    ax.text(50, 92, "structural outcome-isolation boundary",
            ha="center", va="center", fontsize=8, color=ACCENT)
    ax.text(50, 3.2,
            "development cannot use 2012 outcomes "
            "(enforced by the poison-token sentinel)",
            ha="center", va="center", fontsize=8, color=GRAY_D)

    # right column
    r1 = box(60, 36, 89, "One governed final run",
             ["frozen protocol only; refuses dirty",
              "source tree or modified config;",
              "posterior-median decision rule"], emph=True)
    r2 = box(60, 36, r1 - ARROW_GAP, "Frozen evidence bundle",
             ["content-addressed, overwrite-refusing;",
              "run final_8af9d5bc23d8;",
              "aggregate + de-identified evidence"])
    arrow((78, r1 - 0.3), (78, r1 - ARROW_GAP + 0.3))

    # freeze-to-final elbow: across the boundary low, then up the clear
    # corridor left of the evaluation column
    run_mid = (89 + r1) / 2
    ax.plot([33, 55, 55], [13.5, 13.5, run_mid], color=GRAY_M, lw=0.9,
            solid_capstyle="round")
    arrow((55, run_mid), (59.4, run_mid), color=GRAY_M)
    ax.text(41, 15.1, "frozen artifacts only", fontsize=8,
            color=GRAY_M, ha="center", va="bottom")

    # caveat strip
    ax.text(78, 25,
            "2012 was historically available to the project:\nexposed retrospective evaluation,\nnot a prospective or sealed holdout",
            ha="center", va="center", fontsize=8, color=GRAY_D,
            bbox=dict(boxstyle="square,pad=0.45", facecolor="white",
                      edgecolor=GRAY_L, linewidth=0.7))
    save(fig, out, "fig02_governed_workflow.pdf")


# ----------------------------------------------------------------------
# Figure 3 (file fig09) - re-analysis of the prior confusion matrices
# ----------------------------------------------------------------------
def fig09_reanalysis(repo: Path, out: Path):
    rows = {}
    with open(repo / "remediation/reanalysis/reanalysis_metrics.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = {k: (float(v) if v not in ("", "nan") else np.nan)
                                for k, v in r.items() if k != "model"}
    label = {"majority": "Majority baseline", "decision_tree": "Decision tree",
             "xgboost": "XGBoost", "mlrf_random_forest": "MLRF / random forest",
             "ebm": "EBM"}
    maj_acc = rows["majority"]["accuracy"]        # 0.6762
    maj_mae = rows["majority"]["ordinal_mae"]     # 0.3594

    fig = new_fig(3.20)
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.90, bottom=0.155,
                          wspace=0.33)
    axL = fig.add_subplot(gs[0])
    axR = fig.add_subplot(gs[1])

    # Left: accuracy vs ordinal error, majority baselines as quadrant lines
    models = ["decision_tree", "xgboost", "mlrf_random_forest", "ebm"]
    offs = {"decision_tree": (0.005, 0.010, "left"),
            "xgboost": (0.004, 0.013, "left"),
            "ebm": (0.004, 0.013, "left")}
    axL.axvline(maj_acc, color=BASELINE, ls="--", lw=0.9, zorder=1)
    axL.axhline(maj_mae, color=BASELINE, ls="--", lw=0.9, zorder=1)
    for m in models:
        x, y = rows[m]["accuracy"], rows[m]["ordinal_mae"]
        axL.scatter(x, y, s=42, color=ACCENT, zorder=3,
                    edgecolors="white", linewidths=0.6)
        if m == "mlrf_random_forest":
            # this point sits just right of the dashed accuracy line; end
            # the label left of the line and bridge with a short leader so
            # neither line strikes the text
            axL.text(maj_acc - 0.005, y, label[m], fontsize=8,
                     ha="right", va="center")
            axL.plot([maj_acc + 0.0025, x - 0.0065], [y, y], lw=0.6,
                     color=GRAY_L, zorder=2)
            continue
        dx, dy, ha = offs[m]
        axL.annotate(label[m], (x, y), xytext=(x + dx, y + dy),
                     fontsize=8, ha=ha, va="bottom")
    axL.scatter(maj_acc, maj_mae, s=46, marker="D", facecolors="white",
                edgecolors=GRAY_D, linewidths=0.9, zorder=3)
    axL.annotate("Majority baseline", (maj_acc, maj_mae),
                 xytext=(maj_acc - 0.004, maj_mae + 0.012), fontsize=8,
                 ha="right", va="top")
    axL.text(maj_acc - 0.006, 0.517, "majority accuracy 0.676", fontsize=8,
             color=GRAY_M, ha="right", va="bottom")
    axL.text(0.5485, maj_mae - 0.0085, "majority ordinal error 0.359",
             fontsize=8, color=GRAY_M, ha="left", va="bottom")
    axL.set_xlabel("Reported accuracy")
    axL.set_ylabel("Ordinal MAE (recomputed; lower is better)")
    axL.set_xlim(0.545, 0.745)
    axL.set_ylim(0.525, 0.28)   # inverted: lower (better) ordinal error on top
    axL.grid(alpha=0.22, lw=0.5)

    # Right: severe-class precision vs recall tradeoff
    offs2 = {"decision_tree": (-0.018, 0.000, "right", "center"),
             "xgboost": (0.014, 0.012, "left", "bottom"),
             "mlrf_random_forest": (0.015, 0.000, "left", "center"),
             "ebm": (-0.015, 0.014, "right", "bottom")}
    for m in models:
        x, y = rows[m]["severe_recall"], rows[m]["severe_precision"]
        axR.scatter(x, y, s=42, color=ACCENT, zorder=3,
                    edgecolors="white", linewidths=0.6)
        dx, dy, ha, va = offs2[m]
        axR.annotate(label[m], (x, y), xytext=(x + dx, y + dy),
                     fontsize=8, ha=ha, va=va)
    axR.annotate("Majority baseline: severe recall 0.000,\nprecision undefined (never predicts severe)",
                 (0.0, 0.02), fontsize=8, color=GRAY_M, ha="left", va="bottom",
                 xytext=(0.02, 0.02))
    axR.set_xlabel("Severe-class recall")
    axR.set_ylabel("Severe-class precision")
    axR.set_xlim(0, 0.8)
    axR.set_ylim(0, 0.5)
    axR.grid(alpha=0.22, lw=0.5)
    save(fig, out, "fig09_original_reanalysis.pdf")


# ----------------------------------------------------------------------
# Figure 4 (file fig03) - corrected 2012 ordinal error with 95% CIs
# ----------------------------------------------------------------------
def fig03_final_mae(res: dict, out: Path):
    models = [m for m in res if "ordinal_mae" in res[m]]
    order = sorted(models, key=lambda m: res[m]["ordinal_mae"])
    maj = res["majority"]["ordinal_mae"]

    fig = new_fig(3.40)
    ax = fig.add_axes([0.315, 0.135, 0.655, 0.845])
    ys = np.arange(len(order))[::-1]
    for y, m in zip(ys, order):
        v = res[m]["ordinal_mae"]
        lo = res[m]["ordinal_mae_ci"]["ci_low"]
        hi = res[m]["ordinal_mae_ci"]["ci_high"]
        role = model_role(m, res)
        primary = m == "ordinal_random_forest"
        col = ACCENT if primary else GRAY_D
        ax.plot([lo, hi], [y, y], lw=1.1, color=col, solid_capstyle="butt",
                zorder=2)
        ax.plot([lo, lo], [y - 0.16, y + 0.16], lw=0.9, color=col, zorder=2)
        ax.plot([hi, hi], [y - 0.16, y + 0.16], lw=0.9, color=col, zorder=2)
        ax.scatter([v], [y], s=30 if primary else 22,
                   marker=ROLE_MARKER[role],
                   facecolors=col if role in ("candidate",) else "white",
                   edgecolors=col, linewidths=1.0, zorder=3)
    ax.axvline(maj, color=BASELINE, ls="--", lw=1.0, zorder=1)
    ax.text(maj + 0.004, len(order) - 0.55, f"majority baseline {maj:.4f}",
            fontsize=8, color=GRAY_M, ha="left", va="top", rotation=0)
    labels = []
    for m in order:
        lab = PRETTY[m]
        labels.append(lab)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    for tl, m in zip(ax.get_yticklabels(), order):
        if m == "ordinal_random_forest":
            tl.set_fontweight("bold")
            tl.set_color(ACCENT)
    ax.set_xlabel("Ordinal mean absolute error, 2012 out-of-time evaluation "
                  "(lower is better)")
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(axis="x", alpha=0.22, lw=0.5)
    handles = [
        Line2D([], [], marker="o", color=ACCENT, ls="", markersize=5,
               label="primary"),
        Line2D([], [], marker="o", color=GRAY_D, ls="", markersize=5,
               label="secondary"),
        Line2D([], [], marker="s", color=GRAY_D, ls="", markersize=5,
               markerfacecolor="white", label="ablation"),
        Line2D([], [], marker="D", color=GRAY_D, ls="", markersize=4.5,
               markerfacecolor="white", label="baseline"),
        Line2D([], [], marker="^", color=GRAY_D, ls="", markersize=5,
               markerfacecolor="white", label="exploratory (final-only)"),
        Line2D([], [], color=GRAY_D, lw=1.1, label="95% case-bootstrap CI"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              borderaxespad=0.2, handletextpad=0.5)
    save(fig, out, "fig03_final_ordinal_mae.pdf")


# ----------------------------------------------------------------------
# Figure 5 (file fig04) - severe recall vs ordinal error tradeoff
# ----------------------------------------------------------------------
def fig04_tradeoff(res: dict, out: Path):
    maj = res["majority"]["ordinal_mae"]
    # every individually labelled model gets a plain offset label directly
    # at its marker (no leader lines between named points)
    pts = {
        "xgboost": (0.012, -0.010, "left", "center"),
        "ebm": (0.012, -0.008, "left", "center"),
        "proportional_odds": (-0.012, -0.011, "right", "center"),
        "frank_hall_logistic": (0.000, -0.020, "center", "center"),
        "multinomial_logistic": (0.012, 0.006, "left", "center"),
        "decision_tree": (0.012, 0.002, "left", "center"),
        "shallow_tree": (0.012, 0.002, "left", "center"),
        "random_forest": (0.013, 0.000, "left", "center"),
        "ordinal_random_forest": (0.012, 0.000, "left", "center"),
    }
    fig = new_fig(3.55)
    ax = fig.add_axes([0.095, 0.125, 0.885, 0.855])
    ax.axhline(maj, color=BASELINE, ls="--", lw=1.0, zorder=1)
    ax.text(0.742, maj + 0.004, f"majority ordinal error {maj:.4f}",
            fontsize=8, color=GRAY_M, ha="right", va="bottom")

    # constant baselines collapse to one reference point
    ax.scatter([0.0], [maj], s=42, marker="D", facecolors="white",
               edgecolors=GRAY_D, linewidths=1.0, zorder=3)
    ax.plot([0.002, 0.020], [maj + 0.004, 0.399], lw=0.6,
            color=GRAY_L, zorder=2)
    ax.text(0.023, 0.403, "Majority / ordinal median /\n"
            "prior probability (constant)", fontsize=8, ha="left", va="center")

    for m, (dx, dy, ha, va) in pts.items():
        x = res[m]["severe_recall"]
        y = res[m]["ordinal_mae"]
        role = model_role(m, res)
        primary = m == "ordinal_random_forest"
        col = ACCENT if primary else GRAY_D
        ax.scatter([x], [y], s=44 if primary else 30,
                   marker=ROLE_MARKER[role],
                   facecolors=col if role == "candidate" else "white",
                   edgecolors=col, linewidths=1.0, zorder=3)
        ax.annotate(PRETTY[m], (x, y), xytext=(x + dx, y + dy), fontsize=8,
                    ha=ha, va=va,
                    fontweight="bold" if primary else "normal",
                    color=ACCENT if primary else "black")

    # the two unweighted ablations nearly coincide at zero recall; both
    # markers are drawn and the pair is named once (individual values are
    # in Figure 4 and Tables 3/5)
    for m in ("ordinal_random_forest_unweighted", "random_forest_unweighted"):
        x = res[m]["severe_recall"]
        y = res[m]["ordinal_mae"]
        ax.scatter([x], [y], s=30, marker=ROLE_MARKER[model_role(m, res)],
                   facecolors="white", edgecolors=GRAY_D, linewidths=1.0,
                   zorder=3)
    ax.plot([0.011, 0.0265], [0.3372, 0.3300], lw=0.6, color=GRAY_L, zorder=2)
    ax.text(0.030, 0.3290, "unweighted forest ablations", fontsize=8,
            ha="left", va="center")
    ax.set_xlabel("Severe-class recall, 2012 out-of-time evaluation")
    ax.set_ylabel("Ordinal MAE (lower is better)")
    ax.set_xlim(-0.03, 0.75)
    ax.set_ylim(0.310, 0.66)
    ax.grid(alpha=0.22, lw=0.5)
    handles = [
        Line2D([], [], marker="o", color=ACCENT, ls="", markersize=5.5,
               label="primary"),
        Line2D([], [], marker="o", color=GRAY_D, ls="", markersize=5,
               label="secondary"),
        Line2D([], [], marker="s", color=GRAY_D, markerfacecolor="white",
               ls="", markersize=5, label="ablation"),
        Line2D([], [], marker="D", color=GRAY_D, markerfacecolor="white",
               ls="", markersize=4.5, label="baseline"),
        Line2D([], [], marker="^", color=GRAY_D, markerfacecolor="white",
               ls="", markersize=5.5, label="exploratory (final-only)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    save(fig, out, "fig04_tradeoff.pdf")


# ----------------------------------------------------------------------
# Figure 6 (file fig06a) - proper scores vs the deterministic prior
# ----------------------------------------------------------------------
def fig06a_scores(r: dict, out: Path):
    res = r["results"]
    prior = res["prior_probability"]
    # exclude the two degenerate one-hot constant baselines from the panel
    models = [m for m in res
              if "log_loss" in res[m] and m not in ("majority", "ordinal_median")]
    order = sorted(models, key=lambda m: res[m]["log_loss"])

    fig = new_fig(3.05)
    metrics = [("log_loss", "Log loss", (0.62, 2.14), [0.75, 1.25, 1.75],
                [0.245, 0.20, 0.215, 0.675]),
               ("brier", "Brier score", (0.400, 0.620), [0.45, 0.55],
                [0.505, 0.20, 0.215, 0.675]),
               ("rps", "Ranked probability score", (0.212, 0.408), [0.25, 0.35],
                [0.765, 0.20, 0.215, 0.675])]
    ys = np.arange(len(order))[::-1]
    axes = []
    for k, ttl, xlim, xticks, rect in metrics:
        ax = fig.add_axes(rect)
        axes.append(ax)
        for y, m in zip(ys, order):
            v = res[m][k]
            beats_all = all(res[m][kk] < prior[kk] for kk in
                            ("log_loss", "brier", "rps"))
            col = ACCENT if beats_all else GRAY_M
            ax.scatter([v], [y], s=26, marker="o",
                       facecolors=col if beats_all else "white",
                       edgecolors=col, linewidths=1.0, zorder=3)
        ax.axvline(prior[k], color=BASELINE, ls="--", lw=1.1, zorder=1)
        ax.set_title(ttl, fontsize=9, pad=3)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.grid(axis="x", alpha=0.22, lw=0.5)
        ax.set_yticks(ys)
        if not axes[:-1]:
            ax.set_yticklabels([PRETTY[m] for m in order])
            for tl, m in zip(ax.get_yticklabels(), order):
                if m == "ordinal_random_forest":
                    tl.set_fontweight("bold")
                    tl.set_color(ACCENT)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="x", labelsize=9)

    axes[1].set_xlabel("lower is better", fontsize=9, color=GRAY_M,
                       labelpad=2)
    fig.text(0.245, 0.016,
             "dashed line = deterministic prior forecast, the trivial "
             "probabilistic floor;\nfilled = beats the prior on all three "
             "proper scores (exactly the four forest variants)",
             fontsize=8.5, color=GRAY_D, ha="left", va="bottom")
    save(fig, out, "fig06a_probability_scores.pdf")


# ----------------------------------------------------------------------
# Figure 7 (file fig06b) - development-only temporal calibration
# ----------------------------------------------------------------------
def fig06b_calibration(r: dict, out: Path):
    cal = r["calibration"]
    fig = plt.figure(figsize=(4.1, 2.60))
    ax = fig.add_axes([0.145, 0.115, 0.835, 0.845])
    names = ["ordinal_random_forest", "majority"]
    labs = ["Ordinal RF (primary)", "Majority (constant)"]
    xx = np.arange(len(names))
    w = 0.32
    raw = [cal[m]["ece_raw"] for m in names]
    calv = [cal[m]["ece_calibrated"] for m in names]
    ax.bar(xx - w / 2, raw, w, color="white", edgecolor=GRAY_D, linewidth=0.9,
           hatch="////", label="raw")
    ax.bar(xx + w / 2, calv, w, color=ACCENT, edgecolor=ACCENT, linewidth=0.9,
           label="calibrated (development-only)")
    for x, v in zip(xx - w / 2, raw):
        ax.text(x, v + 0.010, f"{v:.3f}", ha="center", fontsize=9)
    for x, v in zip(xx + w / 2, calv):
        ax.text(x, v + 0.010, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(xx)
    ax.set_xticklabels(labs, fontsize=9.5)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("Expected calibration error", fontsize=9.5)
    ax.set_ylim(0, 0.50)
    ax.legend(frameon=False, fontsize=9, loc="upper left",
              handlelength=1.2, borderaxespad=0.1, labelspacing=0.4)
    save(fig, out, "fig06b_calibration.pdf")


# ----------------------------------------------------------------------
# Figure 8 (file fig07) - leakage factorial matched contrasts
# ----------------------------------------------------------------------
def fig07_factorial(repo: Path, out: Path):
    lf = json.loads((repo / "remediation/experiment/leakage_factorial.json")
                    .read_text())
    se = lf["simple_effects_vs_reference"]["ordinal_mae"]
    ref = lf["reference_cell_all_defects_off"]["ordinal_mae"]
    allon = next(c for c in lf["cells"]
                 if c["leakage_features"] and c["preprocess_before"]
                 and c["random_split"])["ordinal_mae"]
    rows = [
        ("Outcome-derived features", se["leakage_features_vs_reference"]),
        ("Random rather than\ntemporal validation",
         se["random_split_given_leak_off"]),
        ("Preprocessing before\nsplitting", se["preprocess_before_given_leak_off"]),
        ("All three defects\ncombined (interacting)", allon - ref),
    ]
    fig = new_fig(2.95)
    ax = fig.add_axes([0.255, 0.235, 0.725, 0.745])
    ys = np.arange(len(rows))[::-1]
    vals = [v for _, v in rows]
    bars = ax.barh(ys, vals, height=0.55,
                   edgecolor=[ACCENT if i in (0, 3) else GRAY_D
                              for i in range(len(rows))],
                   linewidth=1.0)
    for i, (b, v) in enumerate(zip(bars, vals)):
        b.set_facecolor("#DCE5EE" if i in (0, 3) else "#EDEDED")
        ax.text(v - 0.006, ys[i], f"{v:+.3f}", ha="right", va="center",
                fontsize=8)
    ax.axvline(0, color=GRAY_D, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([t for t, _ in rows], fontsize=8.5)
    ax.set_xlabel("Change in development ordinal MAE vs the leakage-controlled "
                  "reference (negative = deceptively lower error)\n"
                  "matched same-model contrasts, development years only, "
                  f"mean of 5 seeds; reference ordinal MAE {ref:.3f}",
                  fontsize=8)
    ax.set_xlim(-0.40, 0.012)
    ax.grid(axis="x", alpha=0.22, lw=0.5)
    ax.text(-0.385, 1.5,
            "effects are not additive: once outcome leakage\nis present, the "
            "two smaller defects contribute\nalmost nothing",
            fontsize=8, color=GRAY_D, ha="left", va="center", style="italic")
    save(fig, out, "fig07_leakage_factorial.pdf")


# ----------------------------------------------------------------------
# Figure 9 (file fig05) - effect of protocol correction
# ----------------------------------------------------------------------
def fig05_corrections(repo: Path, out: Path):
    v3 = json.loads((repo / "remediation/evidence_release/final_f27613102c96/"
                            "manifest.json").read_text())
    broad = json.loads((repo / "remediation/experiment/broad_sensitivity/"
                               "final_results.json").read_text())
    strict = json.loads((repo / "remediation/experiment/final_results.json")
                        .read_text())
    stages = ["v3 protocol\nbroad tier, argmax,\nweaker isolation",
              "v4 machinery\nbroad tier,\nmedian rule",
              "v4 primary\nrestricted tier,\nmedian rule"]
    dd = [v3["paired_difference_vs_baseline"]["ordinal_random_forest"],
          broad["paired_difference_vs_baseline"]["ordinal_random_forest"],
          strict["paired_difference_vs_baseline"]["ordinal_random_forest"]]
    rec = [v3["results"]["ordinal_random_forest"]["severe_recall"],
           broad["results"]["ordinal_random_forest"]["severe_recall"],
           strict["results"]["ordinal_random_forest"]["severe_recall"]]

    fig = new_fig(3.30)
    gs = fig.add_gridspec(1, 2, left=0.105, right=0.93, top=0.965, bottom=0.235,
                          wspace=0.42)
    xx = np.arange(3)

    axL = fig.add_subplot(gs[0])
    pt = [d["difference"] for d in dd]
    lo = [d["difference"] - d["ci_low"] for d in dd]
    hi = [d["ci_high"] - d["difference"] for d in dd]
    axL.axhline(0, color=BASELINE, ls="--", lw=0.9)
    axL.text(2.05, 0.0008, "no difference from majority", fontsize=8,
             color=GRAY_M, ha="right", va="bottom")
    axL.errorbar(xx, pt, yerr=[lo, hi], fmt="none", ecolor=GRAY_D,
                 elinewidth=1.0, capsize=3, capthick=0.9, zorder=2)
    axL.plot(xx, pt, color=GRAY_L, lw=0.9, zorder=1)
    axL.scatter(xx, pt, s=[30, 30, 44],
                color=[GRAY_D, GRAY_D, ACCENT], zorder=3)
    for x, d in zip(xx, dd):
        # value labels sit below the lower CI cap, clear of the connecting
        # line and the error bars
        axL.text(x, d["ci_low"] - 0.0018, f"{d['difference']:+.4f}",
                 fontsize=8, ha="center", va="top")
    axL.set_xticks(xx)
    axL.set_xticklabels(stages, fontsize=8)
    axL.set_ylabel("$\\Delta$ ordinal MAE vs majority (95% CI)")
    axL.set_xlim(-0.42, 2.42)
    axL.set_ylim(-0.045, 0.006)
    axL.grid(axis="y", alpha=0.22, lw=0.5)

    axR = fig.add_subplot(gs[1])
    axR.plot(xx, rec, color=GRAY_L, lw=0.9, zorder=1)
    axR.scatter(xx, rec, s=[30, 30, 44], color=[GRAY_D, GRAY_D, ACCENT],
                zorder=3)
    for i, (x, v) in enumerate(zip(xx, rec)):
        if i == len(rec) - 1:
            # last point: label below-left, clear of the steep incoming line
            axR.text(x - 0.10, v - 0.008, f"{v:.3f}", fontsize=8,
                     va="top", ha="right")
        else:
            axR.text(x + 0.10, v + 0.006, f"{v:.3f}", fontsize=8,
                     va="bottom")
    axR.set_xticks(xx)
    axR.set_xticklabels(stages, fontsize=8)
    axR.set_ylabel("Severe-class recall (hard decisions)")
    axR.set_xlim(-0.42, 2.42)
    axR.set_ylim(0, 0.32)
    axR.grid(axis="y", alpha=0.22, lw=0.5)
    save(fig, out, "fig05_protocol_corrections.pdf")


# ----------------------------------------------------------------------
# Figure 8 (file fig06c) - severe-class reliability diagram (raw vs calibrated)
# ----------------------------------------------------------------------
def fig06c_reliability(repo: Path, r: dict, out: Path):
    """Severe-class reliability for the primary model. Calibrated points come from the
    FROZEN run artifact (calibration.ordinal_random_forest.reliability_severe); raw
    points are recomputed from the committed lossless parquet with the SAME binning rule
    (crashsev.calibration.reliability_points, 10 equal-width bins, empty bins omitted).
    Nothing is invented: every plotted value is frozen or derived from frozen row-level
    evidence."""
    import sys
    sys.path.insert(0, str(repo / "remediation"))
    import pandas as pd
    from crashsev.calibration import reliability_points

    cal_pts = r["calibration"]["ordinal_random_forest"]["reliability_severe"]
    df = pd.read_parquet(repo / "remediation/experiment/predictions_lossless.parquet")
    d = df[df["model"] == "ordinal_random_forest"]
    P = d[["proba_0", "proba_1", "proba_2"]].to_numpy()
    y = d["y_true"].to_numpy(int)
    raw_pts = reliability_points(y, P, 2)
    assert sum(p["count"] for p in raw_pts) == len(d) == 11630
    assert sum(p["count"] for p in cal_pts) == 11630

    fig = new_fig(3.30, width=5.0)
    ax = fig.add_axes([0.115, 0.115, 0.86, 0.86])
    ax.plot([0, 1], [0, 1], color=GRAY_M, ls=":", lw=1.0, zorder=1)
    ax.text(0.875, 0.845, "perfect reliability", fontsize=8, color=GRAY_M,
            rotation=33.5, ha="center", va="center")

    # Per-bin count labels: deterministic per-point placement keyed by the frozen
    # bin count (all sixteen counts are unique across both series), replacing the
    # r3 fixed per-series offset that collided in the congested low-probability
    # corner (r4 visual fix; REVISION_MEMO_R4). A KeyError here is intentional
    # fail-closed behavior: if the frozen evidence ever changed, every placement
    # must be re-audited. The thin white halo keeps a label readable where it
    # must cross a grid, reference, or series line.
    from matplotlib import patheffects as _pe
    _halo = [_pe.withStroke(linewidth=2.2, foreground="white")]
    _label_pos = {
        # raw series (open circles, dashed): (dx pt, dy pt, ha)
        8706:  (15, -4, "left"),    # right of the big first-bin circle, under the dashed rise
        1958:  (10, -10, "left"),   # below-right, clear of the calibrated square above
        635:   (-10, 5, "right"),  # above-left: below-right would sit inside the legend
        214:   (9, -10, "left"),
        67:    (10, -3, "left"),    # right of the circle, below the rising dashed segment
        33:    (9, -11, "left"),    # below-right, clear of the diagonal and the square above
        15:    (9, -10, "left"),
        2:     (10, -3, "left"),    # right of the top circle, away from the square's label
        # calibrated series (filled squares, solid)
        11148: (-4, 11, "left"),    # above the big first-bin square
        295:   (-9, 6, "right"),    # above-left; the dotted diagonal passes below-left here
        86:    (0, 8, "center"),    # local maximum: straight above
        42:    (0, -13, "center"),  # valley: straight below, above the distant dashed line
        28:    (0, -13, "center"),  # below; the raw circle sits above this square
        22:    (-9, 5, "right"),    # above-left, clear of the incoming steep segment
        4:     (-9, -2, "right"),   # left of the marker on the steep climb
        5:     (-8, 4, "right"),    # above-left of the top square, away from the raw circle
    }

    def series(pts, color, marker, label, ls):
        xs = [p["mean_predicted"] for p in pts]
        ys_ = [p["empirical_frequency"] for p in pts]
        ns = [p["count"] for p in pts]
        sizes = [14 + 26 * np.log10(n) for n in ns]
        ax.plot(xs, ys_, color=color, lw=1.0, ls=ls, zorder=2, alpha=0.85)
        ax.scatter(xs, ys_, s=sizes, facecolors="white" if marker == "o" else color,
                   edgecolors=color, linewidths=1.1, marker=marker, zorder=3,
                   label=label)
        for x, yv, n in zip(xs, ys_, ns):
            dx, dy, h = _label_pos[n]
            ax.annotate(f"{n:,}", (x, yv), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8, color=color, ha=h,
                        zorder=4, path_effects=_halo)

    series(raw_pts, GRAY_D, "o", "raw probabilities (recomputed from frozen evidence)",
           (0, (4, 2)))
    series(cal_pts, ACCENT, "s",
           "development-only calibrated (frozen run artifact)", "-")

    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.22, lw=0.5)
    ax.set_xlabel("Mean predicted severe-class probability (bin)", fontsize=9.5)
    ax.set_ylabel("Empirical severe frequency", fontsize=9.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9, handlelength=1.8,
              borderaxespad=0.3)
    ax.text(0.025, 0.975,
            "severe = working mapped class 2; 10 equal-width bins,\n"
            "empty bins omitted; numbers = crashes per bin (of 11,630)",
            transform=ax.transAxes, fontsize=8, color=GRAY_D, va="top", ha="left")
    save(fig, out, "fig06c_reliability.pdf")


# ----------------------------------------------------------------------
# Figure 10 (file fig08) - severe one-vs-rest precision-recall curves
# ----------------------------------------------------------------------
def fig08_ranking(repo: Path, out: Path):
    from sklearn.metrics import average_precision_score, precision_recall_curve

    run = repo / "remediation/runs/final_8af9d5bc23d8"
    # one representative per model family; expected AP from the committed
    # experiment/severe_ranking.md (cross-check tolerance 5e-4)
    show = [
        ("ordinal_random_forest", "Ordinal RF (primary)", 0.20377,
         ACCENT, "-"),
        ("random_forest_unweighted", "Random forest (unweighted)", 0.20125,
         "#7A7A7A", (0, (5, 2))),
        ("frank_hall_logistic", "Frank-Hall logit", 0.21829,
         "#B0803C", (0, (1, 1.1))),
        ("xgboost", "XGBoost", 0.21584, "#4E7B62", (0, (4, 1.5, 1, 1.5))),
        ("ebm", "EBM (exploratory)", 0.16860, "#8A6E8A", (0, (2, 1.4))),
        ("decision_tree", "Decision tree", 0.10851, "#444444",
         (0, (6, 2, 1, 2, 1, 2))),
    ]
    fig = new_fig(3.65)
    ax = fig.add_axes([0.093, 0.125, 0.885, 0.855])
    prevalence = None
    handles = []
    for key, lab, ap_expect, col, ls in show:
        ys, ps = [], []
        with open(run / f"predictions_{key}.csv", newline="") as f:
            for r in csv.DictReader(f):
                ys.append(1 if r["y_true"] == "2" else 0)
                ps.append(float(r["proba_2"]))
        ys = np.asarray(ys)
        ps = np.asarray(ps)
        if prevalence is None:
            prevalence = ys.mean()
        ap = average_precision_score(ys, ps)
        assert abs(ap - ap_expect) < 5e-4, (key, ap, ap_expect)
        prec, recall, _ = precision_recall_curve(ys, ps)
        (ln,) = ax.plot(recall, prec, lw=1.4, color=col, ls=ls,
                        label=f"{lab} — AP {ap:.3f}")
        handles.append(ln)
    ax.axhline(prevalence, color=BASELINE, ls="--", lw=1.0)
    ax.text(0.008, prevalence + 0.012,
            f"no-skill prevalence {prevalence:.3f}", fontsize=9,
            color=GRAY_M, ha="left", va="bottom")
    ax.set_xlabel("Severe-class recall", fontsize=10)
    ax.set_ylabel("Severe-class precision", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.22, lw=0.5)
    leg = ax.legend(handles=handles, loc="upper right", frameon=False,
                    fontsize=9, handlelength=2.6, labelspacing=0.55,
                    title="one representative per model family",
                    title_fontsize=9, alignment="left")
    ax.text(0.985, 0.40,
            "curves computed from the frozen 2012 probabilities of\n"
            "run final_8af9d5bc23d8; no operating threshold is selected",
            transform=ax.transAxes, fontsize=9, color=GRAY_M, va="top",
            ha="right", style="italic")
    save(fig, out, "fig08_severe_ranking.pdf")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[4]),
                    help="repository root (default: derived from this file's location)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                    / "figures" / "user"))
    args = ap.parse_args()
    repo = Path(args.repo)
    out = Path(args.out)

    r = json.loads((repo / "remediation/experiment/final_results.json")
                   .read_text())
    res = r["results"]

    fig01_lineage(out)
    fig02_workflow(out)
    fig09_reanalysis(repo, out)
    fig03_final_mae(res, out)
    fig04_tradeoff(res, out)
    fig06a_scores(r, out)
    fig06b_calibration(r, out)
    fig06c_reliability(repo, r, out)
    fig07_factorial(repo, out)
    fig05_corrections(repo, out)
    fig08_ranking(repo, out)
    print("done: 11 figures")


if __name__ == "__main__":
    main()
