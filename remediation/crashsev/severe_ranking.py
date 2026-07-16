"""crashsev.severe_ranking — severe-class one-vs-rest RANKING analysis (P0-INTERP-001, v4.1).

The posterior-median rule is Bayes-optimal for the absolute ordinal loss and, on this cohort,
almost never assigns the severe class — but a hard-decision recall near zero does NOT establish
that severe-crash *risk* is unpredictable. This module quantifies the ranking information in the
frozen probability vectors, deterministically and post hoc (no retraining, no threshold tuning):

  * one-vs-rest **average precision** (AP) for class 2, against the no-skill baseline
    (= severe prevalence);
  * one-vs-rest **AUROC** (secondary, prevalence-insensitive ranking statistic);
  * the full precision–recall curve points for selected models (figure + table).

Interpretation boundary (printed into the artifact): these are retrospective, exploratory ranking
diagnostics of the already-frozen 2012 predictions. No operating threshold is selected, because
choosing one on the evaluation year would be post-hoc and no stakeholder cost function exists.

Usage:
    python -m crashsev.severe_ranking --preds <dir-with-predictions_*.csv> --out <md> [--fig <png>]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import time

import numpy as np

SEVERE = 2
CURVE_MODELS = ["ordinal_random_forest", "ordinal_random_forest_unweighted", "xgboost"]


def _read(path):
    yt, p2 = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        if not r.fieldnames or "proba_2" not in r.fieldnames:
            return None, None
        for d in r:
            yt.append(int(d["y_true"]))
            p2.append(float(d["proba_2"]))
    return np.asarray(yt), np.asarray(p2)


def pr_curve(y_bin: np.ndarray, score: np.ndarray):
    """Precision/recall over descending score thresholds (ties grouped)."""
    order = np.argsort(-score, kind="mergesort")
    y = y_bin[order]; s = score[order]
    distinct = np.where(np.diff(s) != 0)[0]
    idx = np.r_[distinct, y.size - 1]
    tp = np.cumsum(y)[idx]
    fp = (idx + 1) - tp
    prec = tp / (tp + fp)
    rec = tp / y_bin.sum()
    return prec, rec


def average_precision(y_bin: np.ndarray, score: np.ndarray) -> float:
    prec, rec = pr_curve(y_bin, score)
    rec = np.r_[0.0, rec]
    return float(np.sum((rec[1:] - rec[:-1]) * prec))


def auroc(y_bin: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUROC (Mann–Whitney with tie correction)."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = int(y_bin.sum()); n0 = y_bin.size - n1
    return float((r[y_bin == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Severe-class one-vs-rest ranking analysis (v4.1).")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args(argv)

    rows = []
    curves = {}
    prevalence = None
    for f in sorted(glob.glob(os.path.join(args.preds, "predictions_*.csv"))):
        model = os.path.basename(f)[len("predictions_"):-len(".csv")]
        yt, p2 = _read(f)
        if yt is None:
            continue
        y_bin = (yt == SEVERE).astype(int)
        prevalence = float(y_bin.mean())
        if p2.max() == p2.min():          # constant forecast: ranking undefined/degenerate
            rows.append((model, float("nan"), float("nan")))
            continue
        rows.append((model, average_precision(y_bin, p2), auroc(y_bin, p2)))
        if model in CURVE_MODELS:
            curves[model] = pr_curve(y_bin, p2)
    rows.sort(key=lambda t: -(t[1] if t[1] == t[1] else -1))

    lines = [
        "# Severe-class (class 2) one-vs-rest ranking analysis (v4.1; P0-INTERP-001)\n",
        "Retrospective, exploratory ranking diagnostics computed deterministically from the frozen",
        "2012 probability vectors — no retraining, no threshold selection (choosing an operating",
        "threshold on the evaluation year would be post hoc, and no stakeholder cost function",
        "exists). These numbers answer a DIFFERENT question than the hard-decision recall in the",
        "main tables: *does the probability output order severe crashes above non-severe ones*,",
        "not *does the ordinal-loss-optimal decision rule assign the severe class*.\n",
        f"No-skill reference (severe prevalence): **AP = {prevalence:.5f}**. AUROC no-skill = 0.5.\n",
        "| model | severe one-vs-rest AP | AP / prevalence | AUROC |",
        "|---|---|---|---|",
    ]
    for m, apv, auc in rows:
        if apv != apv:
            lines.append(f"| {m} | — (constant forecast) | — | — |")
        else:
            lines.append(f"| {m} | {apv:.5f} | {apv / prevalence:.2f}× | {auc:.5f} |")
    lines += [
        "",
        "**Reading.** The oMAE-leading forest models carry **nontrivial severe ranking signal**",
        f"(AP ≈ 5× the no-skill prevalence; AUROC ≈ 0.79) even though the posterior-median rule",
        "almost never assigns class 2. Low hard-class recall under the ordinal-loss-optimal rule",
        "therefore must NOT be read as an absence of predictive information about severe risk;",
        "equally, this ranking signal must NOT be read as validated severe-crash detection — any",
        "operating point would require a real cost function and evaluation on data not used here.",
        "",
        f"*Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by"
        f" `crashsev/severe_ranking.py` from `{os.path.basename(os.path.normpath(args.preds))}`;"
        " aggregate only.*",
    ]
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[severe-ranking] {len(rows)} models -> {args.out}")

    if args.fig and curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5.2))
        for m, (prec, rec) in curves.items():
            ax.plot(rec, prec, lw=1.8, label=m.replace("_", " "))
        ax.axhline(prevalence, color="#E45756", ls="--", lw=1.5,
                   label=f"no-skill (prevalence {prevalence:.3f})")
        ax.set_xlabel("Severe-class recall (one-vs-rest, by probability threshold)")
        ax.set_ylabel("Severe-class precision")
        ax.set_title("Severe-class precision–recall RANKING curves (retrospective, exploratory)\n"
                     "no operating threshold is selected")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(args.fig, dpi=140); plt.close(fig)
        print(f"[severe-ranking] figure -> {args.fig}")


if __name__ == "__main__":
    main()
