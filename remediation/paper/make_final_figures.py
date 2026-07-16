"""Generate the paper's final-result figures from the committed aggregate
(``experiment/final_results.json``). No per-crash data is read, so figures are reproducible
from the committed artifact alone (AA2-022)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PKG = Path(__file__).resolve().parents[1]
RESULTS = PKG / "experiment" / "final_results.json"
FIGS = PKG / "paper" / "generated"
FIGS.mkdir(parents=True, exist_ok=True)

PRETTY = {
    "majority": "Majority", "ordinal_median": "Ordinal median", "empirical_prior": "Empirical prior",
    "prior_probability": "Prior probability",
    "multinomial_logistic": "Multinomial logit", "proportional_odds": "Proportional-odds logit",
    "frank_hall_logistic": "Frank–Hall logit", "shallow_tree": "Shallow tree",
    "decision_tree": "Decision tree", "random_forest": "Random forest",
    "ordinal_random_forest": "Ordinal RF", "xgboost": "XGBoost", "ebm": "EBM",
    "random_forest_unweighted": "Random forest (unweighted)",
    "ordinal_random_forest_unweighted": "Ordinal RF (unweighted)",
}


def main():
    r = json.loads(RESULTS.read_text())
    res = r["results"]
    models = [m for m in res if "ordinal_mae" in res[m]]
    maj = res["majority"]["ordinal_mae"]

    # Fig A: ordinal MAE with 95% crash-level bootstrap CI, sorted, vs majority baseline
    order = sorted(models, key=lambda m: res[m]["ordinal_mae"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ys = np.arange(len(order))
    pts = [res[m]["ordinal_mae"] for m in order]
    los = [res[m]["ordinal_mae"] - res[m]["ordinal_mae_ci"]["ci_low"] for m in order]
    his = [res[m]["ordinal_mae_ci"]["ci_high"] - res[m]["ordinal_mae"] for m in order]
    colors = ["#54A24B" if res[m]["ordinal_mae"] < maj else "#B0B0B0" for m in order]
    ax.barh(ys, pts, color=colors)
    ax.errorbar(pts, ys, xerr=[los, his], fmt="none", ecolor="#333", capsize=3, lw=1)
    ax.axvline(maj, color="#E45756", ls="--", lw=2, label=f"majority baseline ({maj:.3f})")
    ax.set_yticks(ys); ax.set_yticklabels([PRETTY[m] for m in order])
    ax.set_xlabel("Ordinal MAE on held-out out-of-time 2012 (lower is better)")
    ax.set_title("Out-of-time ordinal MAE with 95% crash-level bootstrap CIs")
    ax.legend(); ax.invert_yaxis(); fig.tight_layout()
    fig.savefig(FIGS / "fig_final_ordinal_mae_ci.png", dpi=140); plt.close(fig)

    # Fig B: the core tension — ordinal MAE vs severe-class recall. PDF-002: per-point text
    # labels overlapped illegibly; points are now NUMBERED (sorted by ordinal MAE) with a
    # side legend, so no two labels can collide.
    order2 = sorted(models, key=lambda m: res[m]["ordinal_mae"])
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    # deterministic x-offsets for coincident points so every number stays legible
    # deterministic collision resolution: any marker landing too close to an already-placed one
    # is shifted right in fixed steps, so every number stays legible (coincident AND
    # near-coincident points; the shift is visual only and noted in the side key)
    placed = []          # (x, y) of already-drawn markers
    shifted = []         # ranks that were offset
    for i, m in enumerate(order2, 1):
        x, y = float(res[m]["severe_recall"]), float(res[m]["ordinal_mae"])
        x_draw = x
        while any(abs(x_draw - px) < 0.022 and abs(y - py) < 0.007 for px, py in placed):
            x_draw += 0.026
            if i not in shifted:
                shifted.append(i)
        placed.append((x_draw, y))
        kind = res[m].get("kind", "")
        col = ("#4C78A8" if kind == "candidate"
               else "#9C755F" if kind == "ablation" else "#F58518")
        ax.scatter(x_draw, y, s=170, color=col, zorder=3, edgecolors="white", linewidths=0.8)
        ax.annotate(str(i), (x_draw, y), ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold", zorder=4)
    coincident = shifted
    ax.axhline(maj, color="#E45756", ls="--", lw=1.5, label=f"majority ordinal MAE ({maj:.3f})")
    key = ("blue = candidate · brown = ablation control\norange = baseline/exploratory\n\n"
           + "\n".join(f"{i:>2}. {PRETTY[m]}" for i, m in enumerate(order2, 1)))
    if coincident:
        key += ("\n\npoints " + "·".join(map(str, coincident))
                + " offset horizontally\nfor visibility (overlapping positions)")
    ax.text(1.03, 0.5, key, transform=ax.transAxes, fontsize=8, va="center", family="monospace")
    ax.set_xlabel("Severe-class (2) recall on the 2012 evaluation")
    ax.set_ylabel("Ordinal MAE (lower better)")
    ax.set_title("Assigning severe crashes vs. average ordinal error\n(numbers = rank by ordinal MAE)",
                 fontsize=11, loc="left")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.76, 1))
    fig.savefig(FIGS / "fig_final_tradeoff.png", dpi=140); plt.close(fig)

    # Fig C: severe-class precision–recall (real final test)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for m in models:
        rec, prec = res[m]["severe_recall"], res[m]["severe_precision"]
        # skip degenerate points: never-predicts-severe models have UNDEFINED precision (NaN)
        if not (np.isfinite(rec) and np.isfinite(prec)) or (rec == 0 and prec == 0):
            continue
        ax.scatter(rec, prec, s=80, zorder=3)
        ax.annotate(PRETTY[m], (rec, prec), textcoords="offset points", xytext=(6, 3), fontsize=8)
    ax.set_xlabel("Severe-class (2) recall"); ax.set_ylabel("Severe-class (2) precision")
    ax.set_title("Severe-class precision–recall on the held-out out-of-time 2012")
    ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.7); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIGS / "fig_final_severe_pr.png", dpi=140); plt.close(fig)

    # Fig D: probabilistic quality (proper scores) + calibration ECE raw vs calibrated.
    # ALL models with probabilities, sorted by log loss. PDF-002: the rotated x-labels were too
    # small/dense — panel A is now a HORIZONTAL bar chart with full-size labels; the axis is
    # log-scaled because the hard one-hot baselines sit at 11+.
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    probm = sorted([m for m in models if "log_loss" in res[m]],
                   key=lambda m: res[m]["log_loss"], reverse=True)
    ll = [res[m]["log_loss"] for m in probm]
    cols = ["#4C78A8" if res[m].get("kind") == "candidate"
            else "#9C755F" if res[m].get("kind") == "ablation" else "#72B7B2" for m in probm]
    ypos = np.arange(len(probm))
    ax[0].barh(ypos, ll, color=cols)
    ax[0].set_yticks(ypos); ax[0].set_yticklabels([PRETTY[m] for m in probm], fontsize=9)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Log loss (lower better; log scale)")
    ax[0].set_title("Probabilistic quality (log loss) — all models\n"
                    "(blue = candidates, brown = unweighted ablation controls)")
    for y, v in zip(ypos, ll):
        ax[0].text(v * 1.05, y, f"{v:.3f}", va="center", fontsize=8)
    cal = r.get("calibration", {})
    cm = [m for m in cal if cal[m].get("calibrated")]
    if cm:
        raw = [cal[m]["ece_raw"] for m in cm]; caln = [cal[m]["ece_calibrated"] for m in cm]
        xx = np.arange(len(cm)); w = 0.35
        ax[1].bar(xx - w/2, raw, w, label="raw", color="#E45756")
        ax[1].bar(xx + w/2, caln, w, label="dev-fit calibrated", color="#54A24B")
        ax[1].set_xticks(xx); ax[1].set_xticklabels([PRETTY[m] for m in cm])
        ax[1].set_ylabel("Expected calibration error"); ax[1].set_title("Calibration (fit on development only)")
        ax[1].legend()
    fig.tight_layout(); fig.savefig(FIGS / "fig_final_probabilistic.png", dpi=140); plt.close(fig)

    print("wrote 4 final-result figures to", FIGS)


if __name__ == "__main__":
    main()
