"""crashsev.seed_robustness — final-year seed/refit robustness of the forest results (STAT-002b).

The primary bootstrap interval is CONDITIONAL on one fitted model: it quantifies case-sampling
uncertainty, not training-seed/refit variability. This module measures the latter directly, as a
**retrospective, exploratory robustness addendum** requested by the fourth-round review: the four
forest variants are refit on the development data under five seeds (nothing else varies — same
frozen cohort, tier, preprocessing, and decision rule) and scored once each on the 2012 year.

Governance status, stated plainly: this analysis READS the final-year outcomes (like the other
post-benchmark diagnostics) and REFITS models, so it is not part of the frozen one-shot
benchmark; it does not modify any frozen artifact, selects nothing, and is reported only as
seed-variability evidence for the already-published primary comparison.

Usage:
    python -m crashsev.seed_robustness --data _local_data/modeling_table_09_12.csv \
        --config configs/route_r_09_12.yml --out experiment --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import cli as CLI
from . import metrics as M
from . import models as Models
from . import preprocessing as pp

PKG_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROBUST = ["random_forest_unweighted", "ordinal_random_forest_unweighted",
                 "ordinal_random_forest", "random_forest"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Final-year seed/refit robustness (retrospective addendum).")
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(PKG_ROOT / "experiment"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args(argv)
    cfg = CLI.load_config(args.config)
    n_classes = int(cfg["n_classes"])

    prep = CLI.prepare(args.data, cfg, seal_final=True)
    y_test = prep.y_test.to_numpy()
    maj = float(M.ordinal_mae_from_cm(M.confusion_matrix_from_preds(
        y_test, np.zeros_like(y_test), n_classes)))   # majority == constant class 0 here

    results = {m: [] for m in MODELS_ROBUST}
    for seed in args.seeds:
        reg = Models.build_registry(include_optional=False, seed=int(seed))
        for name in MODELS_ROBUST:
            pipe, _ = CLI.fit_model(reg[name], prep.X_dev, prep.y_dev,
                                    prep.numeric_cols, prep.categorical_cols, cfg)
            yp_arg, proba = CLI.predict_aligned(pipe, prep.X_test, n_classes)
            yp = CLI.decide_labels(proba, yp_arg, cfg)
            omae = float(M.ordinal_mae_from_cm(M.confusion_matrix_from_preds(y_test, yp, n_classes)))
            results[name].append(omae)
            print(f"[seed-robustness] seed={seed} {name}: oMAE={omae:.4f} (d vs majority {omae-maj:+.4f})")

    lines = [
        "# Final-year seed/refit robustness — forest variants, five seeds (v4.1 addendum)\n",
        "Retrospective, exploratory robustness analysis (reads the final year; refits models; is",
        "NOT part of the frozen one-shot benchmark and modifies no frozen artifact). Same cohort,",
        f"tier (`{cfg.get('feature_tier')}`), preprocessing, and decision rule "
        f"(`{cfg.get('decision_rule')}`) as the primary run; only the estimator seed varies "
        f"(seeds {args.seeds}). Majority-baseline ordinal MAE = {maj:.4f} (deterministic).\n",
        "| model | per-seed oMAE | mean | sd | range | Δ vs majority (all seeds) | direction persists |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in MODELS_ROBUST:
        v = np.array(results[name])
        d = v - maj
        per = ", ".join(f"{x:.4f}" for x in v)
        persists = "yes" if (d < 0).all() else ("no" if (d > 0).all() else "MIXED")
        sign_note = "better than baseline" if (d < 0).all() else ("worse than baseline" if (d > 0).all() else "mixed")
        lines.append(f"| {name} | {per} | {v.mean():.4f} | {v.std():.4f} | "
                     f"{v.max() - v.min():.4f} | {d.min():+.4f} … {d.max():+.4f} | {persists} ({sign_note}) |")
    lines += [
        "",
        "**Reading.** The frozen primary run used seed 42; these five refits show how much of the",
        "reported margins is training-noise. A 'yes' in the last column means every refit lands on",
        "the same side of the majority baseline as the published result. Seed variability adds to,",
        "and is not captured by, the case-bootstrap interval — which remains conditional on one",
        "fitted model.",
        "",
        f"*Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by "
        f"`crashsev/seed_robustness.py`; aggregates only.*",
    ]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "seed_robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[seed-robustness] wrote {out / 'seed_robustness.md'}")


if __name__ == "__main__":
    main()
