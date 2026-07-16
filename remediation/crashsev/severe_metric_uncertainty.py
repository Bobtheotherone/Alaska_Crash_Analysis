"""crashsev.severe_metric_uncertainty — descriptive uncertainty for the frozen primary model's
severe-class metrics (SEV-UNC-001, v4.1 revision).

POST HOC, retrospective, deterministic. NOT part of the frozen one-shot benchmark; modifies no
frozen artifact; selects no threshold; implies no operational validation. The published severe
recall/precision and the severe one-vs-rest AP/AUROC were point estimates; this addendum attaches
crash-level case-bootstrap percentile intervals to them for the frozen primary model
(``ordinal_random_forest``, run ``final_8af9d5bc23d8``).

Bootstrap: rows are resampled with replacement (each crash is exactly one row, so case- and
crash-level resampling coincide), ``numpy.random.default_rng(seed)`` with ``rng.integers``;
2,000 replicates; 2.5/97.5 percentiles. Undefined-value rule: a replicate in which the primary
model predicts no severe crash has UNDEFINED severe precision, and a replicate with no true
severe crash has UNDEFINED recall/AP/AUROC; such replicates are counted and EXCLUDED from the
percentiles (with 50 predicted-severe and 450 true-severe rows the expected count is ~0).

Usage (from remediation/):
    python -m crashsev.severe_metric_uncertainty --runs runs/final_8af9d5bc23d8 --out experiment
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PKG_ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=PKG_ROOT, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Severe-metric case-bootstrap uncertainty (post hoc).")
    ap.add_argument("--runs", default=str(PKG_ROOT / "runs" / "final_8af9d5bc23d8"))
    ap.add_argument("--out", default=str(PKG_ROOT / "experiment"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(args.runs) / "predictions_ordinal_random_forest.csv",
                     float_precision="round_trip")
    y = df["y_true"].to_numpy(int)
    yp = df["y_pred"].to_numpy(int)
    s2 = df["proba_2"].to_numpy("float64")
    pos = (y == 2).astype(int)
    hit = ((y == 2) & (yp == 2)).astype(int)
    pred2 = (yp == 2).astype(int)
    n = len(y)

    point = {
        "severe_recall": float(hit.sum() / pos.sum()),
        "severe_precision": float(hit.sum() / pred2.sum()),
        "severe_average_precision": float(average_precision_score(pos, s2)),
        "severe_auroc": float(roc_auc_score(pos, s2)),
        "n": n, "true_severe": int(pos.sum()), "predicted_severe": int(pred2.sum()),
    }

    rng = np.random.default_rng(args.seed)
    reps = {k: [] for k in ("severe_recall", "severe_precision",
                            "severe_average_precision", "severe_auroc")}
    undef = {k: 0 for k in reps}
    for _ in range(args.resamples):
        idx = rng.integers(0, n, size=n)
        p_, h_, d_, s_ = pos[idx], hit[idx], pred2[idx], s2[idx]
        tp = int(h_.sum())
        if p_.sum() == 0:
            for k in ("severe_recall", "severe_average_precision", "severe_auroc"):
                undef[k] += 1
        else:
            reps["severe_recall"].append(tp / p_.sum())
            if p_.sum() == len(p_):
                undef["severe_average_precision"] += 1; undef["severe_auroc"] += 1
            else:
                reps["severe_average_precision"].append(average_precision_score(p_, s_))
                reps["severe_auroc"].append(roc_auc_score(p_, s_))
        if d_.sum() == 0:
            undef["severe_precision"] += 1
        else:
            reps["severe_precision"].append(tp / d_.sum())

    intervals = {}
    for k, vals in reps.items():
        v = np.asarray(vals, dtype="float64")
        intervals[k] = {"point": point[k],
                        "ci95_percentile": [float(np.percentile(v, 2.5)),
                                            float(np.percentile(v, 97.5))],
                        "defined_replicates": int(len(v)),
                        "undefined_replicates_excluded": int(undef[k])}

    res = {
        "analysis": "severe_metric_uncertainty",
        "status": "POST HOC (v4.1 revision addendum): descriptive case-bootstrap intervals for the "
                  "frozen primary model's severe-class metrics on the historically exposed 2012 "
                  "cohort. NOT prespecified; no threshold is selected; no operational validation "
                  "is implied; no frozen artifact is modified.",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "source_run": "final_8af9d5bc23d8",
        "model": "ordinal_random_forest (frozen primary; posterior-median labels; raw probabilities)",
        "bootstrap": {"unit": "crash (== row)", "method": "percentile", "seed": args.seed,
                      "replicates": args.resamples,
                      "undefined_rule": "undefined replicate values counted and excluded"},
        "metrics": intervals,
    }
    jpath = out / "severe_metric_uncertainty.json"
    jpath.write_text(json.dumps(res, indent=2), encoding="utf-8")

    lines = [
        "# Severe-class metric uncertainty — frozen primary model (SEV-UNC-001; post hoc, v4.1 revision)\n",
        res["status"] + "\n",
        "| metric | point | 95% case-bootstrap CI | undefined replicates |",
        "|---|---|---|---|",
    ]
    for k, v in intervals.items():
        lines.append(f"| {k.replace('_', ' ')} | {v['point']:.4f} | "
                     f"[{v['ci95_percentile'][0]:.4f}, {v['ci95_percentile'][1]:.4f}] | "
                     f"{v['undefined_replicates_excluded']} |")
    lines += [
        "",
        "**Reading.** These intervals quantify case-sampling variability only, under the same "
        "cross-crash independence approximation as the headline interval; they are conditional "
        "on the one fitted model, the frozen protocol, the researcher-defined target, and the "
        "exposed cohort. They do not validate any operating point.",
        "",
        f"*Generated {res['generated_utc']} by `crashsev/severe_metric_uncertainty.py` "
        f"(commit {res['git_commit'][:8]}); seed {args.seed}, {args.resamples} replicates; aggregates only.*",
    ]
    (out / "severe_metric_uncertainty.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[severe-unc] wrote {jpath}")
    for k, v in intervals.items():
        print(f"   {k}: {v['point']:.4f} [{v['ci95_percentile'][0]:.4f}, {v['ci95_percentile'][1]:.4f}]")


if __name__ == "__main__":
    main()
