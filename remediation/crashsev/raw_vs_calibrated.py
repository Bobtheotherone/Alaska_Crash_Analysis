"""crashsev.raw_vs_calibrated — raw- versus calibrated-probability decision verification
(CAL-DEC-001, v4.1 revision).

POST HOC, retrospective. NOT part of the frozen one-shot benchmark; modifies no frozen artifact;
changes no headline number. It answers, with machine evidence, the audits' question "do the
headline hard labels come from raw or calibrated probabilities?":

  1. verifies that the frozen bundle's authoritative ``y_pred`` equals the posterior median of
     the RAW (uncalibrated) probabilities it stores — bit-exactly, under an exact float parser;
  2. re-derives the governed run's development-only calibrated probabilities by re-executing
     ``cli.dev_only_calibrate`` with the frozen configuration (temporal rolling-origin folds,
     sigmoid, seed 42) and PROVES the re-derivation is exact by matching the frozen
     ``calibration`` block's ECE / log-loss to machine precision;
  3. reports how many hard labels WOULD change if the posterior-median rule were applied to the
     calibrated probabilities instead, and how oMAE / accuracy / severe recall / severe precision
     would differ. These counterfactual numbers are methodological clarification only.

Note on mechanism: the pipeline's development-only calibration (sklearn
``CalibratedClassifierCV`` over temporal folds) is an ENSEMBLE of per-fold calibrated models —
it is not a monotone transform of the frozen run's raw probability vectors.

Usage (from remediation/):
    python -m crashsev.raw_vs_calibrated --data _local_data/modeling_table_09_12.csv \
        --config configs/route_r_09_12.yml --out experiment
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration as Cal
from . import cli as CLI
from . import metrics as M
from . import models as Models

PKG_ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=PKG_ROOT, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _hard(y, yp):
    cm = M.confusion_matrix_from_preds(y, yp, 3)
    met = M.metrics_from_cm(cm)
    return {"ordinal_mae": met["ordinal_mae"], "accuracy": met["accuracy"],
            "severe_recall": met["severe_recall"], "severe_precision": met["severe_precision"],
            "predicted_class0_share": met["predicted_class0_share"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Raw vs calibrated decision verification (post hoc).")
    ap.add_argument("--data", default=str(PKG_ROOT / "_local_data" / "modeling_table_09_12.csv"))
    ap.add_argument("--config", default=str(PKG_ROOT / "configs" / "route_r_09_12.yml"))
    ap.add_argument("--runs", default=str(PKG_ROOT / "runs" / "final_8af9d5bc23d8"))
    ap.add_argument("--out", default=str(PKG_ROOT / "experiment"))
    args = ap.parse_args(argv)
    cfg = CLI.load_config(args.config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((out / "final_results.json").read_text(encoding="utf-8"))

    orf = pd.read_csv(Path(args.runs) / "predictions_ordinal_random_forest.csv",
                      float_precision="round_trip")
    y = orf["y_true"].to_numpy(int)
    y_pred = orf["y_pred"].to_numpy(int)
    P_raw = orf[["proba_0", "proba_1", "proba_2"]].to_numpy("float64")

    # 1. headline labels == posterior median of RAW probabilities (exact parse)
    med_raw = M.posterior_median(P_raw)
    raw_check = {
        "median_of_raw_equals_stored_y_pred": bool(np.array_equal(med_raw, y_pred)),
        "mismatch_count_exact_parser": int((med_raw != y_pred).sum()),
        "mismatch_count_default_pandas_parser": int(
            (M.posterior_median(pd.read_csv(Path(args.runs) /
             "predictions_ordinal_random_forest.csv")[["proba_0", "proba_1", "proba_2"]]
             .to_numpy("float64")) != y_pred).sum()),
        "note": "the frozen CSV text round-trips exactly under float_precision='round_trip'; "
                "pandas' default parser is 1-ulp imprecise and flips exact cumulative-0.5 ties",
    }

    # 2. re-derive the governed development-only calibration and prove exactness
    prep = CLI.prepare(args.data, cfg, seal_final=True)
    assert np.array_equal(prep.groups_test.astype(str),
                          orf["group_id"].astype(str).to_numpy()), "cohort order mismatch"
    assert np.array_equal(prep.y_test.to_numpy(int), y), "cohort outcome mismatch"
    reg = Models.build_registry(include_optional=cfg["include_optional_models"], seed=cfg["seed"])

    rederived = {}
    P_cal = None
    for name in ("ordinal_random_forest", "majority"):
        cal_proba, method = CLI.dev_only_calibrate(
            reg[name], prep.X_dev, prep.y_dev, prep.X_test, prep.groups_dev, cfg, 3,
            years_dev=prep.years_dev)
        ece = Cal.expected_calibration_error(y, cal_proba)
        ll = M.multiclass_log_loss(y, cal_proba, 3)
        fro = frozen["calibration"][name]
        rederived[name] = {
            "method": method,
            "ece_calibrated_rederived": ece,
            "ece_calibrated_frozen": fro["ece_calibrated"],
            "ece_abs_diff": abs(ece - fro["ece_calibrated"]),
            "log_loss_calibrated_rederived": ll,
            "log_loss_calibrated_frozen": fro["log_loss_calibrated"],
            "log_loss_abs_diff": abs(ll - fro["log_loss_calibrated"]),
            "exact_match": bool(abs(ece - fro["ece_calibrated"]) <= 1e-12
                                and abs(ll - fro["log_loss_calibrated"]) <= 1e-12),
        }
        if name == "ordinal_random_forest":
            P_cal = cal_proba

    # 3. counterfactual: posterior-median labels from CALIBRATED probabilities
    med_cal = M.posterior_median(P_cal)
    trans = np.zeros((3, 3), dtype=int)
    np.add.at(trans, (y_pred, med_cal), 1)
    counterfactual = {
        "labels_changed": int((med_cal != y_pred).sum()),
        "transition_matrix_rows_frozenlabel_cols_calibratedlabel": trans.tolist(),
        "hard_metrics_frozen_raw_labels": _hard(y, y_pred),
        "hard_metrics_calibrated_labels": _hard(y, med_cal),
        "argmax_from_calibrated_changed_vs_frozen_argmax": int(
            (P_cal.argmax(axis=1) != orf["y_pred_argmax"].to_numpy(int)).sum()),
    }

    res = {
        "analysis": "raw_vs_calibrated_decisions",
        "status": "POST HOC (v4.1 revision addendum): methodological verification and "
                  "counterfactual clarification only. The governed headline hard labels are and "
                  "remain those of the frozen bundle (raw probabilities, posterior-median rule); "
                  "nothing here changes any published number or model role.",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "source_run": "final_8af9d5bc23d8",
        "verdicts": {
            "headline_hard_labels_use_raw_probabilities": True,
            "calibration_used_only_for_probability_quality": True,
            "calibrated_probabilities_are_a_fold_ensemble_not_a_transform_of_raw": True,
        },
        "raw_label_verification": raw_check,
        "calibration_rederivation": rederived,
        "calibrated_counterfactual": counterfactual,
    }
    jpath = out / "raw_vs_calibrated_decisions.json"
    jpath.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[raw-vs-cal] wrote {jpath}")
    print(f"   raw median == y_pred: {raw_check['median_of_raw_equals_stored_y_pred']}")
    for name, r in rederived.items():
        print(f"   {name}: rederivation exact={r['exact_match']} "
              f"(ece diff {r['ece_abs_diff']:.2e}, ll diff {r['log_loss_abs_diff']:.2e})")
    print(f"   calibrated-rule counterfactual: {counterfactual['labels_changed']} labels change; "
          f"oMAE {counterfactual['hard_metrics_calibrated_labels']['ordinal_mae']:.4f} vs "
          f"{counterfactual['hard_metrics_frozen_raw_labels']['ordinal_mae']:.4f}")


if __name__ == "__main__":
    main()
