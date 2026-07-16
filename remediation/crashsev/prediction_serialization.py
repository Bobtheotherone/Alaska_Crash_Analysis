"""crashsev.prediction_serialization — release-safe lossless prediction evidence (SER-001, v4.1).

POST HOC packaging step; modifies NO frozen artifact. The frozen run bundle's prediction CSVs
already serialise probabilities at full ``repr`` precision (pandas ``to_csv`` default), so the
text round-trips bit-exactly — but ONLY when the reader parses with an exact float parser.
pandas' DEFAULT parser (``float_precision=None``) is fast-but-1-ulp-imprecise, which flips the
posterior-median label on rows whose cumulative probability equals exactly 0.5 (forest
probabilities are rationals with denominator 300) and flips argmax on exact two-way ties.
This module therefore

  1. re-reads every prediction file with ``float_precision='round_trip'`` (exact),
  2. verifies ZERO posterior-median and ZERO argmax label mismatches against the stored,
     authoritative ``y_pred`` / ``y_pred_argmax`` columns for all models,
  3. writes ONE de-identified, binary-lossless parquet (surrogate ids per the PRIV-001
     witness transform; float64 columns — no text parsing on the consumer side),
  4. re-reads the parquet and proves bitwise probability equality, exact hard-metric
     reproduction against the frozen manifest, probability-metric agreement, and exact
     reproduction of the primary paired-bootstrap interval, and
  5. records everything in ``experiment/prediction_roundtrip_test.json``.

``y_pred`` remains the authoritative governed hard label; the parquet exists so that a
reviewer can reconstruct it exactly without knowing the parser caveat.

Usage (from remediation/):
    python -m crashsev.prediction_serialization --runs runs/final_8af9d5bc23d8 --out experiment
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M

PKG_ROOT = Path(__file__).resolve().parents[1]
PERM_SEED = 20260712


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=PKG_ROOT, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _exact_primary_pair_ci(y, yp, ym, groups, seed=42, reps=2000):
    """Reproduce crashsev.uncertainty.paired_difference_ci's exact RNG draw sequence
    (rng.choice over the sorted unique groups) with a fast row lookup."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    order = np.argsort(groups, kind="stable")
    sg = groups[order]
    ea, eb = np.abs(y - yp).astype(float), np.abs(y - ym).astype(float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for i in range(reps):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        rows = order[np.searchsorted(sg, sampled)]
        diffs[i] = ea[rows].mean() - eb[rows].mean()
    return float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lossless prediction serialization + roundtrip test.")
    ap.add_argument("--runs", default=str(PKG_ROOT / "runs" / "final_8af9d5bc23d8"))
    ap.add_argument("--out", default=str(PKG_ROOT / "experiment"))
    args = ap.parse_args(argv)
    run = Path(args.runs); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))

    pred_files = sorted(run.glob("predictions_*.csv"))
    frames = {}
    per_model = {}
    ids0 = None
    for p in pred_files:
        name = p.stem.replace("predictions_", "")
        df = pd.read_csv(p, float_precision="round_trip")
        frames[name] = df
        ids = df["group_id"].astype(str).to_numpy()
        if ids0 is None:
            ids0 = ids
        assert np.array_equal(ids, ids0), f"{name}: row order differs across prediction files"
        P = df[["proba_0", "proba_1", "proba_2"]].to_numpy("float64")
        med = M.posterior_median(P)
        am = P.argmax(axis=1)
        per_model[name] = {
            "rows": int(len(df)),
            "median_label_mismatches_exact_parse": int((med != df["y_pred"].to_numpy(int)).sum()),
            "argmax_label_mismatches_exact_parse": int((am != df["y_pred_argmax"].to_numpy(int)).sum()),
        }
        # default-parser comparison, to document the pitfall explicitly
        dflt = pd.read_csv(p)
        Pd = dflt[["proba_0", "proba_1", "proba_2"]].to_numpy("float64")
        per_model[name]["median_label_mismatches_DEFAULT_parser"] = int(
            ((np.cumsum(Pd, axis=1) < 0.5).sum(axis=1) != dflt["y_pred"].to_numpy(int)).sum())

    # de-identified surrogate ids (identical recipe + seed to the published witness)
    ordered = sorted(set(ids0.tolist()))
    rng = random.Random(PERM_SEED)
    rng.shuffle(ordered)
    mapping = {cid: f"T{ix:05d}" for ix, cid in enumerate(ordered)}
    surr = np.array([mapping[i] for i in ids0])

    long = []
    for name, df in frames.items():
        long.append(pd.DataFrame({
            "model": name, "row_surrogate": surr,
            "y_true": df["y_true"].to_numpy(int),
            "y_pred": df["y_pred"].to_numpy(int),
            "y_pred_argmax": df["y_pred_argmax"].to_numpy(int),
            "proba_0": df["proba_0"].to_numpy("float64"),
            "proba_1": df["proba_1"].to_numpy("float64"),
            "proba_2": df["proba_2"].to_numpy("float64"),
        }))
    tbl = pd.concat(long, ignore_index=True)
    pq = out / "predictions_lossless.parquet"
    tbl.to_parquet(pq, index=False)

    # roundtrip verification ---------------------------------------------------
    back = pd.read_parquet(pq)
    checks = {"parquet_rows": int(len(back)),
              "expected_rows": int(len(pred_files) * len(ids0)),
              "models": {}}
    all_ok = True
    for name, df in frames.items():
        b = back[back["model"] == name].reset_index(drop=True)
        Pb = b[["proba_0", "proba_1", "proba_2"]].to_numpy("float64")
        Pc = df[["proba_0", "proba_1", "proba_2"]].to_numpy("float64")
        bitwise = bool(np.ascontiguousarray(Pb).tobytes() == np.ascontiguousarray(Pc).tobytes())
        labels_ok = bool(np.array_equal(b["y_pred"].to_numpy(int), df["y_pred"].to_numpy(int))
                         and np.array_equal(b["y_true"].to_numpy(int), df["y_true"].to_numpy(int)))
        med = M.posterior_median(Pb)
        med_ok = int((med != b["y_pred"].to_numpy(int)).sum())
        y, yp = b["y_true"].to_numpy(int), b["y_pred"].to_numpy(int)
        omae = float(np.mean(np.abs(y - yp)))
        rec = manifest["results"][name]
        hard_ok = (abs(omae - rec["ordinal_mae"]) == 0.0
                   and abs(float((y == yp).mean()) - rec["accuracy"]) == 0.0)
        ll = M.multiclass_log_loss(y, Pb, 3)
        prob_ok = abs(ll - rec["log_loss"]) <= 1e-12
        entry = {"bitwise_probability_roundtrip": bitwise,
                 "label_columns_identical": labels_ok,
                 "posterior_median_mismatches_from_parquet": med_ok,
                 "hard_metrics_exact_vs_manifest": bool(hard_ok),
                 "log_loss_vs_manifest_abs_diff": float(abs(ll - rec["log_loss"]))}
        entry.update(per_model[name])
        checks["models"][name] = entry
        all_ok &= (bitwise and labels_ok and med_ok == 0 and hard_ok and prob_ok
                   and per_model[name]["median_label_mismatches_exact_parse"] == 0
                   and per_model[name]["argmax_label_mismatches_exact_parse"] == 0)

    # exact reproduction of the primary paired-bootstrap interval from the parquet
    borf = back[back["model"] == "ordinal_random_forest"].reset_index(drop=True)
    bmaj = back[back["model"] == "majority"].reset_index(drop=True)
    lo, hi = _exact_primary_pair_ci(borf["y_true"].to_numpy(int), borf["y_pred"].to_numpy(int),
                                    bmaj["y_pred"].to_numpy(int), ids0,
                                    seed=manifest["config"]["seed"],
                                    reps=manifest["config"]["bootstrap_resamples"])
    recci = manifest["paired_difference_vs_baseline"]["ordinal_random_forest"]
    ci_ok = (abs(lo - recci["ci_low"]) <= 1e-15 and abs(hi - recci["ci_high"]) <= 1e-15)
    checks["primary_bootstrap_interval"] = {
        "recomputed": [lo, hi], "manifest": [recci["ci_low"], recci["ci_high"]],
        "exact_match": bool(ci_ok),
        "note": "row_surrogate is a bijection of the crash id, so surrogate-level resampling is "
                "identical; the RNG stream here uses the ORIGINAL sorted-id order (documented).",
    }
    all_ok &= ci_ok

    result = {
        "analysis": "prediction_roundtrip_test",
        "status": "POST HOC packaging verification (v4.1 revision); no frozen artifact modified; "
                  "y_pred in the frozen bundle remains the authoritative governed hard label.",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "source_run": manifest["run_id"],
        "parser_note": "Frozen CSVs are full-repr text and round-trip EXACTLY under an exact "
                       "float parser (pandas float_precision='round_trip' or Python float()); "
                       "pandas' DEFAULT parser is 1-ulp imprecise and flips labels on exact "
                       "cumulative-0.5 / argmax-tie rows. The parquet removes the pitfall.",
        "lossless_parquet": {"path": str(pq.relative_to(PKG_ROOT)).replace("\\", "/"),
                             "sha256": _sha256(pq),
                             "deid": "row_surrogate per PRIV-001 witness transform (seeded "
                                     "permutation, mapping discarded)",
                             "kept_out_of_git": "yes (per-crash rows; hash pinned here)"},
        "all_checks_pass": bool(all_ok),
        "checks": checks,
    }
    jpath = out / "prediction_roundtrip_test.json"
    jpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[serialization] all_checks_pass={all_ok}; wrote {jpath}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
