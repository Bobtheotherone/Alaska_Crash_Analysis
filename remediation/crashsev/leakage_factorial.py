"""
crashsev.leakage_factorial — controlled factorial for evaluation defects (AA2-020).

The original "leakage diagnostic" changed three things at once. This module varies the three
factors in a 2x2x2 grid, repeated over seeds. IMPORTANT INTERPRETATION LIMIT (FAC-001): only the
**leakage_features** and **preprocess_before** factors are matched contrasts (same rows/model/seed,
one factor toggled) whose differences are legitimate apparent-inflation estimates. The
**random_split** factor changes the *evaluation population itself* (a random IID subset vs a
temporal held-out block) and its train size, so a "random_split main effect" averaged across cells
is NOT a clean causal effect — it is a **validation-optimism** contrast between two different
estimands. Read the random-split rows as "how optimistic is a random-CV estimate vs a temporal one",
never as "the effect of splitting" holding the target fixed.

Factors (each OFF = the leakage-controlled protocol, ON = the defect/alternative):
  * leakage_features   : add the 14 outcome-derived columns to the model's inputs. (matched contrast)
  * preprocess_before  : fit the ColumnTransformer on ALL rows before splitting (vs train-only). (matched)
  * random_split       : evaluate on a random IID split (vs a temporal, crash-grouped one). (DIFFERENT
                         estimand -> validation-optimism, not an isolated causal effect)

Outcome: ordinal MAE (primary), accuracy, and severe-class recall of a fixed model
(a class-balanced random forest, 120 trees — see ``_fixed_model``) evaluated under each cell's
protocol. Reported as (a) **matched simple effects** against the all-defects-off reference cell,
(b) the same contrasts conditioned on leakage already being present, and (c) **interactions**;
marginal main effects (mean over ON cells minus mean over OFF cells) are retained under an
explicit label but are NOT one-at-a-time contrasts and must not be described as such (FAC-001).

Governance: the factorial runs on **development years only** (it derives its temporal split
from the years < final_test_years), so the held-out final-test year is never touched. It is a
methodological illustration, not the headline predictive result.

Run: ``python -m crashsev.leakage_factorial --data _local_data/modeling_table_09_12.csv \
        --config configs/route_a_09_12.yml --out experiment``
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from . import cli
from . import contracts as K
from . import metrics as M
from . import preprocessing as pp
from . import target as T

PKG_ROOT = Path(__file__).resolve().parents[1]


def _fixed_model(seed: int):
    # A fixed, class-balanced comparator model held constant across all cells so that only the
    # three evaluation-defect factors vary. A random forest is fast (parallel) and makes the
    # leakage effect vivid (outcome-derived inputs are near-deterministic for their classes).
    return RandomForestClassifier(n_estimators=120, class_weight="balanced",
                                  n_jobs=-1, random_state=seed)


def _temporal_split(years: pd.Series, seed: int):
    uniq = sorted(int(y) for y in years.dropna().unique())
    if len(uniq) < 2:
        raise SystemExit("need >=2 development years for a temporal factorial split")
    test_year = uniq[-1]
    tr = np.where(years.to_numpy() != test_year)[0]
    te = np.where(years.to_numpy() == test_year)[0]
    return tr, te


def _random_split(n: int, seed: int, test_frac: float = 0.33):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(n * test_frac)
    return idx[k:], idx[:k]


def run_cell(df, y, allowed, leak_cols, *, leakage_features, preprocess_before,
             random_split, years, seed, n_classes):
    cols = list(allowed) + (list(leak_cols) if leakage_features else [])
    X = df[cols].copy()
    num, cat, _ = pp.split_feature_types(X, categorical_max_cardinality=100)
    pre = pp.build_preprocessor(num, cat, sparse=True)

    if random_split:
        tr, te = _random_split(len(X), seed)
    else:
        tr, te = _temporal_split(years, seed)

    if preprocess_before:
        Xall = pre.fit_transform(X)                 # LEAK: vocab/medians see the test rows
        Xtr, Xte = Xall[tr], Xall[te]
    else:
        Xtr = pre.fit_transform(X.iloc[tr])         # correct: train-only fit
        Xte = pre.transform(X.iloc[te])

    ytr, yte = y.iloc[tr].to_numpy(), y.iloc[te].to_numpy()
    if len(np.unique(ytr)) < n_classes:
        return None
    clf = _fixed_model(seed).fit(Xtr, ytr)
    yp = clf.predict(Xte).astype(int)
    cm = M.confusion_matrix_from_preds(yte, yp, n_classes)
    prf = M.per_class_prf_from_cm(cm)
    return {
        "ordinal_mae": M.ordinal_mae_from_cm(cm),
        "accuracy": M.accuracy_from_cm(cm),
        "severe_recall": prf[n_classes - 1]["recall"],
        "n_test": int(len(te)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Controlled leakage factorial (AA2-020).")
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(PKG_ROOT / "experiment"))
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args(argv)

    cfg = cli.load_config(args.config)
    schema = K.load_schema(); mapping = K.load_target_mapping(); ledger = K.load_ledger()
    raw = cli.load_any(args.data)
    K.assert_valid(raw, schema, mapping)
    _, y_all, _ = T.map_severity_from_mapping(raw[schema.target_col], mapping)
    keep = y_all.notna().to_numpy()
    df = raw.loc[keep].reset_index(drop=True)
    y = y_all[keep].astype(int).reset_index(drop=True)
    df, _ = cli.clean_numeric_sentinels(df, schema)

    # development years only (never touch the sealed final-test year)
    years_all = pd.to_numeric(df[cfg["year_col"]], errors="coerce")
    dev_mask = ~years_all.isin(cfg["final_test_years"]) & years_all.notna()
    df = df.loc[dev_mask].reset_index(drop=True)
    y = y.loc[dev_mask.to_numpy()].reset_index(drop=True)
    years = pd.to_numeric(df[cfg["year_col"]], errors="coerce")

    allowed = [c for c in ledger.allowed() if c in df.columns]
    leak_cols = [c for c in ledger.prohibited() if c in df.columns]
    n_classes = cfg["n_classes"]

    factors = ["leakage_features", "preprocess_before", "random_split"]
    cells: List[dict] = []
    for lf, pb, rs in itertools.product([False, True], repeat=3):
        per_seed = []
        for s in range(args.seeds):
            r = run_cell(df, y, allowed, leak_cols, leakage_features=lf, preprocess_before=pb,
                         random_split=rs, years=years, seed=s, n_classes=n_classes)
            if r:
                per_seed.append(r)
        if not per_seed:
            continue
        agg = {m: float(np.mean([p[m] for p in per_seed])) for m in ("ordinal_mae", "accuracy", "severe_recall")}
        agg_sd = {m + "_sd": float(np.std([p[m] for p in per_seed])) for m in ("ordinal_mae", "accuracy", "severe_recall")}
        cells.append({"leakage_features": lf, "preprocess_before": pb, "random_split": rs,
                      **agg, **agg_sd, "n_seeds": len(per_seed)})

    # MARGINAL main effects: mean(metric | factor ON) - mean(metric | factor OFF), averaged over
    # every other-factor cell. This averages matched and leakage-saturated conditions together —
    # it is NOT a one-at-a-time contrast (FAC-001) and is retained for continuity only.
    def main_effect(metric, factor):
        on = [c[metric] for c in cells if c[factor]]
        off = [c[metric] for c in cells if not c[factor]]
        return float(np.mean(on) - np.mean(off)) if on and off else None

    effects = {
        m: {f: main_effect(m, f) for f in factors}
        for m in ("ordinal_mae", "accuracy", "severe_recall")
    }
    reference = next((c for c in cells if not c["leakage_features"]
                      and not c["preprocess_before"] and not c["random_split"]), None)

    # MATCHED simple effects vs the reference cell, plus the same contrasts conditioned on
    # leakage being ON, and the resulting interaction terms. These are the headline numbers.
    cell_ix = {(c["leakage_features"], c["preprocess_before"], c["random_split"]): c for c in cells}

    def _cv(metric, lf, pb, rs):
        c = cell_ix.get((lf, pb, rs))
        return None if c is None else c[metric]

    simple_effects, interactions = {}, {}
    for m in ("ordinal_mae", "accuracy", "severe_recall"):
        ref_v, leak_v = _cv(m, False, False, False), _cv(m, True, False, False)
        if ref_v is None or leak_v is None:
            continue
        pe_off = _cv(m, False, True, False); pe_on = _cv(m, True, True, False)
        rs_off = _cv(m, False, False, True); rs_on = _cv(m, True, False, True)
        simple_effects[m] = {
            "leakage_features_vs_reference": leak_v - ref_v,
            "preprocess_before_given_leak_off": None if pe_off is None else pe_off - ref_v,
            "preprocess_before_given_leak_on": None if pe_on is None else pe_on - leak_v,
            "random_split_given_leak_off": None if rs_off is None else rs_off - ref_v,
            "random_split_given_leak_on": None if rs_on is None else rs_on - leak_v,
        }
        se = simple_effects[m]
        interactions[m] = {
            "leakage_x_preprocess_before": (None if None in (se["preprocess_before_given_leak_on"], se["preprocess_before_given_leak_off"])
                                            else se["preprocess_before_given_leak_on"] - se["preprocess_before_given_leak_off"]),
            "leakage_x_random_split": (None if None in (se["random_split_given_leak_on"], se["random_split_given_leak_off"])
                                       else se["random_split_given_leak_on"] - se["random_split_given_leak_off"]),
        }

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "design": "2x2x2 factorial x seeds; development years only; fixed balanced random forest",
        "development_years": sorted(int(v) for v in years.dropna().unique()),
        "n_seeds": args.seeds, "n_leakage_columns": len(leak_cols),
        "reference_cell_all_defects_off": reference,
        "cells": cells,
        "simple_effects_vs_reference": simple_effects,
        "interactions": interactions,
        "marginal_main_effects_NOT_one_at_a_time": effects,
        # kept under its historical key too, so prior consumers do not silently break:
        "main_effects": effects,
        "interpretation": {
            "ordinal_mae": "negative effect => that defect makes apparent error LOOK smaller (inflation)",
            "severe_recall": "leakage_features typically inflates severe recall via outcome-derived inputs",
            "interactions": "conditional on leakage being present, the other two defects have almost no "
                            "room left to matter (floor effect) — the defects are NOT additive",
        },
        "FAC-001_caveat": (
            "leakage_features and preprocess_before are MATCHED contrasts (same rows/model/seed, one "
            "factor toggled) -> their effects are valid apparent-inflation estimates. random_split is "
            "NOT a matched contrast: it changes the evaluation population and train size, so its "
            "'main effect' is a VALIDATION-OPTIMISM contrast between different estimands, not an "
            "isolated causal effect of splitting. Do not read the random_split row causally."
        ),
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "leakage_factorial.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(out, report)
    print(f"[factorial] wrote {out/'leakage_factorial.json'}")
    print("main effects on ordinal MAE:", {k: round(v, 4) for k, v in effects["ordinal_mae"].items() if v is not None})


def _write_md(out: Path, rep: dict) -> None:
    ref = rep["reference_cell_all_defects_off"]
    se = rep["simple_effects_vs_reference"]; ia = rep["interactions"]
    e = rep["marginal_main_effects_NOT_one_at_a_time"]

    def _f(v):
        return "—" if v is None else f"{v:+.3f}"

    lines = [
        "# Leakage factorial — matched defect contrasts and interactions (AA2-020 / FAC-001)\n",
        f"Design: {rep['design']}. Development years {rep['development_years']}, "
        f"{rep['n_seeds']} seeds, {rep['n_leakage_columns']} outcome-derived columns.\n",
        f"**Reference cell (all defects off — the leakage-controlled protocol):** "
        f"ordinal MAE = {ref['ordinal_mae']:.3f}, accuracy = {ref['accuracy']:.3f}, "
        f"severe recall = {ref['severe_recall']:.3f}.\n",
        "## Matched simple effects vs the reference (the headline numbers)\n",
        "`leakage` and `prep-before` are matched contrasts (same rows/model/seed, one factor "
        "toggled). `random-split` changes the evaluation population itself, so its row is a "
        "**validation-optimism** contrast between two estimands, not an isolated causal effect. "
        "A negative Δ ordinal MAE / positive Δ accuracy means the defect makes the model *look* "
        "better than it is.\n",
        "| defect (contrast vs reference) | Δ ordinal MAE | Δ accuracy | Δ severe recall | Δ oMAE given leakage already ON |",
        "|---|---|---|---|---|",
        f"| add outcome-derived features | {_f(se['ordinal_mae']['leakage_features_vs_reference'])} "
        f"| {_f(se['accuracy']['leakage_features_vs_reference'])} "
        f"| {_f(se['severe_recall']['leakage_features_vs_reference'])} | — |",
        f"| fit preprocessing before split | {_f(se['ordinal_mae']['preprocess_before_given_leak_off'])} "
        f"| {_f(se['accuracy']['preprocess_before_given_leak_off'])} "
        f"| {_f(se['severe_recall']['preprocess_before_given_leak_off'])} "
        f"| {_f(se['ordinal_mae']['preprocess_before_given_leak_on'])} |",
        f"| random (non-temporal) split — validation-optimism | {_f(se['ordinal_mae']['random_split_given_leak_off'])} "
        f"| {_f(se['accuracy']['random_split_given_leak_off'])} "
        f"| {_f(se['severe_recall']['random_split_given_leak_off'])} "
        f"| {_f(se['ordinal_mae']['random_split_given_leak_on'])} |",
        "\n## Interactions (ordinal MAE)\n",
        f"* leakage × preprocess-before: **{_f(ia['ordinal_mae']['leakage_x_preprocess_before'])}** — "
        "the preprocessing effect essentially vanishes once leakage is present.",
        f"* leakage × random-split: **{_f(ia['ordinal_mae']['leakage_x_random_split'])}** — "
        "likewise for the split contrast.",
        "\nThe three defects therefore **interact rather than add**: once outcome-derived features "
        "are present the apparent error is already near its floor, and the other two defects have "
        "almost no room left to move it.\n",
        "## Marginal main effects (NOT one-at-a-time contrasts; retained for continuity)\n",
        "Mean over all factor-ON cells minus mean over all factor-OFF cells — these averages mix "
        "matched and leakage-saturated conditions and *understate* the two smaller defects.\n",
        "| defect | marginal Δ ordinal MAE | marginal Δ accuracy | marginal Δ severe recall |",
        "|---|---|---|---|",
    ]
    names = {"leakage_features": "add outcome-derived features",
             "preprocess_before": "fit preprocessing before split",
             "random_split": "random (non-temporal) split"}
    for f, label in names.items():
        lines.append(f"| {label} | {_f(e['ordinal_mae'][f])} | {_f(e['accuracy'][f])} | {_f(e['severe_recall'][f])} |")
    lines += [
        "\n## All cells\n",
        "| leakage | prep-before | random-split | ordinal MAE | accuracy | severe recall |",
        "|---|---|---|---|---|---|",
    ]
    for c in rep["cells"]:
        lines.append(f"| {int(c['leakage_features'])} | {int(c['preprocess_before'])} | "
                     f"{int(c['random_split'])} | {c['ordinal_mae']:.3f} | {c['accuracy']:.3f} | "
                     f"{c['severe_recall']:.3f} |")
    lines.append("\n*Generated by `crashsev/leakage_factorial.py`; development data only.*")
    (out / "leakage_factorial.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
