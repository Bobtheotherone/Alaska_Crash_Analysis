"""
crashsev.leakage_demo — A1/A2 diagnostic: how much does the contaminated protocol inflate
apparent performance? (VAL-001, VAL-002, METH-001)

This runs the SAME model two ways on the synthetic fixture and reports the optimism gap:

  * CONTAMINATED (mimics the original project): a random stratified split, NO
    feature-availability denylist (outcome-derived columns kept), and preprocessing
    (one-hot vocabulary + median fill) computed on the FULL dataframe before the split.
  * CORRECTED (crashsev): a chronological + crash-grouped split, the leakage denylist,
    and a train-only fit/transform pipeline.

The gap is a DIAGNOSTIC of methodological optimism. It is not a benchmark; the contaminated
number must never be reported as a result. It runs on synthetic data, so the magnitudes are
illustrative of the mechanism, not a claim about Alaska.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import leakage, metrics as M, models as Models, preprocessing as pp, schema, splits, synth, target as T


def _contaminated_matrix(df: pd.DataFrame, y: pd.Series):
    """Mimic the integrated web pipeline: get_dummies + median fill on FULL data, keep all
    non-target columns (including outcome-derived), then a random split AFTER encoding."""
    X = df.drop(columns=[schema.TARGET_COLUMN], errors="ignore").copy()
    X = X.drop(columns=["Crash Number"], errors="ignore")
    num = X.select_dtypes(include=[np.number])
    cat = X.select_dtypes(exclude=[np.number])
    dummies = pd.get_dummies(cat, dummy_na=True) if not cat.empty else pd.DataFrame(index=X.index)
    Xf = pd.concat([num, dummies], axis=1)
    for c in Xf.columns:
        if Xf[c].isna().any():
            Xf[c] = Xf[c].fillna(float(Xf[c].median()))
    return Xf


def run(seed: int = 42, n: int = 12000) -> Dict:
    df = synth.make_synthetic_crash_df(n=n, seed=seed)
    _, y_all, _ = T.map_severity(df[schema.TARGET_COLUMN])
    keep = y_all.notna().to_numpy()
    df = df.loc[keep].reset_index(drop=True)
    y = y_all[keep].astype(int).reset_index(drop=True)

    model_names = ["random_forest", "xgboost"]
    registry = Models.build_registry(include_optional=True)
    out: Dict[str, Dict] = {}

    for name in model_names:
        if name not in registry:
            continue
        spec = registry[name]

        # ---- CONTAMINATED ----
        Xf = _contaminated_matrix(df, y)
        Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, random_state=42, stratify=y)
        est = spec.make_estimator()
        est.fit(Xtr, ytr)
        yp = est.predict(Xte).astype(int)
        contaminated = M.compute_all(yte, yp, schema.N_CLASSES)

        # ---- CORRECTED ----
        allowed = leakage.allowed_feature_columns(df.columns, use_case="post_crash_triage")
        X = df[allowed].copy()
        dev_idx, test_idx, _ = splits.chronological_group_split(
            df, y, year_col="Year", group_col="Crash Number",
            final_test_years=[2016, 2017], row_id_col="Crash Number",
        )
        num_c, cat_c, _ = pp.split_feature_types(X.iloc[dev_idx])
        pipe = Models.make_pipeline(spec, num_c, cat_c)
        pipe.fit(X.iloc[dev_idx], y.iloc[dev_idx])
        yp2 = pipe.predict(X.iloc[test_idx]).astype(int)
        corrected = M.compute_all(y.iloc[test_idx], yp2, schema.N_CLASSES)

        out[name] = {
            "contaminated": {k: contaminated[k] for k in
                             ["accuracy", "ordinal_mae", "qwk", "macro_f1", "severe_recall", "severe_precision"]},
            "corrected": {k: corrected[k] for k in
                          ["accuracy", "ordinal_mae", "qwk", "macro_f1", "severe_recall", "severe_precision"]},
            "optimism_gap": {
                "accuracy": contaminated["accuracy"] - corrected["accuracy"],
                "ordinal_mae": contaminated["ordinal_mae"] - corrected["ordinal_mae"],
                "severe_recall": contaminated["severe_recall"] - corrected["severe_recall"],
                "qwk": contaminated["qwk"] - corrected["qwk"],
            },
        }
    return {
        "note": ("SYNTHETIC diagnostic. Illustrates the mechanism by which the original "
                 "protocol (random split + outcome-derived features + full-data preprocessing) "
                 "overstates performance. Not an Alaska result."),
        "models": out,
    }


def main():
    res = run()
    out_dir = Path(__file__).resolve().parents[1] / "reanalysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "leakage_optimism_gap.json").write_text(json.dumps(res, indent=2))
    for name, d in res["models"].items():
        g = d["optimism_gap"]
        print(f"{name}: contaminated vs corrected optimism gap -> "
              f"acc +{g['accuracy']:.3f}, oMAE {g['ordinal_mae']:+.3f}, "
              f"QWK +{g['qwk']:.3f}, severe_recall {g['severe_recall']:+.3f}")


if __name__ == "__main__":
    main()
