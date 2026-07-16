"""Split-overlap, tiny-sample, unknown-category, and metric-regression tests
(VAL-002, EVAL-003, EVAL-004, EVAL-002)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crashsev import metrics as M, models as Models, preprocessing as pp, schema, splits, synth, target as T


def _prepare(n=1500, seed=3):
    df = synth.make_synthetic_crash_df(n=n, seed=seed)
    _, y_all, _ = T.map_severity(df[schema.TARGET_COLUMN])
    keep = y_all.notna().to_numpy()
    df = df.loc[keep].reset_index(drop=True)
    y = y_all[keep].astype(int).reset_index(drop=True)
    return df, y


def test_chronological_split_has_zero_group_overlap_and_stable_hash():
    df, y = _prepare()
    dev_idx, test_idx, manifest, assignment = splits.chronological_group_split(
        df, y, year_col="Year", group_col="Crash Number",
        final_test_years=[2016, 2017], row_id_col="Crash Number",
    )
    assert manifest.group_overlap_count == 0
    assert manifest.reconciles is True
    # final test contains only the sealed years
    assert set(pd.to_numeric(df.iloc[test_idx]["Year"]).unique()) <= {2016, 2017}
    assert not (set(pd.to_numeric(df.iloc[dev_idx]["Year"]).unique()) & {2016, 2017})
    # the saved assignment reproduces the manifest hash
    assert splits.assignment_hash(assignment) == manifest.assignment_sha256
    # hash is deterministic across identical calls
    _, _, manifest2, _ = splits.chronological_group_split(
        df, y, year_col="Year", group_col="Crash Number",
        final_test_years=[2016, 2017], row_id_col="Crash Number",
    )
    assert manifest.assignment_sha256 == manifest2.assignment_sha256


def test_tiny_sample_fails_closed_instead_of_train_equals_test():
    # The original code returns (X, X, y, y) when len(X) < 10; we must refuse.
    X = pd.DataFrame({"a": range(8)})
    y = pd.Series([0, 1] * 4)
    with pytest.raises(pp.InsufficientDataError):
        pp.check_sufficient(X, y, min_rows=50)
    # single-class training target also fails closed
    X2 = pd.DataFrame({"a": range(100)})
    y2 = pd.Series([0] * 100)
    with pytest.raises(pp.InsufficientDataError):
        pp.check_sufficient(X2, y2, min_rows=50)


def test_pipeline_handles_unseen_categories_at_inference():
    df, y = _prepare()
    dev_idx, test_idx, _, _ = splits.chronological_group_split(
        df, y, year_col="Year", group_col="Crash Number",
        final_test_years=[2016, 2017], row_id_col="Crash Number",
    )
    from crashsev import leakage
    X = df[leakage.allowed_feature_columns(df.columns)].copy()
    X.loc[test_idx, "Crash Type"] = "NEVER_SEEN_IN_TRAINING"
    num_c, cat_c, _ = pp.split_feature_types(X.iloc[dev_idx])
    reg = Models.build_registry(include_optional=False)
    pipe = Models.make_pipeline(reg["random_forest"], num_c, cat_c)
    pipe.fit(X.iloc[dev_idx], y.iloc[dev_idx])
    preds = pipe.predict(X.iloc[test_idx])   # must not raise
    assert len(preds) == len(test_idx)
    assert set(np.unique(preds)) <= {0, 1, 2}


def test_metrics_reproduce_published_confusion_matrices():
    """Regression test: our metric code reproduces the reconnaissance's recomputation
    from the paper's confusion matrices, to 3 decimals."""
    data = json.loads((ROOT / "data" / "confusion_matrices_from_paper.json").read_text())
    expected = {  # from the reconnaissance report's diagnostic table
        "decision_tree": dict(accuracy=0.5797, ordinal_mae=0.4510, qwk=0.2210, macro_f1=0.4515),
        "xgboost": dict(accuracy=0.6274, ordinal_mae=0.4013, qwk=0.3674, macro_f1=0.5142),
        "mlrf_random_forest": dict(accuracy=0.7098, ordinal_mae=0.3089, qwk=0.3650, macro_f1=0.5064),
        "ebm": dict(accuracy=0.5785, ordinal_mae=0.4902, qwk=0.3107, macro_f1=0.4604),
    }
    for name, exp in expected.items():
        m = M.metrics_from_cm(data["reported"][name]["confusion_matrix"])
        for k, v in exp.items():
            assert abs(m[k] - v) < 1e-3, f"{name}.{k}: {m[k]} vs {v}"


def test_majority_baseline_matches_prevalence():
    m = M.metrics_from_cm([[7395, 0, 0], [3152, 0, 0], [389, 0, 0]])
    assert abs(m["accuracy"] - 0.6762) < 1e-3
    assert m["severe_recall"] == 0.0
    assert m["predicted_class0_share"] == 1.0
