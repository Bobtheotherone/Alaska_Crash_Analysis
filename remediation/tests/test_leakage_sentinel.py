"""Leakage sentinel tests (VAL-001, METH-001): the denylist excludes outcome-derived
columns, and a test-only category cannot change trained preprocessing state."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crashsev import leakage, models as Models, preprocessing as pp, schema, splits, synth, target as T


def test_denylist_excludes_outcome_derived_and_identifiers():
    df = synth.make_synthetic_crash_df(n=1000, seed=1)
    deny = leakage.denylist_columns(df.columns, use_case="post_crash_triage")
    assert "Number of Fatalities" in deny
    assert "Number of Serious Injuries" in deny
    assert "Crash Number" in deny            # identifier
    allowed = leakage.allowed_feature_columns(df.columns)
    assert "Number of Fatalities" not in allowed
    assert schema.TARGET_COLUMN not in allowed
    assert "Weather" in allowed and "AADT" in allowed  # legitimate features kept


def test_unknown_timing_column_is_excluded_by_default():
    # A column not in the ledger must fail closed (excluded), not silently used.
    cols = ["Weather", "AADT", "Some Undocumented Column"]
    allowed = leakage.allowed_feature_columns(cols)
    assert "Some Undocumented Column" not in allowed


def test_test_only_category_cannot_alter_trained_preprocessing():
    """
    The core leakage sentinel: introduce a category value that appears ONLY in the test
    partition. A correct train-only pipeline must (a) not learn it, and (b) still transform
    the test rows deterministically. If preprocessing were fit on the full data (the bug),
    the trained feature space would depend on test-only values.
    """
    df = synth.make_synthetic_crash_df(n=1500, seed=2)
    _, y_all, _ = T.map_severity(df[schema.TARGET_COLUMN])
    keep = y_all.notna().to_numpy()
    df = df.loc[keep].reset_index(drop=True)
    y = y_all[keep].astype(int).reset_index(drop=True)

    dev_idx, test_idx, _, _ = splits.chronological_group_split(
        df, y, year_col="Year", group_col="Crash Number",
        final_test_years=[2016, 2017], row_id_col="Crash Number",
    )
    allowed = leakage.allowed_feature_columns(df.columns)
    X = df[allowed].copy()
    # inject a test-only sentinel category into a categorical feature
    X.loc[test_idx, "Weather"] = "SENTINEL_TEST_ONLY_VALUE"

    num_c, cat_c, _ = pp.split_feature_types(X.iloc[dev_idx])
    pre = pp.build_preprocessor(num_c, cat_c)
    pre.fit(X.iloc[dev_idx])                       # fit on DEV only
    feature_names = list(pre.get_feature_names_out())

    # The sentinel value must NOT appear anywhere in the trained feature space.
    assert not any("SENTINEL_TEST_ONLY_VALUE" in f for f in feature_names)

    # And transforming the test set (which contains the sentinel) must still work and
    # yield exactly the trained number of columns (unknown category folded, not added).
    Xt = pre.transform(X.iloc[test_idx])
    assert Xt.shape[1] == len(feature_names)


def test_near_perfect_sentinel_flags_a_planted_leak_on_train_only():
    # A diagnostic probe fit on TRAIN rows should flag a column that perfectly encodes y.
    y = pd.Series(np.repeat([0, 1, 2], 100))
    X = pd.DataFrame({"leak": y.astype(str), "noise": np.random.default_rng(0).integers(0, 5, 300)})
    flagged = dict(leakage.near_perfect_sentinel(X, y, min_accuracy=0.98))
    assert "leak" in flagged and flagged["leak"] >= 0.98
    assert "noise" not in flagged
