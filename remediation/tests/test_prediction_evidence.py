"""R-004: the lossless parquet is the authoritative row-level evidence.

Two guarantees, each pinned by a test:

  1. every hard metric of every model recomputes with ZERO mismatch from
     ``experiment/predictions_lossless.parquet`` against the frozen
     ``final_results.json`` values;
  2. the documented CSV hazard is real and stays documented: reading the frozen
     prediction CSVs with pandas' DEFAULT float parser perturbs exact cumulative-0.5
     ties and flips posterior-median labels — three for the primary model, four across
     all models, exactly as recorded per model in
     ``experiment/prediction_roundtrip_test.json`` — while
     ``float_precision="round_trip"`` reproduces every stored label exactly.

Both tests skip with an explicit reason when their (deliberately uncommitted /
local-only) inputs are absent, e.g. in a bare source checkout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from crashsev.metrics import posterior_median

REM = Path(__file__).resolve().parents[1]
PARQUET = REM / "experiment" / "predictions_lossless.parquet"
RUN = REM / "runs" / "final_8af9d5bc23d8"


def _final_results() -> dict:
    return json.loads((REM / "experiment" / "final_results.json").read_text("utf-8"))


def test_parquet_reproduces_all_hard_metrics_and_labels_exactly():
    if not PARQUET.exists():
        pytest.skip("lossless parquet not present in this checkout "
                    "(shipped in the handoff evidence/ tier)")
    pd = pytest.importorskip("pandas")
    fr = _final_results()
    df = pd.read_parquet(PARQUET)
    assert len(df) == 162_820  # 14 models x 11,630 crashes
    mismatch_total = 0
    for model, rec in fr["results"].items():
        d = df[df["model"] == model]
        assert len(d) == fr["split"]["n_final_test"], model
        y = d["y_true"].to_numpy(int)
        yp = d["y_pred"].to_numpy(int)
        assert float(np.mean(np.abs(y - yp))) == rec["ordinal_mae"], model
        assert float(np.mean(y == yp)) == rec["accuracy"], model
        # posterior-median reconstruction from the stored probabilities
        P = d[["proba_0", "proba_1", "proba_2"]].to_numpy()
        mismatch_total += int((posterior_median(P) != yp).sum())
    assert mismatch_total == 0, (
        f"posterior-median reconstruction mismatches: {mismatch_total} of 162,820")


def test_released_probability_vectors_are_valid_distributions():
    """FH-CONV-001 acceptance (manuscript §6.2.1): every released probability vector of
    every model — including both Frank–Hall variants — is finite, nonnegative, sums to
    one within tolerance, and has monotone reconstructed cumulative probabilities."""
    if not PARQUET.exists():
        pytest.skip("lossless parquet not present in this checkout "
                    "(shipped in the handoff evidence/ tier)")
    pd = pytest.importorskip("pandas")
    df = pd.read_parquet(PARQUET)
    P = df[["proba_0", "proba_1", "proba_2"]].to_numpy()
    assert np.isfinite(P).all(), "non-finite released probability"
    assert (P >= 0).all(), "negative released probability"
    assert np.abs(P.sum(axis=1) - 1.0).max() < 1e-9, "released probability rows must sum to 1"
    cum = np.cumsum(P, axis=1)
    assert (np.diff(cum, axis=1) >= -1e-12).all(), "reconstructed cumulative must be monotone"
    # Frank–Hall rows specifically: the 1e-9 floor implies strictly positive entries
    fh = df["model"].isin(["frank_hall_logistic", "ordinal_random_forest",
                           "ordinal_random_forest_unweighted"])
    assert (P[fh.to_numpy()] > 0).all(), "Frank–Hall probabilities must be strictly positive"


def test_default_csv_parser_perturbs_exact_ties_roundtrip_parser_does_not():
    if not RUN.exists():
        pytest.skip("local-only run bundle not present in this checkout "
                    "(per-crash CSVs stay local by policy; the parquet is authoritative)")
    pd = pytest.importorskip("pandas")
    fr = _final_results()
    recorded = json.loads(
        (REM / "experiment" / "prediction_roundtrip_test.json").read_text("utf-8")
    )["checks"]["models"]
    flips_roundtrip = 0
    for model in fr["results"]:
        f = RUN / f"predictions_{model}.csv"
        exact = pd.read_csv(f, float_precision="round_trip")
        naive = pd.read_csv(f)  # default (fast, 1-ulp imprecise) parser
        stored = exact["y_pred"].to_numpy(int)
        P_exact = exact[["proba_0", "proba_1", "proba_2"]].to_numpy()
        P_naive = naive[["proba_0", "proba_1", "proba_2"]].to_numpy()
        flips_roundtrip += int((posterior_median(P_exact) != stored).sum())
        flips = int((posterior_median(P_naive) != stored).sum())
        assert flips == recorded[model]["median_label_mismatches_DEFAULT_parser"], (
            f"{model}: default-parser flips {flips} != recorded "
            f"{recorded[model]['median_label_mismatches_DEFAULT_parser']}; "
            "update README/R-004 wording and the round-trip witness if pandas changed")
    assert flips_roundtrip == 0, "round_trip parser must reproduce every stored label"
    # the documented headline numbers: 3 primary-model flips, 4 across all models
    assert recorded["ordinal_random_forest"]["median_label_mismatches_DEFAULT_parser"] == 3
    assert sum(c["median_label_mismatches_DEFAULT_parser"] for c in recorded.values()) == 4
