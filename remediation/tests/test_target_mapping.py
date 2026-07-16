"""Target-mapping tests (DATA-002, DATA-005): fail-closed, no silent coercion, reconciles."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crashsev import schema, target as T


def test_kabco_letters_map_to_expected_ordinal():
    raw = pd.Series(["O", "C", "B", "A", "K"])
    kabco, y, audit = T.map_severity(raw)
    assert list(y.astype("Int64")) == [0, 1, 1, 2, 2]
    assert audit.n_quarantined == 0
    assert audit.to_dict()["reconciles"] is True


def test_blank_and_unknown_never_coerced_to_zero():
    # The original importer returns "O" (class 0) for blank/unknown -> DATA-005.
    raw = pd.Series(["K", "", None, np.nan, "UNKNOWN", "N/A", "not applicable", "garbage"])
    kabco, y, audit = T.map_severity(raw)
    # Only the single "K" maps; everything else is quarantined (NA), never class 0.
    assert audit.n_mapped == 1
    assert audit.n_quarantined == 7
    assert int(y.dropna().iloc[0]) == 2
    assert y.isna().sum() == 7
    # No unmapped value silently became class 0.
    assert (y.dropna() == 0).sum() == 0


def test_textual_labels_map_by_meaning_not_sort_order():
    raw = pd.Series([
        "Fatal Injury (Killed)", "Suspected Serious Injury", "Suspected Minor Injury",
        "Possible Injury", "No Apparent Injury", "Property Damage Only",
    ])
    _, y, audit = T.map_severity(raw)
    assert list(y.astype("Int64")) == [2, 2, 1, 1, 0, 0]
    assert audit.n_quarantined == 0


def test_numeric_codes_fail_closed_without_codebook():
    # Bare numeric codes must NOT be guessed by sort order (the DATA-002 inversion defect).
    raw = pd.Series(["1", "2", "3", "4", "5"])
    _, y, audit = T.map_severity(raw)
    assert audit.n_mapped == 0          # nothing guessed
    assert audit.n_quarantined == 5

    # With a verified codebook (here severity descends as code ascends), mapping is correct
    # and NOT inverted: code 1 = K (fatal) -> class 2, code 5 = O -> class 0.
    codebook = {"1": "K", "2": "A", "3": "B", "4": "C", "5": "O"}
    _, y2, audit2 = T.map_severity(raw, numeric_code_map=codebook)
    assert list(y2.astype("Int64")) == [2, 2, 1, 1, 0]
    assert audit2.n_quarantined == 0


def test_reconciliation_invariant_always_holds():
    rng = np.random.default_rng(0)
    raw = pd.Series(rng.choice(["O", "A", "K", "", "weird", "B", "C"], size=500))
    _, y, audit = T.map_severity(raw)
    d = audit.to_dict()
    assert d["n_mapped"] + d["n_quarantined"] == d["n_total"] == 500
    assert d["reconciles"] is True
