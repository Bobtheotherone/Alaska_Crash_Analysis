"""Gate-0 contract tests (AA2-003/004): impostor rejection, undocumented target values,
and fail-closed aborts on malformed/incomplete/ambiguous/unevidenced target mappings."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crashsev import contracts as K


@pytest.fixture(scope="module")
def artifacts():
    return K.load_schema(), K.load_target_mapping(), K.load_ledger()


def test_all_contract_files_load_and_hash(artifacts):
    schema, mapping, ledger = artifacts
    h = K.hash_contract_inputs(schema, mapping, ledger)
    assert set(h) == {"schema_sha256", "target_mapping_sha256", "feature_ledger_sha256"}
    assert all(len(v) == 64 for v in h.values())
    assert mapping.blank_maps_to == "O"           # verified source override
    assert len(ledger.allowed()) == 66 and len(ledger.prohibited()) == 14


def test_three_column_impostor_is_rejected(artifacts):
    schema, mapping, _ = artifacts
    imp = pd.DataFrame({"Crash Number": [1, 2], "Year": [2011, 2012],
                        "Crash Severity": ["Fatal", "Possible"], "Bogus": [9, 9]})
    res = K.validate_dataframe(imp, schema, mapping)
    assert res.ok is False
    assert any("required column" in e for e in res.errors)
    with pytest.raises(K.ContractViolation):
        K.assert_valid(imp, schema, mapping)


def test_undocumented_target_value_is_rejected(artifacts):
    schema, mapping, _ = artifacts
    df = pd.DataFrame({c: ["x"] for c in schema.required_columns})
    df["Crash Number"] = [1]; df["Year"] = [2011]; df["Crash Severity"] = ["Catastrophic"]
    res = K.validate_dataframe(df, schema, mapping)
    assert res.ok is False
    assert any("undocumented" in e for e in res.errors)


def test_valid_frame_is_accepted(artifacts):
    schema, mapping, _ = artifacts
    df = pd.DataFrame({c: (["Null value"] * 4) for c in schema.required_columns})
    df["Crash Number"] = [1, 2, 3, 4]; df["Year"] = [2009, 2010, 2011, 2012]
    df["Crash Severity"] = ["Fatal", "Possible", "Non-Incapacitating", np.nan]  # nan = blank = O
    res = K.assert_valid(df, schema, mapping)
    assert res.ok is True


MALFORMED = {
    "blank_override_without_evidence":
        "target_column: x\ntext_to_kabco: {Fatal: K}\nblank_maps_to: O\n"
        "kabco_to_ordinal: {O: 0, C: 1, B: 1, A: 2, K: 2}\n",
    "non_kabco_letter":
        "target_column: x\ntext_to_kabco: {Fatal: Z}\n"
        "kabco_to_ordinal: {O: 0, C: 1, B: 1, A: 2, K: 2}\n",
    "incomplete_ordinal_map":
        "target_column: x\ntext_to_kabco: {Fatal: K}\nkabco_to_ordinal: {O: 0, C: 1}\n",
    "ambiguous_mappable_and_quarantined":
        "target_column: x\ntext_to_kabco: {Unknown: K}\nquarantine_labels: [Unknown]\n"
        "kabco_to_ordinal: {O: 0, C: 1, B: 1, A: 2, K: 2}\n",
    "noncontiguous_classes":
        "target_column: x\ntext_to_kabco: {Fatal: K}\n"
        "kabco_to_ordinal: {O: 0, C: 2, B: 2, A: 5, K: 5}\n",
}


@pytest.mark.parametrize("name", list(MALFORMED))
def test_malformed_mapping_aborts(tmp_path, name):
    p = tmp_path / "m.yml"
    p.write_text(MALFORMED[name], encoding="utf-8")
    with pytest.raises(K.ContractViolation):
        K.load_target_mapping(p)


def test_real_mapping_still_loads():
    m = K.load_target_mapping()
    assert m.kabco_to_ordinal == {"O": 0, "C": 1, "B": 1, "A": 2, "K": 2}
