"""Immutable-bundle and split-correctness tests (AA2-006/017)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crashsev import cli, splits as S


# ---- split correctness (AA2-006) ----

def _toy(years):
    df = pd.DataFrame({"Crash Number": list(range(10, 10 + len(years))), "Year": years})
    y = pd.Series([i % 3 for i in range(len(years))])
    return df, y


def test_missing_year_row_is_excluded_not_development():
    df, y = _toy([2009, 2010, 2011, 2012, np.nan])
    dev, test, man, asg = S.chronological_group_split(
        df, y, year_col="Year", group_col="Crash Number",
        final_test_years=[2012], row_id_col="Crash Number")
    part = dict(zip(asg["row_id"], asg["partition"]))
    assert part["14"] == "excluded"            # the AA2-006 regression guard
    assert man.n_excluded == 1 and man.reconciles is True
    assert man.excluded_reason_counts.get("null_year") == 1


def test_split_reconciles_and_hash_reproduces():
    df, y = _toy([2009, 2010, 2011, 2012, 2012, np.nan])
    dev, test, man, asg = S.chronological_group_split(
        df, y, year_col="Year", group_col="Crash Number",
        final_test_years=[2012], row_id_col="Crash Number")
    assert man.n_development + man.n_final_test + man.n_excluded == man.n_total
    assert S.assignment_hash(asg) == man.assignment_sha256


def test_duplicate_and_null_ids_abort():
    with pytest.raises(ValueError):
        S.chronological_group_split(
            pd.DataFrame({"Crash Number": [1, 1], "Year": [2009, 2012]}),
            pd.Series([0, 1]), year_col="Year", group_col="Crash Number",
            final_test_years=[2012], row_id_col="Crash Number")
    with pytest.raises(ValueError):
        S.chronological_group_split(
            pd.DataFrame({"Crash Number": [1, None], "Year": [2009, 2012]}),
            pd.Series([0, 1]), year_col="Year", group_col="Crash Number",
            final_test_years=[2012], row_id_col="Crash Number")


# ---- immutable bundle (AA2-017) ----

def test_bundle_is_atomic_hashed_and_refuses_overwrite(tmp_path):
    files = {"predictions_x.csv": "a,b\n1,2\n", "target_audit.json": "{}"}
    manifest = {"run_id": "final_deadbeef", "note": "t"}
    d = cli.write_bundle(tmp_path, "final_deadbeef", files, manifest)
    assert (d / "STATUS").read_text().strip() == "COMPLETE"
    m = json.loads((d / "manifest.json").read_text())
    # every artifact hashed
    import hashlib
    for fname, content in files.items():
        assert m["artifact_sha256"][fname] == hashlib.sha256(content.encode()).hexdigest()
    # refuses to overwrite an existing run id
    with pytest.raises(FileExistsError):
        cli.write_bundle(tmp_path, "final_deadbeef", files, manifest)


def test_clean_numeric_sentinels_neutralises_int32_min():
    from crashsev import contracts as K
    schema = K.load_schema()
    df = pd.DataFrame({c: [0] for c in schema.numeric_sentinels})
    df["AADT"] = [-2147483648]                 # the int32-min sentinel
    cleaned, flagged = cli.clean_numeric_sentinels(df, schema)
    assert pd.isna(cleaned["AADT"].iloc[0])
    assert flagged["AADT"] == 1


def test_bundle_files_bytes_match_manifest_hashes_exactly(tmp_path):
    """BUNDLE-CRLF-001 (v4): the bundle writer must write the EXACT bytes it hashes - no
    platform newline translation between the hashed content and the materialised file."""
    import hashlib as _hashlib
    import json as _json
    files = {"a.csv": "x,y\n1,2\n", "b.txt": "line1\nline2\n"}
    d = cli.write_bundle(tmp_path, "final_beef01", files, {})
    m = _json.loads((d / "manifest.json").read_text())
    for fname, content in files.items():
        on_disk = (d / fname).read_bytes()
        assert on_disk == content.encode("utf-8")
        assert _hashlib.sha256(on_disk).hexdigest() == m["artifact_sha256"][fname]
