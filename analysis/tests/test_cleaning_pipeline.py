# analysis/tests/test_cleaning_pipeline.py

from django.test import SimpleTestCase
import pandas as pd

from analysis.ml_core.cleaning import (
    UNKNOWN_THRESHOLD,
    YES_NO_THRESHOLD,
    DEFAULT_UNKNOWN_STRINGS,
    validate_config_values,
    build_ml_ready_dataset,
    map_numeric_severity,
    map_text_severity,
)
from analysis.ml_core.models import _ensure_cleaning_params


class EnsureCleaningParamsTests(SimpleTestCase):
    def test_passthrough_of_known_keys(self):
        raw = {
            "severity_col": "Severity",
            "base_unknowns": ["unk"],
            "unknown_threshold": 12.5,
            "yes_no_threshold": 7.5,
            "columns_to_drop": ["foo", "bar"],
            "leakage_columns": ["baz"],
            "unexpected": "ignore",
        }

        resolved = _ensure_cleaning_params(raw)

        self.assertEqual(resolved["severity_col"], "Severity")
        self.assertEqual(resolved["base_unknowns"], ["unk"])
        self.assertEqual(resolved["unknown_threshold"], 12.5)
        self.assertEqual(resolved["yes_no_threshold"], 7.5)
        self.assertEqual(resolved["columns_to_drop"], ["foo", "bar"])
        self.assertEqual(resolved["leakage_columns"], ["baz"])
        self.assertNotIn("unexpected", resolved)


class CleaningConfigTests(SimpleTestCase):
    def test_validate_config_values_accepts_defaults(self):
        # Should not raise
        validate_config_values()

    def test_validate_config_values_rejects_bad_unknown_threshold(self):
        with self.assertRaises(ValueError):
            validate_config_values(unknown_threshold=-0.1)
        with self.assertRaises(ValueError):
            validate_config_values(unknown_threshold=100.1)

    def test_validate_config_values_rejects_bad_yes_no_threshold(self):
        with self.assertRaises(ValueError):
            validate_config_values(yes_no_threshold=-0.1)
        with self.assertRaises(ValueError):
            validate_config_values(yes_no_threshold=100.1)

    def test_default_thresholds_match_expectations(self):
        self.assertEqual(UNKNOWN_THRESHOLD, 10.0)
        self.assertEqual(YES_NO_THRESHOLD, 1.0)


class BuildMLReadyDatasetTests(SimpleTestCase):
    def _make_toy_df(self):
        return pd.DataFrame(
            {
                "Severity": [
                    "Minor injury",
                    "Serious injury",
                    "Fatal crash",
                    "Minor injury",
                    "Serious injury",
                ],
                "Speed_Limit": [25, 35, 45, 55, 65],
                "Num_Vehicles": [1, 2, 2, 3, 1],
                # Categorical column is fine here because we’re only testing
                # build_ml_ready_dataset, not sklearn.fit.
                "Weather": ["Clear", "Rain", "Snow", "unknown", "N/A"],
            }
        )

    def test_build_ml_ready_dataset_basic_contract(self):
        df = self._make_toy_df()
        X, y, meta = build_ml_ready_dataset(
            df,
            severity_col="Severity",
        )

        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X), 0)
        self.assertNotIn("Severity", X.columns)

        self.assertEqual(meta["severity_column"], "Severity")
        self.assertIn("severity_mapping", meta)
        self.assertIn("unknown_values", meta)
        self.assertIn("leakage_columns", meta)
        self.assertIn("n_rows_before_target_filter", meta)
        self.assertIn("n_rows_after_target_filter", meta)
        self.assertIn("n_features_before_leakage", meta)
        self.assertIn("n_features_after_leakage", meta)
        self.assertIn("cleaning_meta", meta)

        unknowns = set(meta["unknown_values"])
        self.assertTrue({"unknown", "unspecified"} <= unknowns)

    def test_default_unknown_strings_are_reasonable(self):
        core = {
            "unknown",
            "missing",
            "unspecified",
            "not specified",
            "not applicable",
            "n/a",
            "na",
            "null",
            "blank",
            "tbd",
            "tba",
            "to be determined",
            "refused",
            "prefer not to say",
            "no data",
            "no value",
        }
        self.assertEqual(core, set(DEFAULT_UNKNOWN_STRINGS))

    def test_build_ml_ready_dataset_respects_cleaning_knobs(self):
        df = pd.DataFrame(
            {
                "severity": ["a", "b", "b", "a", "a"],
                "drop_me": ["mystery", "mystery", "ok", "ok", "ok"],
                "leaker": [1, 2, 3, 4, 5],
                "keep_me": [10, 11, 12, 13, 14],
            }
        )

        X, y, meta = build_ml_ready_dataset(
            df,
            severity_col="severity",
            base_unknowns={"mystery"},
            unknown_threshold=40.0,
            yes_no_threshold=5.0,
            columns_to_drop=["drop_me"],
            leakage_columns=["leaker"],
        )

        self.assertEqual(len(X), len(y))
        self.assertNotIn("drop_me", X.columns)
        self.assertNotIn("leaker", X.columns)
        self.assertIn("keep_me", X.columns)
        self.assertIn("mystery", meta["cleaning_meta"]["unknown_values"])
        self.assertEqual(
            meta["cleaning_meta"]["cleaning_config"]["unknown_threshold"], 40.0
        )
        self.assertEqual(
            meta["cleaning_meta"]["cleaning_config"]["yes_no_threshold"], 5.0
        )
        self.assertIn("drop_me", meta["cleaning_meta"]["user_specified_drops"])
        self.assertIn("leaker", meta["leakage_columns"])


class SeverityMappingTests(SimpleTestCase):
    def test_map_numeric_severity_three_levels(self):
        mapping = map_numeric_severity([1, 2, 3])

        # Lowest value must be mapped to 0, highest to 2.
        self.assertEqual(mapping[1], 0)
        self.assertEqual(mapping[3], 2)

        # The middle value should fall between them in severity.
        self.assertIn(mapping[2], {0, 1, 2})
        self.assertLessEqual(mapping[1], mapping[2])
        self.assertLessEqual(mapping[2], mapping[3])

    def test_map_text_severity_keywords(self):
        mapping = map_text_severity(
            ["No injury", "Minor injury", "Serious injury", "Fatal crash"]
        )

        self.assertEqual(mapping["No injury"], 0)
        self.assertEqual(mapping["Minor injury"], 0)
        self.assertEqual(mapping["Serious injury"], 2)
        self.assertEqual(mapping["Fatal crash"], 2)
