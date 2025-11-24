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
        # Guardrail so future edits can't silently drift
        self.assertEqual(UNKNOWN_THRESHOLD, 10.0)
        self.assertEqual(YES_NO_THRESHOLD, 1.0)


class BuildMLReadyDatasetTests(SimpleTestCase):
    def _make_toy_df(self):
        # Small but realistic-ish crash dataset
        return pd.DataFrame(
            {
                "Severity": ["Minor injury", "Serious injury", "Fatal crash", "Unknown", "unspecified"],
                "Speed_Limit": [25, 35, 45, 55, 65],
                "Weather": ["Clear", "Rain", "Snow", "unknown", "N/A"],
            }
        )

    def test_build_ml_ready_dataset_basic_contract(self):
        df = self._make_toy_df()
        X, y, meta = build_ml_ready_dataset(
            df,
            severity_col="Severity",
        )

        # Shapes consistent
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X), 0)

        # Severity column should not be in features
        self.assertNotIn("Severity", X.columns)

        # Meta structure is present
        self.assertEqual(meta["severity_column"], "Severity")
        self.assertIn("severity_mapping", meta)
        self.assertIn("unknown_values", meta)
        self.assertIn("leakage_columns", meta)
        self.assertIn("n_rows_before_target_filter", meta)
        self.assertIn("n_rows_after_target_filter", meta)
        self.assertIn("n_features_before_leakage", meta)
        self.assertIn("n_features_after_leakage", meta)
        self.assertIn("cleaning_meta", meta)

        # Unknown tokens should include things like "unknown" and "unspecified"
        unknowns = set(meta["unknown_values"])
        self.assertTrue({"unknown", "unspecified"} <= unknowns)

    def test_default_unknown_strings_are_reasonable(self):
        # This asserts we didn't accidentally drop core unknown tokens
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


class SeverityMappingTests(SimpleTestCase):
    def test_map_numeric_severity_three_levels(self):
        # For three numeric levels we expect a clean 0/1/2 mapping
        mapping = map_numeric_severity([1, 2, 3])
        self.assertEqual(mapping[1], 0)
        self.assertEqual(mapping[2], 1)
        self.assertEqual(mapping[3], 2)

    def test_map_text_severity_keywords(self):
        mapping = map_text_severity(
            ["No injury", "Minor injury", "Serious injury", "Fatal crash"]
        )

        # Low severity
        self.assertEqual(mapping["No injury"], 0)
        self.assertEqual(mapping["Minor injury"], 0)

        # High severity: "serious injury" is explicitly in high_keywords,
        # and "fatal" is obviously high severity.
        self.assertEqual(mapping["Serious injury"], 2)
        self.assertEqual(mapping["Fatal crash"], 2)
