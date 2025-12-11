# analysis/tests/test_model_training.py

from django.test import SimpleTestCase
import numpy as np
import pandas as pd

from analysis.ml_core import models as ml_models
from analysis.ml_core.models import MODEL_REGISTRY


class ModelTrainingTests(SimpleTestCase):
    def _make_toy_df(self, n: int = 80) -> pd.DataFrame:
        rng = np.random.RandomState(0)
        severities = np.array(["Minor injury", "Serious injury", "Fatal crash"])

        return pd.DataFrame(
            {
                "Severity": rng.choice(severities, size=n),
                # All remaining columns numeric so sklearn is happy
                "Speed_Limit": rng.choice([25, 35, 45, 55], size=n),
                "Num_Vehicles": rng.randint(1, 4, size=n),
                "AADT": rng.randint(100, 5000, size=n),
            }
        )

    def test_all_registered_models_train_on_toy_data(self):
        df = self._make_toy_df(n=80)

        for name, spec in MODEL_REGISTRY.items():
            with self.subTest(model=name):
                # Skip EBM if interpret isn't installed
                if name.startswith("ebm") and getattr(
                    ml_models, "ExplainableBoostingClassifier", None
                ) is None:
                    continue

                # Skip XGB if xgboost isn't installed
                if name.startswith("xgb") and getattr(
                    ml_models, "XGBClassifier", None
                ) is None:
                    continue

                result = spec.trainer(
                    df,
                    cleaning_params={"severity_col": "Severity"},
                    model_params={},
                )

                self.assertIn("model", result)
                self.assertIn("metrics", result)
                self.assertIn("feature_importances", result)
                self.assertIn("cleaning_meta", result)
                self.assertIn("leakage_warnings", result)

                metrics = result["metrics"]
                self.assertIn("train_accuracy", metrics)
                self.assertGreaterEqual(metrics["train_accuracy"], 0.0)
                self.assertLessEqual(metrics["train_accuracy"], 1.0)
