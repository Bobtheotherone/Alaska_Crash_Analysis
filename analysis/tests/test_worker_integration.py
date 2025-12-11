# analysis/tests/test_worker_integration.py

from django.test import TestCase
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
import numpy as np
import pandas as pd

from ingestion.models import UploadedDataset
from analysis.models import ModelJob
from analysis.ml_core.worker import run_model_job


class WorkerIntegrationTests(TestCase):
    def _make_csv_bytes(self) -> bytes:
        """
        Generate a small-but-realistic numeric dataset that will survive
        the cleaning + leakage detection pipeline.

        We avoid tiny, perfectly correlated data so that
        find_leakage_columns() does not drop all features.
        """
        rng = np.random.RandomState(123)
        n = 60

        severities = np.array(
            ["Minor injury", "Serious injury", "Fatal crash"]
        )

        df = pd.DataFrame(
            {
                "Severity": rng.choice(severities, size=n),
                # Numeric features with enough noise that no single
                # column is an almost-perfect predictor of Severity.
                "Speed_Limit": rng.choice([25, 35, 45, 55], size=n),
                "Num_Vehicles": rng.randint(1, 5, size=n),
                "AADT": rng.randint(100, 5000, size=n),
            }
        )

        return df.to_csv(index=False).encode("utf-8")

    def test_run_model_job_success_path(self):
        User = get_user_model()
        user = User.objects.create_user(username="test-user", password="x")

        csv_bytes = self._make_csv_bytes()

        upload = UploadedDataset.objects.create(
            owner=user,
            original_filename="test.csv",
            size_bytes=len(csv_bytes),
            mime_type="text/csv",
            status=UploadedDataset.Status.ACCEPTED,
        )
        upload.raw_file.save("test.csv", ContentFile(csv_bytes), save=True)

        job = ModelJob.objects.create(
            upload=upload,
            owner=user,
            model_name="crash_severity_risk_v1",
            status=ModelJob.Status.QUEUED,
            parameters={
                "cleaning": {"severity_col": "Severity"},
                "model_params": {},
            },
        )

        # Run the worker synchronously
        run_model_job(job.id)

        job.refresh_from_db()
        self.assertEqual(job.status, ModelJob.Status.SUCCEEDED)
        self.assertIsNotNone(job.result_metadata)

        meta = job.result_metadata
        self.assertIn("metrics", meta)
        self.assertIn("feature_importances", meta)
        self.assertIn("leakage_warnings", meta)
        self.assertIn("cleaning_meta", meta)
