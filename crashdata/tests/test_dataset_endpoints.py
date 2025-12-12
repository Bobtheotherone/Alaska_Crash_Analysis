import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase
from django.urls import resolve
from rest_framework.test import APIRequestFactory, force_authenticate

from crashdata.importer import ImportError
from crashdata.views import dataset_stats_view, import_crash_records_view


class DatasetEndpointTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.user = SimpleNamespace(is_authenticated=True)

    @patch("crashdata.views.CrashRecord")
    @patch("crashdata.views._get_dataset_for_user")
    def test_dataset_stats_returns_counts(self, mock_get_dataset, mock_crashrecord):
        uuid_str = "11111111-1111-1111-1111-111111111111"
        dataset = SimpleNamespace(id=uuid_str)
        mock_get_dataset.return_value = dataset

        qs = SimpleNamespace()

        def aggregate(**kwargs):
            return {
                "crashrecord_count": 5,
                "mappable_count": 3,
                "min_dt": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "max_dt": datetime(2024, 2, 1, tzinfo=timezone.utc),
            }

        qs.aggregate = aggregate
        mock_crashrecord.objects.filter.return_value = qs

        request = self.factory.get(f"/api/crashdata/datasets/{uuid_str}/stats/")
        force_authenticate(request, user=self.user)
        response = dataset_stats_view(request, upload_id=uuid_str)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["crashrecord_count"], 5)
        self.assertEqual(data["mappable_count"], 3)
        self.assertEqual(data["upload_id"], uuid_str)
        self.assertIsNotNone(data["min_crash_datetime"])
        self.assertIsNotNone(data["max_crash_datetime"])

    def test_dataset_stats_url_resolves_for_uuid(self):
        uuid_str = "11111111-1111-1111-1111-111111111111"
        match = resolve(f"/api/crashdata/datasets/{uuid_str}/stats/")
        self.assertEqual(match.func, dataset_stats_view)

    @patch("crashdata.views.import_crash_records_for_dataset")
    @patch("crashdata.views._get_dataset_for_user")
    def test_import_endpoint_returns_counts(self, mock_get_dataset, mock_import):
        dataset = SimpleNamespace(id="abc")
        mock_get_dataset.return_value = dataset
        mock_import.return_value = (10, 8)

        request = self.factory.post("/api/crashdata/datasets/abc/import-crash-records/")
        force_authenticate(request, user=self.user)
        response = import_crash_records_view(request, upload_id="abc")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["imported"], 10)
        self.assertEqual(data["mappable"], 8)

    @patch("crashdata.views.import_crash_records_for_dataset")
    @patch("crashdata.views._get_dataset_for_user")
    def test_import_endpoint_handles_error(self, mock_get_dataset, mock_import):
        dataset = SimpleNamespace(id="abc")
        mock_get_dataset.return_value = dataset
        mock_import.side_effect = ImportError("oops")

        request = self.factory.post("/api/crashdata/datasets/abc/import-crash-records/")
        force_authenticate(request, user=self.user)
        response = import_crash_records_view(request, upload_id="abc")
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"oops", response.content)
