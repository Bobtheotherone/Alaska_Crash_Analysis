from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase
from django.utils import timezone
from rest_framework.test import APIRequestFactory, force_authenticate

from crashdata.views import crashes_within_bbox_view, heatmap_view
from crashdata.views import export_crashes_csv


class DummyQS:
    def __init__(self):
        self.filters = []
        self.only_fields = []

    def filter(self, **kwargs):
        new = DummyQS()
        new.filters = self.filters + [kwargs]
        new.only_fields = self.only_fields
        return new

    def select_related(self, *args, **kwargs):
        return self

    def only(self, *args, **kwargs):
        self.only_fields = list(args)
        return self

    def __getitem__(self, key):
        return self

    def __iter__(self):
        return iter([])

    def iterator(self):
        return iter([])

    def count(self):
        return 0


class DateFilterBehaviorTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.user = SimpleNamespace(is_authenticated=True)

    @patch("crashdata.views.queries.crashes_within_bbox")
    def test_bbox_missing_dates_not_filtered(self, mock_query):
        mock_query.return_value = DummyQS()
        request = self.factory.get(
            "/api/crashdata/crashes-within-bbox/",
            {"min_lon": "0", "min_lat": "0", "max_lon": "1", "max_lat": "1"},
        )
        force_authenticate(request, user=self.user)

        response = crashes_within_bbox_view(request)
        self.assertEqual(response.status_code, 200)

        kwargs = mock_query.call_args.kwargs
        self.assertIsNone(kwargs.get("start_datetime"))
        self.assertIsNone(kwargs.get("end_datetime"))

    @patch("crashdata.views.queries.crashes_within_bbox")
    def test_heatmap_missing_dates_not_filtered(self, mock_query):
        mock_query.return_value = DummyQS()
        request = self.factory.get(
            "/api/crashdata/heatmap/",
            {"min_lon": "0", "min_lat": "0", "max_lon": "1", "max_lat": "1"},
        )
        force_authenticate(request, user=self.user)

        response = heatmap_view(request)
        self.assertEqual(response.status_code, 200)

        kwargs = mock_query.call_args.kwargs
        self.assertIsNone(kwargs.get("start_datetime"))
        self.assertIsNone(kwargs.get("end_datetime"))

    def test_invalid_start_datetime_returns_400(self):
        request = self.factory.get(
            "/api/crashdata/heatmap/",
            {
                "min_lon": "0",
                "min_lat": "0",
                "max_lon": "1",
                "max_lat": "1",
                "start_datetime": "not-a-date",
            },
        )
        force_authenticate(request, user=self.user)

        response = heatmap_view(request)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"start_datetime", response.content)
        self.assertIn(b"InvalidDatetime", response.content)

    def test_export_missing_dates(self):
        with patch("crashdata.views._get_dataset_for_user") as mock_get_dataset:
            dataset = SimpleNamespace(id="abc")
            mock_get_dataset.return_value = dataset
            with patch("crashdata.views.CrashRecord.objects.filter") as mock_filter:
                qs = DummyQS()
                qs.count = lambda: 0
                qs.iterator = lambda: iter([])
                mock_filter.return_value = qs
                request = self.factory.get(
                    "/api/crashdata/exports/crashes.csv", {"upload_id": "abc"}
                )
                force_authenticate(request, user=self.user)
                response = export_crashes_csv(request)
                self.assertEqual(response.status_code, 200)

    @patch("crashdata.views.queries.crashes_within_bbox")
    def test_bbox_datetime_local_accepted_and_aware(self, mock_query):
        mock_query.return_value = DummyQS()
        request = self.factory.get(
            "/api/crashdata/crashes-within-bbox/",
            {
                "min_lon": "-1",
                "min_lat": "0",
                "max_lon": "1",
                "max_lat": "1",
                "start_datetime": "2013-01-01T00:00",
                "end_datetime": "2017-12-31T23:50",
            },
        )
        force_authenticate(request, user=self.user)

        response = crashes_within_bbox_view(request)
        self.assertEqual(response.status_code, 200)

        kwargs = mock_query.call_args.kwargs
        self.assertTrue(timezone.is_aware(kwargs["start_datetime"]))
        self.assertTrue(timezone.is_aware(kwargs["end_datetime"]))
