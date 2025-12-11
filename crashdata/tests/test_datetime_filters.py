from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase
from rest_framework.test import APIRequestFactory, force_authenticate

from crashdata.views import crashes_within_bbox_view, heatmap_view


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
