from __future__ import annotations

from typing import Any, Dict

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase, APIClient
from unittest import mock

from ingestion.models import UploadedDataset
from ingestion import validation


User = get_user_model()


def make_csv_content(rows: list[list[Any]]) -> str:
    lines = []
    for row in rows:
        # join with commas, coercing to string
        line = ",".join("" if v is None else str(v) for v in row)
        lines.append(line)
    return "\n".join(lines)


class IngestionGatewayTests(APITestCase):
    def setUp(self) -> None:
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="user",
            password="password",
        )
        self.other_user = User.objects.create_user(
            username="other",
            password="password",
        )
        self.admin = User.objects.create_user(
            username="admin",
            password="password",
            is_staff=True,
        )

        # Stub out MIME sniffing so tests are stable regardless of python-magic.
        sniff_patcher = mock.patch(
            "ingestion.views.validation.sniff_mime_type",
            side_effect=self._sniff_mime_type_passthrough,
        )
        self.mock_sniff = sniff_patcher.start()
        self.addCleanup(sniff_patcher.stop)

        # Stub out AV so tests do not depend on a running ClamAV daemon.
        av_patcher = mock.patch(
            "ingestion.views.scan_bytes_with_clamav",
            return_value={"status": "passed", "details": "AV scan stubbed for tests."},
        )
        self.mock_av = av_patcher.start()
        self.addCleanup(av_patcher.stop)

    def _sniff_mime_type_passthrough(
        self,
        file_bytes: bytes,
        original_name: str,
        declared_mime: str | None,
    ) -> Dict[str, Any]:
        # Default behaviour: always "passed" with the declared MIME (if any).
        return {
            "status": "passed",
            "detected_mime_type": declared_mime or "text/csv",
            "details": "sniff_mime_type stubbed for tests.",
        }

    def _upload_csv_as(self, user: User, filename: str, rows: list[list[Any]]) -> str:
        """Helper to upload a small CSV file as a given user and return upload_id."""
        self.client.force_authenticate(user=user)
        csv_content = make_csv_content(rows)
        upload = SimpleUploadedFile(
            filename,
            csv_content.encode("utf-8"),
            content_type="text/csv",
        )
        response = self.client.post(
            reverse("ingest-upload"),
            {"file": upload},
            format="multipart",
        )
        self.assertEqual(
            response.status_code,
            status.HTTP_201_CREATED,
            msg=f"Expected 201 for upload, got {response.status_code}: {response.content!r}",
        )
        data = response.json()
        self.assertEqual(data["overall_status"], "accepted")
        return data["upload_id"]

    def test_upload_valid_csv_happy_path(self) -> None:
        """Uploading a valid CSV produces an accepted upload with the full report."""
        self.client.force_authenticate(user=self.user)
        csv_rows = [
            ["crash_id", "crash_date", "severity", "latitude", "longitude"],
            [1, "2024-01-01", "K", 60.0, -150.0],
        ]
        csv_content = make_csv_content(csv_rows)
        upload = SimpleUploadedFile(
            "valid.csv",
            csv_content.encode("utf-8"),
            content_type="text/csv",
        )
        response = self.client.post(
            reverse("ingest-upload"),
            {"file": upload},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        data = response.json()

        self.assertEqual(data["overall_status"], "accepted")
        self.assertIn("upload_id", data)
        self.assertEqual(data["schema_version"], validation.SCHEMA_VERSION)

        # Steps should include all expected pipeline stages.
        step_names = {s["step"] for s in data["steps"]}
        expected_steps = {
            "PAYLOAD",
            "EXTENSION_CHECK",
            "FILE_SIZE",
            "MIME_SNIFF",
            "AV_SCAN",
            "PARSE_TABLE",
            "SCHEMA_CHECK",
            "TYPE_AND_RANGE_CHECKS",
            "GEO_CHECKS",
        }
        self.assertEqual(step_names, expected_steps)

        # Each step should expose severity and is_hard_fail for the UI.
        for step in data["steps"]:
            self.assertIn("severity", step)
            self.assertIn(step["severity"], {"info", "warning", "error"})
            self.assertIn("is_hard_fail", step)

        # UploadedDataset row should be persisted with a matching report.
        upload_id = data["upload_id"]
        dataset = UploadedDataset.objects.get(id=upload_id)
        self.assertEqual(dataset.status, UploadedDataset.Status.ACCEPTED)
        self.assertIsNotNone(dataset.validation_report)
        self.assertEqual(dataset.validation_report["overall_status"], "accepted")

    def test_missing_file_hard_fail(self) -> None:
        """If no file is sent, the upload is rejected with PAYLOAD_MISSING_FILE."""
        self.client.force_authenticate(user=self.user)
        response = self.client.post(reverse("ingest-upload"), {}, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertEqual(data["overall_status"], "rejected")
        self.assertEqual(data["error_code"], "PAYLOAD_MISSING_FILE")

        step = next(s for s in data["steps"] if s["step"] == "PAYLOAD")
        self.assertEqual(step["status"], "failed")
        self.assertEqual(step["code"], "PAYLOAD_MISSING_FILE")
        self.assertEqual(step["severity"], "error")
        self.assertTrue(step["is_hard_fail"])

    def test_disallowed_extension_hard_fail(self) -> None:
        """A file with an extension not in INGESTION_ALLOWED_EXTENSIONS is rejected."""
        self.client.force_authenticate(user=self.user)
        bad_file = SimpleUploadedFile(
            "bad.txt",
            b"not,important\n",
            content_type="text/plain",
        )
        response = self.client.post(
            reverse("ingest-upload"),
            {"file": bad_file},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertEqual(data["overall_status"], "rejected")
        self.assertEqual(data["error_code"], "EXTENSION_NOT_ALLOWED")

        step = next(s for s in data["steps"] if s["step"] == "EXTENSION_CHECK")
        self.assertEqual(step["status"], "failed")
        self.assertEqual(step["code"], "EXTENSION_NOT_ALLOWED")
        self.assertEqual(step["severity"], "error")
        self.assertTrue(step["is_hard_fail"])

    @override_settings(INGESTION_MAX_FILE_SIZE_BYTES=10)
    def test_file_too_large_hard_fail(self) -> None:
        """Files larger than INGESTION_MAX_FILE_SIZE_BYTES are rejected."""
        self.client.force_authenticate(user=self.user)
        # Content longer than the 10-byte limit in the override_settings above.
        big_content = b"x" * 100
        upload = SimpleUploadedFile(
            "big.csv",
            big_content,
            content_type="text/csv",
        )
        response = self.client.post(
            reverse("ingest-upload"),
            {"file": upload},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertEqual(data["overall_status"], "rejected")
        self.assertEqual(data["error_code"], "FILE_TOO_LARGE")

        step = next(s for s in data["steps"] if s["step"] == "FILE_SIZE")
        self.assertEqual(step["status"], "failed")
        self.assertEqual(step["code"], "FILE_TOO_LARGE")
        self.assertEqual(step["severity"], "error")
        self.assertTrue(step["is_hard_fail"])

    def test_mime_mismatch_hard_fail(self) -> None:
        """A MIME sniff mismatch results in a MIME_MISMATCH error."""
        self.client.force_authenticate(user=self.user)

        def fake_sniff(file_bytes: bytes, original_name: str, declared_mime: str | None):
            return {
                "status": "failed",
                "detected_mime_type": "application/octet-stream",
                "details": "Declared type does not match detected MIME (test stub).",
            }

        self.mock_sniff.side_effect = fake_sniff

        csv_rows = [
            ["crash_id", "crash_date", "severity", "latitude", "longitude"],
            [1, "2024-01-01", "K", 60.0, -150.0],
        ]
        csv_content = make_csv_content(csv_rows)
        upload = SimpleUploadedFile(
            "mime.csv",
            csv_content.encode("utf-8"),
            content_type="text/csv",
        )
        response = self.client.post(
            reverse("ingest-upload"),
            {"file": upload},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertEqual(data["overall_status"], "rejected")
        self.assertEqual(data["error_code"], "MIME_MISMATCH")

        step = next(s for s in data["steps"] if s["step"] == "MIME_SNIFF")
        self.assertEqual(step["status"], "failed")
        self.assertEqual(step["code"], "MIME_MISMATCH")
        self.assertEqual(step["severity"], "error")
        self.assertTrue(step["is_hard_fail"])

    def test_av_detects_malware_hard_fail(self) -> None:
        """If AV marks the upload as infected, it is rejected with UPLOAD_INFECTED."""
        self.client.force_authenticate(user=self.user)
        self.mock_av.return_value = {
            "status": "failed",
            "details": "EICAR test signature detected (test stub).",
        }

        csv_rows = [
            ["crash_id", "crash_date", "severity", "latitude", "longitude"],
            [1, "2024-01-01", "K", 60.0, -150.0],
        ]
        csv_content = make_csv_content(csv_rows)
        upload = SimpleUploadedFile(
            "infected.csv",
            csv_content.encode("utf-8"),
            content_type="text/csv",
        )
        response = self.client.post(
            reverse("ingest-upload"),
            {"file": upload},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertEqual(data["overall_status"], "rejected")
        self.assertEqual(data["error_code"], "UPLOAD_INFECTED")

        step = next(s for s in data["steps"] if s["step"] == "AV_SCAN")
        self.assertEqual(step["status"], "failed")
        self.assertEqual(step["code"], "UPLOAD_INFECTED")
        self.assertEqual(step["severity"], "error")
        self.assertTrue(step["is_hard_fail"])

    @override_settings(INGESTION_REQUIRE_AV=True)
    def test_av_unavailable_when_required_hard_fail(self) -> None:
        """If AV is required but unavailable, uploads are rejected (AV_UNAVAILABLE_REQUIRED)."""
        self.client.force_authenticate(user=self.user)
        self.mock_av.return_value = {
            "status": "skipped",
            "details": "AV unavailable (test stub).",
        }

        csv_rows = [
            ["crash_id", "crash_date", "severity", "latitude", "longitude"],
            [1, "2024-01-01", "K", 60.0, -150.0],
        ]
        csv_content = make_csv_content(csv_rows)
        upload = SimpleUploadedFile(
            "noav.csv",
            csv_content.encode("utf-8"),
            content_type="text/csv",
        )
        response = self.client.post(
            reverse("ingest-upload"),
            {"file": upload},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertEqual(data["overall_status"], "rejected")
        self.assertEqual(data["error_code"], "AV_UNAVAILABLE_REQUIRED")

        step = next(s for s in data["steps"] if s["step"] == "AV_SCAN")
        self.assertEqual(step["status"], "skipped")
        self.assertEqual(step["code"], "AV_UNAVAILABLE_REQUIRED")
        self.assertEqual(step["severity"], "error")
        self.assertTrue(step["is_hard_fail"])

    def test_missing_required_schema_columns_hard_fail(self) -> None:
        """Missing required columns causes SCHEMA_MISSING_COLUMNS and a hard-fail."""
        self.client.force_authenticate(user=self.user)
        # A CSV with no MMUCC fields at all.
        csv_rows = [
            ["foo", "bar"],
            [1, 2],
        ]
        csv_content = make_csv_content(csv_rows)
        upload = SimpleUploadedFile(
            "schema.csv",
            csv_content.encode("utf-8"),
            content_type="text/csv",
        )
        response = self.client.post(
            reverse("ingest-upload"),
            {"file": upload},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertEqual(data["overall_status"], "rejected")
        self.assertEqual(data["error_code"], "SCHEMA_MISSING_COLUMNS")

        step = next(s for s in data["steps"] if s["step"] == "SCHEMA_CHECK")
        self.assertEqual(step["status"], "failed")
        self.assertEqual(step["code"], "SCHEMA_MISSING_COLUMNS")
        self.assertEqual(step["severity"], "error")
        self.assertTrue(step["is_hard_fail"])

    def test_soft_fail_type_and_geo_checks_still_accepted(self) -> None:
        """Type/range and geo issues increase invalid counts but do not reject upload."""
        self.client.force_authenticate(user=self.user)
        csv_rows = [
            ["crash_id", "crash_date", "severity", "latitude", "longitude", "driver_age"],
            [1, "2024-01-01", "K", 60.0, -150.0, 30],   # valid
            [2, "2024-01-02", "K", 80.0, -150.0, 200],  # bad lat + out-of-range age
        ]
        csv_content = make_csv_content(csv_rows)
        upload = SimpleUploadedFile(
            "softfail.csv",
            csv_content.encode("utf-8"),
            content_type="text/csv",
        )
        response = self.client.post(
            reverse("ingest-upload"),
            {"file": upload},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        data = response.json()
        self.assertEqual(data["overall_status"], "accepted")

        row_checks = data["row_checks"]
        self.assertEqual(row_checks["total_rows"], 2)
        self.assertGreater(row_checks["invalid_row_count"], 0)
        self.assertGreater(row_checks["invalid_geo_row_count"], 0)

        type_step = next(s for s in data["steps"] if s["step"] == "TYPE_AND_RANGE_CHECKS")
        geo_step = next(s for s in data["steps"] if s["step"] == "GEO_CHECKS")

        self.assertEqual(type_step["status"], "passed")
        self.assertEqual(type_step["severity"], "warning")
        self.assertFalse(type_step["is_hard_fail"])

        self.assertEqual(geo_step["status"], "passed")
        self.assertEqual(geo_step["severity"], "warning")
        self.assertFalse(geo_step["is_hard_fail"])

    def test_list_uploads_respects_ownership_and_admin(self) -> None:
        """GET /api/ingest/uploads/ returns only own uploads for normal users and all for admins."""
        # Create one upload for the primary user and one for another user.
        user_upload_id = self._upload_csv_as(
            self.user,
            "mine.csv",
            [["crash_id", "crash_date", "severity", "latitude", "longitude"],
             [1, "2024-01-01", "K", 60.0, -150.0]],
        )
        other_upload_id = self._upload_csv_as(
            self.other_user,
            "theirs.csv",
            [["crash_id", "crash_date", "severity", "latitude", "longitude"],
             [2, "2024-01-02", "K", 60.0, -151.0]],
        )

        # As a normal user, we should only see our own upload.
        self.client.force_authenticate(user=self.user)
        response = self.client.get(reverse("ingest-upload-list"))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["count"], 1)
        ids = {item["id"] for item in data["results"]}
        self.assertEqual(ids, {str(user_upload_id)})

        # As an admin, we should see both uploads.
        self.client.force_authenticate(user=self.admin)
        response = self.client.get(reverse("ingest-upload-list"))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["count"], 2)
        ids = {item["id"] for item in data["results"]}
        self.assertEqual(ids, {str(user_upload_id), str(other_upload_id)})

    def test_get_upload_status_permissions(self) -> None:
        """Only owners (and admins) can read a specific upload's status."""
        upload_id = self._upload_csv_as(
            self.user,
            "mine.csv",
            [["crash_id", "crash_date", "severity", "latitude", "longitude"],
             [1, "2024-01-01", "K", 60.0, -150.0]],
        )

        # Owner can fetch status.
        self.client.force_authenticate(user=self.user)
        url = reverse("ingest-upload-status", args=[upload_id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(str(data["upload_id"]), str(upload_id))

        # Non-owner cannot.
        self.client.force_authenticate(user=self.other_user)
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

        # Admin can.
        self.client.force_authenticate(user=self.admin)
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
