from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from django.test import SimpleTestCase

from ingestion import antivirus


class AntivirusScanTests(SimpleTestCase):
    def _fake_module(self, instream_result=None, instream_exception=None, ping_exception=None):
        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def ping(self):
                if ping_exception:
                    raise ping_exception
                return "PONG"

            def instream(self, buff):
                # consume buffer to mirror real client
                buff.read()
                if instream_exception:
                    raise instream_exception
                return instream_result or {"stream": ("OK", None)}

        return SimpleNamespace(
            ClamdUnixSocket=FakeClient,
            ClamdNetworkSocket=FakeClient,
        )

    def test_av_scan_passes_when_no_threats_found(self):
        fake_clamd = self._fake_module(instream_result={"stream": ("OK", None)})
        with mock.patch.object(antivirus, "clamd", fake_clamd):
            result = antivirus.scan_bytes_with_clamav(b"data")

        self.assertEqual(result["status"], "passed")
        self.assertIn("ClamAV did not find", result["details"])

    def test_av_scan_marks_failure_when_threat_found(self):
        fake_clamd = self._fake_module(
            instream_result={"stream": ("FOUND", "Eicar-Test-Signature")}
        )
        with mock.patch.object(antivirus, "clamd", fake_clamd):
            result = antivirus.scan_bytes_with_clamav(b"data")

        self.assertEqual(result["status"], "failed")
        self.assertIn("Eicar-Test-Signature", result["details"])

    def test_av_scan_handles_attribute_error_gracefully(self):
        fake_clamd = self._fake_module(instream_exception=AttributeError("clamd_socket"))
        with mock.patch.object(antivirus, "clamd", fake_clamd):
            result = antivirus.scan_bytes_with_clamav(b"data")

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(
            result["details"],
            "ClamAV client is misconfigured; antivirus scanning was skipped.",
        )
        self.assertNotIn("clamd_socket", result["details"])

    def test_av_scan_skipped_when_python_clamd_missing(self):
        with mock.patch.object(antivirus, "clamd", None):
            result = antivirus.scan_bytes_with_clamav(b"data")

        self.assertEqual(result["status"], "skipped")
        self.assertIn("python-clamd is not installed", result["details"])

    def test_av_scan_skipped_when_daemon_unreachable(self):
        fake_clamd = self._fake_module(ping_exception=ConnectionError("refused"))
        with mock.patch.object(antivirus, "clamd", fake_clamd):
            with self.settings(CLAMAV_TCP_HOST="clamav", CLAMAV_TCP_PORT=3310):
                result = antivirus.scan_bytes_with_clamav(b"data")

        self.assertEqual(result["status"], "skipped")
        self.assertIn("ClamAV daemon is unreachable", result["details"])
        self.assertIn("clamav:3310", result["details"])

    def test_av_scan_skipped_on_unexpected_response(self):
        fake_clamd = self._fake_module(instream_result={"stream": ("WEIRD", None)})
        with mock.patch.object(antivirus, "clamd", fake_clamd):
            result = antivirus.scan_bytes_with_clamav(b"data")

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(
            result["details"],
            "Unexpected response from ClamAV; antivirus scanning was skipped.",
        )
