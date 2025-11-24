import io
import logging
from typing import Dict

from django.conf import settings

try:
    import clamd  # type: ignore[import]
except Exception:  # noqa: BLE001
    clamd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def scan_bytes_with_clamav(data: bytes) -> Dict[str, str]:
    """
    Scan the provided bytes using a ClamAV daemon if one is available.

    Returns a dict with keys:
    - status: "passed", "failed", or "skipped"
    - details: human-readable string describing the outcome
    """
    if not data:
        return {"status": "passed", "details": "Empty payload; nothing to scan."}

    if clamd is None:
        return {
            "status": "skipped",
            "details": (
                "python-clamd is not installed; antivirus scanning was skipped. "
                "Install ClamAV and the clamd Python library to enable scanning."
            ),
        }

    try:
        unix_socket = getattr(settings, "CLAMAV_UNIX_SOCKET", None)
        tcp_host = getattr(settings, "CLAMAV_TCP_HOST", None)
        tcp_port = getattr(settings, "CLAMAV_TCP_PORT", 3310)

        if unix_socket:
            client = clamd.ClamdUnixSocket(path=unix_socket)
        elif tcp_host:
            client = clamd.ClamdNetworkSocket(host=tcp_host, port=tcp_port)
        else:
            # Fall back to default local Unix socket path
            client = clamd.ClamdUnixSocket()

        result = client.instream(io.BytesIO(data))
        # ClamAV returns {'stream': ('OK', None)} or ('FOUND', 'Malware-Name')
        stream_result = result.get("stream")
        if not stream_result:
            return {
                "status": "skipped",
                "details": f"Unexpected response from ClamAV: {result!r}",
            }

        outcome, signature = stream_result
        if outcome == "OK":
            return {
                "status": "passed",
                "details": "ClamAV did not find any malware in the uploaded file.",
            }
        if outcome == "FOUND":
            return {
                "status": "failed",
                "details": f"ClamAV detected malware: {signature}",
            }

        return {
            "status": "skipped",
            "details": f"Unexpected ClamAV outcome '{outcome}' with signature {signature!r}.",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("ClamAV scan failed: %s", exc)
        return {
            "status": "skipped",
            "details": f"ClamAV scan could not be completed: {exc}",
        }
