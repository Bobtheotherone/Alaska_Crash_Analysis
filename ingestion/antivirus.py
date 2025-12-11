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
    def _skip(details: str) -> Dict[str, str]:
        return {"status": "skipped", "details": details}

    if not data:
        return {"status": "passed", "details": "Empty payload; nothing to scan."}

    if clamd is None:
        return _skip("python-clamd is not installed; antivirus scanning was skipped.")

    try:
        unix_socket = getattr(settings, "CLAMAV_UNIX_SOCKET", None)
        tcp_host = getattr(settings, "CLAMAV_TCP_HOST", None)
        tcp_port = getattr(settings, "CLAMAV_TCP_PORT", 3310)

        if unix_socket:
            client = clamd.ClamdUnixSocket(path=unix_socket)
            target_desc = f"unix socket {unix_socket}"
        elif tcp_host:
            client = clamd.ClamdNetworkSocket(host=tcp_host, port=tcp_port)
            target_desc = f"{tcp_host}:{tcp_port}"
        else:
            # Fall back to default local Unix socket path
            client = clamd.ClamdUnixSocket()
            target_desc = "default ClamAV unix socket"

        try:
            client.ping()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ClamAV ping failed (%s): %s", target_desc, exc)
            detail = "ClamAV daemon is unreachable"
            if target_desc:
                detail += f" at {target_desc}"
            detail += "; antivirus scanning was skipped."
            return _skip(detail)

        result = client.instream(io.BytesIO(data))
        # ClamAV returns {'stream': ('OK', None)} or ('FOUND', 'Malware-Name')
        stream_result = result.get("stream")
        if (
            not stream_result
            or not isinstance(stream_result, (list, tuple))
            or len(stream_result) < 2
        ):
            logger.warning("Unexpected ClamAV response shape: %r", result)
            return _skip(
                "Unexpected response from ClamAV; antivirus scanning was skipped."
            )

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
            "details": "Unexpected response from ClamAV; antivirus scanning was skipped.",
        }
    except AttributeError as exc:
        logger.warning("ClamAV client attribute error during scan: %s", exc)
        return _skip(
            "ClamAV client is misconfigured; antivirus scanning was skipped."
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ClamAV scan failed: %s", exc, exc_info=True)
        return _skip(
            "Antivirus scan could not be completed; antivirus scanning was skipped."
        )
