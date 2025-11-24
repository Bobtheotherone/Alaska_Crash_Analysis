from django.utils.deprecation import MiddlewareMixin


class SecurityHeadersMiddleware(MiddlewareMixin):
    """Attach a small set of OWASP-aligned security headers.

    These headers complement Django's SecurityMiddleware and are safe to run
    in both development and production. More restrictive CSP policies can be
    configured at the reverse proxy layer if needed.
    """

    def process_response(self, request, response):  # noqa: D401
        # Protect against MIME confusion attacks.
        response.setdefault("X-Content-Type-Options", "nosniff")
        # Clickjacking protection.
        response.setdefault("X-Frame-Options", "DENY")
        # Limit referrer leakage.
        response.setdefault("Referrer-Policy", "same-origin")
        # Cross-origin isolation for modern browsers.
        response.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        # Simple CSP tuned for a React single-page app served from the same origin.
        csp = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "font-src 'self' data:; "
            "object-src 'none'; "
            "frame-ancestors 'none'"
        )
        response.setdefault("Content-Security-Policy", csp)
        return response
