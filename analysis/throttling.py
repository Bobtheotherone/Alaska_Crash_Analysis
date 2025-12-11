# analysis/throttling.py

from rest_framework.throttling import UserRateThrottle


class BurstRateThrottle(UserRateThrottle):
    """
    Simple per-user 'burst' throttle for the analysis API endpoints.

    The throttle rate is controlled by the DRF setting:
        REST_FRAMEWORK['DEFAULT_THROTTLE_RATES']['burst']

    Example in settings.py:
        REST_FRAMEWORK = {
            # ...
            "DEFAULT_THROTTLE_RATES": {
                "burst": "20/min",
            },
        }

    If no rate is configured for 'burst', throttling is effectively disabled.
    """

    scope = "burst"
