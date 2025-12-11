"""
Bridge module that re-exports Peyton's global cleaning configuration.

The canonical values live in ``peyton_original.DataCleaning.config``;
this module simply provides a stable import path for the rest of the
Django codebase.
"""

from __future__ import annotations

from peyton_original.DataCleaning import config as peyton_config

UNKNOWN_THRESHOLD: float = peyton_config.UNKNOWN_THRESHOLD
YES_NO_THRESHOLD: float = peyton_config.YES_NO_THRESHOLD
UNKNOWN_STRINGS = peyton_config.UNKNOWN_STRINGS

__all__ = ["UNKNOWN_THRESHOLD", "YES_NO_THRESHOLD", "UNKNOWN_STRINGS", "peyton_config"]
