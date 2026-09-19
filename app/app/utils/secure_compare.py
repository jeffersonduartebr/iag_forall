# -*- coding: utf-8 -*-
# Objective: Constant-time comparison of secrets supplied by clients.
"""Timing-safe equality for tokens and passwords.

``secrets.compare_digest(str, str)`` raises ``TypeError`` when either string
has non-ASCII characters, so a header like ``x-api-key: ç`` turned into a 500
(and a non-ASCII admin password could never match). Comparing the UTF-8 bytes
keeps the constant-time guarantee for any input.
"""

from __future__ import annotations

import hmac


def secret_equals(provided: str, expected: str) -> bool:
    """Constant-time ``provided == expected`` over UTF-8 bytes; never raises for str input."""
    return hmac.compare_digest(str(provided).encode("utf-8"), str(expected).encode("utf-8"))
