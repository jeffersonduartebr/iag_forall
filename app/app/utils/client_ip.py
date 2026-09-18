# -*- coding: utf-8 -*-
# Objective: Resolve the real client IP honoring only trusted proxies.
"""Client IP resolution shared by the rate-limit middleware and login throttling.

``X-Forwarded-For`` is set by the client unless a proxy we trust overwrites
it, so it is honored only when the direct peer is listed in
``TRUSTED_PROXY_IPS``. Otherwise anyone could rotate the header to dodge
per-IP limits.
"""

from __future__ import annotations

from typing import Iterable, Optional


def parse_trusted_proxies(raw: Optional[str]) -> set[str]:
    """Parse the comma-separated ``TRUSTED_PROXY_IPS`` setting."""
    return {ip.strip() for ip in (raw or "").split(",") if ip.strip()}


def resolve_client_ip(
    direct_ip: Optional[str],
    forwarded_for: Optional[str],
    trusted_proxies: Iterable[str],
) -> str:
    """First ``X-Forwarded-For`` hop when the peer is a trusted proxy, else the peer itself."""
    peer = (direct_ip or "").strip() or "unknown"
    if peer in set(trusted_proxies) and forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    return peer
