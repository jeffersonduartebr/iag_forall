# Objective: Rate-limit identity taken only from verified credentials, never from client-chosen headers.
"""Who a request is charged to, for the tenant and adaptive rate limiters.

Both limiters used to read the tenant from ``X-Tenant-ID`` / ``X-Tenant`` /
``X-School-ID`` (and the adaptive one from ``?tenant_id=``) whenever there was
no JWT. Those values are chosen by the client, so rotating them per request
opened a fresh bucket every time and the quota never applied.

A tenant is now trusted only when verified auth vouches for it: the
``tenant_id`` claim of a signed JWT, else the JWT subject, else the API key
that authenticated the call. Everyone else — including every caller when
``REQUIRE_API_AUTH=0`` — is bucketed by client IP, which the limiters already
did for anonymous traffic.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Request

from app.api.auth import resolve_auth
from app.settings_dynamic import settings
from app.utils.client_ip import parse_trusted_proxies, resolve_client_ip


def verified_identity(request: Request) -> Optional[str]:
    """Tenant (or principal) proven by a JWT or API key; ``None`` when unauthenticated."""
    try:
        ctx = resolve_auth(
            authorization=request.headers.get("authorization"),
            x_api_key=request.headers.get("x-api-key"),
        )
    except Exception:
        return None
    if not ctx.authenticated:
        return None
    if ctx.tenant_id:
        return str(ctx.tenant_id)[:128]
    key = ctx.owner_key
    return None if key == "anonymous" else key[:128]


def client_ip(request: Request) -> str:
    """Client IP, honoring ``X-Forwarded-For`` only behind ``TRUSTED_PROXY_IPS``."""
    return resolve_client_ip(
        request.client.host if request.client else None,
        request.headers.get("X-Forwarded-For"),
        parse_trusted_proxies(settings.get("TRUSTED_PROXY_IPS", "")),
    )
