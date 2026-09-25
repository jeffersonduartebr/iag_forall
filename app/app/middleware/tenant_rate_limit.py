# Objective: Steady-state per-tenant rate limiting for production fairness.
"""Always-on tenant rate limits independent of Ollama pressure."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.middleware.trusted_identity import client_ip, verified_identity
from app.settings_dynamic import settings
from app.utils.redis_distributed import redis_sliding_window_limit

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return str(settings.get("TENANT_RATE_LIMIT_ENABLED", "1")).strip().lower() in {"1", "true", "yes", "on"}


def _rpm() -> int:
    try:
        return max(1, int(settings.get("TENANT_RATE_LIMIT_RPM", 120)))
    except Exception:
        return 120


def _window_s() -> int:
    try:
        return max(10, int(settings.get("TENANT_RATE_LIMIT_WINDOW_S", 60)))
    except Exception:
        return 60


class TenantRateLimitMiddleware(BaseHTTPMiddleware):
    """Apply per-tenant request quotas on query endpoints."""

    QUERY_PATHS = frozenset({"/query", "/query/stream", "/v1/query", "/v1/chat/completions"})

    async def dispatch(self, request: Request, call_next):
        if not _enabled() or request.url.path not in self.QUERY_PATHS:
            return await call_next(request)

        tenant_id = await self._resolve_tenant(request)
        identity = tenant_id or client_ip(request)
        scope = f"tenant:{identity}"
        limited = await redis_sliding_window_limit(
            scope,
            max_requests=_rpm(),
            window_seconds=_window_s(),
        )
        if limited:
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "category": "tenant_rate_limit",
                    "message": "Limite de requisições por tenant excedido.",
                    "tenant_id": tenant_id,
                    "limit_rpm": _rpm(),
                    "window_s": _window_s(),
                },
                headers={
                    "X-RateLimit-Limit": str(_rpm()),
                    "X-RateLimit-Window": str(_window_s()),
                    "Retry-After": str(_window_s()),
                },
            )

        response = await call_next(request)
        response.headers["X-Tenant-RateLimit-Limit"] = str(_rpm())
        return response

    async def _resolve_tenant(self, request: Request) -> Optional[str]:
        """Tenant from verified auth only — ``X-Tenant-ID`` & co. are client-chosen and ignored."""
        return verified_identity(request)
