# Objective: Test coverage for adaptive request admission and overload-aware rate limiting.
"""Unit tests for the adaptive request-limiter middleware."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.responses import Response


def _build_request(
    path: str,
    *,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    body: bytes = b"",
    query_string: bytes = b"",
    client_host: str = "127.0.0.1",
) -> Request:
    """Create a lightweight Starlette request for middleware unit tests."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": query_string,
        "headers": [(k.lower().encode("utf-8"), v.encode("utf-8")) for k, v in (headers or {}).items()],
        "client": (client_host, 1234),
        "server": ("testserver", 80),
    }

    sent = False

    async def _receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive=_receive)


@pytest.mark.asyncio
async def test_resolve_identity_falls_back_to_ip_without_header_or_query(monkeypatch):
    """The middleware must not parse JSON bodies to resolve tenant identity."""
    from app.middleware import rate_limit as rl

    middleware = rl.RateLimitMiddleware(app=lambda scope, receive, send: None)
    request = _build_request(
        "/query",
        headers={"content-type": "application/json"},
        body=b'{"query":"hello","tenant_id":"tenant-42"}',
        client_host="10.0.0.8",
    )

    identity, identity_type = await middleware._resolve_identity(request)

    assert identity == "10.0.0.8"
    assert identity_type == "ip"


@pytest.mark.asyncio
async def test_normal_pressure_does_not_limit_interactive_queries(monkeypatch):
    """Interactive traffic should not receive 429 while Ollama pressure is normal."""
    from app.middleware import rate_limit as rl

    rl.rate_limit_store = rl.RateLimitStore()
    rl._pressure_tracker = rl.PressureStateTracker()
    monkeypatch.setattr(
        rl,
        "get_ollama_admission_snapshot",
        lambda: {
            "current_limit": 2,
            "total_inflight": 0,
            "max_queue_wait_ms": 0.0,
            "utilization": 0.0,
            "vram_ratio": 0.0,
            "pressure_state": "normal",
        },
    )
    monkeypatch.setattr(rl, "get_backpressure", lambda: SimpleNamespace(get_stats=lambda: {"utilization": 0.0}))

    middleware = rl.RateLimitMiddleware(app=lambda scope, receive, send: None)

    async def _call_next(_request):
        return Response("ok", status_code=200)

    for _ in range(5):
        response = await middleware.dispatch(_build_request("/query"), _call_next)
        assert response.status_code == 200
        assert response.headers["X-Admission-State"] == "normal"


@pytest.mark.asyncio
async def test_congested_pressure_marks_interactive_queries_for_async_queue(monkeypatch):
    """Under congestion interactive queries should be preempted to the async queue before timing out."""
    from app.middleware import rate_limit as rl

    rl.rate_limit_store = rl.RateLimitStore()
    rl.rate_limit_store._use_redis = False
    rl._pressure_tracker = rl.PressureStateTracker()

    settings_map = {
        "ADAPTIVE_LIMITER_ENABLED": "1",
        "ADAPTIVE_LIMITER_HYSTERESIS_WINDOWS": "1",
        "ADAPTIVE_LIMITER_WINDOW_SECONDS": "15",
        "ADAPTIVE_LIMITER_INTERACTIVE_PER_SLOT_CONGESTED": "1",
        "ADAPTIVE_LIMITER_INTERACTIVE_PER_SLOT_ELEVATED": "2",
    }
    monkeypatch.setattr(rl.settings, "get", lambda key, default=None: settings_map.get(key, default))
    monkeypatch.setattr(
        rl,
        "get_ollama_admission_snapshot",
        lambda: {
            "current_limit": 1,
            "total_inflight": 1,
            "max_queue_wait_ms": 2000.0,
            "utilization": 1.0,
            "vram_ratio": 0.9,
            "pressure_state": "congested",
        },
    )
    monkeypatch.setattr(rl, "get_backpressure", lambda: SimpleNamespace(get_stats=lambda: {"utilization": 0.0}))

    middleware = rl.RateLimitMiddleware(app=lambda scope, receive, send: None)

    async def _call_next(_request):
        return Response("ok", status_code=200)

    request = _build_request(
        "/query",
        headers={"content-type": "application/json"},
        body=b'{"query":"hello","tenant_id":"school-1"}',
    )
    response = await middleware.dispatch(request, _call_next)

    assert response.status_code == 200
    assert getattr(request.state, "defer_to_query_job", False) is True
    assert response.headers["X-Admission-State"] == "congested"
    assert response.headers["X-RateLimit-Scope"] == "interactive_query"
    assert response.headers["X-RateLimit-Reason"] == "ollama_queue_wait"


@pytest.mark.asyncio
async def test_congested_pressure_still_rejects_admin_routes(monkeypatch):
    """Non-interactive routes should still receive 429 under adaptive-limiter overload."""
    from app.middleware import rate_limit as rl

    rl.rate_limit_store = rl.RateLimitStore()
    rl.rate_limit_store._use_redis = False
    rl._pressure_tracker = rl.PressureStateTracker()

    settings_map = {
        "ADAPTIVE_LIMITER_ENABLED": "1",
        "ADAPTIVE_LIMITER_HYSTERESIS_WINDOWS": "1",
        "ADAPTIVE_LIMITER_WINDOW_SECONDS": "15",
        "ADAPTIVE_LIMITER_ADMIN_PER_SLOT_CONGESTED": "1",
    }
    monkeypatch.setattr(rl.settings, "get", lambda key, default=None: settings_map.get(key, default))
    monkeypatch.setattr(
        rl,
        "get_ollama_admission_snapshot",
        lambda: {
            "current_limit": 1,
            "total_inflight": 1,
            "max_queue_wait_ms": 2000.0,
            "utilization": 1.0,
            "vram_ratio": 0.9,
            "pressure_state": "congested",
        },
    )
    monkeypatch.setattr(rl, "get_backpressure", lambda: SimpleNamespace(get_stats=lambda: {"utilization": 0.0}))

    middleware = rl.RateLimitMiddleware(app=lambda scope, receive, send: None)

    async def _call_next(_request):
        return Response("ok", status_code=200)

    first = await middleware.dispatch(
        _build_request("/admin/settings", headers={"x-tenant-id": "tenant-a"}, method="GET"),
        _call_next,
    )
    second = await middleware.dispatch(
        _build_request(
            "/admin/settings",
            headers={"x-tenant-id": "tenant-a"},
            method="GET",
        ),
        _call_next,
    )
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["X-Admission-State"] == "congested"
    assert second.headers["X-RateLimit-Scope"] == "admin_eval_governance"
    assert second.headers["X-RateLimit-Reason"] == "ollama_overloaded"


@pytest.mark.asyncio
async def test_admin_routes_are_limited_before_interactive_queries(monkeypatch):
    """Lower-priority admin traffic should receive a stricter quota than /query."""
    from app.middleware import rate_limit as rl

    rl.rate_limit_store = rl.RateLimitStore()
    rl.rate_limit_store._use_redis = False
    rl._pressure_tracker = rl.PressureStateTracker()

    settings_map = {
        "ADAPTIVE_LIMITER_ENABLED": "1",
        "ADAPTIVE_LIMITER_HYSTERESIS_WINDOWS": "1",
        "ADAPTIVE_LIMITER_WINDOW_SECONDS": "15",
        "ADAPTIVE_LIMITER_INTERACTIVE_PER_SLOT_ELEVATED": "2",
        "ADAPTIVE_LIMITER_ADMIN_PER_SLOT_ELEVATED": "1",
    }
    monkeypatch.setattr(rl.settings, "get", lambda key, default=None: settings_map.get(key, default))
    monkeypatch.setattr(
        rl,
        "get_ollama_admission_snapshot",
        lambda: {
            "current_limit": 1,
            "total_inflight": 1,
            "max_queue_wait_ms": 600.0,
            "utilization": 0.9,
            "vram_ratio": 0.6,
            "pressure_state": "elevated",
        },
    )
    monkeypatch.setattr(rl, "get_backpressure", lambda: SimpleNamespace(get_stats=lambda: {"utilization": 0.0}))

    middleware = rl.RateLimitMiddleware(app=lambda scope, receive, send: None)

    async def _call_next(_request):
        return Response("ok", status_code=200)

    admin_first = await middleware.dispatch(
        _build_request("/admin/settings", headers={"x-tenant-id": "tenant-a"}, method="GET"),
        _call_next,
    )
    admin_second = await middleware.dispatch(
        _build_request("/admin/settings", headers={"x-tenant-id": "tenant-a"}, method="GET"),
        _call_next,
    )
    interactive_first = await middleware.dispatch(
        _build_request("/query", headers={"content-type": "application/json"}, body=b'{"query":"q","tenant_id":"tenant-a"}'),
        _call_next,
    )
    interactive_second = await middleware.dispatch(
        _build_request("/query", headers={"content-type": "application/json"}, body=b'{"query":"q2","tenant_id":"tenant-a"}'),
        _call_next,
    )

    assert admin_first.status_code == 200
    assert admin_second.status_code == 429
    assert interactive_first.status_code == 200
    assert interactive_second.status_code == 200
