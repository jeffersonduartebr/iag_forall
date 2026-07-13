# Objective: Test coverage for backpressure middleware behavior and regressions.
"""Unit tests for the global backpressure semaphore and middleware helpers."""

from __future__ import annotations

import pytest
from starlette.requests import Request
from starlette.responses import Response


def _request(path: str) -> Request:
    """Create a minimal request object for middleware tests."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive=_receive)


@pytest.mark.asyncio
async def test_backpressure_semaphore_respects_capacity(monkeypatch):
    """The semaphore should reject requests immediately once capacity is exhausted."""
    from app.middleware import backpressure as bp

    bp.BackpressureSemaphore._instance = None
    monkeypatch.setattr(bp.settings, "get", lambda key, default=None: 2 if key == "MAX_CONCURRENT_REQUESTS" else default)

    semaphore = bp.BackpressureSemaphore()
    assert await semaphore.acquire() is True
    assert await semaphore.acquire() is True
    assert await semaphore.acquire() is False

    await semaphore.release()
    assert semaphore.current_load == 1
    assert await semaphore.acquire() is True


@pytest.mark.asyncio
async def test_backpressure_release_tolerates_extra_release(monkeypatch):
    """Releasing more times than acquired should not crash or underflow counters."""
    from app.middleware import backpressure as bp

    bp.BackpressureSemaphore._instance = None
    monkeypatch.setattr(bp.settings, "get", lambda key, default=None: 1 if key == "MAX_CONCURRENT_REQUESTS" else default)

    semaphore = bp.BackpressureSemaphore()
    await semaphore.release()
    assert semaphore.current_load == 0


@pytest.mark.asyncio
async def test_backpressure_marks_query_for_async_queue_when_capacity_exhausted(monkeypatch):
    """Interactive query routes should be deferred instead of rejected when backpressure is full."""
    from app.middleware import backpressure as bp

    bp.BackpressureSemaphore._instance = None
    monkeypatch.setattr(bp.settings, "get", lambda key, default=None: 1 if key == "MAX_CONCURRENT_REQUESTS" else default)
    middleware = bp.BackpressureMiddleware(app=lambda scope, receive, send: None)
    semaphore = bp.get_backpressure()
    assert await semaphore.acquire() is True

    async def _call_next(request):
        return Response(status_code=200, headers={"X-Test": "ok"})

    request = _request("/query")
    response = await middleware.dispatch(request, _call_next)

    assert response.status_code == 200
    assert getattr(request.state, "defer_to_query_job", False) is True
    await semaphore.release()
