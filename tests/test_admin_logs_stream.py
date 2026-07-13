# Objective: Test coverage for admin log streaming endpoint.
"""Tests for admin SSE log stream."""

import json

import pytest
from app.api import admin_dashboard_routes as routes


@pytest.mark.asyncio
async def test_logs_stream_yields_sse_events(monkeypatch):
    """Log stream should emit SSE data lines."""
    monkeypatch.setattr(routes, "resolve_admin_session", lambda **kwargs: {"username": "admin"})

    async def _fake_stream(*, query):
        yield {"event": "startup", "level": "info", "container": "api"}
        yield {"event": "ready", "level": "info", "container": "api"}

    monkeypatch.setattr(routes, "stream_logs", _fake_stream)

    response = await routes.logs_stream()
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

    body = "".join(chunks)
    assert "data:" in body
    assert "startup" in body
    assert "ready" in body

    # Validate JSON payload inside SSE
    for line in body.splitlines():
        if line.startswith("data:"):
            payload = json.loads(line.replace("data:", "").strip())
            assert "event" in payload


@pytest.mark.asyncio
async def test_logs_stream_requires_auth(monkeypatch):
    """Unauthorized log stream should fail."""
    from fastapi import HTTPException

    def _deny(**kwargs):
        raise HTTPException(status_code=401, detail="Não autorizado.")

    monkeypatch.setattr(routes, "resolve_admin_session", _deny)
    with pytest.raises(HTTPException) as exc:
        await routes.logs_stream()
    assert exc.value.status_code == 401
