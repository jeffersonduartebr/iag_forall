# Objective: Test coverage for admin log streaming endpoint.
"""Tests for admin SSE log stream."""

import json

import pytest
from app.api import admin_dashboard_routes as routes


@pytest.mark.asyncio
async def test_logs_stream_yields_sse_events(monkeypatch):
    """Log stream should emit SSE data lines."""

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
async def test_logs_stream_requires_auth():
    """Unauthorized log stream should fail.

    Pela app: a autenticação é agora uma dependência do router, e chamar o
    handler directamente já não a exercita.
    """
    from fastapi.testclient import TestClient

    from app import main

    with TestClient(main.app) as client:
        assert client.get("/admin/logs/stream").status_code in {401, 403}
