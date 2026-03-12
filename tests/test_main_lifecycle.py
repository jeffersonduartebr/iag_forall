"""Tests for app lifecycle and operational routes after router extraction."""

from types import SimpleNamespace

import pytest
from fastapi import Response


@pytest.mark.asyncio
async def test_preload_ollama_models_download_flow(monkeypatch):
    """preload_ollama_models should exercise the download path without real IO."""
    from app import main

    monkeypatch.setenv("CANDIDATE_MODELS_LIST", '["ollama/phi4:latest"]')
    monkeypatch.setenv("JUDGE_MODELS", "[]")
    monkeypatch.setenv("OLLAMA_MODEL", "")

    monkeypatch.setattr(main, "VLM_OLLAMA_MODELS", [])
    monkeypatch.setattr(main.settings, "get", lambda k, d=None: "nomic-ai/nomic-embed-text-v1.5" if k == "EMBED_TEXT_MODEL" else d)

    class _Resp:
        def __init__(self, status_code=200, payload=None):
            self.status_code = status_code
            self._payload = payload or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("http error")

        def json(self):
            return self._payload

    class _StreamCtx:
        async def __aenter__(self):
            return _Resp(200, {})

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            if False:
                yield ""

    class _Client:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, _url):
            return _Resp(200, {"models": []})

        def stream(self, method, url, json):
            return _StreamCtx()

    monkeypatch.setattr(main.httpx, "AsyncClient", _Client)
    await main.preload_ollama_models()


@pytest.mark.asyncio
async def test_preload_ollama_models_skips_available_and_handles_lookup_error(monkeypatch):
    """preload_ollama_models should skip cached tags and tolerate tag lookup failures."""
    from app import main

    monkeypatch.setenv("CANDIDATE_MODELS_LIST", "ollama/phi4:latest")
    monkeypatch.setenv("JUDGE_MODELS", "[]")
    monkeypatch.setenv("OLLAMA_MODEL", "ollama/gemma3:4b")
    monkeypatch.setattr(main, "VLM_OLLAMA_MODELS", ["llava:7b"])
    monkeypatch.setattr(main.settings, "get", lambda k, d=None: "ollama/custom-embed" if k == "EMBED_TEXT_MODEL" else d)

    class _Resp:
        def raise_for_status(self):
            raise RuntimeError("lookup failed")

        def json(self):
            return {}

    class _StreamResp:
        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            if False:
                yield ""

    class _StreamCtx:
        async def __aenter__(self):
            return _StreamResp()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    calls = {"stream": 0}

    class _Client:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, _url):
            return _Resp()

        def stream(self, method, url, json):
            calls["stream"] += 1
            return _StreamCtx()

    monkeypatch.setattr(main.httpx, "AsyncClient", _Client)
    await main.preload_ollama_models()
    assert calls["stream"] >= 1


def test_metrics_endpoint_wraps_prometheus_payload(monkeypatch):
    """metrics should return the Prometheus payload with the reported content type."""
    from app import main

    monkeypatch.setattr(main, "render_metrics_response", lambda: ("metric 1\n", "text/plain"))
    resp = main.metrics()
    assert resp.body == b"metric 1\n"
    assert resp.media_type == "text/plain"


@pytest.mark.asyncio
async def test_startup_event_executes_warmup_and_shutdown(monkeypatch):
    """startup_event and shutdown_event should schedule warmup/cleanup hooks."""
    from app import main

    async def _cleanup():
        return None

    async def _preload():
        return None

    async def _add_doc(**kwargs):
        return True

    coros = []

    def _create_task(coro):
        coros.append(coro)
        return SimpleNamespace()

    monkeypatch.setattr(main, "settings", SimpleNamespace(ADMIN_TOKEN="abc", get=lambda k, d=None: d))
    monkeypatch.setattr(main.asyncio, "get_running_loop", lambda: SimpleNamespace(set_default_executor=lambda ex: None))
    monkeypatch.setattr(main, "ensure_runtime_support_tables", lambda: None)
    monkeypatch.setattr(main, "start_reload_listener", lambda: None)
    monkeypatch.setattr(main, "start_background_services", lambda: None)
    monkeypatch.setattr(main, "rate_limit_cleanup", _cleanup)
    monkeypatch.setattr(main.asyncio, "create_task", _create_task)
    monkeypatch.setattr(main, "get_redis", lambda: None)
    monkeypatch.setattr(main, "_ensure_model_metrics_table", lambda: None)
    monkeypatch.setattr(main, "init_vectorstore", lambda: None)
    monkeypatch.setattr(main, "preload_ollama_models", _preload)
    monkeypatch.setattr(main, "vs_add_document", _add_doc)
    monkeypatch.setenv("ENABLE_SMOKE_TESTS", "0")

    await main.startup_event()
    assert coros

    for coro in coros:
        await coro

    monkeypatch.setattr(main, "stop_background_services", lambda: None)
    monkeypatch.setattr(main, "stop_reload_listener", lambda: None)
    monkeypatch.setattr(main, "close_http_client", _preload)
    monkeypatch.setattr(main, "close_redis", lambda: None)
    monkeypatch.setattr(main, "close_engine", lambda: None)
    await main.shutdown_event()


@pytest.mark.asyncio
async def test_lifespan_runs_startup_and_shutdown(monkeypatch):
    """lifespan should bracket the app context with startup and shutdown hooks."""
    from app import main

    calls = []

    async def _startup():
        calls.append("startup")

    async def _shutdown():
        calls.append("shutdown")

    monkeypatch.setattr(main, "startup_event", _startup)
    monkeypatch.setattr(main, "shutdown_event", _shutdown)

    async with main.lifespan(main.app):
        calls.append("inside")

    assert calls == ["startup", "inside", "shutdown"]


@pytest.mark.asyncio
async def test_startup_event_rejects_invalid_settings(monkeypatch):
    """startup_event should fail fast when critical settings validation reports errors."""
    from app import main

    monkeypatch.setattr(main, "settings", SimpleNamespace(ADMIN_TOKEN="abc"))
    monkeypatch.setattr(main, "validate_critical_settings", lambda _settings: ["bad timeout"])

    with pytest.raises(RuntimeError) as exc:
        await main.startup_event()
    assert "Invalid critical settings" in str(exc.value)


@pytest.mark.asyncio
async def test_shutdown_event_tolerates_cleanup_failures(monkeypatch):
    """shutdown_event should swallow cleanup errors and keep shutting down."""
    from app import main

    async def _boom():
        raise RuntimeError("http")

    monkeypatch.setattr(main, "stop_background_services", lambda: (_ for _ in ()).throw(RuntimeError("bg")))
    monkeypatch.setattr(main, "stop_reload_listener", lambda: None)
    monkeypatch.setattr(main, "close_http_client", _boom)
    monkeypatch.setattr(main, "close_redis", lambda: (_ for _ in ()).throw(RuntimeError("redis")))
    monkeypatch.setattr(main, "close_engine", lambda: (_ for _ in ()).throw(RuntimeError("db")))

    await main.shutdown_event()


@pytest.mark.asyncio
async def test_route_query_stream_uses_query_runtime(monkeypatch):
    """Streaming endpoint should delegate to process_query_request and emit SSE frames."""
    from app import main

    async def _process(_req):
        return {
            "result": {
                "answer": "ola mundo",
                "model": "ollama/test",
                "modality": "text",
                "metadata": {"prompt_tokens": 1},
            }
        }

    monkeypatch.setattr(main, "process_query_request", _process)
    response = await main.route_query_stream(SimpleNamespace())

    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    payload = "".join(chunks)
    assert "event: meta" in payload
    assert "event: token" in payload
    assert "event: done" in payload


@pytest.mark.asyncio
async def test_health_and_versioned_endpoints(monkeypatch):
    """Operational routes and versioned wrappers should use extracted routers."""
    from app import main
    from app.api import ops_routes

    async def _h():
        return {"status": "unhealthy", "components": {"redis": {"healthy": False}}}

    monkeypatch.setattr(ops_routes, "get_full_health_check", _h)
    resp = await ops_routes.health()
    assert resp.status_code == 503

    async def _live():
        return {"status": "alive"}

    async def _ready():
        return {"status": "not_ready"}

    monkeypatch.setattr(ops_routes, "get_liveness_check", _live)
    monkeypatch.setattr(ops_routes, "get_readiness_check", _ready)
    assert (await ops_routes.liveness())["status"] == "alive"
    assert (await ops_routes.readiness()).status_code == 503

    async def _route_query(req):
        return {"ok": True}

    monkeypatch.setattr(main, "route_query", _route_query)
    out = await main.v1_route_query(SimpleNamespace())
    assert out["ok"] is True

    monkeypatch.setattr(ops_routes, "health", _ready)
    health_resp = await main.v1_health()
    assert health_resp["status"] == "not_ready"
