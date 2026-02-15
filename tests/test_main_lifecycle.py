from types import SimpleNamespace

import pytest

@pytest.mark.asyncio
async def test_preload_ollama_models_download_flow(monkeypatch):
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
async def test_startup_event_executes_warmup_and_shutdown(monkeypatch):
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
    assert coros  # cleanup + warmup tasks were scheduled

    for c in coros:
        await c

    monkeypatch.setattr(main, "stop_background_services", lambda: None)
    monkeypatch.setattr(main, "stop_reload_listener", lambda: None)
    monkeypatch.setattr(main, "close_http_client", _preload)
    monkeypatch.setattr(main, "close_redis", lambda: None)
    monkeypatch.setattr(main, "close_engine", lambda: None)
    await main.shutdown_event()


@pytest.mark.asyncio
async def test_health_and_versioned_endpoints(monkeypatch):
    from app import main

    async def _h():
        return {"status": "unhealthy", "components": {"redis": {"healthy": False}}}

    monkeypatch.setattr(main, "get_full_health_check", _h)
    resp = await main.health()
    assert resp.status_code == 503

    async def _live():
        return {"status": "alive"}

    async def _ready():
        return {"status": "not_ready"}

    monkeypatch.setattr(main, "get_liveness_check", _live)
    monkeypatch.setattr(main, "get_readiness_check", _ready)
    assert (await main.liveness())["status"] == "alive"
    assert (await main.readiness()).status_code == 503

    async def _route_query(req):
        return {"ok": True}

    monkeypatch.setattr(main, "route_query", _route_query)
    out = await main.v1_route_query(SimpleNamespace())
    assert out["ok"] is True
