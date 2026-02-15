"""Módulo `tests/test_main_lifecycle.py`: descreve responsabilidades e integrações deste arquivo."""

from types import SimpleNamespace

import pytest

@pytest.mark.asyncio
async def test_preload_ollama_models_download_flow(monkeypatch):
    """Testa preload ollama models download flow."""
    from app import main

    monkeypatch.setenv("CANDIDATE_MODELS_LIST", '["ollama/phi4:latest"]')
    monkeypatch.setenv("JUDGE_MODELS", "[]")
    monkeypatch.setenv("OLLAMA_MODEL", "")

    monkeypatch.setattr(main, "VLM_OLLAMA_MODELS", [])
    monkeypatch.setattr(main.settings, "get", lambda k, d=None: "nomic-ai/nomic-embed-text-v1.5" if k == "EMBED_TEXT_MODEL" else d)

    class _Resp:
        """Classe `_Resp`: concentra responsabilidades de test main lifecycle."""
        def __init__(self, status_code=200, payload=None):
            """Inicializa estado interno necessário para uso da classe."""
            self.status_code = status_code
            self._payload = payload or {}

        def raise_for_status(self):
            """Executa raise for status."""
            if self.status_code >= 400:
                raise RuntimeError("http error")

        def json(self):
            """Executa json."""
            return self._payload

    class _StreamCtx:
        """Classe `_StreamCtx`: concentra responsabilidades de test main lifecycle."""
        async def __aenter__(self):
            """Executa aenter."""
            return _Resp(200, {})

        async def __aexit__(self, exc_type, exc, tb):
            """Executa aexit."""
            return False

        async def aiter_lines(self):
            """Executa aiter lines."""
            if False:
                yield ""

    class _Client:
        """Classe `_Client`: concentra responsabilidades de test main lifecycle."""
        def __init__(self, timeout=None):
            """Inicializa estado interno necessário para uso da classe."""
            self.timeout = timeout

        async def __aenter__(self):
            """Executa aenter."""
            return self

        async def __aexit__(self, exc_type, exc, tb):
            """Executa aexit."""
            return False

        async def get(self, _url):
            """Executa get."""
            return _Resp(200, {"models": []})

        def stream(self, method, url, json):
            """Executa stream."""
            return _StreamCtx()

    monkeypatch.setattr(main.httpx, "AsyncClient", _Client)
    await main.preload_ollama_models()


@pytest.mark.asyncio
async def test_startup_event_executes_warmup_and_shutdown(monkeypatch):
    """Testa startup event executes warmup and shutdown."""
    from app import main

    async def _cleanup():
        """Executa cleanup."""
        return None

    async def _preload():
        """Executa preload."""
        return None

    async def _add_doc(**kwargs):
        """Executa add doc."""
        return True

    coros = []

    def _create_task(coro):
        """Executa create task."""
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
    """Testa health and versioned endpoints."""
    from app import main

    async def _h():
        """Executa h."""
        return {"status": "unhealthy", "components": {"redis": {"healthy": False}}}

    monkeypatch.setattr(main, "get_full_health_check", _h)
    resp = await main.health()
    assert resp.status_code == 503

    async def _live():
        """Executa live."""
        return {"status": "alive"}

    async def _ready():
        """Executa ready."""
        return {"status": "not_ready"}

    monkeypatch.setattr(main, "get_liveness_check", _live)
    monkeypatch.setattr(main, "get_readiness_check", _ready)
    assert (await main.liveness())["status"] == "alive"
    assert (await main.readiness()).status_code == 503

    async def _route_query(req):
        """Executa route query."""
        return {"ok": True}

    monkeypatch.setattr(main, "route_query", _route_query)
    out = await main.v1_route_query(SimpleNamespace())
    assert out["ok"] is True
