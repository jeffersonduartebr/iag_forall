"""Módulo `tests/test_health_components.py`: descreve responsabilidades e integrações deste arquivo."""

import sys
from types import SimpleNamespace

import pytest

from app import health


class _Response:
    """Classe `_Response`: concentra responsabilidades de test health components."""
    def __init__(self, data=None, fail=False):
        """Inicializa estado interno necessário para uso da classe."""
        self._data = data or {}
        self._fail = fail

    def raise_for_status(self):
        """Executa raise for status."""
        if self._fail:
            raise RuntimeError("bad status")

    def json(self):
        """Executa json."""
        return self._data


@pytest.mark.asyncio
async def test_component_health_checks_success_paths(monkeypatch):
    """Testa component health checks success paths."""
    monkeypatch.setattr(
        "app.utils.redis_client.check_redis_health",
        lambda: {"healthy": True, "latency_ms": 1.2, "pool_size": 4},
    )

    class _DBConn:
        """Classe `_DBConn`: concentra responsabilidades de test health components."""
        def execute(self, _):
            """Executa execute."""
            return None

    class _DBCtx:
        """Classe `_DBCtx`: concentra responsabilidades de test health components."""
        def __enter__(self):
            """Executa enter."""
            return _DBConn()

        def __exit__(self, exc_type, exc, tb):
            """Executa exit."""
            return False

    class _Engine:
        """Classe `_Engine`: concentra responsabilidades de test health components."""
        def connect(self):
            """Executa connect."""
            return _DBCtx()

    fake_sqlalchemy = SimpleNamespace(create_engine=lambda *a, **k: _Engine(), text=lambda q: q)
    monkeypatch.setitem(sys.modules, "sqlalchemy", fake_sqlalchemy)

    fake_chromadb = SimpleNamespace(
        PersistentClient=lambda path: SimpleNamespace(list_collections=lambda: ["c1", "c2"])
    )
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    class _AsyncClient:
        """Classe `_AsyncClient`: concentra responsabilidades de test health components."""
        def __init__(self, timeout):
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
            return _Response({"models": [{"name": "m1"}, {"name": "m2"}]})

    monkeypatch.setattr(health.httpx, "AsyncClient", _AsyncClient)

    manager = SimpleNamespace(get_all_statuses=lambda: [{"model": "a", "state": "closed"}])
    monkeypatch.setattr("app.reliability.get_circuit_breaker_manager", lambda: manager)

    redis = await health.check_redis_health()
    db = await health.check_database_health()
    chroma = await health.check_vectorstore_health()
    ollama = await health.check_ollama_health()
    cb = await health.check_circuit_breakers_health()

    assert redis.healthy and redis.details["pool_size"] == 4
    assert db.healthy
    assert chroma.healthy and chroma.details["collections"] == 2
    assert ollama.healthy and ollama.details["models_loaded"] == 2
    assert cb.healthy is True


@pytest.mark.asyncio
async def test_component_health_error_paths_and_cache(monkeypatch):
    """Testa component health error paths and cache."""
    monkeypatch.setattr("app.utils.redis_client.check_redis_health", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    redis = await health.check_redis_health()
    assert redis.healthy is False

    class _BadAsyncClient:
        """Classe `_BadAsyncClient`: concentra responsabilidades de test health components."""
        def __init__(self, timeout):
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
            return _Response(fail=True)

    monkeypatch.setattr(health.httpx, "AsyncClient", _BadAsyncClient)
    ollama = await health.check_ollama_health()
    assert ollama.healthy is False

    async def _ok(name, healthy=True):
        """Executa ok."""
        return health.ComponentHealth(name=name, healthy=healthy)

    monkeypatch.setattr(health, "check_redis_health", lambda: _ok("redis", True))
    monkeypatch.setattr(health, "check_database_health", lambda: _ok("mariadb", True))
    monkeypatch.setattr(health, "check_vectorstore_health", lambda: _ok("chromadb", True))
    monkeypatch.setattr(health, "check_ollama_health", lambda: _ok("ollama", True))
    monkeypatch.setattr(health, "check_circuit_breakers_health", lambda: _ok("circuit_breakers", True))

    health.invalidate_health_cache()
    first = await health.get_full_health_check(force_refresh=True)
    second = await health.get_full_health_check(force_refresh=False)

    assert first["status"] == "healthy"
    assert second["cached"] is True


@pytest.mark.asyncio
async def test_component_to_dict_and_liveness():
    """Testa component to dict and liveness."""
    comp = health.ComponentHealth(name="x", healthy=False, latency_ms=1.2345, error="err", details={"a": 1})
    d = comp.to_dict()
    assert d["latency_ms"] == 1.23
    assert d["error"] == "err"
    assert d["details"]["a"] == 1

    live = await health.get_liveness_check()
    assert live["status"] == "alive"
