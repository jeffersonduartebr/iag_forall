import sys
from types import SimpleNamespace

import pytest

from app import health


class _Response:
    def __init__(self, data=None, fail=False):
        self._data = data or {}
        self._fail = fail

    def raise_for_status(self):
        if self._fail:
            raise RuntimeError("bad status")

    def json(self):
        return self._data


@pytest.mark.asyncio
async def test_component_health_checks_success_paths(monkeypatch):
    monkeypatch.setattr(
        "app.utils.redis_client.check_redis_health",
        lambda: {"healthy": True, "latency_ms": 1.2, "pool_size": 4},
    )

    class _DBConn:
        def execute(self, _):
            return None

    class _DBCtx:
        def __enter__(self):
            return _DBConn()

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Engine:
        def connect(self):
            return _DBCtx()

    fake_sqlalchemy = SimpleNamespace(create_engine=lambda *a, **k: _Engine(), text=lambda q: q)
    monkeypatch.setitem(sys.modules, "sqlalchemy", fake_sqlalchemy)

    fake_chromadb = SimpleNamespace(
        PersistentClient=lambda path: SimpleNamespace(list_collections=lambda: ["c1", "c2"])
    )
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    class _AsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, _url):
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
    monkeypatch.setattr("app.utils.redis_client.check_redis_health", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    redis = await health.check_redis_health()
    assert redis.healthy is False

    class _BadAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, _url):
            return _Response(fail=True)

    monkeypatch.setattr(health.httpx, "AsyncClient", _BadAsyncClient)
    ollama = await health.check_ollama_health()
    assert ollama.healthy is False

    async def _ok(name, healthy=True):
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
    comp = health.ComponentHealth(name="x", healthy=False, latency_ms=1.2345, error="err", details={"a": 1})
    d = comp.to_dict()
    assert d["latency_ms"] == 1.23
    assert d["error"] == "err"
    assert d["details"]["a"] == 1

    live = await health.get_liveness_check()
    assert live["status"] == "alive"
