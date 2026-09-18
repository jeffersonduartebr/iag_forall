# Objective: Test coverage for bulk settings cache priming.
"""settings_dynamic._prime_cache: one DB query + one MGET, same precedence as per-key reads."""

from contextlib import contextmanager
from types import SimpleNamespace

import fakeredis
import pytest

import app.settings_dynamic as sd


class _Engine:
    def __init__(self, rows):
        self.rows = rows
        self.queries = 0

    @contextmanager
    def connect(self):
        engine = self

        def execute(*_a, **_k):
            engine.queries += 1
            return SimpleNamespace(fetchall=lambda: list(engine.rows))

        yield SimpleNamespace(execute=execute)


@pytest.fixture
def sources(monkeypatch):
    server = fakeredis.FakeRedis()
    engine = _Engine([("NSGA_W_QUALITY", "2.0"), ("NSGA_W_LATENCY", "0.9"), ("CUSTOM_ONLY_IN_DB", "x")])
    monkeypatch.setattr(sd, "_get_rds", lambda: server)
    monkeypatch.setattr(sd, "engine", engine)
    sd._lru.clear()
    sd._last_prime = 0.0
    yield SimpleNamespace(redis=server, engine=engine)
    sd._lru.clear()
    sd._last_prime = 0.0


def test_prime_applies_redis_db_env_default_precedence(monkeypatch, sources):
    sources.redis.set("settings:NSGA_W_QUALITY", "3.0")  # Redis vence o DB
    monkeypatch.setenv("NSGA_W_COST", "77")  # env vence o default
    monkeypatch.setenv("FORCE_NSGA_W_LATENCY", "5")  # FORCE_ vence tudo

    assert sd._prime_cache() is True
    assert sd._lru.get("NSGA_W_QUALITY") == "3.0"
    assert sd._lru.get("NSGA_W_LATENCY") == "5"
    assert sd._lru.get("NSGA_W_COST") == "77"
    assert sd._lru.get("CUSTOM_ONLY_IN_DB") == "x"
    assert sd._lru.get("UNCERTAINTY_THRESHOLD") == sd.SETTINGS_DEFAULTS["UNCERTAINTY_THRESHOLD"]
    assert sources.engine.queries == 1


def test_prime_never_reads_secrets_from_redis_or_db(monkeypatch, sources):
    sources.redis.set("settings:JWT_SECRET", "from-redis")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    sd._prime_cache()
    assert sd._lru.get("JWT_SECRET") != "from-redis"


def test_prime_is_rate_limited_and_reset_by_invalidation(sources):
    assert sd._prime_cache() is True
    assert sd._prime_cache() is False  # dentro do intervalo mínimo
    sd._invalidate_cache()
    assert sd._prime_cache() is True
    assert sources.engine.queries == 2


def test_get_primes_once_then_serves_from_memory(sources):
    assert sd.settings.get("NSGA_W_QUALITY") == "2.0"
    assert sd.settings.get("NSGA_W_LATENCY") == "0.9"
    assert sources.engine.queries == 1  # a segunda leitura não toca o DB


def test_prime_skips_when_sources_down(monkeypatch):
    monkeypatch.setattr(sd, "_get_rds", lambda: None)
    monkeypatch.setattr(sd, "_all_from_db", lambda: None)
    sd._lru.clear()
    sd._last_prime = 0.0
    assert sd._prime_cache() is False
