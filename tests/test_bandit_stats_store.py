# Objective: Test coverage for bandit statistics persistence.
"""services.bandit_stats_store: parsing, DB fallback with negative cache, batch upsert."""

import json
from contextlib import contextmanager

import fakeredis
import pytest
from app.services import bandit_stats_store as store


def test_sanitize_clamps_and_defaults():
    out = store.sanitize_model_stats({"avg": 1.7, "count": -3, "var": float("nan"), "alpha": 0})
    assert out == {"mean": 1.0, "count": 0, "var": 0.0, "M2": 0.0, "alpha": 1e-6, "beta": 1.0}


def test_parse_redis_stats_tolerates_bytes_and_garbage():
    raw = {b"m1": json.dumps({"mean": 0.4, "count": 2}), "m2": "{nope", "m3": "[1]"}
    assert set(store.parse_redis_stats(raw)) == {"m1"}
    assert store.parse_redis_stats(None) == {}


def test_db_fallback_backfills_hits_and_caches_misses():
    cache = store.ColdContextCache(ttl_s=60)
    calls, backfilled = [], {}

    def loader(ctx):
        calls.append(ctx)
        return {"m": store.sanitize_model_stats({"mean": 0.5})} if ctx == "warm" else {}

    assert store.db_fallback("warm", loader, backfilled.__setitem__, cache)
    assert "warm" in backfilled

    assert store.db_fallback("cold", loader, backfilled.__setitem__, cache) == {}
    assert store.db_fallback("cold", loader, backfilled.__setitem__, cache) == {}
    assert calls == ["warm", "cold"]  # o contexto frio só consulta o DB uma vez por janela

    cache.forget("cold")
    store.db_fallback("cold", loader, backfilled.__setitem__, cache)
    assert calls == ["warm", "cold", "cold"]


def test_cold_cache_expires_and_is_bounded(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(store.time, "monotonic", lambda: now[0])
    cache = store.ColdContextCache(ttl_s=10, maxsize=2)
    cache.mark_cold("a")
    assert cache.is_cold("a")
    now[0] += 11
    assert not cache.is_cold("a")
    for ctx in ("x", "y", "z"):
        cache.mark_cold(ctx)
    assert not cache.is_cold("x") and cache.is_cold("z")


class _Engine:
    def __init__(self):
        self.executed = []

    @contextmanager
    def begin(self):
        yield self

    def execute(self, sql, params):
        self.executed.append(params)


def test_upsert_uses_single_executemany():
    engine = _Engine()
    store.upsert_stats_db(lambda: engine, [("c", "m1", {"mean": 0.5, "count": 2}), ("c", "m2", {"mean": 0.1})])
    assert len(engine.executed) == 1 and [p["model"] for p in engine.executed[0]] == ["m1", "m2"]
    store.upsert_stats_db(lambda: engine, [])
    assert len(engine.executed) == 1


def test_engine_creation_failure_is_contained():
    def broken():
        raise RuntimeError("sem banco")

    store.upsert_stats_db(broken, [("c", "m", {"mean": 0.5})])  # não propaga
    assert store.load_stats_from_db(broken, "c") == {}


def test_bandits_cold_context_hits_db_once(monkeypatch):
    from app import bandits

    store.cold_contexts.clear()
    server = fakeredis.FakeRedis()
    queries = []
    monkeypatch.setattr(bandits, "_get_rds", lambda: server)
    monkeypatch.setattr(bandits, "_get_ctx_stats_from_db", lambda ctx: queries.append(ctx) or {})

    assert bandits._get_ctx_stats("ctx:novo") == {}
    assert bandits._get_ctx_stats("ctx:novo") == {}
    assert queries == ["ctx:novo"]

    bandits._set_ctx_stats("ctx:novo", {"m": {"mean": 0.6, "count": 1}})
    assert bandits._get_ctx_stats("ctx:novo")["m"]["mean"] == pytest.approx(0.6)
    store.cold_contexts.clear()
