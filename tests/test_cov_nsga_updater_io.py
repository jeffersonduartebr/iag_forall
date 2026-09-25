# Objective: Coverage for the NSGA-II worker's I/O helpers (Redis, EMA reads, persistence, health).
"""nsga_weights_updater: candidates, EMA aggregation, efficiency history, persistence, reward shares."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import fakeredis
import pytest

from app import nsga_weights_updater as nsga


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


def _engine(rows=None, fail=False, calls=None):
    class _Conn:
        def execute(self, stmt, params=None):
            if calls is not None:
                calls.append(params)
            return _Result(rows or [])

    @contextmanager
    def ctx():
        if fail:
            raise RuntimeError("db down")
        yield _Conn()

    return SimpleNamespace(connect=ctx, begin=ctx)


@pytest.fixture
def rds(monkeypatch):
    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(nsga, "redis_client", client)
    return client


def test_get_redis_client_returns_client_only_when_ping_succeeds(monkeypatch):
    monkeypatch.setattr(nsga.redis, "Redis", lambda **kw: SimpleNamespace(ping=lambda: True))
    assert nsga.get_redis_client() is not None
    monkeypatch.setattr(nsga.redis, "Redis", lambda **kw: SimpleNamespace(ping=lambda: False))
    assert nsga.get_redis_client() is None

    def refuse(**kw):
        raise ConnectionError("no redis")

    monkeypatch.setattr(nsga.redis, "Redis", refuse)
    assert nsga.get_redis_client() is None


def test_init_db_tables_runs_ddl_and_survives_db_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(nsga, "_db_engine", lambda: _engine(calls=calls))
    nsga.init_db_tables()
    assert len(calls) == 1
    monkeypatch.setattr(nsga, "_db_engine", lambda: _engine(fail=True))
    nsga.init_db_tables()  # só registra o erro


def test_candidates_prefer_redis_then_settings_then_fallback(monkeypatch, rds):
    rds.set(nsga.REDIS_KEY_CANDIDATES["text"], json.dumps(["a", "b"]))
    assert nsga.load_candidate_models("text") == ["a", "b"]

    rds.set(nsga.REDIS_KEY_CANDIDATES["text"], "{not json")
    monkeypatch.setattr(type(nsga.settings), "CANDIDATE_MODELS_LIST", property(lambda s: ["x", "", "y", "x"]))
    assert nsga.load_candidate_models("text") == ["x", "y"]

    rds.set(nsga.REDIS_KEY_CANDIDATES["vision"], json.dumps({"not": "a list"}))
    monkeypatch.setattr(type(nsga.settings), "CANDIDATE_VISION_MODELS_LIST", property(lambda s: []))
    assert nsga.load_candidate_models("vision") == list(nsga._FALLBACK_CANDIDATES["vision"])
    assert nsga.load_candidate_models("audio") == []


def test_aggregate_ema_uses_db_rows_and_cold_start(monkeypatch):
    row = {"model": "m1", "ema_latency": "1.5", "ema_cost": 0.02, "ema_quality": 8, "ema_alignment": 0.9}
    monkeypatch.setattr(nsga, "_db_engine", lambda: _engine(rows=[row]))
    data = nsga.aggregate_ema_by_model("text", ["m1", "m2"])
    assert data["m1"] == {"latency": 1.5, "cost": 0.02, "quality": 8.0, "alignment": 0.9}
    assert data["m2"] == {"latency": 2.0, "cost": 0.001, "quality": 5.0, "alignment": 1.0}

    monkeypatch.setattr(nsga, "_db_engine", lambda: _engine(fail=True))
    assert nsga.aggregate_ema_by_model("text", ["m1"])["m1"]["quality"] == 5.0


def test_efficiency_history_is_bounded_and_newest_first(monkeypatch, rds):
    monkeypatch.setattr(nsga.settings, "get", lambda key, fallback=None: 3 if "HISTORY" in key else fallback)
    for eff in (1.0, 2.0, 3.0, 4.0):
        nsga.store_efficiency_history("text", eff)
    assert nsga.get_efficiency_history("text") == [4.0, 3.0, 2.0]


def test_efficiency_history_without_or_with_broken_redis(monkeypatch):
    monkeypatch.setattr(nsga, "redis_client", None)
    nsga.store_efficiency_history("text", 1.0)
    assert nsga.get_efficiency_history("text") == []

    broken = SimpleNamespace(lpush=None, lrange=None)  # chamar None levanta TypeError
    monkeypatch.setattr(nsga, "redis_client", broken)
    nsga.store_efficiency_history("text", 1.0)
    assert nsga.get_efficiency_history("text") == []


@pytest.mark.parametrize(
    "health, n_warnings, marker",
    [(-1.0, 1, "STUCK"), (0.0, 1, "degraded"), (1.0, 0, None)],
)
def test_check_optimization_health_warns_by_health(monkeypatch, rds, health, n_warnings, marker):
    metrics = {"trend": -0.5, "variance": 2.0, "health": health}
    monkeypatch.setattr(nsga, "compute_convergence_metrics", lambda history: metrics)
    status = nsga.check_optimization_health("vision", 3.5)
    assert status["history_size"] == 1 and status["current_efficiency"] == 3.5
    assert len(status["warnings"]) == n_warnings
    if marker:
        assert marker in status["warnings"][0]


def test_persist_results_writes_db_rows_and_redis(monkeypatch, rds):
    calls = []
    monkeypatch.setattr(nsga, "_db_engine", lambda: _engine(calls=calls))
    nsga.persist_results("text", {"a": 0.6, "b": 0.4})
    assert calls == [{"mod": "text", "m": "a", "w": 0.6}, {"mod": "text", "m": "b", "w": 0.4}]
    assert json.loads(rds.get("nsga:weights:text")) == {"a": 0.6, "b": 0.4}


def test_persist_results_tolerates_db_and_redis_failures(monkeypatch):
    monkeypatch.setattr(nsga, "_db_engine", lambda: _engine(fail=True))
    monkeypatch.setattr(nsga, "redis_client", SimpleNamespace(set=None))
    nsga.persist_results("text", {"a": 1.0})  # nenhum erro propagado


def test_publish_reward_shares_skipped_under_frozen_policy(monkeypatch, rds):
    monkeypatch.setattr(nsga, "is_frozen_policy_active", lambda: True)
    nsga.publish_reward_shares("text", (1.0, 0.01, 8.0))
    assert rds.keys() == []


def test_publish_reward_shares_falls_back_on_invalid_min_share(monkeypatch, rds):
    seen = {}
    monkeypatch.setattr(nsga, "is_frozen_policy_active", lambda: False)
    monkeypatch.setattr(nsga.settings, "get", lambda key, fallback=None: "abc" if "MIN_SHARE" in key else fallback)

    def derive(*args, min_share):
        seen["min_share"] = min_share
        return (0.5, 0.3, 0.2)

    monkeypatch.setattr(nsga, "derive_reward_weights", derive)
    nsga.publish_reward_shares("text", (1.0, 0.01, 8.0))
    assert seen["min_share"] == nsga.DEFAULT_MIN_SHARE
    payload = json.loads(rds.get(rds.keys()[0]))
    assert payload["quality"] == 0.5 and payload["sys_metrics"] == {"quality": 8.0, "latency_s": 1.0, "cost_usd": 0.01}
