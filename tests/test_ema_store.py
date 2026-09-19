# Objective: Test coverage for the shared EMA store and its use in routing.
"""services.ema_store: atomic shared EMA, cached snapshot and S(m) latency/cost estimates."""

import json

import fakeredis
import pytest
from app.services import ema_store


@pytest.fixture(autouse=True)
def _fresh():
    ema_store.reset_ema_snapshots()
    yield
    ema_store.reset_ema_snapshots()


def test_updates_from_two_processes_accumulate():
    shared = fakeredis.FakeServer()
    worker_a, worker_b = fakeredis.FakeRedis(server=shared), fakeredis.FakeRedis(server=shared)
    ema_store.update_shared_ema("text", "m", 2.0, 8.0, 0.01, rds=worker_a)
    second = ema_store.update_shared_ema("text", "m", 4.0, 6.0, 0.03, rds=worker_b)
    assert second["updates"] == 2
    assert second["ema_latency"] == pytest.approx(0.2 * 4.0 + 0.8 * 2.0)
    assert json.loads(worker_a.hget("ema:text", "m"))["updates"] == 2


def test_update_retries_on_concurrent_write(monkeypatch):
    server = fakeredis.FakeRedis()
    real_pipeline = server.pipeline
    interfered = []

    def pipeline(*a, **k):
        pipe = real_pipeline(*a, **k)
        real_multi = pipe.multi

        def multi():
            if not interfered:  # outro "processo" grava entre o WATCH e o MULTI
                interfered.append(1)
                server.hset("ema:text", "m", json.dumps(ema_store.next_ema(None, 9.0, 1.0, 0.5)))
            return real_multi()

        pipe.multi = multi
        return pipe

    monkeypatch.setattr(server, "pipeline", pipeline)
    entry = ema_store.update_shared_ema("text", "m", 1.0, 9.0, 0.0, rds=server)
    assert entry["updates"] == 2  # recomputou sobre a gravação concorrente


def test_snapshot_is_cached_and_parsed(monkeypatch):
    server = fakeredis.FakeRedis()
    server.hset("ema:text", "m", json.dumps({"ema_latency": 1.5, "ema_cost": 0.002, "ema_quality": 8, "updates": 4}))
    server.hset("ema:text", "bad", "{nope")
    snap = ema_store.load_ema_snapshot("text", rds=server)
    assert set(snap) == {"m"}
    server.hset("ema:text", "m2", json.dumps({"ema_latency": 1.0, "ema_cost": 0.0, "updates": 1}))
    assert set(ema_store.load_ema_snapshot("text", rds=server)) == {"m"}  # ainda no cache de 5 s


def test_routing_latency_cost_uses_ema_after_enough_updates():
    entry = {"ema_latency": 3.2, "ema_cost": 0.004, "updates": 3}
    assert ema_store.routing_latency_cost(entry, is_local=False, is_sota=True) == (3.2, 0.004)
    few = {**entry, "updates": 2}
    assert ema_store.routing_latency_cost(few, is_local=False, is_sota=True) == (2.0, 0.01)
    assert ema_store.routing_latency_cost(None, is_local=True, is_sota=False) == (0.5, 0.000001)


def test_choose_top2_prefers_faster_cheaper_model_with_real_ema(monkeypatch):
    from app import router_strategy

    models = ["openai/gpt-5.5", "anthropic/claude-sonnet-5"]
    snapshot = {m: {"mean": 0.8, "count": 50, "alpha": 40.0, "beta": 10.0} for m in models}
    ema = {
        "openai/gpt-5.5": {"ema_latency": 20.0, "ema_cost": 0.05, "updates": 10},
        "anthropic/claude-sonnet-5": {"ema_latency": 2.0, "ema_cost": 0.005, "updates": 10},
    }
    monkeypatch.setattr(router_strategy, "get_snapshot", lambda: snapshot)
    monkeypatch.setattr(router_strategy, "sample_metrics_from_snapshot", lambda snap: {m: 8.0 for m in models})
    monkeypatch.setattr(router_strategy, "load_ema_snapshot", lambda modality: ema)
    weights = {"w_quality": 1.0, "w_latency": 0.5, "w_cost": 50.0}
    top2 = router_strategy.choose_top2_models(models, weights, "q", "text", 0.1)
    assert top2[0] == "anthropic/claude-sonnet-5"  # antes, latência/custo eram constantes iguais para ambos
