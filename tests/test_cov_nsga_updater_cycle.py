# Objective: Coverage for the NSGA-II worker cycle, HTTP endpoints, loop and judge feedback.
"""nsga_weights_updater: run_optimization_cycle, FastAPI routes, loop scheduling, judge-feedback tuning."""

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app import nsga_weights_updater as nsga


@pytest.fixture
def cycle(monkeypatch):
    """Tiny DEAP run with every side effect captured."""
    calls = []
    monkeypatch.setattr(nsga, "redis_client", fakeredis.FakeRedis(decode_responses=True))
    monkeypatch.setattr(nsga, "load_candidate_models", lambda m: ["fast", "slow"])
    monkeypatch.setattr(
        nsga,
        "aggregate_ema_by_model",
        lambda m, models: {
            "fast": {"latency": 0.5, "cost": 0.0001, "quality": 6.0, "alignment": 1.0},
            "slow": {"latency": 6.0, "cost": 0.01, "quality": 9.0, "alignment": 1.0},
        },
    )
    real_run = nsga.run_nsga_optimization
    monkeypatch.setattr(nsga, "run_nsga_optimization", lambda mod, ms, met: real_run(mod, ms, met, n_pop=8, n_gen=2))
    monkeypatch.setattr(nsga, "persist_results", lambda m, w: calls.append(("persist", m, w)))
    monkeypatch.setattr(nsga, "tune_uncertainty_threshold", lambda eff: calls.append(("uq", eff)))
    monkeypatch.setattr(nsga, "tune_global_strategy_weights", lambda sm: calls.append(("strategy", sm)))
    monkeypatch.setattr(nsga, "tune_weights_from_judge_feedback", lambda: calls.append(("judge",)))
    monkeypatch.setattr(nsga, "publish_reward_shares", lambda m, sm: calls.append(("reward", m)))
    return calls


def test_text_cycle_runs_global_tuning_and_publishes(cycle):
    weights = nsga.run_optimization_cycle("text")
    assert set(weights) == {"fast", "slow"} and sum(weights.values()) == pytest.approx(1.0)
    kinds = [c[0] for c in cycle]
    assert kinds == ["persist", "uq", "strategy", "judge", "reward"]


def test_non_text_cycle_skips_global_tuning(cycle):
    nsga.run_optimization_cycle("vision")
    assert [c[0] for c in cycle] == ["persist", "reward"]


def test_cycle_without_models_is_skipped(monkeypatch, cycle):
    monkeypatch.setattr(nsga, "load_candidate_models", lambda m: [])
    assert nsga.run_optimization_cycle("text") is None
    assert cycle == []


def test_http_routes(monkeypatch):
    monkeypatch.setattr(nsga, "run_optimization_cycle", lambda m: {"a": 1.0})
    monkeypatch.setattr(nsga, "calibration_status_payload", lambda: {"risk_factors": {}})
    client = TestClient(nsga.app)
    assert client.post("/run/audio").status_code == 400
    assert client.post("/run/text").json() == {"status": "ok", "weights": {"a": 1.0}}
    assert client.get("/health").json() == {"status": "ok"}
    assert "nsga" in client.get("/metrics").text.lower()
    assert client.get("/calibration/status").json() == {"risk_factors": {}}

    def boom(*_a):
        raise RuntimeError("kaput")

    monkeypatch.setattr(nsga, "run_optimization_cycle", boom)
    resp = client.post("/run/text")
    assert resp.status_code == 500 and resp.json() == {"error": "kaput"}

    ran = []
    monkeypatch.setattr(nsga, "run_calibration_cycle", lambda: ran.append(1))
    assert client.post("/calibration/run").json()["status"] == "ok" and ran == [1]
    monkeypatch.setattr(nsga, "run_calibration_cycle", boom)
    assert client.post("/calibration/run").status_code == 500


def test_loop_iteration_isolates_errors_and_calibrates_every_third(monkeypatch):
    seen = []

    def cycle(m):
        seen.append(m)
        if m == "vision":
            raise RuntimeError("vision down")

    calib = []
    monkeypatch.setattr(nsga, "run_optimization_cycle", cycle)
    monkeypatch.setattr(nsga, "run_calibration_cycle", lambda: calib.append(1))
    nsga.run_loop_iteration(1)
    assert seen == nsga.MODALITIES and calib == []
    nsga.run_loop_iteration(3)
    assert calib == [1]

    def broken():
        raise RuntimeError("calib down")

    monkeypatch.setattr(nsga, "run_calibration_cycle", broken)
    nsga.run_loop_iteration(6)  # erro só registrado


def test_background_loop_honours_stop_event(monkeypatch):
    iterations = []
    stop = threading.Event()

    def iterate(i):
        iterations.append(i)
        if i == 2:
            stop.set()

    monkeypatch.setattr(nsga, "run_loop_iteration", iterate)
    monkeypatch.setattr(nsga, "UPDATE_INTERVAL_S", 0)
    nsga.background_loop(stop, initial_delay_s=0)
    assert iterations == [1, 2]

    nsga.background_loop(stop, initial_delay_s=0)  # já parado: retorna antes da 1ª iteração
    assert iterations == [1, 2]


def _judge_engine(judged, proxy):
    results = iter([judged, proxy])

    class _Conn:
        def execute(self, *_a, **_k):
            return SimpleNamespace(fetchone=lambda: next(results))

    @contextmanager
    def connect():
        yield _Conn()

    return SimpleNamespace(connect=connect)


@pytest.fixture
def judge(monkeypatch):
    updates = []
    monkeypatch.setattr(nsga, "is_frozen_policy_active", lambda: False)
    monkeypatch.setattr(nsga.settings, "get", lambda key, fallback=None: fallback)
    monkeypatch.setattr(nsga.settings, "set", lambda k, v, actor="": updates.append((k, v)))
    return updates


@pytest.mark.parametrize(
    "judged, proxy, w_quality, expected",
    [
        (None, None, 1.0, []),  # sem dados
        ((10, 9), (4,), 1.0, []),  # abaixo do mínimo de amostras
        ((40, 4), (0,), 1.0, []),  # taxa de erro baixa
        ((40, 30), None, 5.0, []),  # já no teto
        ((40, 30), (2,), 1.0, [("NSGA_W_QUALITY", "1.3")]),
    ],
)
def test_judge_feedback_branches(monkeypatch, judge, judged, proxy, w_quality, expected):
    monkeypatch.setattr(nsga, "_db_engine", lambda: _judge_engine(judged, proxy))
    monkeypatch.setattr(type(nsga.settings), "NSGA_W_QUALITY", property(lambda s: w_quality))
    nsga.tune_weights_from_judge_feedback()
    assert judge == expected


def test_judge_feedback_frozen_or_db_failure_changes_nothing(monkeypatch, judge):
    def broken():
        raise RuntimeError("db down")

    monkeypatch.setattr(nsga, "_db_engine", broken)
    nsga.tune_weights_from_judge_feedback()
    monkeypatch.setattr(nsga, "is_frozen_policy_active", lambda: True)
    monkeypatch.setattr(nsga, "_db_engine", lambda: pytest.fail("frozen policy must not read the DB"))
    nsga.tune_weights_from_judge_feedback()
    assert judge == []
