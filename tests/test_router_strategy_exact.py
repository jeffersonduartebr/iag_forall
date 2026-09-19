# Objective: Exact-value tests for the routing score S(m), risk factors and candidate filters.
"""router_strategy helpers pinned to numbers (complements the ranking tests)."""

from types import SimpleNamespace

import pytest

from app import router_strategy as rs

SOTA, LOCAL, OTHER = "openai/gpt-5.5", "ollama/gemma3:4b", "mistral/medium"
RISKS = (1.3, 0.6, 1.1)


def test_model_score_formula():
    assert rs.model_score(8.0, 2.0, 0.01, (1.0, 0.5, 50.0)) == pytest.approx(8.0 - 1.0 - 0.5)
    assert rs.model_score(6.0, 0.5, 0.0, (2.0, 1.0, 0.0)) == pytest.approx(11.5)


@pytest.mark.parametrize(
    ("model", "high_uq", "expected"),
    [
        (SOTA, True, 1.3),
        (LOCAL, True, 0.6),
        (OTHER, True, 1.0),
        (SOTA, False, 1.0),
        (LOCAL, False, 1.1),
        (OTHER, False, 1.0),
    ],
)
def test_risk_factor(model, high_uq, expected):
    assert rs._risk_factor(model, high_uq, RISKS) == expected


class _Breakers:
    def __init__(self, statuses=None, unavailable=(), broken=False):
        self.statuses = statuses or {}
        self.unavailable = set(unavailable)
        self.broken = broken

    def is_available(self, model):
        return model not in self.unavailable

    def get_status(self, model):
        if self.broken:
            raise RuntimeError("registry down")
        return self.statuses.get(model, {})


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ({"state": "open"}, 0.0),
        ({"state": "half-open"}, 0.5),
        ({}, 1.0),  # defaults: fechado, sem falhas
        ({"state": "closed", "fail_counter": 1, "fail_max": 5}, 0.9),
        ({"state": "closed", "fail_counter": 1}, 0.9),  # fail_max padrão 5
        ({"state": "closed", "fail_counter": 5, "fail_max": 5}, 0.5),
        ({"state": "closed", "fail_counter": 9, "fail_max": 5}, 0.3),  # piso
        ({"state": "closed", "fail_counter": 0, "fail_max": 5}, 1.0),
        ({"state": "closed", "fail_counter": 3, "fail_max": 0}, 1.0),
        ({"state": "closed", "fail_counter": 2, "fail_max": 1}, 0.3),
    ],
)
def test_circuit_breaker_penalty(monkeypatch, status, expected):
    monkeypatch.setattr(rs, "get_circuit_breaker_manager", lambda: _Breakers({"m": status}))
    assert rs._get_circuit_breaker_penalty("m") == pytest.approx(expected)


def test_circuit_breaker_penalty_on_error(monkeypatch):
    monkeypatch.setattr(rs, "get_circuit_breaker_manager", lambda: _Breakers(broken=True))
    assert rs._get_circuit_breaker_penalty("m") == 1.0


def test_filter_candidates(monkeypatch):
    monkeypatch.setattr(rs, "model_supports_vision", lambda m: "vl" in m or "4o" in m)
    monkeypatch.setattr(rs, "is_vision_only_model", lambda m: "vl" in m)
    models = ["a/text", "b/qwen-vl", "c/gpt-4o"]
    assert rs._filter_candidates(models, "text", _Breakers()) == ["a/text", "c/gpt-4o"]
    assert rs._filter_candidates(models, "vision", _Breakers()) == ["b/qwen-vl", "c/gpt-4o"]
    assert rs._filter_candidates(models, "multimodal", _Breakers(unavailable={"c/gpt-4o"})) == ["b/qwen-vl"]
    assert rs._filter_candidates(["a/text"], "vision", _Breakers()) == ["ollama/qwen3-vl:4b"]
    assert rs._filter_candidates(["b/qwen-vl"], "text", _Breakers()) == ["ollama/gemma3:4b"]
    assert rs._filter_candidates(models, "audio", _Breakers(unavailable={"a/text"})) == ["b/qwen-vl", "c/gpt-4o"]
    everything_open = _Breakers(unavailable=set(models))
    assert rs._filter_candidates(models, "audio", everything_open) == models  # todos abertos: mantém a lista


@pytest.fixture
def strategy(monkeypatch):
    env = SimpleNamespace(
        qualities={SOTA: 8.0, LOCAL: 6.0, OTHER: 7.0},
        ema={},
        threshold="0.45",
        breakers=_Breakers(),
        emergency=None,
    )
    monkeypatch.setattr(
        rs,
        "get_cascade_detector",
        lambda: SimpleNamespace(
            is_emergency_mode=env.emergency is not None, get_emergency_fallback=lambda: env.emergency
        ),
    )
    monkeypatch.setattr(rs, "get_snapshot", lambda: {})
    monkeypatch.setattr(rs, "sample_metrics_from_snapshot", lambda snap: dict(env.qualities))
    monkeypatch.setattr(rs, "load_ema_snapshot", lambda modality: env.ema)
    monkeypatch.setattr(rs, "get_circuit_breaker_manager", lambda: env.breakers)
    monkeypatch.setattr(rs, "is_vision_only_model", lambda m: False)
    monkeypatch.setattr(
        rs,
        "settings",
        SimpleNamespace(
            get=lambda key, default=None: env.threshold if key == "UNCERTAINTY_THRESHOLD" else default,
            RISK_FACTOR_SOTA_HIGH_UQ=RISKS[0],
            RISK_FACTOR_LOCAL_HIGH_UQ=RISKS[1],
            RISK_FACTOR_LOCAL_LOW_UQ=RISKS[2],
        ),
    )
    return env


W = {"w_quality": 1.0, "w_latency": 0.5, "w_cost": 50.0}


def test_top2_low_uncertainty_uses_heuristic_latency_cost(strategy):
    # S(sota) = 8 - 2*0,5 - 0,01*50 = 6,5; S(local) = 6*1,1 - 0,25 - 5e-5 = 6,34995; S(other) = 7 - 1 - 5e-5
    assert rs.choose_top2_models([OTHER, LOCAL, SOTA], W, "q", uncertainty_score=0.45) == [SOTA, LOCAL]


def test_top2_high_uncertainty_applies_risk_factors(strategy):
    # u > 0,45: S(sota) = 8*1,3 - 1,5 = 8,9; S(other) = 5,99995; S(local) = 3,6 - 0,25 = 3,34995
    assert rs.choose_top2_models([LOCAL, OTHER, SOTA], W, "q", uncertainty_score=0.46) == [SOTA, OTHER]
    strategy.threshold = "0.5"
    assert rs.choose_top2_models([LOCAL, OTHER, SOTA], W, "q", uncertainty_score=0.46) == [SOTA, LOCAL]


def test_top2_uses_mature_shared_ema_and_default_weights(strategy):
    # EMA madura do sota: 10 s e 0,05 USD -> S = 8 - 5 - 2,5 = 0,5 (pesos padrão quando ausentes)
    strategy.ema = {SOTA: {"ema_latency": 10.0, "ema_cost": 0.05, "updates": 3}}
    assert rs.choose_top2_models([SOTA, LOCAL, OTHER], {}, "q") == [LOCAL, OTHER]
    strategy.ema[SOTA]["updates"] = 2  # imatura: volta às heurísticas
    assert rs.choose_top2_models([SOTA, LOCAL, OTHER], {}, "q") == [SOTA, LOCAL]


def test_top2_breaker_penalty_and_unknown_quality(strategy):
    strategy.breakers = _Breakers({SOTA: {"state": "half-open"}})  # 8*0,5 - 1,5 = 2,5
    assert rs.choose_top2_models([SOTA, LOCAL, OTHER], W, "q") == [LOCAL, OTHER]
    strategy.breakers = _Breakers()
    strategy.qualities = {}  # sem amostra: qualidade 5 para todos
    assert rs.choose_top2_models([OTHER, LOCAL], W, "q") == [LOCAL, OTHER]


def test_emergency_mode_short_circuits(strategy):
    strategy.emergency = "ollama/phi4:latest"
    assert rs.choose_top2_models([SOTA, LOCAL], W, "q") == ["ollama/phi4:latest"]
    strategy.emergency = ""
    assert rs.choose_top2_models([SOTA, LOCAL], W, "q") == [SOTA, LOCAL]
