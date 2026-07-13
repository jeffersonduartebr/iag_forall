# Objective: Test coverage for router strategy edge cases and resilience branches.
"""Additional tests for router strategy helper branches."""

from types import SimpleNamespace

import pytest

from app import router_strategy as rs


def test_router_strategy_helper_predicates_and_penalties(monkeypatch):
    """Helper predicates and circuit-breaker penalty logic should cover normal and degraded states."""
    assert rs._is_sota("openai/gpt-5") is True
    assert rs._is_sota("ollama/gemma3:4b") is False
    assert rs._is_local("ollama/qwen3.5:4b") is True
    assert rs._is_local("openai/gpt-4o") is False

    class _Manager:
        def __init__(self, status):
            self._status = status

        def get_status(self, model):
            return self._status

    monkeypatch.setattr(rs, "get_circuit_breaker_manager", lambda: _Manager({"state": "open"}))
    assert rs._get_circuit_breaker_penalty("m") == 0.0

    monkeypatch.setattr(rs, "get_circuit_breaker_manager", lambda: _Manager({"state": "half-open"}))
    assert rs._get_circuit_breaker_penalty("m") == 0.5

    monkeypatch.setattr(rs, "get_circuit_breaker_manager", lambda: _Manager({"state": "closed", "fail_counter": 2, "fail_max": 4}))
    assert rs._get_circuit_breaker_penalty("m") == pytest.approx(0.75)

    monkeypatch.setattr(rs, "get_circuit_breaker_manager", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert rs._get_circuit_breaker_penalty("m") == 1.0


def test_choose_top2_models_emergency_and_filters(monkeypatch):
    """Strategy selection should honor emergency mode and modality-specific hard filters."""
    monkeypatch.setattr(rs, "get_snapshot", lambda: {})
    monkeypatch.setattr(rs, "sample_metrics_from_snapshot", lambda snapshot: {})
    monkeypatch.setattr(rs, "settings", SimpleNamespace(
        get=lambda key, default=None: 0.45 if key == "UNCERTAINTY_THRESHOLD" else default,
        RISK_FACTOR_SOTA_HIGH_UQ=1.3,
        RISK_FACTOR_LOCAL_HIGH_UQ=0.6,
        RISK_FACTOR_LOCAL_LOW_UQ=1.1,
    ))

    class _Cascade:
        is_emergency_mode = True

        def get_emergency_fallback(self):
            return "ollama/emergency:latest"

    monkeypatch.setattr(rs, "get_cascade_detector", lambda: _Cascade())
    assert rs.choose_top2_models(["ollama/a"], {"w_quality": 1, "w_latency": 1, "w_cost": 1}, "q") == ["ollama/emergency:latest"]

    class _CascadeOff:
        is_emergency_mode = False

    class _BreakerManager:
        def is_available(self, model):
            return not model.endswith("blocked")

        def get_status(self, model):
            return {"state": "closed", "fail_counter": 0, "fail_max": 5}

    monkeypatch.setattr(rs, "get_cascade_detector", lambda: _CascadeOff())
    monkeypatch.setattr(rs, "get_circuit_breaker_manager", lambda: _BreakerManager())

    out = rs.choose_top2_models(
        ["ollama/gemma3:4b", "ollama/llava:7b", "openai/gpt-5.1-blocked"],
        {"w_quality": 1.0, "w_latency": 1.0, "w_cost": 1.0},
        "img",
        modality="vision",
    )
    assert "ollama/llava:7b" in out
    assert "openai/gpt-5.1-blocked" not in out

    text_out = rs.choose_top2_models(
        ["ollama/llava:7b", "ollama/moondream:latest"],
        {"w_quality": 1.0, "w_latency": 1.0, "w_cost": 1.0},
        "texto simples",
        modality="text",
    )
    assert text_out == ["ollama/gemma3:4b"]


def test_choose_top2_models_handles_all_breakers_open(monkeypatch):
    """If every candidate is unavailable, the strategy should fall back to the original list."""
    monkeypatch.setattr(rs, "get_snapshot", lambda: {"openai/gpt-5.1": {"avg_latency": 2.0, "avg_cost": 0.01}})
    monkeypatch.setattr(rs, "sample_metrics_from_snapshot", lambda snapshot: {"openai/gpt-5.1": 9.0})
    monkeypatch.setattr(rs, "settings", SimpleNamespace(
        get=lambda key, default=None: 0.45 if key == "UNCERTAINTY_THRESHOLD" else default,
        RISK_FACTOR_SOTA_HIGH_UQ=1.3,
        RISK_FACTOR_LOCAL_HIGH_UQ=0.6,
        RISK_FACTOR_LOCAL_LOW_UQ=1.1,
    ))
    monkeypatch.setattr(rs, "get_cascade_detector", lambda: SimpleNamespace(is_emergency_mode=False))

    class _BreakerManager:
        def is_available(self, model):
            return False

        def get_status(self, model):
            return {"state": "closed", "fail_counter": 0, "fail_max": 5}

    monkeypatch.setattr(rs, "get_circuit_breaker_manager", lambda: _BreakerManager())
    out = rs.choose_top2_models(["openai/gpt-5.1"], {"w_quality": 1.0, "w_latency": 0.0, "w_cost": 0.0}, "q")
    assert out == ["openai/gpt-5.1"]
