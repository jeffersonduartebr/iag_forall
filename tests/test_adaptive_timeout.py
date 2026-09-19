# Objective: Test coverage for adaptive per-model request timeouts.
"""Adaptive timeouts: EMA lookup (Redis then DB), per-family defaults and clamping."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app import adaptive_timeout as at


def _settings(enabled=True, min_t=30, max_t=1200, mult=2.0, reasoning=3.0):
    return SimpleNamespace(
        ADAPTIVE_TIMEOUT_ENABLED=enabled,
        MIN_TIMEOUT=min_t,
        MAX_TIMEOUT=max_t,
        ADAPTIVE_TIMEOUT_MULTIPLIER=mult,
        ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER=reasoning,
        get=lambda key, default=None: {"REQUEST_TIMEOUT_SECONDS": 90}.get(key, default),
    )


@pytest.fixture
def adaptive(monkeypatch):
    monkeypatch.setattr(at, "settings", _settings())
    monkeypatch.setattr(at, "get_ema_latency", lambda model, modality="text": None)


def _engine_returning(row):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = row
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    return engine


def test_model_families():
    assert at.is_reasoning_model("openai/o3-mini")
    assert at.is_reasoning_model("anthropic/claude-opus-4")
    assert not at.is_reasoning_model("openai/gpt-4o")
    assert at.is_fast_local_model("ollama/gemma3:4b")
    assert not at.is_fast_local_model("gemma-cloud")  # sem "ollama" não é local
    assert not at.is_fast_local_model("ollama/qwen2.5:14b")


def test_disabled_returns_request_timeout(monkeypatch):
    monkeypatch.setattr(at, "settings", _settings(enabled=False))
    assert at.calculate_adaptive_timeout("ollama/gemma3:4b") == 90


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("ollama/gemma3:4b", 30),  # 2 s × 2 × 1,2 = 4,8 → piso de 30
        ("ollama/deepseek-r1", 108),  # raciocínio: 30 s × 3 × 1,2
        ("ollama/qwen2.5:14b", 30),  # local padrão: 5 s × 2 × 1,2 = 12 → piso
        ("openai/gpt-4o", 30),  # nuvem: 10 s × 2 × 1,2 = 24 → piso
    ],
)
def test_default_latency_per_family(adaptive, model, expected):
    assert at.calculate_adaptive_timeout(model) == expected


def test_uses_ema_and_clamps(adaptive):
    assert at.calculate_adaptive_timeout("openai/gpt-4o", ema_latency=50.0) == 120
    assert at.calculate_adaptive_timeout("openai/o1", ema_latency=50.0) == 180
    assert at.calculate_adaptive_timeout("openai/o1", ema_latency=10_000.0) == 1200


@given(
    ema=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1e6, allow_nan=False)),
    model=st.sampled_from(["ollama/gemma3:4b", "openai/o3", "ollama/qwen", "gemini/flash"]),
    min_t=st.integers(min_value=1, max_value=300),
    span=st.integers(min_value=0, max_value=3000),
)
def test_timeout_always_within_bounds(ema, model, min_t, span):
    original = (at.settings, at.get_ema_latency)
    at.settings = _settings(min_t=min_t, max_t=min_t + span)
    at.get_ema_latency = lambda model, modality="text": None
    try:
        timeout = at.calculate_adaptive_timeout(model, ema_latency=ema)
    finally:
        at.settings, at.get_ema_latency = original
    assert min_t <= timeout <= min_t + span


def test_user_timeout_is_clamped(adaptive):
    assert at.get_timeout_for_request("m", user_timeout=5) == 30
    assert at.get_timeout_for_request("m", user_timeout=5000) == 1200
    assert at.get_timeout_for_request("m", user_timeout=45) == 45
    assert at.get_timeout_for_request("ollama/deepseek-r1") == 108


def test_ema_latency_prefers_redis_then_db(monkeypatch, fake_redis):
    monkeypatch.setattr(at, "get_redis", lambda: fake_redis)
    engine = _engine_returning((7.5,))
    monkeypatch.setattr("app.db.get_engine", lambda: engine)

    assert at.get_ema_latency("m", "text") == 7.5  # DB, depois grava no Redis
    assert float(fake_redis.get("ema_latency:text:m")) == 7.5
    assert 0 < fake_redis.ttl("ema_latency:text:m") <= at.EMA_CACHE_TTL_S

    engine.connect.side_effect = AssertionError("não deveria consultar o DB")
    assert at.get_ema_latency("m", "text") == 7.5


def test_ema_latency_missing_everywhere(monkeypatch, fake_redis):
    monkeypatch.setattr(at, "get_redis", lambda: fake_redis)
    monkeypatch.setattr("app.db.get_engine", lambda: _engine_returning(None))
    assert at.get_ema_latency("m") is None

    monkeypatch.setattr("app.db.get_engine", MagicMock(side_effect=RuntimeError("db down")))
    assert at.get_ema_latency("m") is None


def test_redis_helpers_tolerate_failures(monkeypatch):
    monkeypatch.setattr(at, "get_redis", lambda: None)
    assert at._get_ema_from_redis("m", "text") is None
    at._set_ema_to_redis("m", "text", 1.0)

    broken = MagicMock()
    broken.get.side_effect = ConnectionError
    broken.setex.side_effect = ConnectionError
    monkeypatch.setattr(at, "get_redis", lambda: broken)
    assert at._get_ema_from_redis("m", "text") is None
    at._set_ema_to_redis("m", "text", 1.0)


def test_timeout_status(monkeypatch, adaptive):
    status = at.get_timeout_status("openai/o3")
    assert status["is_reasoning_model"] is True
    assert status["multiplier"] == 3.0
    assert status["calculated_timeout"] == 108
    assert (status["min_timeout"], status["max_timeout"]) == (30, 1200)
