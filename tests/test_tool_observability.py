# -*- coding: utf-8 -*-
"""Testes do reward/observabilidade de turnos de tool calling (roadmap item #2)."""

from __future__ import annotations

import pytest
from app.services import tool_observability as to


def _call(name: str, arguments):
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}


def test_tool_call_quality_all_valid():
    calls = [_call("get_weather", '{"city": "Paris"}'), _call("search", '{"q": "x"}')]
    assert to.tool_call_quality(calls) == 1.0


def test_tool_call_quality_partial_invalid():
    calls = [_call("get_weather", '{"city": "Paris"}'), _call("bad", "{not json")]
    assert to.tool_call_quality(calls) == 0.5


def test_tool_call_quality_empty_or_malformed_scores_zero():
    assert to.tool_call_quality([]) == 0.0
    assert to.tool_call_quality(None) == 0.0
    assert to.tool_call_quality("not-a-list") == 0.0
    assert to.tool_call_quality([{"function": {"name": "x", "arguments": "["}}]) == 0.0


def test_tool_call_quality_no_argument_call_is_wellformed():
    assert to.tool_call_quality([_call("ping", "")]) == 1.0
    assert to.tool_call_quality([_call("ping", "{}")]) == 1.0
    assert to.tool_call_quality([_call("ping", {"already": "parsed"})]) == 1.0


def test_arguments_valid_rejects_non_dict_function():
    assert to._arguments_valid({"function": "nope"}) is False
    assert to._arguments_valid({"function": {"name": "x", "arguments": 123}}) is False


def test_record_tool_turn_feeds_bandit_and_returns_quality(monkeypatch):
    seen = {}

    def fake_compute_reward(model, quality, latency_s, cost_per_1k=None):
        seen["compute"] = (model, quality, latency_s, cost_per_1k)
        return 0.42

    def fake_bandit_update(model, query, reward, modality="text"):
        seen["update"] = {"model": model, "query": query, "reward": reward, "modality": modality}

    import app.bandits as bandits

    monkeypatch.setattr(bandits, "compute_reward", fake_compute_reward)
    monkeypatch.setattr(bandits, "bandit_update", fake_bandit_update)

    quality = to.record_tool_turn(
        chosen_model="openai/gpt-4o",
        tool_calls=[_call("get_weather", '{"city": "Paris"}')],
        modality="text",
        latency_s=1.5,
        cost_val=0.001,
        query="weather in paris?",
        conversation_depth=3,
    )

    assert quality == 1.0
    # quality 1.0 é escalado para 0..10 antes de compute_reward
    assert seen["compute"] == ("openai/gpt-4o", 10.0, 1.5, 0.001)
    assert seen["update"]["model"] == "openai/gpt-4o"
    assert seen["update"]["reward"] == 0.42
    assert seen["update"]["modality"] == "text"


def test_record_tool_turn_never_raises_on_bandit_failure(monkeypatch):
    import app.bandits as bandits

    def boom(*a, **k):
        raise RuntimeError("bandit down")

    monkeypatch.setattr(bandits, "compute_reward", boom)
    # Não deve propagar: reward é best-effort.
    quality = to.record_tool_turn(
        chosen_model="m",
        tool_calls=[_call("f", "{}")],
        query="q",
    )
    assert quality == 1.0


def test_tool_metrics_registered_on_shared_registry():
    from app.observability import registry

    names = set(registry._names_to_collectors.keys())  # type: ignore[attr-defined]
    assert "router_tool_calls_total" in names
    assert "router_tool_call_args_invalid_total" in names


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
