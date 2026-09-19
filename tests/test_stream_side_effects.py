# Objective: Test coverage for usage and feedback side effects of streamed answers.
"""services.query_http._record_stream_side_effects: usage accounting and feedback dispatch."""

from types import SimpleNamespace

import pytest
from app.schemas import QueryRequest
from app.services import query_http


@pytest.fixture
def effects(monkeypatch):
    usage, feedback = [], []
    monkeypatch.setattr("app.providers_async.get_model_cost", lambda model, p, c: 0.002)
    monkeypatch.setattr("app.utils.pricing.impute_local_cost", lambda seconds: seconds * 0.001)
    monkeypatch.setattr("app.services.governance_runtime.schedule_runtime_usage", lambda **kw: usage.append(kw))
    monkeypatch.setattr("app.tasks.task_process_feedback", SimpleNamespace(delay=lambda **kw: feedback.append(kw)))
    return usage, feedback


def _req():
    return QueryRequest(query="pergunta", modality="text", tenant_id="t1")


def test_local_stream_imputes_occupancy_in_feedback_cost(effects):
    usage, feedback = effects
    query_http._record_stream_side_effects(_req(), "ollama/gemma3:4b", "resposta", 2.0, 10, 20)

    assert usage == [{"tenant_id": "t1", "cost_usd": 0.002, "tokens_in": 10, "tokens_out": 20, "requests": 1}]
    (fb,) = feedback
    assert fb["cost_val"] == pytest.approx(0.004)  # caixa + 2 s de ocupação imputada
    assert (fb["chosen_model"], fb["latency_s"], fb["raw_payload"]) == ("ollama/gemma3:4b", 2.0, {"streamed": True})


def test_cloud_stream_uses_cash_cost_and_skips_empty_answers(effects):
    usage, feedback = effects
    query_http._record_stream_side_effects(_req(), "openai/gpt-4o", "ok", 2.0, None, None)
    assert feedback[0]["cost_val"] == 0.002 and feedback[0]["prompt_tokens"] == 0

    query_http._record_stream_side_effects(_req(), "openai/gpt-4o", "   ", 2.0, 1, 1)
    assert len(usage) == 2 and len(feedback) == 1  # uso registrado, feedback não


def test_side_effect_failures_are_contained(monkeypatch, effects):
    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr("app.providers_async.get_model_cost", boom)
    monkeypatch.setattr("app.utils.pricing.impute_local_cost", boom)
    monkeypatch.setattr("app.services.governance_runtime.schedule_runtime_usage", boom)
    monkeypatch.setattr("app.tasks.task_process_feedback", SimpleNamespace(delay=boom))
    query_http._record_stream_side_effects(_req(), "ollama/x", "resposta", 1.0, 1, 1)
