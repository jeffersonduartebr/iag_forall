"""Unit tests for query runtime orchestration."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def _allow_guardrails():
    return SimpleNamespace(allowed=True, reasons=[])


def _allow_budget():
    return SimpleNamespace(allowed=True)


def _base_result():
    return {
        "answer": "raw-answer",
        "model": "ollama/test",
        "modality": "text",
        "latency_s": 0.2,
        "cost_per_1k": 0.01,
        "metadata": {"raw_payload": "{}", "prompt_tokens": 11, "completion_tokens": 7},
        "route": {"chosen_model": "ollama/test", "objectives": {}, "pareto_front": [], "explanation": "", "fallback": {"used": False, "models_tried": [], "errors": []}},
        "candidates": [],
    }


def _request(**overrides):
    values = {
        "query": "Como funciona?",
        "tenant_id": "school-1",
        "modality": "text",
        "image_b64": None,
        "images": [],
        "policy_version": None,
        "experiment_id": None,
        "user_key": None,
        "system_prompt": "",
        "enable_rag_for_answer": False,
        "enable_rag_for_image": False,
        "max_tokens": None,
        "temperature": None,
        "rag_modality": "text",
        "use_cache": True,
        "timeout_seconds": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _settings_map(monkeypatch, qr, mapping):
    """Patch dynamic settings lookups through the stable get() surface."""
    monkeypatch.setattr(qr.settings, "get", lambda key, fallback=None: mapping.get(key, fallback))


@pytest.mark.asyncio
async def test_process_query_request_blocks_guardrails(monkeypatch):
    """Blocked input should short-circuit before routing."""
    from app.services import query_runtime as qr

    monkeypatch.setattr(qr, "check_input_guardrails", lambda _q: SimpleNamespace(allowed=False, reasons=["policy"]))

    with pytest.raises(HTTPException) as exc:
        await qr.process_query_request(_request())
    assert exc.value.status_code == 400
    assert exc.value.detail["category"] == "guardrail_block"


@pytest.mark.asyncio
async def test_process_query_request_blocks_budget(monkeypatch):
    """Budget denial should map to HTTP 429 with budget detail."""
    from app.services import query_runtime as qr

    monkeypatch.setattr(qr, "check_input_guardrails", lambda _q: _allow_guardrails())
    monkeypatch.setattr(
        qr,
        "check_tenant_budget",
        lambda _tenant: SimpleNamespace(
            allowed=False,
            reason="daily_limit",
            daily_spent=1.0,
            monthly_spent=3.0,
            daily_limit=1.0,
            monthly_limit=10.0,
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await qr.process_query_request(_request())
    assert exc.value.status_code == 429
    assert exc.value.detail["category"] == "tenant_budget_exceeded"


@pytest.mark.asyncio
async def test_process_query_request_applies_policy_and_variant(monkeypatch):
    """Assigned experiments should override selected policy metadata."""
    from app.services import query_runtime as qr

    monkeypatch.setattr(qr, "check_input_guardrails", lambda _q: _allow_guardrails())
    monkeypatch.setattr(qr, "check_tenant_budget", lambda _tenant: _allow_budget())
    monkeypatch.setattr(qr, "get_active_policy", lambda: {"version": "policy-v1"})
    _settings_map(monkeypatch, qr, {"AB_TESTING_ENABLED": "1"})
    monkeypatch.setattr(
        qr,
        "get_ab_test_manager",
        lambda: SimpleNamespace(get_assignment=lambda exp_id, user_key: ("B", {"policy_version": "policy-v2"})),
    )
    monkeypatch.setattr(qr, "sanitize_output_guardrails", lambda answer: (f"clean:{answer}", ["safe"]))

    async def _route(**kwargs):
        return _base_result()

    monkeypatch.setattr(qr, "route_and_answer", _route)
    out = await qr.process_query_request(_request(experiment_id="exp-1", user_key="teacher-1"))

    assert out["result"]["answer"] == "clean:raw-answer"
    assert out["result"]["metadata"]["policy_version"] == "policy-v2"
    assert out["result"]["metadata"]["experiment_variant"]["name"] == "B"
    assert out["result"]["metadata"]["guardrail_output_tags"] == ["safe"]


@pytest.mark.asyncio
async def test_process_query_request_uses_first_image_for_vision(monkeypatch):
    """Image requests should switch modality and pass the first image to routing."""
    from app.services import query_runtime as qr

    captured = {}
    monkeypatch.setattr(qr, "check_input_guardrails", lambda _q: _allow_guardrails())
    monkeypatch.setattr(qr, "check_tenant_budget", lambda _tenant: _allow_budget())
    monkeypatch.setattr(qr, "get_active_policy", lambda: None)
    _settings_map(monkeypatch, qr, {"AB_TESTING_ENABLED": "0"})
    monkeypatch.setattr(qr, "sanitize_output_guardrails", lambda answer: (answer, []))

    async def _route(**kwargs):
        captured.update(kwargs)
        result = _base_result()
        result["modality"] = kwargs["modality"]
        return result

    monkeypatch.setattr(qr, "route_and_answer", _route)
    out = await qr.process_query_request(_request(images=["img1"], use_cache=False))

    assert captured["modality"] == "vision"
    assert captured["image_b64"] == "img1"
    assert out["modality"] == "vision"


def test_record_query_side_effects_dispatches_feedback_usage_and_experiment(monkeypatch):
    """Side effects should dispatch feedback, usage accounting, and experiment metrics."""
    from app.services import query_runtime as qr

    calls = {"delay": None, "usage": None, "metrics": []}
    monkeypatch.setattr(
        qr.task_process_feedback,
        "delay",
        lambda **kwargs: calls.__setitem__("delay", kwargs),
    )
    monkeypatch.setattr(
        qr,
        "record_tenant_usage",
        lambda tenant_id, cost_usd, tokens_in, tokens_out, requests: calls.__setitem__(
            "usage",
            (tenant_id, cost_usd, tokens_in, tokens_out, requests),
        ),
    )
    _settings_map(monkeypatch, qr, {"AB_TESTING_ENABLED": "1"})
    monkeypatch.setattr(
        qr,
        "get_ab_test_manager",
        lambda: SimpleNamespace(record_result=lambda *args: calls["metrics"].append(args)),
    )

    result = _base_result()
    result["metadata"]["experiment_variant"] = {"name": "B"}
    result["metadata"]["quality"] = 0.8
    qr.record_query_side_effects(_request(experiment_id="exp-1"), result, "img")

    assert calls["delay"]["image_b64"] == "img"
    assert calls["usage"] == ("school-1", 0.01, 11, 7, 1)
    assert len(calls["metrics"]) == 3


def test_record_query_side_effects_tolerates_feedback_failures(monkeypatch):
    """Side effects should not raise when downstream hooks fail."""
    from app.services import query_runtime as qr

    monkeypatch.setattr(qr.task_process_feedback, "delay", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("queue")))
    monkeypatch.setattr(qr, "record_tenant_usage", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("usage")))
    _settings_map(monkeypatch, qr, {"AB_TESTING_ENABLED": "1"})
    monkeypatch.setattr(qr, "get_ab_test_manager", lambda: (_ for _ in ()).throw(RuntimeError("ab")))

    result = _base_result()
    result["metadata"]["experiment_variant"] = {"name": "B"}
    qr.record_query_side_effects(_request(experiment_id="exp-1"), result, None)


def test_governance_runtime_adapters_delegate(monkeypatch):
    """Runtime governance adapter should isolate roadmap imports behind thin wrappers."""
    from app.services import governance_runtime as gr

    monkeypatch.setattr(gr, "ensure_roadmap_tables", lambda: "ok")
    monkeypatch.setattr(gr, "get_active_policy", lambda: {"version": "v1"})
    monkeypatch.setattr(gr, "check_tenant_budget", lambda tenant_id: {"tenant_id": tenant_id, "allowed": True})

    calls = {}

    def _record(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(gr, "record_tenant_usage", _record)

    assert gr.ensure_runtime_support_tables() == "ok"
    assert gr.get_runtime_active_policy()["version"] == "v1"
    assert gr.check_runtime_budget("school-1")["tenant_id"] == "school-1"
    gr.record_runtime_usage(tenant_id="school-1", cost_usd=1.0, tokens_in=2, tokens_out=3, requests=1)
    assert calls["tenant_id"] == "school-1"
