# Objective: Test coverage for query runtime behavior and regressions.
"""Unit tests for query runtime orchestration."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock


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
        "metadata": {"raw_payload": "{}", "prompt_tokens": 11, "completion_tokens": 7, "uncertainty_score": 0.2},
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


class _Metric:
    def __init__(self):
        self.values = []

    def labels(self, **kwargs):
        self.values.append(("labels", kwargs))
        return self

    def inc(self, value=1):
        self.values.append(("inc", value))

    def set(self, value):
        self.values.append(("set", value))


def test_enrich_result_reliability_marks_low_confidence_answer_for_review():
    """Low-confidence, unsupported answers should be abstained and routed to review."""
    from app.services import query_runtime as qr

    result = _base_result()
    result["answer"] = "talvez"
    result["metadata"].update(
        {
            "uncertainty_score": 0.92,
            "grounded": False,
            "retrieval_mode": "light_retrieval",
            "workload_class": "knowledge_lookup",
            "guardrail_output_tags": [],
        }
    )
    enriched = qr.enrich_result_reliability(result)

    assert enriched["abstained"] is True
    assert enriched["verification_status"] == "unsupported"
    assert enriched["review_status"] == "needs_review"
    assert "evidencia suficiente" in enriched["answer"].lower()


def test_classify_query_workload_recognizes_fast_and_heavy_paths():
    """Workload classification should distinguish simple, retrieval-heavy, reasoning, and vision traffic."""
    from app.services import query_runtime as qr

    assert qr.classify_query_workload(_request(query="Quanto é 2+2?"), modality="text", image_input=None) == "simple_text"
    assert (
        qr.classify_query_workload(
            _request(query="Cite a policy e o documento de referência sobre avaliação escolar."),
            modality="text",
            image_input=None,
        )
        == "knowledge_lookup"
    )
    assert (
        qr.classify_query_workload(
            _request(query="Explique passo a passo como resolver uma equação do segundo grau."),
            modality="text",
            image_input=None,
        )
        == "reasoning"
    )
    assert qr.classify_query_workload(_request(query="O que há na imagem?"), modality="vision", image_input="img") == "vision"


def test_apply_query_runtime_profile_bypasses_rag_for_simple_text(monkeypatch):
    """Simple text traffic should use the fast path when performance mode and bypass are enabled."""
    from app.services import query_runtime as qr

    _settings_map(
        monkeypatch,
        qr,
        {
            "ROUTER_PERF_MODE": "1",
            "ROUTER_SIMPLE_QUERY_MAX_TOKENS": "128",
            "RAG_SIMPLE_QUERY_BYPASS_ENABLED": "1",
        },
    )
    profile = qr.apply_query_runtime_profile(
        _request(query="Quanto é 2+2?", enable_rag_for_answer=True, max_tokens=600),
        modality="text",
        image_input=None,
    )

    assert profile["workload_class"] == "simple_text"
    assert profile["perf_mode_enabled"] is True
    assert profile["use_rag"] is False
    assert profile["max_tokens"] == 128
    assert profile["runtime_hints"]["retrieval_mode"] == "no_retrieval"
    assert profile["runtime_hints"]["max_fallbacks"] == 1
    assert profile["runtime_hints"]["needs_retrieval"] is False
    assert profile["runtime_hints"]["interactive_priority"] == "high"
    assert profile["runtime_hints"]["provider_timeout_seconds"] == 20
    assert profile["runtime_hints"]["sync_deadline_seconds"] == 25


def test_apply_query_runtime_profile_keeps_vision_rag(monkeypatch):
    """Vision traffic should not silently lose retrieval just because simple-query bypass is enabled."""
    from app.services import query_runtime as qr

    _settings_map(
        monkeypatch,
        qr,
        {
            "ROUTER_PERF_MODE": "1",
            "ROUTER_SIMPLE_QUERY_MAX_TOKENS": "64",
            "RAG_SIMPLE_QUERY_BYPASS_ENABLED": "1",
        },
    )
    profile = qr.apply_query_runtime_profile(
        _request(query="O que aparece aqui?", enable_rag_for_image=True, max_tokens=256),
        modality="vision",
        image_input="img",
    )

    assert profile["workload_class"] == "vision"
    assert profile["use_rag"] is True
    assert profile["max_tokens"] == 256
    assert profile["runtime_hints"]["retrieval_mode"] == "full_retrieval"


def test_apply_query_runtime_profile_uses_light_retrieval_for_knowledge_lookup(monkeypatch):
    """Knowledge-lookup traffic should request a lighter retrieval profile by default."""
    from app.services import query_runtime as qr

    _settings_map(
        monkeypatch,
        qr,
        {
            "ROUTER_PERF_MODE": "1",
            "RAG_LIGHT_TOP_K": "2",
            "RAG_LIGHT_CONTEXT_TOKEN_BUDGET": "320",
            "RERANK_ENABLED_FOR_LIGHT_RETRIEVAL": "0",
        },
    )
    profile = qr.apply_query_runtime_profile(
        _request(query="Cite a policy oficial sobre recuperação paralela.", enable_rag_for_answer=False, max_tokens=700),
        modality="text",
        image_input=None,
    )

    assert profile["workload_class"] == "knowledge_lookup"
    assert profile["use_rag"] is True
    assert profile["runtime_hints"]["retrieval_mode"] == "light_retrieval"
    assert profile["runtime_hints"]["rag_top_k"] == 2
    assert profile["runtime_hints"]["rag_context_token_budget"] == 320
    assert profile["runtime_hints"]["rag_rerank_enabled"] is False
    assert profile["runtime_hints"]["needs_retrieval"] is True
    assert profile["runtime_hints"]["interactive_priority"] == "high"
    assert profile["runtime_hints"]["provider_timeout_seconds"] == 35
    assert profile["runtime_hints"]["sync_deadline_seconds"] == 40


def test_apply_query_runtime_profile_promotes_source_seeking_queries_to_full_retrieval(monkeypatch):
    """Source-heavy knowledge lookups should keep full retrieval despite the cheap path defaults."""
    from app.services import query_runtime as qr

    _settings_map(
        monkeypatch,
        qr,
        {
            "ROUTER_PERF_MODE": "1",
            "RAG_LIGHT_TOP_K": "2",
            "RAG_LIGHT_CONTEXT_TOKEN_BUDGET": "320",
            "RAG_FULL_CONTEXT_TOKEN_BUDGET": "900",
        },
    )
    profile = qr.apply_query_runtime_profile(
        _request(query="Cite as fontes e o artigo oficial sobre avaliação formativa.", enable_rag_for_answer=False),
        modality="text",
        image_input=None,
    )

    assert profile["workload_class"] == "knowledge_lookup"
    assert profile["runtime_hints"]["retrieval_mode"] == "full_retrieval"
    assert profile["runtime_hints"]["rag_context_token_budget"] == 900
    assert profile["runtime_hints"]["needs_rerank"] is True


def test_classify_query_workload_respects_disabled_classifier(monkeypatch):
    """Disabling the classifier should fall back to the safest reasoning profile."""
    from app.services import query_runtime as qr

    _settings_map(monkeypatch, qr, {"ROUTER_QUERY_CLASSIFIER_ENABLED": "0"})
    assert qr.classify_query_workload(_request(query="Quanto é 2+2?"), modality="text", image_input=None) == "reasoning"


def test_enrich_result_reliability_marks_supported_grounded_answers():
    """Grounded confident answers should remain supported and auto-approved."""
    from app.services import query_runtime as qr

    result = _base_result()
    result["answer"] = "Resposta com apoio documental."
    result["metadata"].update({"uncertainty_score": 0.1, "grounded": True, "retrieval_mode": "full_retrieval"})
    enriched = qr.enrich_result_reliability(result)
    assert enriched["verification_status"] == "supported"
    assert enriched["review_status"] == "auto_approved"
    assert enriched["abstained"] is False


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
    outcome_metric = _Metric()
    policy_counter = _Metric()
    policy_gauge = _Metric()

    monkeypatch.setattr(qr, "ROUTER_QUERY_OUTCOME", outcome_metric)
    monkeypatch.setattr(qr, "QUERY_POLICY_APPLIED", policy_counter)
    monkeypatch.setattr(qr, "POLICY_VERSION_ACTIVE", policy_gauge)
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
    assert out["result"]["confidence_band"] in {"high", "medium", "low"}
    assert out["result"]["verification_status"] in {"supported", "weakly_supported", "unsupported"}
    assert any(item[0] == "inc" for item in policy_counter.values)
    assert any(item[0] == "inc" for item in outcome_metric.values)
    assert any(item[0] == "set" for item in policy_gauge.values)


@pytest.mark.asyncio
async def test_process_query_request_uses_first_image_for_vision(monkeypatch):
    """Image requests should switch modality and pass the first image to routing."""
    from app.services import query_runtime as qr
    outcome_metric = _Metric()

    monkeypatch.setattr(qr, "ROUTER_QUERY_OUTCOME", outcome_metric)
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
    assert any(item[0] == "labels" and item[1]["outcome"] == "success" for item in outcome_metric.values)


@pytest.mark.asyncio
async def test_process_query_request_records_empty_answer_outcome(monkeypatch):
    """Empty sanitized answers should be tracked as a distinct query outcome."""
    from app.services import query_runtime as qr

    outcome_metric = _Metric()
    monkeypatch.setattr(qr, "ROUTER_QUERY_OUTCOME", outcome_metric)
    monkeypatch.setattr(qr, "check_input_guardrails", lambda _q: _allow_guardrails())
    monkeypatch.setattr(qr, "check_tenant_budget", lambda _tenant: _allow_budget())
    monkeypatch.setattr(qr, "get_active_policy", lambda: None)
    _settings_map(monkeypatch, qr, {"AB_TESTING_ENABLED": "0"})
    monkeypatch.setattr(qr, "sanitize_output_guardrails", lambda answer: ("", []))

    async def _route(**kwargs):
        return _base_result()

    monkeypatch.setattr(qr, "route_and_answer", _route)
    await qr.process_query_request(_request())

    assert any(item[0] == "labels" and item[1]["outcome"] == "empty_answer" for item in outcome_metric.values)


@pytest.mark.asyncio
async def test_process_query_request_error_paths(monkeypatch):
    """Provider failures should be mapped to the expected HTTP status codes."""
    import asyncio as _asyncio
    from app.services import query_runtime as qr

    monkeypatch.setattr(qr, "check_input_guardrails", lambda _q: _allow_guardrails())
    monkeypatch.setattr(qr, "check_tenant_budget", lambda _tenant: _allow_budget())
    monkeypatch.setattr(qr, "get_active_policy", lambda: None)
    monkeypatch.setattr(
        qr,
        "settings",
        SimpleNamespace(
            AB_TESTING_ENABLED=False,
            TEMPERATURE_DEFAULT=0.2,
            MAX_TOKENS_DEFAULT=256,
            get=lambda key, default=None: default,
        ),
    )
    monkeypatch.setattr(
        qr,
        "apply_query_runtime_profile",
        lambda req, modality, image_input: {
            "use_rag": False,
            "max_tokens": 32,
            "workload_class": "simple_text",
            "perf_mode_enabled": False,
            "runtime_hints": {"retrieval_mode": "no_retrieval"},
        },
    )
    monkeypatch.setattr(qr, "sanitize_output_guardrails", lambda answer: (answer, []))
    monkeypatch.setattr(
        qr,
        "create_error_response",
        lambda err: {"error": True, "category": err.category.value if hasattr(err.category, "value") else str(err.category)},
    )

    req = _request()

    monkeypatch.setattr(qr, "route_and_answer", AsyncMock(side_effect=_asyncio.TimeoutError()))
    with pytest.raises(HTTPException) as exc:
        await qr.process_query_request(req)
    assert exc.value.status_code == 504

    monkeypatch.setattr(qr, "route_and_answer", AsyncMock(side_effect=qr.ProviderCircuitOpenError("ollama/x", "open")))
    with pytest.raises(HTTPException) as exc:
        await qr.process_query_request(req)
    assert exc.value.status_code == 503

    monkeypatch.setattr(qr, "route_and_answer", AsyncMock(side_effect=qr.ProviderCallError("ollama/x", "limited", category="provider_rate_limit")))
    with pytest.raises(HTTPException) as exc:
        await qr.process_query_request(req)
    assert exc.value.status_code == 429

    monkeypatch.setattr(qr, "route_and_answer", AsyncMock(side_effect=RuntimeError("boom")))
    with pytest.raises(HTTPException) as exc:
        await qr.process_query_request(req)
    assert exc.value.status_code == 500


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
    monkeypatch.setattr(qr, "create_response_review", lambda **kwargs: calls.__setitem__("review", kwargs))
    monkeypatch.setattr(
        qr,
        "get_ab_test_manager",
        lambda: SimpleNamespace(record_result=lambda *args: calls["metrics"].append(args)),
    )

    result = _base_result()
    result["metadata"].update(
        {
            "experiment_variant": {"name": "B"},
            "quality": 0.8,
            "confidence_score": 0.31,
            "confidence_band": "low",
            "grounded": False,
            "abstained": True,
            "abstain_reason": "low_confidence",
            "verification_status": "unsupported",
            "knowledge_version": "kv1",
            "review_status": "needs_review",
            "citations": [{"doc_id": "doc-1", "rank": 1, "source": "kb"}],
            "evidence_snippets": [{"doc_id": "doc-1", "rank": 1, "text": "trecho"}],
        }
    )
    qr.record_query_side_effects(_request(experiment_id="exp-1"), result, "img")

    assert calls["delay"]["image_b64"] == "img"
    assert calls["usage"] == ("school-1", 0.01, 11, 7, 1)
    assert calls["review"]["review_reason"] == "low_confidence"
    assert len(calls["metrics"]) == 3


def test_record_query_side_effects_perf_mode_skips_heavy_payload_and_ab_metrics(monkeypatch):
    """Performance mode should keep async feedback lean and suppress experiment accounting work."""
    from app.services import query_runtime as qr

    calls = {"delay": None, "metrics": []}
    monkeypatch.setattr(qr.task_process_feedback, "delay", lambda **kwargs: calls.__setitem__("delay", kwargs))
    monkeypatch.setattr(qr, "record_tenant_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(qr, "create_response_review", lambda **kwargs: calls.__setitem__("review", kwargs))
    _settings_map(monkeypatch, qr, {"AB_TESTING_ENABLED": "1", "ROUTER_PERF_MODE": "1"})
    monkeypatch.setattr(
        qr,
        "get_ab_test_manager",
        lambda: SimpleNamespace(record_result=lambda *args: calls["metrics"].append(args)),
    )

    result = _base_result()
    result["metadata"].update({"experiment_variant": {"name": "B"}, "review_status": "auto_approved"})
    qr.record_query_side_effects(_request(experiment_id="exp-1"), result, None)

    assert "raw_payload" not in calls["delay"]["raw_payload"]
    assert calls["metrics"] == []
    assert "review" not in calls


def test_record_query_side_effects_tolerates_feedback_failures(monkeypatch):
    """Side effects should not raise when downstream hooks fail."""
    from app.services import query_runtime as qr

    monkeypatch.setattr(qr.task_process_feedback, "delay", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("queue")))
    monkeypatch.setattr(qr, "record_tenant_usage", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("usage")))
    monkeypatch.setattr(qr, "create_response_review", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("review")))
    _settings_map(monkeypatch, qr, {"AB_TESTING_ENABLED": "1"})
    monkeypatch.setattr(qr, "get_ab_test_manager", lambda: (_ for _ in ()).throw(RuntimeError("ab")))

    result = _base_result()
    result["metadata"]["experiment_variant"] = {"name": "B"}
    qr.record_query_side_effects(_request(experiment_id="exp-1"), result, None)


def test_record_query_side_effects_review_and_experiment_failures(monkeypatch):
    """Side effects should tolerate review queue and experiment metric failures."""
    from app.services import query_runtime as qr

    warnings = []
    delayed = []
    recorded_usage = []
    result = _base_result()
    result.update(
        {
            "answer": "Resposta segura",
            "review_status": "needs_review",
            "estimated_cost_usd": 0.02,
            "metadata": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "raw_payload": "{}",
                "uncertainty_score": 0.1,
                "confidence_score": 0.4,
                "confidence_band": "low",
                "grounded": False,
                "abstained": False,
                "abstain_reason": None,
                "verification_status": "weakly_supported",
                "knowledge_version": "kv1",
                "review_status": "needs_review",
                "citations": [],
                "evidence_snippets": [],
                "correlation_id": "corr-1",
                "experiment_variant": {"name": "b"},
                "quality": 7.0,
                "workload_class": "knowledge_lookup",
            },
        }
    )

    monkeypatch.setattr(qr, "settings", SimpleNamespace(AB_TESTING_ENABLED=True, get=lambda key, default=None: "0"))
    monkeypatch.setattr(qr.task_process_feedback, "delay", lambda **kwargs: delayed.append(kwargs))
    monkeypatch.setattr(qr, "record_tenant_usage", lambda **kwargs: recorded_usage.append(kwargs))
    monkeypatch.setattr(qr, "create_response_review", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("review fail")))
    monkeypatch.setattr(qr, "get_ab_test_manager", lambda: SimpleNamespace(record_result=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ab fail"))))
    monkeypatch.setattr(qr.logger, "warning", lambda msg: warnings.append(msg))

    qr.record_query_side_effects(_request(experiment_id="exp-1"), result, image_input=None)
    assert delayed
    assert recorded_usage
    assert any("review" in msg.lower() for msg in warnings)
    assert any("experiment" in msg.lower() for msg in warnings)


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
