# Objective: Test coverage for the routing orchestrator (route_and_answer_internal_impl).
"""Cache, uncertainty, candidate selection, RAG, provider execution and result metadata."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.services.router_execution import route_and_answer_internal_impl
from router_fakes import Metric, deps_for_execution


@pytest.mark.asyncio
async def test_router_execution_fallback_chain_success():
    """Fallback chain success should report chosen fallback model and errors."""
    deps = deps_for_execution()
    deps["check_cache"] = lambda *a, **k: None
    deps["call_model"] = lambda **kwargs: ("ignored", {})
    deps["_safe_setting_bool"] = lambda key, default=False: True
    deps["_safe_setting_int"] = lambda key, default=2: 2

    async def _fallback(**kwargs):
        return SimpleNamespace(
            success=True,
            result=("fallback-answer", {"prompt_tokens": 3, "completion_tokens": 2, "cost_per_1k": 0.1}),
            model_used="ollama/phi4:latest",
            models_tried=["openai/gpt-4o", "ollama/phi4:latest"],
            errors=[{"error": "upstream down", "category": "provider_unavailable"}],
        )

    deps["execute_with_fallback"] = _fallback

    out = await route_and_answer_internal_impl(
        deps=deps,
        query="pergunta",
        system_prompt="SYS",
        use_rag=False,
        max_tokens=None,
        temperature=None,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=False,
    )
    assert out["answer"] == "fallback-answer"
    assert out["model"] == "ollama/phi4:latest"
    assert out["route"]["fallback"]["used"] is True


@pytest.mark.asyncio
async def test_router_execution_fallback_chain_failure_raises():
    """Fallback chain failure should raise the injected provider error type."""
    deps = deps_for_execution()
    deps["check_cache"] = lambda *a, **k: None
    deps["_safe_setting_bool"] = lambda key, default=False: True

    class _ProviderError(Exception):
        def __init__(self, model, message, category=None, retryable=True):
            super().__init__(message)

    deps["ProviderCallError"] = _ProviderError

    async def _fallback(**kwargs):
        return SimpleNamespace(
            success=False,
            result=None,
            model_used="openai/gpt-4o",
            models_tried=["openai/gpt-4o"],
            errors=[{"error": "timeout", "category": "provider_timeout"}],
        )

    deps["execute_with_fallback"] = _fallback

    with pytest.raises(_ProviderError):
        await route_and_answer_internal_impl(
            deps=deps,
            query="pergunta",
            system_prompt="SYS",
            use_rag=False,
            max_tokens=None,
            temperature=None,
            modality="text",
            image_b64=None,
            rag_modality="text",
            use_cache=False,
        )


@pytest.mark.asyncio
async def test_router_execution_skips_ollama_ensure_when_model_is_verified():
    """Verified Ollama models should not trigger background `/api/tags` checks."""
    deps = deps_for_execution()
    deps["check_cache"] = lambda *a, **k: None

    async def _select_ollama(models, query, modality):
        return "ollama/phi4:latest"

    deps["select_model"] = lambda models, query, modality: "ollama/phi4:latest"
    deps["select_model_async"] = _select_ollama
    deps["is_ollama_model_verified"] = lambda name: True
    ensure_calls = []
    deps["_ensure_ollama_model"] = lambda name: ensure_calls.append(name)

    async def _call_model(**kwargs):
        return "ok", {"prompt_tokens": 1, "completion_tokens": 1, "cost_per_1k": 0.0}

    deps["call_model"] = _call_model

    out = await route_and_answer_internal_impl(
        deps=deps,
        query="pergunta",
        system_prompt="SYS",
        use_rag=False,
        max_tokens=None,
        temperature=None,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=False,
    )

    assert out["answer"] == "ok"
    assert ensure_calls == []


@pytest.mark.asyncio
async def test_router_execution_covers_cache_uq_and_metadata_error_paths():
    """Execution helper should handle cache/UQ metric failures and metadata fallback."""
    deps = deps_for_execution()
    inc_metric = Metric()
    warnings = []

    async def _cache(*args, **kwargs):
        return {"text": "cached", "similarity": 0.91}

    class _UQBreakerDown:
        """Só o breaker de UQ falha; o da cache é um pybreaker real."""

        current_state = "closed"

        def call(self, fn, *args, **kwargs):
            raise RuntimeError("uq down")

    deps["_dep_uq_breaker"] = _UQBreakerDown()
    deps["check_cache"] = _cache
    deps["DEPENDENCY_FAILURES"] = inc_metric
    deps["logger"] = SimpleNamespace(info=lambda *a, **k: None, warning=lambda msg: warnings.append(msg))
    deps["ROUTER_ROUTE_COST"] = SimpleNamespace(
        labels=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("metric down"))
    )
    metric_calls = {"count": 0}

    def _record_metrics():
        metric_calls["count"] += 1
        if metric_calls["count"] == 1:
            raise RuntimeError("metrics fail")

    deps["_record_dependency_breaker_metrics"] = _record_metrics

    out = await route_and_answer_internal_impl(
        deps=deps,
        query="pergunta",
        system_prompt="SYS",
        use_rag=False,
        max_tokens=None,
        temperature=None,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=True,
    )
    assert out["model"] == "semantic_cache"
    assert inc_metric.values

    deps = deps_for_execution()
    deps["check_cache"] = lambda *a, **k: None
    deps["_dep_uq_breaker"] = SimpleNamespace(
        current_state="closed", call=lambda fn, *a, **k: (_ for _ in ()).throw(RuntimeError("uq down"))
    )
    deps["DEPENDENCY_FAILURES"] = inc_metric
    deps["logger"] = SimpleNamespace(info=lambda *a, **k: None, warning=lambda msg: warnings.append(msg))

    async def _call_model(**kwargs):
        return ("ok", {"ignored": True})

    deps["call_model"] = _call_model
    deps["parse_meta_cost"] = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad meta"))
    deps["ROUTER_ROUTE_COST"] = SimpleNamespace(
        labels=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("metric down"))
    )

    out = await route_and_answer_internal_impl(
        deps=deps,
        query="pergunta",
        system_prompt="SYS",
        use_rag=False,
        max_tokens=None,
        temperature=None,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=False,
    )
    assert out["metadata"]["prompt_tokens"] == 0
    assert out["cost_per_1k"] == 0.0
    assert any("UQ fail" in msg or "Metadata error" in msg for msg in warnings)


@pytest.mark.asyncio
async def test_router_execution_passes_light_retrieval_hints():
    """Execution helper should forward light-retrieval hints to the RAG builder."""
    deps = deps_for_execution()
    deps["check_cache"] = lambda *a, **k: None
    captured = {}

    async def _build_augmented_prompt(query, modality="text", image_b64=None, **kwargs):
        captured.update(kwargs)
        return "ctx"

    async def _call_model(**kwargs):
        return "ok", {"prompt_tokens": 1, "completion_tokens": 1, "cost_per_1k": 0.0}

    deps["build_augmented_prompt"] = _build_augmented_prompt
    deps["call_model"] = _call_model

    out = await route_and_answer_internal_impl(
        deps=deps,
        query="Cite a policy oficial sobre recuperação paralela.",
        system_prompt="SYS",
        use_rag=True,
        max_tokens=None,
        temperature=None,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=False,
        runtime_hints={
            "workload_class": "knowledge_lookup",
            "retrieval_mode": "light_retrieval",
            "rag_top_k": 2,
            "rag_context_token_budget": 320,
            "rag_rerank_enabled": False,
            "max_fallbacks": 1,
        },
    )

    assert out["answer"] == "ok"
    assert captured["k"] == 2
    assert captured["retrieval_mode"] == "light_retrieval"
    assert captured["context_token_budget"] == 320
    assert captured["rerank_enabled"] is False


@pytest.mark.asyncio
async def test_router_execution_avoids_augmented_prompt_when_retrieval_bundle_is_weak():
    """Weak retrieval bundles should fall back to the plain query prompt and record the skip reason."""
    deps = deps_for_execution()
    deps["check_cache"] = lambda *a, **k: None

    async def _build_retrieval_bundle(*args, **kwargs):
        return {
            "augmented_prompt": "ctx",
            "context": "",
            "citations": [],
            "evidence_snippets": [],
            "grounded": False,
            "knowledge_version": "kv1",
            "retrieval_mode": "light_retrieval",
            "retrieval_skipped_reason": "insufficient_context_quality",
        }

    async def _call_model(**kwargs):
        return "ok", {"prompt_tokens": 1, "completion_tokens": 1, "cost_per_1k": 0.0}

    deps["build_retrieval_bundle"] = _build_retrieval_bundle
    deps["call_model"] = _call_model

    out = await route_and_answer_internal_impl(
        deps=deps,
        query="Cite a policy oficial sobre recuperação paralela.",
        system_prompt="SYS",
        use_rag=True,
        max_tokens=None,
        temperature=None,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=False,
        runtime_hints={
            "workload_class": "knowledge_lookup",
            "retrieval_mode": "light_retrieval",
            "rag_top_k": 2,
            "rag_context_token_budget": 320,
            "rag_rerank_enabled": False,
            "needs_retrieval": True,
        },
    )

    assert out["answer"] == "ok"
    assert out["metadata"]["retrieval_skipped_reason"] == "insufficient_context_quality"
    assert out["metadata"]["grounded"] is False


@pytest.mark.asyncio
async def test_router_execution_tracks_empty_answer_reason():
    """Execution helper should mark empty responses with a classified reason."""
    deps = deps_for_execution()
    deps["check_cache"] = lambda *a, **k: None

    async def _call_model(**kwargs):
        return "", {"reasoning": "internal trace", "prompt_tokens": 1, "completion_tokens": 0, "cost_per_1k": 0.0}

    deps["call_model"] = _call_model

    out = await route_and_answer_internal_impl(
        deps=deps,
        query="pergunta",
        system_prompt="SYS",
        use_rag=False,
        max_tokens=None,
        temperature=None,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=False,
    )
    assert out["answer"] == ""
    assert deps["ROUTER_RESPONSE_EMPTY"].values


@pytest.mark.asyncio
async def test_router_execution_skips_rag_for_simple_low_uq_query():
    """Simple low-uncertainty text queries should bypass retrieval even when use_rag is requested."""
    deps = deps_for_execution()
    deps["check_cache"] = lambda *a, **k: None
    deps["settings"] = SimpleNamespace(
        MAX_TOKENS_DEFAULT=128,
        TEMPERATURE_DEFAULT=0.3,
        CANDIDATE_MODELS_LIST=["ollama/phi4:latest"],
        CANDIDATE_VISION_MODELS_LIST=[],
        CANDIDATE_MULTIMODAL_MODELS_LIST=[],
        get=lambda key, default=None: {"RAG_SIMPLE_QUERY_BYPASS_ENABLED": "1"}.get(key, default),
    )
    deps["get_uncertainty_score"] = lambda query, modality: 0.1

    async def _select_ollama(models, query, modality):
        return "ollama/phi4:latest"

    deps["select_model"] = lambda models, query, modality: "ollama/phi4:latest"
    deps["select_model_async"] = _select_ollama
    rag_called = {"value": False}

    async def _rag(*args, **kwargs):
        rag_called["value"] = True
        return "context"

    async def _call_model(**kwargs):
        return ("direct-answer", {"prompt_tokens": 2, "completion_tokens": 1, "cost_per_1k": 0.1})

    deps["build_augmented_prompt"] = _rag
    deps["call_model"] = _call_model

    out = await route_and_answer_internal_impl(
        deps=deps,
        query="Quanto é 2+2?",
        system_prompt="SYS",
        use_rag=True,
        max_tokens=None,
        temperature=None,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=False,
    )

    assert out["answer"] == "direct-answer"
    assert rag_called["value"] is False


@pytest.mark.asyncio
async def test_router_execution_keeps_rag_for_high_uq_query():
    """Higher-uncertainty text queries should still execute retrieval when RAG is enabled."""
    deps = deps_for_execution()
    deps["check_cache"] = lambda *a, **k: None
    deps["settings"] = SimpleNamespace(
        MAX_TOKENS_DEFAULT=128,
        TEMPERATURE_DEFAULT=0.3,
        CANDIDATE_MODELS_LIST=["ollama/phi4:latest"],
        CANDIDATE_VISION_MODELS_LIST=[],
        CANDIDATE_MULTIMODAL_MODELS_LIST=[],
        get=lambda key, default=None: {"RAG_SIMPLE_QUERY_BYPASS_ENABLED": "1"}.get(key, default),
    )
    deps["get_uncertainty_score"] = lambda query, modality: 0.8

    async def _select_ollama(models, query, modality):
        return "ollama/phi4:latest"

    deps["select_model"] = lambda models, query, modality: "ollama/phi4:latest"
    deps["select_model_async"] = _select_ollama
    rag_called = {"value": False}

    async def _rag(*args, **kwargs):
        rag_called["value"] = True
        return "context"

    async def _call_model(**kwargs):
        return ("rag-answer", {"prompt_tokens": 2, "completion_tokens": 1, "cost_per_1k": 0.1})

    deps["build_augmented_prompt"] = _rag
    deps["call_model"] = _call_model

    out = await route_and_answer_internal_impl(
        deps=deps,
        query="Explique passo a passo por que a resposta faz sentido.",
        system_prompt="SYS",
        use_rag=True,
        max_tokens=None,
        temperature=None,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=False,
    )

    assert out["answer"] == "rag-answer"
    assert rag_called["value"] is True


@pytest.mark.asyncio
async def test_router_execution_covers_execute_provider_and_empty_fallback_errors():
    """Fallback execution should invoke execute_fn and handle empty error lists."""
    deps = deps_for_execution()
    calls = []

    async def _call_model(**kwargs):
        calls.append(kwargs)
        return ("provider-answer", {"prompt_tokens": 2, "completion_tokens": 1, "cost_per_1k": 0.2})

    async def _fallback(**kwargs):
        result = await kwargs["execute_fn"]("openai/gpt-4o")
        return SimpleNamespace(
            success=True,
            result=result,
            model_used="openai/gpt-4o",
            models_tried=["openai/gpt-4o"],
            errors=[],
        )

    deps["call_model"] = _call_model
    deps["execute_with_fallback"] = _fallback
    deps["_safe_setting_bool"] = lambda key, default=False: True

    out = await route_and_answer_internal_impl(
        deps=deps,
        query="pergunta",
        system_prompt="SYS",
        use_rag=False,
        max_tokens=99,
        temperature=0.7,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=False,
    )
    assert calls and calls[0]["max_tokens"] == 99
    assert out["answer"] == "provider-answer"

    class _ProviderError(Exception):
        def __init__(self, model, message, category=None, retryable=True):
            super().__init__(message)

    deps["ProviderCallError"] = _ProviderError

    async def _empty_fail(**kwargs):
        return SimpleNamespace(
            success=False,
            result=None,
            model_used="openai/gpt-4o",
            models_tried=["openai/gpt-4o"],
            errors=[],
        )

    deps["execute_with_fallback"] = _empty_fail
    with pytest.raises(_ProviderError):
        await route_and_answer_internal_impl(
            deps=deps,
            query="pergunta",
            system_prompt="SYS",
            use_rag=False,
            max_tokens=None,
            temperature=None,
            modality="text",
            image_b64=None,
            rag_modality="text",
            use_cache=False,
        )


@pytest.mark.asyncio
async def test_router_execution_uses_hedge_when_enabled():
    """With hedging enabled and a distinct backup, execution routes through execute_with_hedge (perf #22)."""
    deps = deps_for_execution()
    deps["check_cache"] = lambda *a, **k: None
    # Two distinct candidates so top2 has a real backup.
    deps["choose_top2_models"] = (
        lambda candidates, weights, query_text, modality="text", uncertainty_score=0.0, min_quality=0.0: [
            "openai/gpt-4o",
            "ollama/phi4:latest",
        ]
    )

    async def _select(models, query, modality):
        return "openai/gpt-4o"

    deps["select_model_async"] = _select
    deps["_safe_setting_bool"] = lambda key, default=False: key == "REQUEST_HEDGING_ENABLED"
    deps["_safe_setting_float"] = lambda key, default=0.0: default
    deps["get_ema_latency"] = lambda model, modality: 0.01
    hedge_calls = {}

    async def _hedge(models, execute_fn, hedge_delay_s, max_parallel=2):
        hedge_calls["models"] = models
        hedge_calls["delay"] = hedge_delay_s
        out = await execute_fn(models[0])
        return SimpleNamespace(success=True, result=out, model_used=models[0], models_tried=[models[0]], errors=[])

    async def _call_model(**kwargs):
        return "hedged-ok", {"prompt_tokens": 1, "completion_tokens": 1, "cost_per_1k": 0.0}

    deps["execute_with_hedge"] = _hedge
    deps["call_model"] = _call_model
    deps["ROUTER_HEDGE"] = Metric()

    out = await route_and_answer_internal_impl(
        deps=deps,
        query="pergunta",
        system_prompt="SYS",
        use_rag=False,
        max_tokens=None,
        temperature=None,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=False,
    )
    assert out["answer"] == "hedged-ok"
    assert hedge_calls["models"] == ["openai/gpt-4o", "ollama/phi4:latest"]


@pytest.mark.asyncio
async def test_router_execution_reports_stage_timings_breakdown():
    """Normal execution should surface a per-request stage-latency breakdown (perf #21)."""
    deps = deps_for_execution()
    deps["check_cache"] = lambda *a, **k: None

    async def _call_model(**kwargs):
        return "ok", {"prompt_tokens": 1, "completion_tokens": 1, "cost_per_1k": 0.0}

    deps["call_model"] = _call_model

    out = await route_and_answer_internal_impl(
        deps=deps,
        query="pergunta",
        system_prompt="SYS",
        use_rag=False,
        max_tokens=None,
        temperature=None,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=False,
    )
    timings = out["metadata"]["stage_timings_ms"]
    assert isinstance(timings, dict)
    # Selection + provider + postprocess are always on the non-cached path.
    assert "selection" in timings
    assert "provider_call" in timings
    assert "postprocess" in timings
    assert all(isinstance(v, float) and v >= 0.0 for v in timings.values())


@pytest.mark.asyncio
async def test_router_execution_cache_hit_reports_cache_lookup_timing():
    """Cache-hit path should still report a stage-latency breakdown with cache_lookup (perf #21)."""
    deps = deps_for_execution()

    async def _cache(*args, **kwargs):
        return {"text": "cached", "similarity": 0.95}

    deps["check_cache"] = _cache

    out = await route_and_answer_internal_impl(
        deps=deps,
        query="pergunta",
        system_prompt="SYS",
        use_rag=False,
        max_tokens=None,
        temperature=None,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=True,
    )
    assert out["model"] == "semantic_cache"
    assert "cache_lookup" in out["metadata"]["stage_timings_ms"]

