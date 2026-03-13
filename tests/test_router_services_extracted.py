# Objective: Test coverage for router services extracted behavior and regressions.
"""Direct tests for extracted router service helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.router_execution import route_and_answer_internal_impl
from app.services.router_feedback import process_background_feedback_impl


class _Metric:
    def __init__(self):
        self.values = []

    def labels(self, **kwargs):
        return self

    def inc(self, value=1):
        self.values.append(value)

    def set(self, value):
        self.values.append(value)

    def observe(self, value):
        self.values.append(value)


def _deps_for_execution():
    metric = _Metric()
    logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    settings = SimpleNamespace(
        MAX_TOKENS_DEFAULT=128,
        TEMPERATURE_DEFAULT=0.3,
        CANDIDATE_MODELS_LIST=["openai/gpt-4o", "ollama/phi4:latest"],
        CANDIDATE_VISION_MODELS_LIST=[],
        CANDIDATE_MULTIMODAL_MODELS_LIST=[],
    )
    async def _to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    return {
        "asyncio": SimpleNamespace(create_task=lambda coro: coro.close() if coro else None, to_thread=_to_thread),
        "settings": settings,
        "normalize_modality": lambda modality, image_b64: "vision" if image_b64 and modality == "text" else modality,
        "_dep_cache_breaker": SimpleNamespace(current_state="closed", call_async=lambda fn, *a, **k: fn(*a, **k)),
        "_dep_uq_breaker": SimpleNamespace(current_state="closed", call=lambda fn, *a, **k: fn(*a, **k)),
        "check_cache": None,
        "_record_dependency_breaker_metrics": lambda: None,
        "DEPENDENCY_FAILURES": metric,
        "logger": logger,
        "ROUTER_ROUTE_COST": metric,
        "ROUTER_STAGE_LATENCY": metric,
        "ROUTER_ATTEMPTS_PER_QUERY": metric,
        "ROUTER_RETRY_TOTAL": metric,
        "ROUTER_RESPONSE_EMPTY": metric,
        "FALLBACK_USED": metric,
        "get_uncertainty_score": lambda query, modality: 0.2,
        "BLOCKED_PREFIXES": ("nomic-embed", "text-embedding", "bge-", "e5-"),
        "_is_error_budget_exceeded": lambda: False,
        "get_dynamic_strategy_weights": lambda modality: {"w_quality": 1, "w_latency": 1, "w_cost": 1},
        "choose_top2_models": lambda **kwargs: kwargs["candidates"][:2],
        "select_model": lambda models, query, modality: "openai/gpt-4o",
        "_ensure_ollama_model": lambda name: None,
        "build_augmented_prompt": None,
        "build_final_prompt": lambda **kwargs: f"{kwargs['system_prompt']}::{kwargs['query']}::{kwargs.get('rag_text')}",
        "_safe_setting_bool": lambda key, default=False: False,
        "_safe_setting_int": lambda key, default=2: 2,
        "call_model": None,
        "execute_with_fallback": None,
        "ProviderCallError": RuntimeError,
        "parse_meta_cost": lambda meta, chosen_model, cost_lookup: (meta.get("prompt_tokens", 0), meta.get("completion_tokens", 0), meta.get("cost_per_1k", 0.0), 0.0, meta if isinstance(meta, dict) else {}),
        "get_model_cost": lambda model, p, c: 0.0,
    }


@pytest.mark.asyncio
async def test_router_execution_fallback_chain_success():
    """Fallback chain success should report chosen fallback model and errors."""
    deps = _deps_for_execution()
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
    deps = _deps_for_execution()
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
async def test_router_feedback_paths_cover_judge_and_fallback_quality():
    """Feedback helper should cover judge path and no-judge fallback path."""
    metric_quality = _Metric()
    metric_local = _Metric()
    metric_latency = _Metric()
    stored = []
    logged = []

    class _Pred:
        def predict_error_probability(self, emb):
            return 0.8

        def learn(self, emb, is_correct):
            return None

        def record_outcome(self, p, e):
            return None

        def save(self):
            return None

    class _History:
        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value):
            self.data[key] = value

    async def _judge_answer(query, answer):
        return [{"score": 0.8}, {"score": 1.0}]

    async def _store_cache(**kwargs):
        stored.append(kwargs)

    async def _to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    deps = {
        "_get_ctx_stats": lambda _ctx: {"m1": {"count": 4, "mean": 0.6}},
        "get_predictor": lambda model: _Pred(),
        "asyncio": SimpleNamespace(to_thread=_to_thread, create_task=lambda coro: coro.close() if hasattr(coro, "close") else None),
        "random": SimpleNamespace(random=lambda: 0.0),
        "embed_text": lambda q: [0.1, 0.2],
        "compute_judge_probability": lambda **kwargs: 1.0,
        "settings": SimpleNamespace(JUDGE_MIN_SAMPLE_RATE=0.0),
        "logger": SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, exception=lambda *a, **k: None),
        "judge_answer": _judge_answer,
        "compute_reward": lambda *a, **k: 0.7,
        "bandit_update": lambda **kwargs: None,
        "_persist_ema": lambda *a, **k: None,
        "store_cache": _store_cache,
        "insert_query_log": lambda **kwargs: logged.append(kwargs),
        "ROUTER_QUALITY_AVG": metric_quality,
        "ROUTER_LOCAL_USAGE_RATIO": metric_local,
        "FEEDBACK_PROCESSING_LATENCY": metric_latency,
        "FEEDBACK_BACKLOG_AGE": metric_latency,
        "FEEDBACK_TASK_FAILURES": metric_latency,
    }
    state = {"EMA_HISTORY": _History()}

    await process_background_feedback_impl(
        deps=deps,
        state=state,
        query="q",
        answer="a",
        chosen_model="ollama/m1",
        modality="text",
        latency_s=0.2,
        cost_val=0.01,
    )
    assert stored
    assert logged
    assert logged[-1]["quality_source"] == "judge"
    assert logged[-1]["judge_sampled"] is True
    assert metric_quality.values

    deps["random"] = SimpleNamespace(random=lambda: 1.0)
    async def _broken_store(**kwargs):
        raise RuntimeError("cache down")

    deps["store_cache"] = _broken_store
    await process_background_feedback_impl(
        deps=deps,
        state=state,
        query="q2",
        answer="a2",
        chosen_model="m1",
        modality="text",
        latency_s=0.3,
        cost_val=0.02,
    )
    assert logged[-1]["quality_source"] == "bandit_proxy"
    assert logged[-1]["judge_sampled"] is False
    assert metric_latency.values


@pytest.mark.asyncio
async def test_router_execution_covers_cache_uq_and_metadata_error_paths():
    """Execution helper should handle cache/UQ metric failures and metadata fallback."""
    deps = _deps_for_execution()
    inc_metric = _Metric()
    warnings = []

    async def _cache(*args, **kwargs):
        return {"text": "cached", "similarity": 0.91}

    class _Breaker:
        current_state = "closed"

        async def call_async(self, fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        def call(self, fn, *args, **kwargs):
            raise RuntimeError("uq down")

    deps["_dep_cache_breaker"] = _Breaker()
    deps["_dep_uq_breaker"] = _Breaker()
    deps["check_cache"] = _cache
    deps["DEPENDENCY_FAILURES"] = inc_metric
    deps["logger"] = SimpleNamespace(info=lambda *a, **k: None, warning=lambda msg: warnings.append(msg))
    deps["ROUTER_ROUTE_COST"] = SimpleNamespace(labels=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("metric down")))
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

    deps = _deps_for_execution()
    deps["check_cache"] = lambda *a, **k: None
    deps["_dep_uq_breaker"] = SimpleNamespace(current_state="closed", call=lambda fn, *a, **k: (_ for _ in ()).throw(RuntimeError("uq down")))
    deps["DEPENDENCY_FAILURES"] = inc_metric
    deps["logger"] = SimpleNamespace(info=lambda *a, **k: None, warning=lambda msg: warnings.append(msg))

    async def _call_model(**kwargs):
        return ("ok", {"ignored": True})

    deps["call_model"] = _call_model
    deps["parse_meta_cost"] = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad meta"))
    deps["ROUTER_ROUTE_COST"] = SimpleNamespace(labels=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("metric down")))

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
async def test_router_execution_tracks_empty_answer_reason():
    """Execution helper should mark empty responses with a classified reason."""
    deps = _deps_for_execution()
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
    deps = _deps_for_execution()
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
    deps["select_model"] = lambda models, query, modality: "ollama/phi4:latest"
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
    deps = _deps_for_execution()
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
    deps["select_model"] = lambda models, query, modality: "ollama/phi4:latest"
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
    deps = _deps_for_execution()
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
async def test_router_feedback_covers_failure_branches():
    """Feedback helper should tolerate judge, reward, bandit, cache, metric, log, and outer failures."""
    warnings = []
    exceptions = []
    metric = _Metric()

    class _Pred:
        def predict_error_probability(self, emb):
            return 0.8

        def learn(self, emb, is_correct):
            raise RuntimeError("learn fail")

        def record_outcome(self, p, e):
            return None

        def save(self):
            return None

    class _History:
        def get(self, key):
            raise RuntimeError("history fail")

        def set(self, key, value):
            raise RuntimeError("history fail")

    async def _judge_answer(query, answer):
        return [{"score": 0.8}]

    async def _store_cache(**kwargs):
        raise RuntimeError("cache fail")

    async def _to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    deps = {
        "_get_ctx_stats": lambda _ctx: {"m1": {"count": 4, "mean": 0.6}},
        "get_predictor": lambda model: _Pred(),
        "asyncio": SimpleNamespace(to_thread=_to_thread, create_task=lambda coro: coro.close() if hasattr(coro, "close") else None),
        "random": SimpleNamespace(random=lambda: 0.0),
        "embed_text": lambda q: [0.1],
        "compute_judge_probability": lambda **kwargs: 1.0,
        "settings": SimpleNamespace(JUDGE_MIN_SAMPLE_RATE=0.0),
        "logger": SimpleNamespace(info=lambda *a, **k: None, warning=lambda msg: warnings.append(msg), exception=lambda msg: exceptions.append(msg)),
        "judge_answer": _judge_answer,
        "compute_reward": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reward fail")),
        "bandit_update": lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bandit fail")),
        "_persist_ema": lambda *a, **k: None,
        "store_cache": _store_cache,
        "insert_query_log": lambda **kwargs: (_ for _ in ()).throw(RuntimeError("log fail")),
        "ROUTER_QUALITY_AVG": SimpleNamespace(labels=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("metric fail"))),
        "ROUTER_LOCAL_USAGE_RATIO": metric,
        "FEEDBACK_PROCESSING_LATENCY": metric,
        "FEEDBACK_BACKLOG_AGE": metric,
        "FEEDBACK_TASK_FAILURES": metric,
    }
    await process_background_feedback_impl(
        deps=deps,
        state={"EMA_HISTORY": _History()},
        query="q",
        answer="a",
        chosen_model="ollama/m1",
        modality="text",
        latency_s=0.2,
        cost_val=0.01,
    )
    assert warnings

    deps["_get_ctx_stats"] = lambda _ctx: (_ for _ in ()).throw(RuntimeError("ctx fail"))
    await process_background_feedback_impl(
        deps=deps,
        state={"EMA_HISTORY": _History()},
        query="q",
        answer="a",
        chosen_model="m1",
        modality="text",
        latency_s=0.2,
        cost_val=0.01,
    )
    assert exceptions
