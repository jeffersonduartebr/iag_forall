# Objective: Shared fakes for the router execution/feedback orchestration tests.
"""Metric stub and the dependency dict expected by ``route_and_answer_internal_impl``."""

from __future__ import annotations

from types import SimpleNamespace

import pybreaker


class Metric:
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


def deps_for_execution():
    metric = Metric()
    logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None,
            warning=lambda *a, **k: None)
    settings = SimpleNamespace(
        MAX_TOKENS_DEFAULT=128,
        TEMPERATURE_DEFAULT=0.3,
        CANDIDATE_MODELS_LIST=["openai/gpt-4o", "ollama/phi4:latest"],
        CANDIDATE_VISION_MODELS_LIST=[],
        CANDIDATE_MULTIMODAL_MODELS_LIST=[],
    )

    async def _to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    async def _async_weights():
        return {"w_quality": 1, "w_latency": 1, "w_cost": 1}

    async def _select_async(models, query, modality):
        return "openai/gpt-4o"

    return {
        "asyncio": SimpleNamespace(create_task=lambda coro: coro.close() if coro else None, to_thread=_to_thread),
        "settings": settings,
        "normalize_modality": lambda modality, image_b64: "vision" if image_b64 and modality == "text" else modality,
        # pybreaker real: o fake anterior tinha um `call_async` que funcionava,
        # ou seja testava uma interface que a produção não conseguia usar (a do
        # pybreaker é Tornado-only e levanta NameError).
        "_dep_cache_breaker": pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60, name="fake_cache"),
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
        "get_dynamic_strategy_weights": lambda *args, **kwargs: {"w_quality": 1, "w_latency": 1, "w_cost": 1},
        "get_dynamic_strategy_weights_async": lambda modality: _async_weights(),
        "choose_top2_models": lambda candidates, weights, query_text, modality="text", uncertainty_score=0.0, min_quality=0.0: (
            candidates[:2]
        ),
        "select_model": lambda models, query, modality: "openai/gpt-4o",
        "select_model_async": _select_async,
        "is_ollama_model_verified": lambda name: False,
        "_ensure_ollama_model": lambda name: None,
        "build_augmented_prompt": None,
        "build_final_prompt": lambda **kwargs: f"{kwargs['query']}::{kwargs.get('rag_text')}",
        "_safe_setting_bool": lambda key, default=False: False,
        "_safe_setting_int": lambda key, default=2: 2,
        "call_model": None,
        "execute_with_fallback": None,
        "ProviderCallError": RuntimeError,
        "parse_meta_cost": lambda meta, chosen_model, cost_lookup: (
            meta.get("prompt_tokens", 0),
            meta.get("completion_tokens", 0),
            meta.get("cost_per_1k", 0.0),
            0.0,
            meta if isinstance(meta, dict) else {},
        ),
        "get_model_cost": lambda model, p, c: 0.0,
    }
