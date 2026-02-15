"""Módulo `tests/test_router_core_internal.py`: descreve responsabilidades e integrações deste arquivo."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _mock_settings():
    """Executa mock settings."""
    return SimpleNamespace(
        MAX_TOKENS_DEFAULT=128,
        TEMPERATURE_DEFAULT=0.3,
        CANDIDATE_MODELS_LIST=["openai/gpt-4o", "ollama/phi4:latest"],
        CANDIDATE_VISION_MODELS_LIST=[],
        CANDIDATE_MULTIMODAL_MODELS_LIST=[],
    )


@pytest.mark.asyncio
async def test_internal_cache_hit(monkeypatch):
    """Testa internal cache hit."""
    from app import router_core

    monkeypatch.setattr(router_core, "settings", _mock_settings())
    monkeypatch.setattr(router_core, "check_cache", AsyncMock(return_value={"text": "cached", "similarity": 0.9}))

    out = await router_core._route_and_answer_internal("q1", use_cache=True)
    assert out["model"] == "semantic_cache"
    assert out["answer"] == "cached"
    assert out["cost_per_1k"] == 0.0


@pytest.mark.asyncio
async def test_internal_full_flow_with_pricing_fallback(monkeypatch):
    """Testa internal full flow with pricing fallback."""
    from app import router_core

    monkeypatch.setattr(router_core, "settings", _mock_settings())
    monkeypatch.setattr(router_core, "check_cache", AsyncMock(return_value=None))
    monkeypatch.setattr(router_core, "get_uncertainty_score", lambda *_: 0.25)
    monkeypatch.setattr(router_core, "get_dynamic_strategy_weights", lambda modality: {"w_quality": 1, "w_latency": 1, "w_cost": 1})
    monkeypatch.setattr(router_core, "choose_top2_models", lambda **kwargs: ["openai/gpt-4o"])
    monkeypatch.setattr(router_core, "select_model", lambda models, q, modality: "openai/gpt-4o")

    call_model = AsyncMock(
        return_value=(
            "answer",
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cost_per_1k": 0.0,
                "raw_payload": "{}",
            },
        )
    )
    monkeypatch.setattr(router_core, "call_model", call_model)
    monkeypatch.setattr(router_core, "get_model_cost", lambda model, p, c: 0.123)

    out = await router_core._route_and_answer_internal(
        "Pergunta",
        system_prompt="System prompt",
        use_rag=False,
        use_cache=True,
    )
    assert out["answer"] == "answer"
    assert out["cost_per_1k"] == pytest.approx(0.123)
    assert out["metadata"]["prompt_tokens"] == 10
    assert "System prompt" in call_model.await_args.kwargs["prompt"]


@pytest.mark.asyncio
async def test_internal_rag_fallback_when_augmented_prompt_fails(monkeypatch):
    """Testa internal rag fallback when augmented prompt fails."""
    from app import router_core

    monkeypatch.setattr(router_core, "settings", _mock_settings())
    monkeypatch.setattr(router_core, "check_cache", AsyncMock(return_value=None))
    monkeypatch.setattr(router_core, "get_uncertainty_score", lambda *_: 0.4)
    monkeypatch.setattr(router_core, "get_dynamic_strategy_weights", lambda modality: {"w_quality": 1, "w_latency": 1, "w_cost": 1})
    monkeypatch.setattr(router_core, "choose_top2_models", lambda **kwargs: ["openai/gpt-4o"])
    monkeypatch.setattr(router_core, "select_model", lambda models, q, modality: "openai/gpt-4o")
    monkeypatch.setattr(router_core, "build_augmented_prompt", AsyncMock(side_effect=RuntimeError("rag down")))

    call_model = AsyncMock(return_value=("fallback answer", {"cost_per_1k": 0.0}))
    monkeypatch.setattr(router_core, "call_model", call_model)
    monkeypatch.setattr(router_core, "get_model_cost", lambda model, p, c: 0.0)

    out = await router_core._route_and_answer_internal(
        "Pergunta fallback",
        system_prompt="SYS",
        use_rag=True,
    )
    assert out["answer"] == "fallback answer"
    assert "SYS" in call_model.await_args.kwargs["prompt"]
    assert "Pergunta fallback" in call_model.await_args.kwargs["prompt"]


@pytest.mark.asyncio
async def test_internal_handles_non_dict_metadata(monkeypatch):
    """Testa internal handles non dict metadata."""
    from app import router_core

    monkeypatch.setattr(router_core, "settings", _mock_settings())
    monkeypatch.setattr(router_core, "check_cache", AsyncMock(return_value=None))
    monkeypatch.setattr(router_core, "get_uncertainty_score", lambda *_: 0.5)
    monkeypatch.setattr(router_core, "get_dynamic_strategy_weights", lambda modality: {"w_quality": 1, "w_latency": 1, "w_cost": 1})
    monkeypatch.setattr(router_core, "choose_top2_models", lambda **kwargs: ["openai/gpt-4o"])
    monkeypatch.setattr(router_core, "select_model", lambda models, q, modality: "openai/gpt-4o")
    monkeypatch.setattr(router_core, "call_model", AsyncMock(return_value=("answer", "meta-string")))

    out = await router_core._route_and_answer_internal("q", use_cache=False)
    assert out["metadata"]["prompt_tokens"] == 0
    assert out["metadata"]["completion_tokens"] == 0
    assert out["cost_per_1k"] == 0.0


@pytest.mark.asyncio
async def test_internal_fallback_when_all_candidates_blocked(monkeypatch):
    """Testa internal fallback when all candidates blocked."""
    from app import router_core

    settings = _mock_settings()
    settings.CANDIDATE_MODELS_LIST = ["nomic-embed-text", "text-embedding-3-small"]
    monkeypatch.setattr(router_core, "settings", settings)
    monkeypatch.setattr(router_core, "check_cache", AsyncMock(return_value=None))
    monkeypatch.setattr(router_core, "get_uncertainty_score", lambda *_: 0.1)
    monkeypatch.setattr(router_core, "get_dynamic_strategy_weights", lambda modality: {"w_quality": 1, "w_latency": 1, "w_cost": 1})
    monkeypatch.setattr(router_core, "choose_top2_models", lambda **kwargs: ["ollama/phi4:latest"])
    monkeypatch.setattr(router_core, "select_model", lambda models, q, modality: models[0])
    monkeypatch.setattr(router_core, "_ensure_ollama_model", lambda name: None)
    monkeypatch.setattr(router_core, "call_model", AsyncMock(return_value=("ok", {"cost_per_1k": 0.0})))

    # Avoid background task warnings in unit test.
    monkeypatch.setattr("app.router_core.asyncio.create_task", lambda coro: coro.close())

    out = await router_core._route_and_answer_internal("q", use_cache=False)
    assert out["model"] == "ollama/phi4:latest"
