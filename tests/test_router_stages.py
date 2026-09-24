# Objective: Unit tests for the extracted routing stages.
"""services.router_stages / router_provider_stage: candidate resolution, tools, fallback, failures."""

import time
from types import SimpleNamespace

import pytest
from app.services import router_provider_stage as rps
from app.services import router_stages as rs


class _ProviderCallError(Exception):
    def __init__(self, model, message, category, retryable):
        super().__init__(message)
        self.model, self.category = model, category


def _ctx(**deps_overrides):
    settings = SimpleNamespace(
        CANDIDATE_MODELS_LIST=["openai/gpt-5.5", "ollama/gemma3:4b", "blocked/x", "openai/gpt-5.5"],
        CANDIDATE_VISION_MODELS_LIST=[],
        CANDIDATE_MULTIMODAL_MODELS_LIST=[],
        CANDIDATE_TOOL_MODELS_LIST=[],
    )
    deps = {
        "settings": settings,
        "BLOCKED_PREFIXES": ("blocked/",),
        "logger": SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None),
        "_is_error_budget_exceeded": lambda: False,
        "ProviderCallError": _ProviderCallError,
        "_safe_setting_int": lambda key, default: default,
    }
    deps.update(deps_overrides)
    return rs.RouteContext(
        deps=deps,
        query="q",
        system_prompt="",
        use_rag=False,
        max_tokens=10,
        temperature=0.1,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=False,
    )


def test_not_blocked_dedupes_in_config_order():
    deps = {"BLOCKED_PREFIXES": ("blocked/",)}
    assert rs._not_blocked(deps, ["b", "a", "blocked/z", "b", 3, "a"]) == ["b", "a"]


@pytest.mark.asyncio
async def test_resolve_candidates_filters_and_falls_back(monkeypatch):
    monkeypatch.setattr(rs, "filter_configured_model_names", lambda models: list(models))
    assert await rs.resolve_candidates(_ctx()) == ["openai/gpt-5.5", "ollama/gemma3:4b"]

    monkeypatch.setattr(rs, "filter_configured_model_names", lambda models: [])
    assert await rs.resolve_candidates(_ctx()) == ["ollama/phi4:latest"]


@pytest.mark.asyncio
async def test_error_budget_forces_local_models(monkeypatch):
    monkeypatch.setattr(rs, "filter_configured_model_names", lambda models: list(models))

    async def exceeded():
        return True

    assert await rs.resolve_candidates(_ctx(_is_error_budget_exceeded_async=exceeded)) == ["ollama/gemma3:4b"]


def test_restrict_to_tool_models_rejects_when_none_capable(monkeypatch):
    ctx = _ctx()
    ctx.tools = [{"type": "function", "function": {"name": "f"}}]
    monkeypatch.setattr(rs, "filter_tool_capable_model_names", lambda models: [])
    with pytest.raises(_ProviderCallError) as exc:
        rs.restrict_to_tool_models(ctx, ["openai/gpt-5.5"])
    assert exc.value.category == "no_tool_model"
    assert rs.restrict_to_tool_models(_ctx(), ["m"]) == ["m"]  # sem tools: inalterado


def test_fallback_budget_is_zero_when_deadline_is_tight():
    ctx = _ctx(ROUTER_FALLBACK_SKIPPED=SimpleNamespace(labels=lambda **k: SimpleNamespace(inc=lambda: None)))
    ctx.runtime_hints = {"request_deadline_ts": time.monotonic() + 5.0, "provider_timeout_seconds": 10}
    assert rps._fallback_budget(ctx) == 0
    ctx.runtime_hints = {"max_fallbacks": 3}
    assert rps._fallback_budget(ctx) == 3


def test_finalize_maps_failures_to_provider_error():
    ctx = _ctx(ROUTER_RETRY_TOTAL=SimpleNamespace(labels=lambda **k: SimpleNamespace(inc=lambda: None)))
    failed = SimpleNamespace(success=False, errors=[{"error": "boom", "category": "rate_limited"}], model_used="m2")
    with pytest.raises(_ProviderCallError) as exc:
        rps._finalize(ctx, "m1", failed)
    assert exc.value.model == "m2" and exc.value.category == "rate_limited"

    empty = SimpleNamespace(success=False, errors=[], model_used=None)
    with pytest.raises(_ProviderCallError) as exc:
        rps._finalize(ctx, "m1", empty)
    assert exc.value.model == "m1" and exc.value.category == "provider_unavailable"


def test_finalize_success_reports_fallback():
    ctx = _ctx(
        ROUTER_RETRY_TOTAL=SimpleNamespace(labels=lambda **k: SimpleNamespace(inc=lambda: None)),
        FALLBACK_USED=SimpleNamespace(labels=lambda **k: SimpleNamespace(inc=lambda: None)),
    )
    ok = SimpleNamespace(success=True, result=("resp", {}), model_used="m2", models_tried=["m1", "m2"], errors=[{}])
    outcome = rps._finalize(ctx, "m1", ok)
    assert (outcome.chosen, outcome.fallback_used, outcome.retry_count) == ("m2", True, 1)


@pytest.mark.asyncio
async def test_error_budget_never_forces_local_models_with_open_breakers(monkeypatch):
    """Production (2026-09-24): forcing locals whose breaker was open failed every request and kept the error
    budget exceeded forever (89/100 queries in 502). Without a healthy local model, keep all candidates."""
    monkeypatch.setattr(rs, "filter_configured_model_names", lambda models: list(models))
    monkeypatch.setattr(rs, "_saudavel", lambda model: False)

    async def exceeded():
        return True

    assert await rs.resolve_candidates(_ctx(_is_error_budget_exceeded_async=exceeded)) == ["openai/gpt-5.5", "ollama/gemma3:4b"]
    monkeypatch.setattr(rs, "_saudavel", lambda model: True)
    assert await rs.resolve_candidates(_ctx(_is_error_budget_exceeded_async=exceeded)) == ["ollama/gemma3:4b"]


def test_health_check_uses_the_breaker_and_tolerates_errors(monkeypatch):
    from types import SimpleNamespace

    import app.reliability as rel

    monkeypatch.setattr(rel, "get_circuit_breaker_manager", lambda: SimpleNamespace(is_available=lambda m: m == "ollama/ok"))
    assert rs._saudavel("ollama/ok") and not rs._saudavel("ollama/aberto")
    monkeypatch.setattr(rel, "get_circuit_breaker_manager", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert rs._saudavel("ollama/qualquer")
