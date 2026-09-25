# Objective: Judges are never from the same company as the evaluated model.
"""User requirement (2026-09-24): judge models must be from companies different from the evaluated model's."""

from types import SimpleNamespace

import app.providers_async  # noqa: F401  (ordem de import dos provedores)
import pytest
from app.services import judge_context, judge_vendors
from app.services.judge_selection import _choose_two
from app.services.judge_vendors import MODELO_AVALIADO, com_modelo_avaliado, elegiveis, empresa

JUIZES = ["gemini/gemini-3.8-flash", "openrouter/anthropic/claude-sonnet-5", "openrouter/openai/gpt-5-mini"]


@pytest.mark.parametrize(
    ("modelo", "dona"),
    [
        ("ollama/gemma4:12b-it-qat", "google"), ("gemini/gemini-3.8-flash", "google"),
        ("ollama/granite4.2:8b-q4_K_M", "ibm"), ("openrouter/deepseek/deepseek-v4.1-flash", "deepseek"),
        ("openrouter/anthropic/claude-sonnet-5", "anthropic"), ("openrouter/openai/gpt-5-mini", "openai"),
        ("ollama/llama3.1:8b", "meta"), ("ollama/qwen2.5:14b", "alibaba"), ("ollama/phi4:latest", "microsoft"),
        ("openrouter/x-ai/grok-4", "xai"), ("ollama/modelo-novo", "modelo-novo"),
    ],
)
def test_company_of_each_model(modelo, dona):
    assert empresa(modelo) == dona


def test_judges_from_the_evaluated_company_are_excluded():
    assert elegiveis(JUIZES, "ollama/gemma4:12b-it-qat") == JUIZES[1:]  # Gemma -> nada do Google
    assert elegiveis(JUIZES, "openrouter/anthropic/claude-sonnet-5") == [JUIZES[0], JUIZES[2]]
    assert elegiveis(JUIZES, "ollama/granite4.2:8b-q4_K_M") == JUIZES
    assert elegiveis(JUIZES) == JUIZES  # sem modelo avaliado: sem filtro


def test_resolved_judges_and_meta_judge_follow_the_context(monkeypatch):
    monkeypatch.setattr(judge_context, "settings", SimpleNamespace(JUDGE_MODELS=JUIZES))
    monkeypatch.setattr(judge_context, "filter_configured_model_names", lambda ms: list(ms))
    monkeypatch.setattr(judge_context, "is_model_configured", lambda m: True)
    monkeypatch.setattr(judge_context, "META_JUDGE_HINT", "gemini/gemini-3.8-flash")
    token = MODELO_AVALIADO.set("gemini/gemini-3.8-flash")
    try:
        assert judge_context._resolve_judge_models() == JUIZES[1:]
        assert judge_context._resolve_meta_judge_model() == JUIZES[1]  # meta-juiz também não é do Google
    finally:
        MODELO_AVALIADO.reset(token)
    assert judge_context._resolve_meta_judge_model() == "gemini/gemini-3.8-flash"


def test_no_eligible_judge_means_no_judge(monkeypatch):
    monkeypatch.setattr(judge_context, "settings", SimpleNamespace(JUDGE_MODELS=["gemini/gemini-3.8-flash"]))
    monkeypatch.setattr(judge_context, "filter_configured_model_names", lambda ms: list(ms))
    monkeypatch.setattr(judge_context, "_configured_local_fallback", lambda: "ollama/gemma3:4b")
    token = MODELO_AVALIADO.set("ollama/gemma4:12b-it-qat")
    try:
        assert judge_context._resolve_judge_models() == []
    finally:
        MODELO_AVALIADO.reset(token)
    assert _choose_two([], {}) == []


@pytest.mark.asyncio
async def test_decorator_scopes_the_evaluated_model():
    vistos = []

    @com_modelo_avaliado
    async def juiz(x):
        vistos.append((x, judge_vendors.MODELO_AVALIADO.get()))
        return x

    assert await juiz(1, evaluated_model="ollama/gemma4:12b-it-qat") == 1
    assert await juiz(2) == 2
    assert vistos == [(1, "ollama/gemma4:12b-it-qat"), (2, None)] and MODELO_AVALIADO.get() is None
