# Objective: Uniform judge draw (field protocol): pair and tie-breaker random among other-company judges.
"""User requirement (2026-10-19): in the field window, judges are drawn at random, never from the company of
the evaluated model; the tie-breaker is a third judge drawn the same way."""

from types import SimpleNamespace

import app.providers_async  # noqa: F401  (ordem de import dos provedores)
import pytest
from app.services import judge_context, judge_selection
from app.services.judge_selection import PAR_SORTEADO, JudgeStats, _choose_two
from app.services.judge_vendors import MODELO_AVALIADO, empresa

POOL = [
    "openrouter/anthropic/claude-opus-5.5",
    "openrouter/openai/gpt-6-luna",
    "gemini/gemini-4-argon",
    "openrouter/xiaomi/mimo-v2.6-pro",
    "openrouter/xiaomi/mimo-v2.6-flash",
    "openrouter/meta/muse-spark-1.1",
    "openrouter/qwen/qwen3.8-max-0902",
    "openrouter/deepseek/deepseek-v4-pro-0813",
]


@pytest.fixture
def uniforme(monkeypatch):
    monkeypatch.setattr(
        judge_selection.settings, "get", lambda k, d=None: "uniforme" if k == "JUDGES_SELECTION_MODE" else d
    )


@pytest.mark.parametrize(
    ("modelo", "dona"),
    [
        ("openrouter/xiaomi/mimo-v2.6-pro", "xiaomi"),
        ("openrouter/xiaomi/mimo-v2.6-flash", "xiaomi"),
        ("openrouter/meta/muse-spark-1.1", "meta"),
        ("openrouter/ibm-granite/granite-4.0-h-micro", "ibm"),
        ("openrouter/openai/gpt-6-luna", "openai"),
        ("openrouter/qwen/qwen3.8-max-0902", "alibaba"),
    ],
)
def test_field_pool_companies(modelo, dona):
    assert empresa(modelo) == dona


def test_uniform_pair_ignores_fitness(uniforme):
    # Um juiz com aptidão nula seria descartado no modo adaptativo; no uniforme, sai com a mesma frequência.
    stats = {m: JudgeStats(m, fitness=0.0 if i == 0 else 0.9) for i, m in enumerate(POOL)}
    vezes = sum(POOL[0] in [sj.model for sj in _choose_two(POOL, stats)] for _ in range(4000))
    assert vezes / 4000 == pytest.approx(2 / len(POOL), abs=0.03)


def test_uniform_pair_is_two_distinct_judges_and_is_recorded(uniforme):
    par = _choose_two(POOL, {})
    assert len({sj.model for sj in par}) == 2
    assert PAR_SORTEADO.get() == tuple(sj.model for sj in par)
    assert [sj.model for sj in _choose_two(POOL[:1], {})] == POOL[:1]


def test_uniform_tie_breaker_is_another_judge_of_another_company(uniforme, monkeypatch):
    monkeypatch.setattr(judge_context, "settings", SimpleNamespace(JUDGE_MODELS=POOL))
    monkeypatch.setattr(judge_context, "filter_configured_model_names", lambda ms: list(ms))
    token = MODELO_AVALIADO.set("openrouter/xiaomi/mimo-v2.6-flash")
    try:
        for _ in range(200):
            par = [sj.model for sj in _choose_two(judge_context._resolve_judge_models(), {})]
            meta = judge_context._resolve_meta_judge_model()
            assert meta not in par
            assert "xiaomi" not in {empresa(m) for m in [*par, meta]}
    finally:
        MODELO_AVALIADO.reset(token)
