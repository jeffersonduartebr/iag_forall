# Objective: Test coverage for the shadow comparison between an explored model and the incumbent.
"""openrouter_shadow.maybe_run_shadow_comparison stores paired deltas (fakeredis)."""

import json
from types import SimpleNamespace

import fakeredis
import pytest
from app.openrouter_exploration_state import REDIS_MODEL_STATS_PREFIX

from app import openrouter_shadow as shadow


@pytest.fixture
def env(monkeypatch):
    rds = fakeredis.FakeAsyncRedis()

    async def get_redis():
        return rds

    monkeypatch.setattr("app.openrouter_explorer._get_redis", get_redis)
    clock = iter([10.0, 11.5])
    monkeypatch.setattr(shadow.time, "perf_counter", lambda: next(clock))
    calls = []

    async def call_model(**kwargs):
        calls.append(kwargs)
        return "resposta incumbente", {"usage": {}}

    async def judge_answer(query, answer):
        return [{"score": 8.0}, {"score": 6.0}, {"error": "timeout"}]

    deps = {
        "call_model": call_model,
        "parse_meta_cost": lambda meta, chosen_model, cost_lookup: (10, 20, 0.004, None, None),
        "get_model_cost": lambda *a: 0.0,
        "judge_answer": judge_answer,
    }
    return rds, deps, calls


async def _run(deps, incumbent="openai/gpt-4o", explored="openrouter/x/y"):
    await shadow.maybe_run_shadow_comparison(
        query="q",
        explored_model=explored,
        explored_answer="a",
        explored_quality=80.0,
        explored_latency=2.0,
        explored_cost=0.001,
        incumbent_model=incumbent,
        deps=deps,
        settings=SimpleNamespace(TEMPERATURE_DEFAULT=0.1, MAX_TOKENS_DEFAULT=128),
    )


@pytest.mark.asyncio
async def test_shadow_records_paired_deltas(env):
    rds, deps, calls = env
    await _run(deps)

    assert calls == [
        {
            "model": "openai/gpt-4o",
            "prompt": "q",
            "modality": "text",
            "image_b64": None,
            "temperature": 0.1,
            "max_tokens": 128,
        }
    ]
    stats = json.loads(await rds.get(f"{REDIS_MODEL_STATS_PREFIX}openrouter/x/y"))
    (delta,) = stats["shadow_comparisons"]
    assert delta["delta_quality"] == 10.0  # 80 - média(8, 6) * 10
    assert delta["delta_latency_s"] == 0.5  # 2,0 s explorado - 1,5 s medidos no incumbente
    assert delta["delta_cost_usd"] == -0.003
    assert stats["shadow_delta_quality_mean"] == 10.0


def test_append_shadow_keeps_bounded_history():
    stats = {"shadow_comparisons": [{"delta_quality": 1.0}] * 20}
    out = shadow._append_shadow(stats, {"delta_quality": 21.0})
    assert len(out["shadow_comparisons"]) == 20
    assert out["shadow_delta_quality_mean"] == 2.0
    assert (
        shadow._append_shadow({"shadow_comparisons": "corrompido"}, {"delta_quality": 3.0})["shadow_delta_quality_mean"]
        == 3.0
    )


@pytest.mark.parametrize(
    ("incumbent", "explored"),
    [(None, "m"), ("openai/gpt-4o", "openai/gpt-4o"), ("local-model", "m")],
)
@pytest.mark.asyncio
async def test_shadow_skips_ineligible_pairs(env, incumbent, explored):
    rds, deps, calls = env
    await _run(deps, incumbent=incumbent, explored=explored)
    assert calls == [] and await rds.keys() == []


@pytest.mark.asyncio
async def test_shadow_failure_is_logged_not_raised(env):
    rds, deps, _ = env

    async def broken(**kwargs):
        raise TimeoutError("provider")

    deps["call_model"] = broken
    await _run(deps)
    assert await rds.keys() == []
    assert shadow._judged_quality([]) == 50.0
