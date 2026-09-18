# Objective: Test coverage for the three-dimension judge rubric.
"""Rubric parsing/aggregation and the rubric path of the judge pipeline."""

import json
from types import SimpleNamespace

import pytest

from app import judges
from app.services import judge_rubric as jr

WEIGHTS = dict(jr.DEFAULT_RUBRIC_WEIGHTS)


def _scores(c, a, al):
    return f'<reasoning>ok</reasoning><scores>{{"clareza": {c}, "acuracia": {a}, "alinhamento": {al}}}</scores>'


def test_parse_rubric_scores_valid_aliases_and_clamping():
    assert jr.parse_rubric_scores(_scores(8, 6, 7)) == {"clareza": 8.0, "acuracia": 6.0, "alinhamento": 7.0}
    text = 'Resultado: {"Clareza": 12, "Acurácia": -1, "alinhamento_pedagogico": "5"}'
    assert jr.parse_rubric_scores(text) == {"clareza": 10.0, "acuracia": 0.0, "alinhamento": 5.0}


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "<verdict>CORRECT</verdict>",
        '<scores>{"clareza": 8, "acuracia": 6}</scores>',
        "<scores>{oops}</scores>",
    ],
)
def test_parse_rubric_scores_incomplete_is_none_not_zero(text):
    assert jr.parse_rubric_scores(text) is None


def test_parse_rubric_weights():
    assert jr.parse_rubric_weights(None) == WEIGHTS
    assert jr.parse_rubric_weights("not json") == WEIGHTS
    assert jr.parse_rubric_weights(json.dumps({"clareza": 1, "acuracia": 1, "alinhamento": 2})) == {
        "clareza": 1.0,
        "acuracia": 1.0,
        "alinhamento": 2.0,
    }
    assert jr.parse_rubric_weights({"clareza": 1}) == WEIGHTS


def test_weighted_quality_uses_thesis_weights():
    # 0.3*8 + 0.5*6 + 0.2*7 = 6.8
    assert jr.weighted_quality({"clareza": 8, "acuracia": 6, "alinhamento": 7}, WEIGHTS) == pytest.approx(6.8)


def test_combine_ratings_mean_dispersion_and_median():
    a = {"clareza": 8.0, "acuracia": 6.0, "alinhamento": 7.0}
    b = {"clareza": 6.0, "acuracia": 8.0, "alinhamento": 7.0}
    out = jr.combine_ratings([a, b], WEIGHTS)
    assert out["dimensions"] == {"clareza": 7.0, "acuracia": 7.0, "alinhamento": 7.0}
    assert out["quality"] == pytest.approx(7.0)
    assert out["n_judges"] == 2
    assert out["dispersion"]["by_dimension"]["alinhamento"] == 0.0
    assert out["dispersion"]["by_dimension"]["clareza"] == pytest.approx(1.0)

    c = {"clareza": 1.0, "acuracia": 1.0, "alinhamento": 1.0}
    med = jr.combine_ratings([a, b, c], WEIGHTS, aggregate="median")
    assert med["dimensions"] == {"clareza": 6.0, "acuracia": 6.0, "alinhamento": 7.0}
    assert jr.combine_ratings([], WEIGHTS) is None


def test_build_rubric_prompt_anchors_accuracy_on_reference():
    prompt = jr.build_rubric_prompt("q?", "resp", reference="42", rag_context="ctx")
    assert "GABARITO OFICIAL" in prompt and "42" in prompt and "ctx" in prompt
    assert '"clareza": <0-10>' in prompt


@pytest.fixture
def rubric_env(monkeypatch):
    judges._rubric_cache = judges.VerdictCache()
    monkeypatch.setattr(judges, "_load_judge_stats", lambda w: {})
    monkeypatch.setattr(
        judges, "_choose_two", lambda models, stats: [judges.SelectedJudge("j1", 1.0), judges.SelectedJudge("j2", 1.0)]
    )
    monkeypatch.setattr(judges, "_resolve_judge_models", lambda: ["j1", "j2"])
    monkeypatch.setattr(judges, "_resolve_meta_judge_model", lambda: "meta")
    monkeypatch.setattr(judges, "_persist_judge_metrics", lambda **k: None)
    logs = []
    monkeypatch.setattr(judges, "_persist_judge_log", lambda *a, **k: logs.append((a, k)))

    async def _empty(*a, **k):
        return ""

    monkeypatch.setattr(judges, "get_rag_context", _empty)
    monkeypatch.setattr(judges, "_describe_image_if_needed", _empty)
    values = {"JUDGE_SCORING_MODE": "rubric", "JUDGE_RUBRIC_DISAGREEMENT": "3.0"}
    monkeypatch.setattr(judges, "settings", SimpleNamespace(JUDGES_MODE="llm", get=lambda k, d=None: values.get(k, d)))
    return logs


def _fake_call(outputs, calls):
    async def _call_model(**kwargs):
        calls.append(kwargs["model"])
        out = outputs[kwargs["model"]]
        if isinstance(out, Exception):
            raise out
        return out, {"latency": 1.0, "cost_per_1k": 0.0}

    return _call_model


@pytest.mark.asyncio
async def test_rubric_two_judges_agree(monkeypatch, rubric_env):
    calls = []
    monkeypatch.setattr(judges, "call_model", _fake_call({"j1": _scores(8, 6, 7), "j2": _scores(6, 8, 7)}, calls))
    out = await judges.judge_answer("q", "a")
    assert len(out) == 1 and out[0]["judge_id"] == "llm_rubric"
    assert out[0]["score"] == pytest.approx(0.7)  # médias por dimensão 7/7/7 -> Q = 7.0
    assert out[0]["n_judges"] == 2
    assert sorted(calls) == ["j1", "j2"]
    assert all(k.get("rubric") for _, k in rubric_env)  # notas por dimensão persistidas


@pytest.mark.asyncio
async def test_rubric_disagreement_calls_meta_judge_and_takes_median(monkeypatch, rubric_env):
    calls = []
    outputs = {"j1": _scores(9, 9, 9), "j2": _scores(2, 2, 2), "meta": _scores(8, 7, 8)}
    monkeypatch.setattr(judges, "call_model", _fake_call(outputs, calls))
    out = await judges.judge_answer("q", "a")
    assert "meta" in calls
    assert out[0]["dimensions"] == {"clareza": 8.0, "acuracia": 7.0, "alinhamento": 8.0}
    assert out[0]["n_judges"] == 3


@pytest.mark.asyncio
async def test_rubric_failed_judge_is_excluded_not_zeroed(monkeypatch, rubric_env):
    calls = []
    outputs = {"j1": _scores(8, 8, 8), "j2": RuntimeError("timeout")}
    monkeypatch.setattr(judges, "call_model", _fake_call(outputs, calls))
    out = await judges.judge_answer("q", "a")
    assert out[0]["judge_id"] == "llm_rubric"
    assert out[0]["score"] == pytest.approx(0.8)
    assert out[0]["n_judges"] == 1


@pytest.mark.asyncio
async def test_rubric_all_judges_fail_falls_back_to_flagged_heuristic(monkeypatch, rubric_env):
    outputs = {"j1": "sem notas", "j2": RuntimeError("down")}
    monkeypatch.setattr(judges, "call_model", _fake_call(outputs, []))
    monkeypatch.setattr(judges, "heuristic_score", lambda a: 0.4)
    out = await judges.judge_answer("q", "a")
    assert out == [{"judge_id": "heuristic_fallback", "score": 0.4}]


@pytest.mark.asyncio
async def test_rubric_result_is_cached(monkeypatch, rubric_env):
    calls = []
    monkeypatch.setattr(judges, "call_model", _fake_call({"j1": _scores(7, 7, 7), "j2": _scores(7, 7, 7)}, calls))
    await judges.judge_answer("q", "a")
    await judges.judge_answer("q", "a")
    assert len(calls) == 2
