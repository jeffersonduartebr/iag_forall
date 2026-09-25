# Objective: Unit tests for the background feedback stages.
"""services.feedback_stages: judge sampling, quality, EMA and query-log enrichment."""

from types import SimpleNamespace

import pytest
from app.services import feedback_stages as fs

_NOOP_METRIC = SimpleNamespace(labels=lambda **k: SimpleNamespace(inc=lambda: None, set=lambda v: None))
_LOGGER = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)


def _deps(**overrides):
    deps = {
        "settings": SimpleNamespace(JUDGE_MIN_SAMPLE_RATE=0.05, get=lambda k, d=None: d),
        "compute_judge_probability": lambda **k: 0.5,
        "random": SimpleNamespace(random=lambda: 0.1),
        "logger": _LOGGER,
        "BACKGROUND_JUDGE_SKIPPED": _NOOP_METRIC,
        "FEEDBACK_TASK_FAILURES": _NOOP_METRIC,
    }
    deps.update(overrides)
    return deps


def _fb(**overrides):
    base = dict(query="q", answer="a", chosen_model="m", modality="text", latency_s=1.0, cost_val=0.001)
    base.update(overrides)
    return fs.FeedbackRequest(**base)


def _risk(mean=0.6):
    predictor = SimpleNamespace(learned=[], learn=None, record_outcome=None, maybe_save=None)
    predictor.learn = lambda emb, ok: predictor.learned.append(ok)
    predictor.record_outcome = lambda p, err: None
    predictor.maybe_save = lambda: False
    return fs.ErrorRisk({"mean": mean, "count": 3}, predictor, [0.1], 0.2)


def test_judging_is_sampled_and_throttled_by_backlog():
    assert fs.decide_judging(_deps(), _fb(), _risk()).should_judge is True  # 0.1 < 0.5
    throttled = fs.decide_judging(_deps(get_pending_query_jobs_count=lambda: 5), _fb(), _risk())
    assert throttled.should_judge is False and throttled.reasons == ["query_backlog"]


def test_exploration_always_judged_even_when_throttled():
    deps = _deps(get_pending_query_jobs_count=lambda: 5)
    fb = _fb(raw_payload={"openrouter_exploration": True})
    assert fs.decide_judging(deps, fb, _risk()).should_judge is True


@pytest.mark.asyncio
async def test_judge_quality_learns_only_from_real_judgments():
    async def rubric(q, a, **_):
        return [{"judge_id": "llm_rubric", "score": 0.8, "dimensions": {"clareza": 8}}]

    risk = _risk()
    quality = await fs.judge_quality(_deps(judge_answer=rubric), _fb(), risk, fs.JudgeDecision(True, 0.5))
    assert (quality.value, quality.source) == (8.0, "judge") and quality.judge_rubric["dimensions"] == {"clareza": 8}
    assert risk.predictor.learned == [True]

    async def heuristic(q, a, **_):
        return [{"judge_id": "heuristic_fallback", "score": 0.4}]

    risk = _risk()
    quality = await fs.judge_quality(_deps(judge_answer=heuristic), _fb(), risk, fs.JudgeDecision(True, 0.5))
    assert quality.source == "heuristic_fallback" and risk.predictor.learned == []


@pytest.mark.asyncio
async def test_judge_failure_falls_back_to_neutral_quality():
    async def boom(q, a, **_):
        raise RuntimeError("judge down")

    quality = await fs.judge_quality(_deps(judge_answer=boom), _fb(), _risk(), fs.JudgeDecision(True, 0.5))
    assert (quality.value, quality.source) == (5.0, "fallback_default")


def test_proxy_quality_and_ema_step():
    assert fs.proxy_quality(_risk(mean=1.4)).value == 10.0
    first = fs.next_ema(None, 2.0, 8.0, 0.01)
    assert first["updates"] == 1 and first["ema_latency"] == 2.0
    second = fs.next_ema(first, 4.0, 6.0, 0.03)
    assert second["ema_latency"] == pytest.approx(0.2 * 4.0 + 0.8 * 2.0)
    assert second["updates"] == 2


def test_persist_log_adds_rubric_and_reliability_fields():
    rows = []
    deps = _deps(
        ROUTER_QUALITY_AVG=_NOOP_METRIC,
        ROUTER_LOCAL_USAGE_RATIO=_NOOP_METRIC,
        insert_query_log=lambda **k: rows.append(k),
    )
    fb = _fb(raw_payload={"confidence_score": "0.7", "grounded": 1, "tenant_id": "t1"})
    quality = fs.Quality(8.0, "judge", {"score": 0.8})
    fs.persist_log(deps, fb, quality, True, _risk(), 0.9)
    row = rows[0]
    assert row["raw_payload"]["judge_rubric"] == {"score": 0.8}
    assert row["confidence_score"] == 0.7 and row["grounded"] is True and row["tenant_id"] == "t1"
    assert row["confidence_band"] is None and row["judge_sampled"] is True


def test_shared_ema_is_persisted_or_local_fallback():
    persisted = []
    fb = _fb(latency_s=3.0, cost_val=0.02)
    local = fs.next_ema(None, 3.0, 8.0, 0.02)
    shared = {**local, "updates": 7}
    deps = _deps(_persist_ema=lambda mod, model, entry: persisted.append(entry), update_shared_ema=lambda *a: shared)
    fs._share_and_persist_ema(deps, fb, 8.0, local)
    deps["update_shared_ema"] = lambda *a: None  # Redis indisponível
    fs._share_and_persist_ema(deps, fb, 8.0, local)
    assert persisted == [shared, local]
