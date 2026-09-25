# Objective: Test coverage for the background feedback orchestrator (process_background_feedback_impl).
"""Judging decisions, throttling, quality fallback and failure branches of the feedback loop."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.services.feedback_stages import FeedbackPersistError
from app.services.router_feedback import process_background_feedback_impl
from router_fakes import Metric


@pytest.mark.asyncio
async def test_router_feedback_paths_cover_judge_and_fallback_quality():
    """Feedback helper should cover judge path and no-judge fallback path."""
    metric_quality = Metric()
    metric_local = Metric()
    metric_latency = Metric()
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

        def maybe_save(self):
            return False

    class _History:
        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value):
            self.data[key] = value

    async def _judge_answer(query, answer, evaluated_model=None):
        return [{"score": 0.8}, {"score": 1.0}]

    async def _to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    async def _store_cache(**kwargs):
        stored.append(kwargs)

    deps = {
        "_get_ctx_stats": lambda _ctx: {"m1": {"count": 4, "mean": 0.6}},
        "get_predictor": lambda model: _Pred(),
        "asyncio": SimpleNamespace(
            to_thread=_to_thread, create_task=lambda coro: coro.close() if hasattr(coro, "close") else None
        ),
        "random": SimpleNamespace(random=lambda: 0.0),
        "embed_text": lambda q: [0.1, 0.2],
        "compute_judge_probability": lambda **kwargs: 1.0,
        "settings": SimpleNamespace(JUDGE_MIN_SAMPLE_RATE=0.0),
        "logger": SimpleNamespace(
            info=lambda *a, **k: None, error=lambda *a, **k: None,
            warning=lambda *a, **k: None, exception=lambda *a, **k: None
        ),
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
        "should_throttle_background_judge": lambda: False,
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
async def test_router_feedback_skips_judge_when_background_is_throttled():
    """Background judge work should yield under provider pressure when throttling is enabled."""
    metric = Metric()
    logged = []

    class _Pred:
        def predict_error_probability(self, emb):
            return 0.9

    async def _to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    async def _store_cache(**kwargs):
        return None

    deps = {
        "_get_ctx_stats": lambda _ctx: {"m1": {"count": 8, "mean": 0.7}},
        "get_predictor": lambda model: _Pred(),
        "asyncio": SimpleNamespace(
            to_thread=_to_thread, create_task=lambda coro: coro.close() if hasattr(coro, "close") else None
        ),
        "random": SimpleNamespace(random=lambda: 0.0),
        "embed_text": lambda q: [0.1],
        "compute_judge_probability": lambda **kwargs: 1.0,
        "settings": SimpleNamespace(
            JUDGE_MIN_SAMPLE_RATE=0.0,
            get=lambda key, default=None: {"JUDGE_BACKGROUND_THROTTLE_ENABLED": "1"}.get(key, default),
        ),
        "logger": SimpleNamespace(
            info=lambda *a, **k: None, error=lambda *a, **k: None,
            warning=lambda *a, **k: None, exception=lambda *a, **k: None
        ),
        "judge_answer": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("judge should not run")),
        "compute_reward": lambda *a, **k: 0.5,
        "bandit_update": lambda **kwargs: None,
        "_persist_ema": lambda *a, **k: None,
        "store_cache": _store_cache,
        "insert_query_log": lambda **kwargs: logged.append(kwargs),
        "ROUTER_QUALITY_AVG": metric,
        "ROUTER_LOCAL_USAGE_RATIO": metric,
        "FEEDBACK_PROCESSING_LATENCY": metric,
        "FEEDBACK_BACKLOG_AGE": metric,
        "FEEDBACK_TASK_FAILURES": metric,
        "should_throttle_background_judge": lambda: True,
    }
    state = {"EMA_HISTORY": SimpleNamespace(get=lambda key: None, set=lambda key, value: None)}

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
    assert logged[-1]["quality_source"] == "bandit_proxy"
    assert logged[-1]["judge_sampled"] is False


@pytest.mark.asyncio
async def test_router_feedback_skips_judge_when_query_backlog_exists():
    """Queued query backlog should suppress judge sampling even before provider-pressure throttling."""
    metric = Metric()
    logged = []

    class _Pred:
        def predict_error_probability(self, emb):
            return 0.8

    async def _to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    deps = {
        "_get_ctx_stats": lambda _ctx: {"m1": {"count": 8, "mean": 0.7}},
        "get_predictor": lambda model: _Pred(),
        "asyncio": SimpleNamespace(
            to_thread=_to_thread, create_task=lambda coro: coro.close() if hasattr(coro, "close") else None
        ),
        "random": SimpleNamespace(random=lambda: 0.0),
        "embed_text": lambda q: [0.1],
        "compute_judge_probability": lambda **kwargs: 1.0,
        "settings": SimpleNamespace(
            JUDGE_MIN_SAMPLE_RATE=0.0,
            get=lambda key, default=None: {
                "JUDGE_BACKGROUND_THROTTLE_ENABLED": "1",
                "QUERY_JOB_BACKGROUND_THROTTLE_PENDING_THRESHOLD": "1",
            }.get(key, default),
        ),
        "logger": SimpleNamespace(
            info=lambda *a, **k: None, error=lambda *a, **k: None,
            warning=lambda *a, **k: None, exception=lambda *a, **k: None
        ),
        "judge_answer": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("judge should not run")),
        "compute_reward": lambda *a, **k: 0.5,
        "bandit_update": lambda **kwargs: None,
        "_persist_ema": lambda *a, **k: None,
        "store_cache": lambda **kwargs: None,
        "insert_query_log": lambda **kwargs: logged.append(kwargs),
        "ROUTER_QUALITY_AVG": metric,
        "ROUTER_LOCAL_USAGE_RATIO": metric,
        "FEEDBACK_PROCESSING_LATENCY": metric,
        "FEEDBACK_BACKLOG_AGE": metric,
        "FEEDBACK_TASK_FAILURES": metric,
        "BACKGROUND_JUDGE_SKIPPED": metric,
        "should_throttle_background_judge": lambda: False,
        "get_pending_query_jobs_count": lambda: 3,
    }
    state = {"EMA_HISTORY": SimpleNamespace(get=lambda key: None, set=lambda key, value: None)}

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
    assert logged[-1]["quality_source"] == "bandit_proxy"
    assert logged[-1]["judge_sampled"] is False
    assert metric.values


@pytest.mark.asyncio
async def test_router_feedback_covers_failure_branches():
    """Feedback helper should tolerate judge, reward, bandit, cache, metric, log, and outer failures."""
    warnings = []
    exceptions = []
    metric = Metric()

    class _Pred:
        def predict_error_probability(self, emb):
            return 0.8

        def learn(self, emb, is_correct):
            raise RuntimeError("learn fail")

        def record_outcome(self, p, e):
            return None

        def save(self):
            return None

        def maybe_save(self):
            return False

    class _History:
        def get(self, key):
            raise RuntimeError("history fail")

        def set(self, key, value):
            raise RuntimeError("history fail")

    async def _judge_answer(query, answer, evaluated_model=None):
        return [{"score": 0.8}]

    async def _store_cache(**kwargs):
        raise RuntimeError("cache fail")

    async def _to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    deps = {
        "_get_ctx_stats": lambda _ctx: {"m1": {"count": 4, "mean": 0.6}},
        "get_predictor": lambda model: _Pred(),
        "asyncio": SimpleNamespace(
            to_thread=_to_thread, create_task=lambda coro: coro.close() if hasattr(coro, "close") else None
        ),
        "random": SimpleNamespace(random=lambda: 0.0),
        "embed_text": lambda q: [0.1],
        "compute_judge_probability": lambda **kwargs: 1.0,
        "settings": SimpleNamespace(JUDGE_MIN_SAMPLE_RATE=0.0),
        "logger": SimpleNamespace(
            info=lambda *a, **k: None,
            error=lambda *a, **k: None,
            warning=lambda msg: warnings.append(msg),
            exception=lambda msg: exceptions.append(msg),
        ),
        "judge_answer": _judge_answer,
        "compute_reward": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reward fail")),
        "bandit_update": lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bandit fail")),
        "_persist_ema": lambda *a, **k: None,
        "store_cache": _store_cache,
        "insert_query_log": lambda **kwargs: (_ for _ in ()).throw(RuntimeError("log fail")),
        "ROUTER_QUALITY_AVG": SimpleNamespace(
            labels=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("metric fail"))
        ),
        "ROUTER_LOCAL_USAGE_RATIO": metric,
        "FEEDBACK_PROCESSING_LATENCY": metric,
        "FEEDBACK_BACKLOG_AGE": metric,
        "FEEDBACK_TASK_FAILURES": metric,
    }
    # Todos os estágios falham neste cenário; o último a falhar é a escrita do
    # log, e é a única falha que passa a propagar.
    with pytest.raises(FeedbackPersistError):
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
    # Um erro inesperado num estágio inicial continua tolerado e registado:
    # propagá-lo poria a tarefa em retry, e repetir o pipeline volta a pagar
    # os juízes. Só a escrita do log é que faz a tarefa falhar.
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
