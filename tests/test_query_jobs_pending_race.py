# Objective: Regression — concurrent enqueues must not overshoot the per-identity pending limit.
"""The pending counter is reserved atomically (INCR first, DECR on reject)."""

from __future__ import annotations

import app.query_jobs as qj
import pytest
from app.schemas import QueryRequest
from app.services.pending_slots import release_pending_slot, reserve_pending_slot
from fastapi import HTTPException


class _Settings:
    def get(self, key, default=None):
        return {"QUERY_JOB_MAX_PENDING_PER_TENANT": "1"}.get(key, default)


class _Racing:
    """fakeredis proxy that lets a second request run the moment the first opens a pipeline."""

    def __init__(self, inner, hook):
        self._inner, self._hook = inner, hook

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def pipeline(self):
        hook, self._hook = self._hook, None
        if hook:
            hook()
        return self._inner.pipeline()


def _enqueue():
    req = QueryRequest(query="q", tenant_id="escola-1")
    return qj.enqueue_query_job(
        req=req, correlation_id="cid", reason="ollama_overloaded", pressure_state="congested", route_path="/query"
    )


@pytest.fixture
def env(monkeypatch, fake_redis):
    sent = []
    monkeypatch.setattr(qj, "settings", _Settings())
    monkeypatch.setattr(qj.celery_app, "send_task", lambda *a, **k: sent.append(k))
    return fake_redis, sent


def test_interleaved_enqueues_respect_the_tenant_limit(env, monkeypatch):
    fake_redis, sent = env
    outcomes = []

    def _other_request():
        try:
            outcomes.append(_enqueue().queue_depth)
        except HTTPException as exc:
            outcomes.append(exc.status_code)

    racing = _Racing(fake_redis, _other_request)
    monkeypatch.setattr(qj, "_get_job_store", lambda: racing)
    try:
        outcomes.append(_enqueue().queue_depth)
    except HTTPException as exc:
        outcomes.append(exc.status_code)

    # GET-then-INCR aceitava as duas (contador 2 com limite 1).
    assert sorted(outcomes) == [1, 429] and len(sent) == 1
    assert int(fake_redis.get(qj._tenant_pending_key("escola-1"))) == 1


def test_redis_failure_while_reserving_is_503(env, monkeypatch):
    class _Down:
        def pipeline(self):
            raise ConnectionError("redis down")

    monkeypatch.setattr(qj, "_get_job_store", lambda: _Down())
    with pytest.raises(HTTPException) as exc:
        _enqueue()
    assert exc.value.status_code == 503 and exc.value.detail["category"] == "queue_enqueue_failed"


def test_release_never_deletes_other_reservations(fake_redis):
    assert reserve_pending_slot(fake_redis, "k", 5, 60) == (True, 0)
    assert reserve_pending_slot(fake_redis, "k", 5, 60) == (True, 1)
    release_pending_slot(fake_redis, "k")
    assert int(fake_redis.get("k")) == 1
    release_pending_slot(None, "k")  # best effort: erros engolidos
