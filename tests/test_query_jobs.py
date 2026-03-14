# Objective: Test coverage for asynchronous queued query jobs and their Redis-backed status store.
"""Unit tests for queued query job helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException


class FakePipeline:
    """Small Redis pipeline stub used by queued-job tests."""

    def __init__(self, client):
        self.client = client
        self.ops = []

    def setex(self, key, ttl, value):
        self.ops.append(("setex", key, ttl, value))
        return self

    def incr(self, key):
        self.ops.append(("incr", key))
        return self

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))
        return self

    def decr(self, key):
        self.ops.append(("decr", key))
        return self

    def execute(self):
        out = []
        for op in self.ops:
            name = op[0]
            if name == "setex":
                _, key, ttl, value = op
                self.client.setex(key, ttl, value)
                out.append(True)
            elif name == "incr":
                _, key = op
                out.append(self.client.incr(key))
            elif name == "decr":
                _, key = op
                out.append(self.client.decr(key))
            elif name == "expire":
                out.append(True)
        self.ops.clear()
        return out


class FakeRedis:
    """Very small Redis-like store for queued query job tests."""

    def __init__(self):
        self.data = {}
        self.ttls = {}

    def get(self, key):
        return self.data.get(key)

    def setex(self, key, ttl, value):
        self.data[key] = value
        self.ttls[key] = ttl
        return True

    def delete(self, key):
        self.data.pop(key, None)
        self.ttls.pop(key, None)

    def incr(self, key):
        value = int(self.data.get(key) or 0) + 1
        self.data[key] = value
        return value

    def decr(self, key):
        value = int(self.data.get(key) or 0) - 1
        self.data[key] = value
        return value

    def expire(self, key, ttl):
        self.ttls[key] = ttl
        return True

    def ttl(self, key):
        return self.ttls.get(key, 3600)

    def pipeline(self):
        return FakePipeline(self)

    def keys(self, pattern):
        prefix = pattern.rstrip("*")
        return [key for key in self.data.keys() if str(key).startswith(prefix)]


def _request():
    from app.schemas import QueryRequest

    return QueryRequest(query="fila", modality="text", tenant_id="tenant-1")


def test_enqueue_query_job_returns_accepted_payload(monkeypatch):
    """Enqueueing should store the job and return the polling contract."""
    import app.query_jobs as qj

    fake_redis = FakeRedis()
    sent = {}

    monkeypatch.setattr(qj, "_get_job_store", lambda: fake_redis)
    monkeypatch.setattr(qj.celery_app, "send_task", lambda *args, **kwargs: sent.update({"args": args, "kwargs": kwargs}))

    out = qj.enqueue_query_job(
        req=_request(),
        correlation_id="cid-1",
        reason="ollama_overloaded",
        pressure_state="congested",
        route_path="/query",
    )

    assert out.job_id
    assert out.status.value == "queued"
    assert out.queue_depth == 1
    assert out.poll_after_seconds == 1.0
    assert sent["kwargs"]["task_id"] == out.job_id


def test_get_query_job_status_and_result(monkeypatch):
    """Status lookup and completed-result lookup should read the Redis payload."""
    import app.query_jobs as qj
    from app.schemas import QueryJobStatus, QueryResponse

    fake_redis = FakeRedis()
    monkeypatch.setattr(qj, "_get_job_store", lambda: fake_redis)
    fake_redis.setex(
        qj._job_key("job-1"),
        3600,
        qj._json_dumps(
            {
                "job_id": "job-1",
                "status": QueryJobStatus.COMPLETED.value,
                "created_at": 1.0,
                "started_at": 2.0,
                "finished_at": 3.0,
                "expires_at": 100.0,
                "result": QueryResponse(answer="ok", model="ollama/x").model_dump(mode="json"),
                "error": None,
            }
        ),
    )

    status = qj.get_query_job_status("job-1")
    result = qj.get_query_job_result("job-1")

    assert status.status == QueryJobStatus.COMPLETED
    assert status.poll_after_seconds == 1.0
    assert result.answer == "ok"


def test_get_query_job_result_raises_when_not_ready(monkeypatch):
    """Queued jobs should return HTTP 409 until completion."""
    import app.query_jobs as qj
    from app.schemas import QueryJobStatus

    fake_redis = FakeRedis()
    monkeypatch.setattr(qj, "_get_job_store", lambda: fake_redis)
    fake_redis.setex(
        qj._job_key("job-2"),
        3600,
        qj._json_dumps(
            {
                "job_id": "job-2",
                "status": QueryJobStatus.QUEUED.value,
                "created_at": 1.0,
                "expires_at": 100.0,
                "error": None,
                "result": None,
            }
        ),
    )

    with pytest.raises(HTTPException) as exc:
        qj.get_query_job_result("job-2")
    assert exc.value.status_code == 409


def test_enqueue_query_job_uses_ip_limit_without_tenant(monkeypatch):
    """IP fallback identities should use the dedicated pending limit."""
    import app.query_jobs as qj

    fake_redis = FakeRedis()
    monkeypatch.setattr(qj, "_get_job_store", lambda: fake_redis)
    monkeypatch.setattr(qj.celery_app, "send_task", lambda *args, **kwargs: None)
    monkeypatch.setattr(qj.settings, "get", lambda key, default=None: {"QUERY_JOB_MAX_PENDING_PER_IP": "2"}.get(key, default))

    req = _request().model_copy(update={"tenant_id": None})
    qj.enqueue_query_job(
        req=req,
        correlation_id="cid-1",
        reason="ollama_overloaded",
        pressure_state="elevated",
        route_path="/query",
        identity_key="127.0.0.1",
    )
    qj.enqueue_query_job(
        req=req,
        correlation_id="cid-2",
        reason="ollama_overloaded",
        pressure_state="elevated",
        route_path="/query",
        identity_key="127.0.0.1",
    )

    with pytest.raises(HTTPException) as exc:
        qj.enqueue_query_job(
            req=req,
            correlation_id="cid-3",
            reason="ollama_overloaded",
            pressure_state="elevated",
            route_path="/query",
            identity_key="127.0.0.1",
        )
    assert exc.value.status_code == 429
    assert exc.value.detail["identity_type"] == "ip"
    assert exc.value.detail["pending_limit"] == 2


def test_get_pending_query_jobs_count_totals_all_identities(monkeypatch):
    """Global queue depth should sum every identity bucket stored in Redis."""
    import app.query_jobs as qj

    fake_redis = FakeRedis()
    monkeypatch.setattr(qj, "_get_job_store", lambda: fake_redis)
    fake_redis.data[qj._tenant_pending_key("tenant-a")] = 3
    fake_redis.data[qj._tenant_pending_key("ip:1.2.3.4")] = 2

    assert qj.get_pending_query_jobs_count() == 5
    assert qj.get_pending_query_jobs_count("tenant-a") == 3
