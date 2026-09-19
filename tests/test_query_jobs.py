# Objective: Test coverage for asynchronous queued query jobs and their Redis-backed status store.
"""Unit tests for queued query job helpers."""

from __future__ import annotations

import fakeredis
import pytest
from fastapi import HTTPException


def _request():
    from app.schemas import QueryRequest

    return QueryRequest(query="fila", modality="text", tenant_id="tenant-1")


def test_enqueue_query_job_returns_accepted_payload(monkeypatch):
    """Enqueueing should store the job and return the polling contract."""
    import app.query_jobs as qj

    fake_redis = fakeredis.FakeRedis()
    sent = {}

    monkeypatch.setattr(qj, "_get_job_store", lambda: fake_redis)
    monkeypatch.setattr(
        qj.celery_app, "send_task", lambda *args, **kwargs: sent.update({"args": args, "kwargs": kwargs})
    )

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

    monkeypatch.setenv("REQUIRE_API_AUTH", "0")
    try:
        from app.settings_dynamic import _lru

        _lru.clear()
    except Exception:
        pass

    fake_redis = fakeredis.FakeRedis()
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

    monkeypatch.setenv("REQUIRE_API_AUTH", "0")
    try:
        from app.settings_dynamic import _lru

        _lru.clear()
    except Exception:
        pass

    fake_redis = fakeredis.FakeRedis()
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

    fake_redis = fakeredis.FakeRedis()
    monkeypatch.setattr(qj, "_get_job_store", lambda: fake_redis)
    monkeypatch.setattr(qj.celery_app, "send_task", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        qj.settings, "get", lambda key, default=None: {"QUERY_JOB_MAX_PENDING_PER_IP": "2"}.get(key, default)
    )

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

    fake_redis = fakeredis.FakeRedis()
    monkeypatch.setattr(qj, "_get_job_store", lambda: fake_redis)
    fake_redis.set(qj._tenant_pending_key("tenant-a"), 3)
    fake_redis.set(qj._tenant_pending_key("ip:1.2.3.4"), 2)

    assert qj.get_pending_query_jobs_count() == 5
    assert qj.get_pending_query_jobs_count("tenant-a") == 3


def _stored_job(qj, fake_redis, job_id, ttl=600, **fields):
    payload = {"job_id": job_id, "status": "running", "created_at": 100.0, "started_at": 104.0, **fields}
    fake_redis.setex(qj._job_key(job_id), ttl, qj._json_dumps(payload))
    return payload


@pytest.fixture
def finalize_env(monkeypatch):
    import app.query_jobs as qj

    fake_redis = fakeredis.FakeRedis()
    webhooks = []
    monkeypatch.setattr(qj, "_get_job_store", lambda: fake_redis)
    monkeypatch.setattr(qj.time, "time", lambda: 110.0)
    monkeypatch.setattr(
        "app.services.query_webhooks.schedule_query_job_webhook", lambda **kwargs: webhooks.append(kwargs)
    )
    return qj, fake_redis, webhooks


def test_finalize_query_job_persists_result_and_releases_pending(finalize_env):
    from app.schemas import QueryJobStatus

    qj, fake_redis, webhooks = finalize_env
    _stored_job(qj, fake_redis, "j1", pending_identity="tenant-1", request={"webhook_url": "https://hook"})
    fake_redis.set(qj._tenant_pending_key("tenant-1"), 2)

    qj.finalize_query_job("j1", status=QueryJobStatus.COMPLETED, result={"answer": "ok"})

    stored = qj._json_loads(fake_redis.get(qj._job_key("j1")))
    assert (stored["status"], stored["finished_at"], stored["result"]) == ("completed", 110.0, {"answer": "ok"})
    assert 590 < fake_redis.ttl(qj._job_key("j1")) <= 600  # mantém o TTL restante
    assert int(fake_redis.get(qj._tenant_pending_key("tenant-1"))) == 1
    assert webhooks == [
        {
            "webhook_url": "https://hook",
            "job_id": "j1",
            "status": "completed",
            "result": {"answer": "ok"},
            "error": None,
        }
    ]


def test_finalize_query_job_failure_uses_request_tenant_and_default_ttl(finalize_env):
    from app.schemas import QueryJobStatus

    qj, fake_redis, _ = finalize_env
    _stored_job(qj, fake_redis, "j2", request={"tenant_id": "t9"})
    fake_redis.persist(qj._job_key("j2"))  # sem TTL: não pode virar 1 s

    qj.finalize_query_job("j2", status=QueryJobStatus.FAILED, error={"detail": "boom"})

    stored = qj._json_loads(fake_redis.get(qj._job_key("j2")))
    assert stored["error"] == {"detail": "boom"} and "result" not in stored
    assert fake_redis.ttl(qj._job_key("j2")) > 1
    assert int(fake_redis.get(qj._tenant_pending_key("t9"))) == -1


def test_finalize_query_job_missing_job_or_store(finalize_env, monkeypatch):
    from app.schemas import QueryJobStatus

    qj, fake_redis, webhooks = finalize_env
    qj.finalize_query_job("ghost", status=QueryJobStatus.COMPLETED)
    monkeypatch.setattr(qj, "_get_job_store", lambda: None)
    qj.finalize_query_job("ghost", status=QueryJobStatus.COMPLETED)
    assert webhooks == [] and fake_redis.keys() == []


def test_job_timings_and_pending_identity():
    import app.query_jobs as qj

    assert qj._job_timings({"created_at": 100.0, "started_at": 104.0, "finished_at": 110.0}) == (4.0, 6.0)
    assert qj._job_timings({"finished_at": 110.0}) == (0.0, 0.0)
    assert qj._payload_pending_identity({"pending_identity": " ip:1.2.3.4 "}) == "ip:1.2.3.4"
    assert qj._payload_pending_identity({"request": {"tenant_id": "t1"}}) == "t1"
    assert qj._payload_pending_identity({}) == "ip:unknown"
