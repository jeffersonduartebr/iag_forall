# Objective: Async query-job lifecycle — queued→running→completed/failed/expired, quotas, rollback, TTL, redelivery.
"""Coverage of app.query_jobs failure paths against a real-semantics fakeredis store."""

from __future__ import annotations

import app.query_jobs as qj
import pytest
from app.api.auth import AuthContext
from app.schemas import QueryJobStatus, QueryRequest, QueryResponse
from fastapi import HTTPException


class _Settings:
    def __init__(self, values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


@pytest.fixture
def store(monkeypatch, fake_redis, fake_clock):
    """fakeredis job store, captured Celery sends and webhooks, fixed clock."""
    monkeypatch.setattr(qj, "_get_job_store", lambda: fake_redis)
    sent, hooks = [], []
    monkeypatch.setattr(qj.celery_app, "send_task", lambda *a, **k: sent.append(k))
    monkeypatch.setattr("app.services.query_webhooks.schedule_query_job_webhook", lambda **k: hooks.append(k))
    monkeypatch.setattr(qj, "verify_job_access", None)
    fake_redis.sent, fake_redis.hooks, fake_redis.clock = sent, hooks, fake_clock(qj)
    return fake_redis


def _enqueue(tenant="escola-1", identity_key=None, **kw):
    req = QueryRequest(query="q", tenant_id=tenant, webhook_url="https://hook.test/x")
    return qj.enqueue_query_job(req=req, correlation_id="cid", reason="ollama_overloaded", pressure_state="congested",
                                route_path="/query", identity_key=identity_key, **kw)


def _pending(store, identity):
    return int(store.get(qj._tenant_pending_key(identity)) or 0)


def test_enqueue_without_redis_is_503(monkeypatch):
    monkeypatch.setattr(qj, "_get_job_store", lambda: None)
    with pytest.raises(HTTPException) as exc:
        _enqueue()
    assert exc.value.status_code == 503
    assert exc.value.detail["category"] == "queue_unavailable"


def test_full_lifecycle_completed(store):
    accepted = _enqueue()
    assert _pending(store, "escola-1") == 1
    assert store.sent[0]["task_id"] == accepted.job_id
    status = qj.get_query_job_status(accepted.job_id)
    assert status.status == QueryJobStatus.QUEUED
    with pytest.raises(HTTPException) as exc:
        qj.get_query_job_result(accepted.job_id)
    assert exc.value.status_code == 409 and exc.value.detail["status"] == "queued"

    store.clock.advance(2)
    qj.update_query_job_record(accepted.job_id, status="running", started_at=store.clock.now)
    assert qj.get_query_job_status(accepted.job_id).started_at == store.clock.now
    store.clock.advance(3)
    answer = QueryResponse(answer="ok", model="ollama/x").model_dump(mode="json")
    qj.finalize_query_job(accepted.job_id, status=QueryJobStatus.COMPLETED, result=answer)

    assert qj.get_query_job_result(accepted.job_id).answer == "ok"
    record = qj.get_query_job_record(accepted.job_id)
    assert qj._job_timings(record) == (2.0, 3.0)
    assert _pending(store, "escola-1") == 0
    assert store.ttl(qj._job_key(accepted.job_id)) > 60
    assert store.hooks[0]["webhook_url"] == "https://hook.test/x"
    assert store.hooks[0]["status"] == "completed"


def test_failed_job_surfaces_stored_error(store):
    job_id = _enqueue().job_id
    qj.finalize_query_job(job_id, status=QueryJobStatus.FAILED, error={"status_code": 502, "message": "provider"})
    with pytest.raises(HTTPException) as exc:
        qj.get_query_job_result(job_id)
    assert exc.value.status_code == 500
    assert exc.value.detail["message"] == "provider"
    assert qj.get_query_job_status(job_id).error["status_code"] == 502


def test_expired_and_unknown_jobs_are_404(store):
    store.set(qj._job_key("old"), qj._json_dumps({"status": "expired", "created_at": 1.0}))
    with pytest.raises(HTTPException) as exc:
        qj.get_query_job_result("old")
    assert exc.value.detail["message"] == "Queued query expired."
    with pytest.raises(HTTPException) as exc:
        qj.get_query_job_status("never")
    assert exc.value.status_code == 404


def test_redelivered_task_does_not_release_pending_slot_twice(store):
    """Regression: task_acks_late redelivery finalized twice and drove the tenant counter below its real value."""
    first, second = _enqueue(), _enqueue()
    assert _pending(store, "escola-1") == 2
    qj.finalize_query_job(first.job_id, status=QueryJobStatus.COMPLETED, result={"answer": "a", "model": "m"})
    qj.update_query_job_record(first.job_id, status="running")  # redelivery restarts the job
    qj.finalize_query_job(first.job_id, status=QueryJobStatus.FAILED, error={"message": "retry"})
    assert _pending(store, "escola-1") == 1  # `second` is still pending
    assert qj.get_query_job_record(first.job_id)["status"] == "failed"
    assert len(store.hooks) == 1
    assert second.job_id


def test_update_keeps_full_retention_when_key_has_no_ttl(store):
    """Regression: a record without TTL (ttl=-1) was rewritten with a 1-second TTL and vanished."""
    store.set(qj._job_key("j"), qj._json_dumps({"status": "queued"}))
    qj.update_query_job_record("j", status="running")
    assert store.ttl(qj._job_key("j")) > qj.DEFAULT_JOB_TTL_SECONDS - 60


def test_update_and_finalize_are_noops_for_missing_job_or_store(store, monkeypatch):
    qj.update_query_job_record("ghost", status="running")
    qj.finalize_query_job("ghost", status=QueryJobStatus.COMPLETED)
    assert store.get(qj._job_key("ghost")) is None
    monkeypatch.setattr(qj, "_get_job_store", lambda: None)
    qj.update_query_job_record("ghost", status="running")
    qj.finalize_query_job("ghost", status=QueryJobStatus.COMPLETED)
    assert qj.get_pending_query_jobs_count() == 0
    with pytest.raises(HTTPException) as exc:
        qj.get_query_job_record("ghost")
    assert exc.value.status_code == 503


def test_tenant_saturation_is_429_and_does_not_leak_to_other_tenants(store, monkeypatch):
    monkeypatch.setattr(qj, "settings", _Settings({"QUERY_JOB_MAX_PENDING_PER_TENANT": "1"}))
    _enqueue("escola-1")
    with pytest.raises(HTTPException) as exc:
        _enqueue("escola-1")
    assert exc.value.status_code == 429
    assert exc.value.detail["pending_limit"] == 1 and exc.value.detail["identity_type"] == "tenant"
    assert _enqueue("escola-2").queue_depth == 1
    assert len(store.sent) == 2


def test_ip_identity_uses_its_own_limit(store, monkeypatch):
    monkeypatch.setattr(qj, "settings", _Settings({"QUERY_JOB_MAX_PENDING_PER_IP": "1"}))
    _enqueue(tenant=None, identity_key="ip:1.2.3.4")
    with pytest.raises(HTTPException) as exc:
        _enqueue(tenant=None, identity_key="ip:1.2.3.4")
    assert exc.value.detail["identity_key"] == "ip:1.2.3.4" and exc.value.detail["identity_type"] == "ip"
    _enqueue(tenant=None)
    assert _pending(store, "ip:unknown") == 1


@pytest.mark.parametrize("already_pending", [0, 2])
def test_broker_failure_rolls_back_record_and_counter(store, monkeypatch, already_pending):
    if already_pending:
        store.set(qj._tenant_pending_key("escola-1"), already_pending)

    def _broker_down(*a, **k):
        raise ConnectionError("broker down")

    monkeypatch.setattr(qj.celery_app, "send_task", _broker_down)
    with pytest.raises(HTTPException) as exc:
        _enqueue()
    assert exc.value.detail["category"] == "queue_enqueue_failed"
    assert _pending(store, "escola-1") == already_pending
    assert list(store.scan_iter(match="query_job:*")) == []


def test_owner_hash_is_recorded_and_enforced(store, monkeypatch):
    import app.api.auth as auth

    monkeypatch.setattr(qj, "verify_job_access", auth.verify_job_access)
    monkeypatch.setattr(auth.settings, "_get_bool", lambda key, default=False: key == "REQUIRE_API_AUTH")
    owner = AuthContext(authenticated=True, tenant_id="escola-1")
    job_id = _enqueue(auth=owner).job_id
    assert qj.get_query_job_status(job_id, owner).status == QueryJobStatus.QUEUED
    for intruder in (None, AuthContext(authenticated=True, tenant_id="escola-2")):
        with pytest.raises(HTTPException) as exc:
            qj.get_query_job_result(job_id, intruder)
        assert exc.value.status_code == 403


def test_global_pending_count_scans_all_identities_and_survives_errors(store, monkeypatch):
    for i in range(3):
        store.set(qj._tenant_pending_key(f"t{i}"), i + 1)
    store.set(qj._tenant_pending_key("neg"), -4)  # a corrupted counter must not subtract
    assert qj.get_pending_query_jobs_count() == 6
    assert qj.get_pending_query_jobs_count("t2") == 3

    def _boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(store, "scan", _boom)
    assert qj.get_pending_query_jobs_count() == 0


def test_status_without_identity_uses_global_depth(store):
    store.set(qj._tenant_pending_key("t"), 30)
    store.setex(qj._job_key("j"), 100, qj._json_dumps({"status": "running", "created_at": 1.0}))
    assert qj.get_query_job_status("j").poll_after_seconds == 2.0


def test_helpers_are_defensive(monkeypatch):
    assert qj._json_loads(None) == {} and qj._json_loads(b'{"a":1}') == {"a": 1}
    assert qj._json_loads({"b": 2}) == {"b": 2} and qj._json_loads(42) == {}
    monkeypatch.setattr(qj, "settings", _Settings({k: "x" for k in (
        "QUERY_JOB_TTL_SECONDS", "QUERY_JOB_MAX_PENDING_PER_TENANT", "QUERY_JOB_MAX_PENDING_PER_IP")}))
    assert qj._job_ttl_seconds() == qj.DEFAULT_JOB_TTL_SECONDS
    assert qj._job_pending_limit_for("tenant") == qj.DEFAULT_MAX_PENDING_PER_TENANT
    assert qj._job_pending_limit_for("ip") == qj.DEFAULT_MAX_PENDING_PER_IP
    assert [qj._poll_after_seconds(d) for d in (0, 25, 100)] == [1.0, 2.0, 3.0]
    assert qj._payload_pending_identity({"request": {"tenant_id": " t "}}) == "t"
    assert qj._payload_pending_identity({}) == "ip:unknown"


def test_wait_estimate_scales_with_provider_concurrency(monkeypatch):
    monkeypatch.setattr("app.providers_async.get_ollama_admission_snapshot", lambda: {"current_limit": 4})
    assert qj._estimate_wait_seconds(7) == 2.0

    def _boom():
        raise RuntimeError

    monkeypatch.setattr("app.providers_async.get_ollama_admission_snapshot", _boom)
    assert qj._estimate_wait_seconds(7) == 8.0


def test_metric_and_webhook_failures_never_break_finalize(store, monkeypatch):
    monkeypatch.setattr(qj, "QUERY_JOB_QUEUE_SIZE", None)
    monkeypatch.setattr(qj, "QUERY_JOBS_QUEUED", None)
    monkeypatch.setattr(qj, "QUERY_JOB_POLLING_SERVED", None)
    job_id = _enqueue().job_id

    def _hook_down(**k):
        raise RuntimeError("webhook down")

    monkeypatch.setattr("app.services.query_webhooks.schedule_query_job_webhook", _hook_down)
    qj.finalize_query_job(job_id, status=QueryJobStatus.FAILED, error={"message": "x"})
    assert qj.get_query_job_status(job_id).status == QueryJobStatus.FAILED
