# -*- coding: utf-8 -*-
# Objective: Queue overflow queries as asynchronous jobs and persist their status/results in Redis.
"""Manage queued query jobs used when interactive capacity is temporarily exhausted."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import HTTPException

from .celery_app import celery_app
from .correlation import generate_correlation_id
from .observability import (
    QUERY_JOB_EXECUTION_SECONDS,
    QUERY_JOB_POLLING_SERVED,
    QUERY_JOB_QUEUE_SIZE,
    QUERY_JOB_WAIT_SECONDS,
    QUERY_JOBS_COMPLETED,
    QUERY_JOBS_FAILED,
    QUERY_JOBS_QUEUED,
    QUERY_JOBS_SATURATED,
)
from .schemas import QueryJobStatus, QueryJobStatusResponse, QueryRequest, QueuedQueryAcceptedResponse, QueryResponse
from .settings_dynamic import settings
from .utils.redis_client import get_redis_async_safe

logger = logging.getLogger(__name__)

JOB_KEY_PREFIX = "query_job:"
TENANT_PENDING_PREFIX = "query_job_pending:"
DEFAULT_JOB_TTL_SECONDS = 3600
DEFAULT_MAX_PENDING_PER_TENANT = 500
DEFAULT_MAX_PENDING_PER_IP = 250


def _job_key(job_id: str) -> str:
    """Return the Redis key used to store one job record."""
    return f"{JOB_KEY_PREFIX}{job_id}"


def _tenant_pending_key(identity: str) -> str:
    """Return the Redis key used to track pending jobs for one identity."""
    return f"{TENANT_PENDING_PREFIX}{identity}"


def _json_dumps(payload: Dict[str, Any]) -> str:
    """Serialize job payloads consistently."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _json_loads(raw: Any) -> Dict[str, Any]:
    """Deserialize one stored Redis JSON payload."""
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _job_ttl_seconds() -> int:
    """Return the retention period for queued query jobs."""
    try:
        return max(60, int(settings.get("QUERY_JOB_TTL_SECONDS", DEFAULT_JOB_TTL_SECONDS)))
    except Exception:
        return DEFAULT_JOB_TTL_SECONDS


def _job_pending_limit() -> int:
    """Return the maximum number of pending queued jobs per identity."""
    try:
        return max(1, int(settings.get("QUERY_JOB_MAX_PENDING_PER_TENANT", DEFAULT_MAX_PENDING_PER_TENANT)))
    except Exception:
        return DEFAULT_MAX_PENDING_PER_TENANT


def _job_pending_limit_for(identity_type: str) -> int:
    """Return the maximum pending queued jobs for one identity type."""
    setting_key = "QUERY_JOB_MAX_PENDING_PER_IP" if identity_type == "ip" else "QUERY_JOB_MAX_PENDING_PER_TENANT"
    default = DEFAULT_MAX_PENDING_PER_IP if identity_type == "ip" else DEFAULT_MAX_PENDING_PER_TENANT
    try:
        return max(1, int(settings.get(setting_key, default)))
    except Exception:
        return default


def _poll_after_seconds(queue_depth: int) -> float:
    """Return a conservative polling interval suggestion based on backlog depth."""
    if queue_depth >= 100:
        return 3.0
    if queue_depth >= 25:
        return 2.0
    return 1.0


def _estimate_wait_seconds(queue_depth: int) -> float:
    """Estimate queue wait time from current depth and provider concurrency."""
    try:
        from .providers_async import get_ollama_admission_snapshot

        snapshot = get_ollama_admission_snapshot()
        concurrency = max(1, int(snapshot.get("current_limit", 1) or 1))
    except Exception:
        concurrency = 1
    estimated_batches = max(1.0, float(queue_depth + 1) / float(concurrency))
    return round(estimated_batches, 2)


def _resolve_pending_identity(req: QueryRequest, identity_key: Optional[str]) -> tuple[str, str]:
    """Resolve the effective async-queue identity and whether it is tenant- or IP-based."""
    tenant_identity = (req.tenant_id or "").strip()
    if tenant_identity:
        return tenant_identity, "tenant"
    ip_identity = (identity_key or "").strip() or "ip:unknown"
    return ip_identity, "ip"


def get_pending_query_jobs_count(identity: Optional[str] = None) -> int:
    """Return pending queued jobs globally or for one identity."""
    redis_client = _get_job_store()
    if redis_client is None:
        return 0
    try:
        if identity:
            return max(0, int(redis_client.get(_tenant_pending_key(identity)) or 0))
        keys = list(redis_client.keys(f"{TENANT_PENDING_PREFIX}*"))
        total = 0
        for key in keys:
            total += max(0, int(redis_client.get(key) or 0))
        return total
    except Exception:
        return 0


def _build_urls(job_id: str) -> tuple[str, str]:
    """Build polling URLs for one queued query job."""
    return f"/query/jobs/{job_id}", f"/query/jobs/{job_id}/result"


def _get_job_store():
    """Return a non-blocking Redis client for job storage."""
    return get_redis_async_safe()


def enqueue_query_job(
    *,
    req: QueryRequest,
    correlation_id: Optional[str],
    reason: str,
    pressure_state: str,
    route_path: str,
    identity_key: Optional[str] = None,
) -> QueuedQueryAcceptedResponse:
    """Persist and enqueue one query job, returning the public acceptance payload."""
    redis_client = _get_job_store()
    if redis_client is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": True,
                "message": "Fila assíncrona indisponível no momento.",
                "category": "queue_unavailable",
            },
        )

    tenant_identity, identity_type = _resolve_pending_identity(req, identity_key)
    pending_key = _tenant_pending_key(tenant_identity)
    pending_count = 0
    try:
        pending_count = int(redis_client.get(pending_key) or 0)
    except Exception:
        pending_count = 0
    pending_limit = _job_pending_limit_for(identity_type)
    if pending_count >= pending_limit:
        try:
            QUERY_JOBS_SATURATED.labels(identity_type=identity_type).inc()
        except Exception:
            pass
        raise HTTPException(
            status_code=429,
            detail={
                "error": True,
                "message": "Fila de queries saturada para esta identidade. Tente novamente em instantes.",
                "category": "query_queue_saturated",
                "tenant_id": req.tenant_id,
                "identity_type": identity_type,
                "identity_key": tenant_identity,
                "pending_jobs": pending_count,
                "pending_limit": pending_limit,
            },
        )

    job_id = str(uuid.uuid4())
    created_at = time.time()
    expires_at = created_at + _job_ttl_seconds()
    corr_id = correlation_id or generate_correlation_id()
    payload = {
        "job_id": job_id,
        "status": QueryJobStatus.QUEUED.value,
        "created_at": created_at,
        "started_at": None,
        "finished_at": None,
        "expires_at": expires_at,
        "reason": reason,
        "pressure_state": pressure_state,
        "route_path": route_path,
        "correlation_id": corr_id,
        "pending_identity": tenant_identity,
        "pending_identity_type": identity_type,
        "request": req.model_dump(mode="json"),
        "error": None,
        "result": None,
    }

    try:
        pipe = redis_client.pipeline()
        pipe.setex(_job_key(job_id), _job_ttl_seconds(), _json_dumps(payload))
        pipe.incr(pending_key)
        pipe.expire(pending_key, _job_ttl_seconds())
        pipe.execute()
        try:
            QUERY_JOBS_QUEUED.labels(reason=reason).inc()
            QUERY_JOB_QUEUE_SIZE.inc()
        except Exception:
            pass
        celery_app.send_task(
            "app.tasks.task_execute_query_job",
            kwargs={
                "job_id": job_id,
                "request_payload": req.model_dump(mode="json"),
                "correlation_id": corr_id,
            },
            task_id=job_id,
            queue="celery",
        )
    except Exception as exc:
        logger.error("[query_jobs] Failed to enqueue query job %s: %s", job_id, exc)
        try:
            redis_client.delete(_job_key(job_id))
            if pending_count <= 0:
                redis_client.delete(pending_key)
            else:
                redis_client.decr(pending_key)
        except Exception:
            pass
        raise HTTPException(
            status_code=503,
            detail={
                "error": True,
                "message": "Falha ao enfileirar a query.",
                "category": "queue_enqueue_failed",
            },
        )

    poll_url, result_url = _build_urls(job_id)
    queue_depth = max(0, pending_count + 1)
    return QueuedQueryAcceptedResponse(
        job_id=job_id,
        status=QueryJobStatus.QUEUED,
        poll_url=poll_url,
        result_url=result_url,
        expires_at=expires_at,
        queue_depth=queue_depth,
        estimated_wait_seconds=_estimate_wait_seconds(queue_depth),
        poll_after_seconds=_poll_after_seconds(queue_depth),
    )


def get_query_job_record(job_id: str) -> Dict[str, Any]:
    """Return one raw queued-query record from Redis."""
    redis_client = _get_job_store()
    if redis_client is None:
        raise HTTPException(status_code=503, detail={"error": True, "message": "Fila assíncrona indisponível."})
    raw = redis_client.get(_job_key(job_id))
    if raw is None:
        raise HTTPException(status_code=404, detail={"error": True, "message": "Query job not found."})
    return _json_loads(raw)


def get_query_job_status(job_id: str) -> QueryJobStatusResponse:
    """Return the observable status model for one queued query job."""
    record = get_query_job_record(job_id)
    pending_identity = str(record.get("pending_identity") or "").strip() or None
    queue_depth = get_pending_query_jobs_count(pending_identity) if pending_identity else get_pending_query_jobs_count()
    try:
        QUERY_JOB_POLLING_SERVED.labels(endpoint="status").inc()
    except Exception:
        pass
    return QueryJobStatusResponse(
        job_id=job_id,
        status=QueryJobStatus(record.get("status", QueryJobStatus.EXPIRED.value)),
        created_at=float(record.get("created_at") or 0.0),
        started_at=float(record["started_at"]) if record.get("started_at") is not None else None,
        finished_at=float(record["finished_at"]) if record.get("finished_at") is not None else None,
        expires_at=float(record["expires_at"]) if record.get("expires_at") is not None else None,
        error=record.get("error"),
        poll_after_seconds=_poll_after_seconds(max(0, queue_depth)),
    )


def get_query_job_result(job_id: str) -> QueryResponse:
    """Return the completed QueryResponse for one queued query job."""
    record = get_query_job_record(job_id)
    try:
        QUERY_JOB_POLLING_SERVED.labels(endpoint="result").inc()
    except Exception:
        pass
    status = QueryJobStatus(record.get("status", QueryJobStatus.EXPIRED.value))
    if status == QueryJobStatus.COMPLETED:
        result_payload = record.get("result") or {}
        return QueryResponse.model_validate(result_payload)
    if status == QueryJobStatus.FAILED:
        raise HTTPException(status_code=500, detail=record.get("error") or {"error": True, "message": "Queued query failed."})
    if status == QueryJobStatus.EXPIRED:
        raise HTTPException(status_code=404, detail={"error": True, "message": "Queued query expired."})
    raise HTTPException(
        status_code=409,
        detail={
            "error": True,
            "message": "Queued query not finished yet.",
            "status": status.value,
        },
    )


def update_query_job_record(job_id: str, **fields: Any) -> None:
    """Merge new fields into one queued-query record and refresh its TTL."""
    redis_client = _get_job_store()
    if redis_client is None:
        return
    raw = redis_client.get(_job_key(job_id))
    if raw is None:
        return
    payload = _json_loads(raw)
    payload.update(fields)
    ttl = max(1, int(redis_client.ttl(_job_key(job_id)) or _job_ttl_seconds()))
    redis_client.setex(_job_key(job_id), ttl, _json_dumps(payload))


def finalize_query_job(job_id: str, *, status: QueryJobStatus, result: Optional[Dict[str, Any]] = None, error: Optional[Dict[str, Any]] = None) -> None:
    """Persist the terminal state of one queued query job and decrement tenant pending count."""
    redis_client = _get_job_store()
    if redis_client is None:
        return
    raw = redis_client.get(_job_key(job_id))
    if raw is None:
        return
    payload = _json_loads(raw)
    payload["status"] = status.value
    payload["finished_at"] = time.time()
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error
    ttl = max(1, int(redis_client.ttl(_job_key(job_id)) or _job_ttl_seconds()))
    started_at = float(payload.get("started_at") or payload.get("created_at") or payload["finished_at"])
    created_at = float(payload.get("created_at") or started_at)
    wait_seconds = max(0.0, started_at - created_at)
    execution_seconds = max(0.0, float(payload["finished_at"]) - started_at)
    pipe = redis_client.pipeline()
    pipe.setex(_job_key(job_id), ttl, _json_dumps(payload))
    tenant_identity = (
        str(payload.get("pending_identity") or "").strip()
        or ((payload.get("request") or {}).get("tenant_id") or "").strip()
        or "ip:unknown"
    )
    pending_key = _tenant_pending_key(tenant_identity)
    pipe.decr(pending_key)
    pipe.expire(pending_key, ttl)
    pipe.execute()
    try:
        QUERY_JOB_QUEUE_SIZE.dec()
        QUERY_JOB_WAIT_SECONDS.observe(wait_seconds)
        QUERY_JOB_EXECUTION_SECONDS.observe(execution_seconds)
        if status == QueryJobStatus.COMPLETED:
            QUERY_JOBS_COMPLETED.inc()
        elif status == QueryJobStatus.FAILED:
            QUERY_JOBS_FAILED.inc()
    except Exception:
        pass
