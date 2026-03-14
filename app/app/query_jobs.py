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
    QUERY_JOB_QUEUE_SIZE,
    QUERY_JOB_WAIT_SECONDS,
    QUERY_JOBS_COMPLETED,
    QUERY_JOBS_FAILED,
    QUERY_JOBS_QUEUED,
)
from .schemas import QueryJobStatus, QueryJobStatusResponse, QueryRequest, QueuedQueryAcceptedResponse, QueryResponse
from .settings_dynamic import settings
from .utils.redis_client import get_redis_async_safe

logger = logging.getLogger(__name__)

JOB_KEY_PREFIX = "query_job:"
TENANT_PENDING_PREFIX = "query_job_pending:"
DEFAULT_JOB_TTL_SECONDS = 3600
DEFAULT_MAX_PENDING_PER_TENANT = 100


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

    tenant_identity = (req.tenant_id or "").strip() or (identity_key or "").strip() or "ip:unknown"
    pending_key = _tenant_pending_key(tenant_identity)
    pending_count = 0
    try:
        pending_count = int(redis_client.get(pending_key) or 0)
    except Exception:
        pending_count = 0
    pending_limit = _job_pending_limit()
    if pending_count >= pending_limit:
        raise HTTPException(
            status_code=429,
            detail={
                "error": True,
                "message": "Fila do tenant saturada. Tente novamente em instantes.",
                "category": "query_queue_saturated",
                "tenant_id": req.tenant_id,
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
    return QueuedQueryAcceptedResponse(
        job_id=job_id,
        status=QueryJobStatus.QUEUED,
        poll_url=poll_url,
        result_url=result_url,
        expires_at=expires_at,
        estimated_wait_seconds=float(max(1, pending_count)),
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
    return QueryJobStatusResponse(
        job_id=job_id,
        status=QueryJobStatus(record.get("status", QueryJobStatus.EXPIRED.value)),
        created_at=float(record.get("created_at") or 0.0),
        started_at=float(record["started_at"]) if record.get("started_at") is not None else None,
        finished_at=float(record["finished_at"]) if record.get("finished_at") is not None else None,
        expires_at=float(record["expires_at"]) if record.get("expires_at") is not None else None,
        error=record.get("error"),
    )


def get_query_job_result(job_id: str) -> QueryResponse:
    """Return the completed QueryResponse for one queued query job."""
    record = get_query_job_record(job_id)
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
