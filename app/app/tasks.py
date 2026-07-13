# Objective: Application runtime code for tasks.
# app/tasks.py
"""Define Celery tasks for async feedback processing and evaluation runs.

The worker executes async router code from synchronous Celery tasks by reusing a
process-local event loop. That avoids the cost and complexity of creating a new
loop per task while keeping the public task interface conventional for Celery.
"""
import asyncio
import logging
import threading
import time
from typing import Any, Optional

from celery.signals import worker_process_init, worker_process_shutdown
from fastapi import HTTPException

# NB: `process_query_request` é importado de forma tardia (dentro das funções que
# o usam) para evitar o import circular tasks <-> services.query_runtime
# (query_runtime importa `task_process_feedback` deste módulo no nível do módulo).
from .benchmark_catalog import resolve_catalog_asset_path
from .celery_app import celery_app
from .query_jobs import finalize_query_job, update_query_job_record
from .roadmap_features import (
    add_eval_result,
    get_eval_run,
    update_eval_run_status,
)

# Importamos a lógica do core aqui dentro para evitar ciclos de importação no topo
# se o router_core importar tasks.py
from .router_core import process_background_feedback
from .schemas import QueryJobStatus, QueryRequest, WorkloadHints

logger = logging.getLogger("celery_tasks")


# ==============================================================================
# Persistent Event Loop for Celery Workers
# ==============================================================================

_worker_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get or create a persistent event loop for this worker thread.

    Instead of creating a new event loop per task with asyncio.run(),
    we reuse a single loop to avoid overhead and improve performance.
    """
    global _worker_loop

    with _loop_lock:
        if _worker_loop is None or _worker_loop.is_closed():
            _worker_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_worker_loop)
            logger.info("[Celery] Created persistent event loop for worker")

    return _worker_loop


def run_async(coro):
    """
    Run an async coroutine in the worker's persistent event loop.

    This is more efficient than asyncio.run() which creates a new loop each time.
    """
    loop = _get_or_create_event_loop()
    return loop.run_until_complete(coro)


# ==============================================================================
# Celery Task Signal Handlers
# ==============================================================================

@worker_process_init.connect
def on_worker_process_init(**kwargs):
    """Warm the persistent event loop when a worker process boots."""
    logger.info("[Celery] Worker process initializing...")
    _get_or_create_event_loop()


@worker_process_shutdown.connect
def on_worker_process_shutdown(**kwargs):
    """Close the persistent event loop during worker process shutdown."""
    global _worker_loop
    if _worker_loop is not None and not _worker_loop.is_closed():
        try:
            _worker_loop.close()
            logger.info("[Celery] Worker event loop closed")
        except Exception as e:
            logger.warning(f"[Celery] Error closing event loop: {e}")
    _worker_loop = None


# ==============================================================================
# Celery Tasks
# ==============================================================================

@celery_app.task(bind=True, queue="feedback_queue", max_retries=3, retry_backoff=True)
def task_process_feedback(
    self,
    query: str,
    answer: str,
    chosen_model: str,
    modality: str,
    latency_s: float,
    cost_val: float,
    image_b64: Optional[str] = None,
    raw_payload: Optional[Any] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0
):
    """Run the asynchronous feedback pipeline inside a Celery worker.

    The task bridges synchronous Celery execution and the async feedback logic
    in `router_core`. Retries are delegated to Celery so transient persistence
    or provider-side failures can recover with exponential backoff.
    """
    logger.info(f"[Celery] Processando feedback para modelo {chosen_model}...")

    try:
        # Use persistent event loop instead of asyncio.run()
        run_async(
            process_background_feedback(
                query=query,
                answer=answer,
                chosen_model=chosen_model,
                modality=modality,
                latency_s=latency_s,
                cost_val=cost_val,
                image_b64=image_b64,
                raw_payload=raw_payload,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
        )
        logger.info(f"[Celery] Feedback concluído com sucesso para {chosen_model}.")

    except Exception as e:
        logger.error(f"[Celery] Falha ao processar feedback: {e}")
        # Re-lança para o Celery tentar novamente (retry) com backoff exponencial
        raise self.retry(exc=e)


@celery_app.task(bind=True, queue="feedback_queue", max_retries=1, retry_backoff=True)
def task_execute_eval_run(
    self,
    run_id: str,
    modality: str = "text",
    use_cache: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.5,
):
    """Execute one stored evaluation run and persist per-prompt results.

    The task loads the run definition, executes every prompt through the normal
    routing stack, stores each result, and writes a final aggregate summary back
    to the roadmap/evaluation store. Task retries are reserved for failures
    that abort the whole run rather than one individual prompt.
    """
    logger.info("[Celery] Running eval run_id=%s", run_id)
    run = get_eval_run(run_id)
    if not run:
        logger.error("[Celery] Eval run not found: %s", run_id)
        return {"status": "not_found", "run_id": run_id}

    prompts = run.get("prompts") or []
    run_metadata = run.get("metadata") or {}
    prompt_catalog = run_metadata.get("prompt_catalog") or []
    catalog_by_query = {
        str(item.get("query", "")).strip(): item for item in prompt_catalog if isinstance(item, dict)
    }
    update_eval_run_status(run_id, "running", {"started_at": time.time(), "n_prompts": len(prompts)})

    frozen_meta = bool(run_metadata.get("frozen_policy")) or bool(
        (run_metadata.get("experiment_manifest") or {}).get("frozen_policy")
    )
    frozen_snapshot = (run_metadata.get("experiment_manifest") or {}).get("config_snapshot")

    quality_scores = []
    latency_scores = []
    cost_scores = []

    from .services.frozen_policy import frozen_policy_context

    frozen_ctx = frozen_policy_context(run_id, snapshot=frozen_snapshot) if frozen_meta else None
    if frozen_ctx is not None:
        frozen_ctx.__enter__()

    async def _execute():
        """Iterate through prompts and capture metrics for each eval sample."""
        from .services.query_runtime import process_query_request

        for prompt in prompts:
            prompt_text = str(prompt)
            catalog_item = catalog_by_query.get(prompt_text.strip(), {})
            try:
                image_b64: Optional[str] = None
                images: Optional[list[str]] = None
                req_modality = modality
                image_path = catalog_item.get("image_path")
                if image_path:
                    asset_path = resolve_catalog_asset_path(str(image_path))
                    image_bytes = asset_path.read_bytes()
                    import base64

                    image_b64 = base64.b64encode(image_bytes).decode("ascii")
                    images = [image_b64]
                    req_modality = "multimodal"

                theme = catalog_item.get("theme")
                benchmark_id = catalog_item.get("id")
                workload_hints = None
                if theme or benchmark_id:
                    workload_hints = WorkloadHints(  # type: ignore[call-arg]  # pydantic: campos opcionais têm default
                        theme=str(theme) if theme else None,
                        benchmark_id=str(benchmark_id) if benchmark_id else None,
                    )

                req = QueryRequest(  # type: ignore[call-arg]  # pydantic: campos opcionais têm default
                    query=prompt_text,
                    modality=req_modality,
                    images=images,
                    image_b64=image_b64,
                    enable_rag_for_answer=bool(catalog_item.get("enable_rag")),
                    use_cache=use_cache,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    policy_version=run.get("policy_version"),
                    tenant_id=run.get("tenant_id"),
                    workload_hints=workload_hints,
                )
                wrapped = await process_query_request(req)
                resp = wrapped.get("result", wrapped)
                meta = resp.get("metadata", {})
                q = float(meta.get("quality", 0.0) or 0.0)
                lat = float(resp.get("latency_s", 0.0) or 0.0)
                c = float(resp.get("estimated_cost_usd", resp.get("cost_per_1k", 0.0)) or 0.0)
                add_eval_result(
                    run_id=run_id,
                    prompt_text=prompt_text,
                    model=str(resp.get("model") or "unknown"),
                    quality=q,
                    latency_s=lat,
                    cost_usd=c,
                    metadata={
                        "policy_version": run.get("policy_version"),
                        "answer": str(resp.get("answer") or ""),
                        "grounded": bool(meta.get("grounded")),
                        "verification_status": meta.get("verification_status"),
                        "abstained": bool(meta.get("abstained")),
                        "knowledge_version": meta.get("knowledge_version"),
                        "confidence_score": meta.get("confidence_score"),
                        "workload_class": meta.get("workload_class"),
                        "detected_complexity": meta.get("detected_complexity"),
                        "benchmark_theme": catalog_item.get("theme"),
                        "benchmark_id": catalog_item.get("id"),
                        "reference": catalog_item.get("reference"),
                        "attack_strategy": catalog_item.get("attack_strategy"),
                    },
                )
                quality_scores.append(q)
                latency_scores.append(lat)
                cost_scores.append(c)
            except HTTPException as exc:
                add_eval_result(
                    run_id=run_id,
                    prompt_text=prompt_text,
                    model="error",
                    quality=0.0,
                    latency_s=0.0,
                    cost_usd=0.0,
                    metadata={"error": str(exc.detail), "status_code": exc.status_code},
                )
            except Exception as e:
                add_eval_result(
                    run_id=run_id,
                    prompt_text=prompt_text,
                    model="error",
                    quality=0.0,
                    latency_s=0.0,
                    cost_usd=0.0,
                    metadata={"error": str(e)},
                )

    try:
        run_async(_execute())
        summary: dict[str, Any] = {
            "finished_at": time.time(),
            "n": len(prompts),
            "quality_mean": (sum(quality_scores) / len(quality_scores)) if quality_scores else 0.0,
            "latency_mean": (sum(latency_scores) / len(latency_scores)) if latency_scores else 0.0,
            "cost_mean": (sum(cost_scores) / len(cost_scores)) if cost_scores else 0.0,
        }
        if frozen_ctx is not None:
            frozen_ctx.__exit__(None, None, None)
        update_eval_run_status(run_id, "completed", summary)
        try:
            from .services.eval_feedback import apply_eval_run_feedback

            feedback = apply_eval_run_feedback(run_id, summary=summary, metadata=run_metadata)
            summary["eval_feedback"] = feedback
            update_eval_run_status(run_id, "completed", summary)
        except Exception as exc:
            logger.warning("[Celery] Eval feedback tuning skipped for %s: %s", run_id, exc)
        try:
            from .observability import EVAL_RUN_COMPLETED

            EVAL_RUN_COMPLETED.labels(status="completed").inc()
        except Exception:
            pass
        return {"status": "completed", "run_id": run_id, "summary": summary}
    except Exception as e:
        if frozen_ctx is not None:
            frozen_ctx.__exit__(None, None, None)
        update_eval_run_status(run_id, "failed", {"error": str(e), "failed_at": time.time()})
        try:
            from .observability import EVAL_RUN_COMPLETED

            EVAL_RUN_COMPLETED.labels(status="failed").inc()
        except Exception:
            pass
        logger.error("[Celery] Eval run failed %s: %s", run_id, e)
        raise self.retry(exc=e)


@celery_app.task(bind=True, queue="celery", max_retries=0)
def task_execute_query_job(
    self,
    job_id: str,
    request_payload: dict[str, Any],
    correlation_id: Optional[str] = None,
):
    """Execute one queued query job and persist its terminal status/result."""
    from .correlation import CorrelationIdContext
    from .schemas import QueryRequest
    from .services.query_response_builder import build_query_response
    from .services.query_runtime import process_query_request, record_query_side_effects

    logger.info("[Celery] Running queued query job_id=%s", job_id)
    started_at = time.time()
    update_query_job_record(job_id, status="running", started_at=started_at)

    try:
        req = QueryRequest.model_validate(request_payload)
        with CorrelationIdContext(correlation_id):
            processed = run_async(process_query_request(req))
            result = processed["result"]
            image_input = processed["image_input"]
            metadata = result.setdefault("metadata", {})
            metadata["correlation_id"] = correlation_id
            record_query_side_effects(req, result, image_input)
            response_model = build_query_response(result, correlation_id)
        finalize_query_job(
            job_id,
            status=QueryJobStatus.COMPLETED,
            result=response_model.model_dump(mode="json"),
        )
        return {"status": "completed", "job_id": job_id}
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": True, "message": str(exc.detail)}
        finalize_query_job(
            job_id,
            status=QueryJobStatus.FAILED,
            error={"status_code": exc.status_code, **detail},
        )
        logger.warning("[Celery] Queued query job failed %s with HTTPException %s", job_id, exc.status_code)
        return {"status": "failed", "job_id": job_id}
    except Exception as exc:
        logger.exception("[Celery] Queued query job failed %s: %s", job_id, exc)
        finalize_query_job(
            job_id,
            status=QueryJobStatus.FAILED,
            error={"error": True, "message": str(exc), "status_code": 500},
        )
        return {"status": "failed", "job_id": job_id}
