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
from contextlib import nullcontext
from typing import Any, Optional

from celery.signals import worker_process_init, worker_process_shutdown
from fastapi import HTTPException

# NB: `process_query_request` é importado de forma tardia (dentro das funções que
# o usam) para evitar o import circular tasks <-> services.query_runtime
# (query_runtime importa `task_process_feedback` deste módulo no nível do módulo).
from .celery_app import celery_app
from .query_jobs import finalize_query_job, update_query_job_record
from .roadmap_features import (
    add_eval_result,
    get_eval_run,
    update_eval_run_status,
)
from .router_core import process_background_feedback
from .schemas import QueryJobStatus
from .services.eval_runner import EvalRunOptions, frozen_settings, run_eval_prompts, summarize

# Importamos a lógica do core aqui dentro para evitar ciclos de importação no topo
# se o router_core importar tasks.py
from .services.feedback_stages import FeedbackPersistError

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

    except FeedbackPersistError as e:
        # A linha não foi escrita, mas os juízes já correram e já foram pagos.
        # Repetir o pipeline compraria a mesma linha duas vezes, por isso a
        # tarefa falha sem retry: o valor aqui é deixar de reportar SUCCESS.
        logger.error(f"[Celery] Feedback não persistido para {chosen_model}: {e}")
        raise
    except Exception as e:
        logger.error(f"[Celery] Falha ao processar feedback: {e}")
        # Re-lança para o Celery tentar novamente (retry) com backoff exponencial
        raise self.retry(exc=e)


@celery_app.task(queue="shadow_queue", acks_late=False)
def task_shadow_evaluate(job: dict) -> int:
    """Execução em sombra de uma requisição amostrada (services/sombra). Sem retry: a linha registra o status.

    ``acks_late=False``: uma sombra reentregue depois de uma queda do worker pagaria as mesmas chamadas duas
    vezes; perder uma amostra rara é preferível (e a ausência aparece no número de sorteadas × gravadas).
    """
    from app.services.sombra.executor import executar

    try:
        return len(run_async(executar(job)))
    except Exception as exc:  # nunca propaga: a sombra não tem efeito sobre o atendimento
        logger.warning("[sombra] tarefa falhou: %s", type(exc).__name__)
        return 0


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
    update_eval_run_status(run_id, "running", {"started_at": time.time(), "n_prompts": len(prompts)})
    frozen, frozen_snapshot = frozen_settings(run_metadata)
    opts = EvalRunOptions(modality=modality, use_cache=use_cache, max_tokens=max_tokens, temperature=temperature)

    from .services.frozen_policy import frozen_policy_context

    try:
        applied_policy: dict[str, Any] = {}
        with frozen_policy_context(run_id, snapshot=frozen_snapshot) if frozen else nullcontext() as ctx:
            applied_policy = dict(ctx or {})
            totals = run_async(run_eval_prompts(run_id, run, opts, add_eval_result))
        summary: dict[str, Any] = summarize(len(prompts), totals)
        # A política sob a qual a corrida decorreu, gravada de forma durável.
        # O snapshot só existia no Redis com TTL de 24 h e não era lido por
        # nenhum caminho de análise: passadas 24 horas não havia como afirmar
        # sob que pesos um resultado tinha sido produzido.
        summary["frozen_policy"] = applied_policy or {"active": False}
        update_eval_run_status(run_id, "completed", summary)
        _apply_eval_feedback(run_id, summary, run_metadata)
        _count_eval_run("completed")
        return {"status": "completed", "run_id": run_id, "summary": summary}
    except Exception as e:
        update_eval_run_status(run_id, "failed", {"error": str(e), "failed_at": time.time()})
        _count_eval_run("failed")
        logger.error("[Celery] Eval run failed %s: %s", run_id, e)
        raise self.retry(exc=e)


def _apply_eval_feedback(run_id: str, summary: dict[str, Any], run_metadata: dict[str, Any]) -> None:
    """Feed the run summary back into the tuners (best effort) and store what changed."""
    try:
        from .services.eval_feedback import apply_eval_run_feedback

        summary["eval_feedback"] = apply_eval_run_feedback(run_id, summary=summary, metadata=run_metadata)
        update_eval_run_status(run_id, "completed", summary)
    except Exception as exc:
        logger.warning("[Celery] Eval feedback tuning skipped for %s: %s", run_id, exc)


def _count_eval_run(status: str) -> None:
    try:
        from .observability import EVAL_RUN_COMPLETED

        EVAL_RUN_COMPLETED.labels(status=status).inc()
    except Exception:
        pass


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
