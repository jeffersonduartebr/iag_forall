# app/tasks.py
"""
Celery tasks for background processing.

Optimized to reuse event loop instead of creating new one per task.
"""
import asyncio
import logging
import threading
import time
from typing import Optional, Any

from celery.signals import worker_process_init, worker_process_shutdown

from .celery_app import celery_app

# Importamos a lógica do core aqui dentro para evitar ciclos de importação no topo
# se o router_core importar tasks.py
from .router_core import process_background_feedback
from .roadmap_features import (
    get_eval_run,
    update_eval_run_status,
    add_eval_result,
)
from .router_core import route_and_answer

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
    """Initialize event loop when worker process starts."""
    logger.info("[Celery] Worker process initializing...")
    _get_or_create_event_loop()


@worker_process_shutdown.connect
def on_worker_process_shutdown(**kwargs):
    """Clean up event loop when worker process shuts down."""
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
    """
    Executa o feedback loop (Juízes, Bandit Update, Logging) em background via Celery.
    Isso garante que o aprendizado não seja perdido se a API reiniciar.

    Uses persistent event loop for better performance.
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
    """
    Execute an eval run asynchronously and persist per-prompt metrics.
    """
    logger.info("[Celery] Running eval run_id=%s", run_id)
    run = get_eval_run(run_id)
    if not run:
        logger.error("[Celery] Eval run not found: %s", run_id)
        return {"status": "not_found", "run_id": run_id}

    prompts = run.get("prompts") or []
    update_eval_run_status(run_id, "running", {"started_at": time.time(), "n_prompts": len(prompts)})

    quality_scores = []
    latency_scores = []
    cost_scores = []

    async def _execute():
        """Executa execute."""
        for prompt in prompts:
            try:
                resp = await route_and_answer(
                    query=str(prompt),
                    modality=modality,
                    use_cache=use_cache,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                meta = resp.get("metadata", {})
                q = float(meta.get("quality", 0.0) or 0.0)
                l = float(resp.get("latency_s", 0.0) or 0.0)
                c = float(resp.get("cost_per_1k", 0.0) or 0.0)
                add_eval_result(
                    run_id=run_id,
                    prompt_text=str(prompt),
                    model=str(resp.get("model") or "unknown"),
                    quality=q,
                    latency_s=l,
                    cost_usd=c,
                    metadata={"policy_version": run.get("policy_version")},
                )
                quality_scores.append(q)
                latency_scores.append(l)
                cost_scores.append(c)
            except Exception as e:
                add_eval_result(
                    run_id=run_id,
                    prompt_text=str(prompt),
                    model="error",
                    quality=0.0,
                    latency_s=0.0,
                    cost_usd=0.0,
                    metadata={"error": str(e)},
                )

    try:
        run_async(_execute())
        summary = {
            "finished_at": time.time(),
            "n": len(prompts),
            "quality_mean": (sum(quality_scores) / len(quality_scores)) if quality_scores else 0.0,
            "latency_mean": (sum(latency_scores) / len(latency_scores)) if latency_scores else 0.0,
            "cost_mean": (sum(cost_scores) / len(cost_scores)) if cost_scores else 0.0,
        }
        update_eval_run_status(run_id, "completed", summary)
        return {"status": "completed", "run_id": run_id, "summary": summary}
    except Exception as e:
        update_eval_run_status(run_id, "failed", {"error": str(e), "failed_at": time.time()})
        logger.error("[Celery] Eval run failed %s: %s", run_id, e)
        raise self.retry(exc=e)
