# app/tasks.py
"""
Celery tasks for background processing.

Optimized to reuse event loop instead of creating new one per task.
"""
import asyncio
import logging
import threading
from typing import Optional, Any

from celery.signals import worker_process_init, worker_process_shutdown

from .celery_app import celery_app

# Importamos a lógica do core aqui dentro para evitar ciclos de importação no topo
# se o router_core importar tasks.py
from .router_core import process_background_feedback

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