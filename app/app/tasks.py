# app/tasks.py
import asyncio
import logging
from typing import Optional, Any

from .celery_app import celery_app

# Importamos a lógica do core aqui dentro para evitar ciclos de importação no topo
# se o router_core importar tasks.py
from .router_core import process_background_feedback

logger = logging.getLogger("celery_tasks")

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
    """
    logger.info(f"[Celery] Processando feedback para modelo {chosen_model}...")
    
    try:
        # Como process_background_feedback é async (usa await judge_answer),
        # precisamos rodar um event loop dentro do worker do Celery.
        asyncio.run(
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