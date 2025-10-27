# app/tasks.py
from .celery_app import celery_app
from .providers import safe_call_model

@celery_app.task(bind=True, max_retries=2, autoretry_for=(Exception,), retry_backoff=True)
def task_call_model(self, *, model: str, prompt: str, max_tokens: int, temperature: float, api_base: str | None):
    """
    Tarefa Celery para chamar o modelo via providers.safe_call_model.
    Retorna {"text": str, "meta": dict}
    """
    text, meta = safe_call_model(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        api_base=api_base
    )
    return {"text": text, "meta": meta}
