# Objective: Application runtime code for celery app.
"""Configure the shared Celery application used by background workers.

This module centralizes broker and result-backend wiring for asynchronous tasks
executed outside the FastAPI request lifecycle. The application imports
``app.tasks`` eagerly so that worker startup always registers the task set used
by feedback processing and evaluation jobs.

The configuration intentionally reads connection details from environment
variables at import time because this module is loaded directly by the Celery
CLI. That makes it the single source of truth for:

- Redis broker and result backend URLs
- Task serialization policy
- Queue routing for background workloads
- Startup retry behavior for broker availability

The resulting ``celery_app`` object is imported both by the worker container and
by runtime code that needs to enqueue background jobs.
"""

# app/celery_app.py
import os
from celery import Celery

# Configurações do Broker (Redis)
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# Monta URL de conexão segura
if REDIS_PASSWORD:
    BROKER_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
    RESULT_BACKEND = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/1"
else:
    BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
    RESULT_BACKEND = f"redis://{REDIS_HOST}:{REDIS_PORT}/1"

# ==============================================================================
# 🚨 CORREÇÃO AQUI: Adicionado include=['app.tasks']
# Isso força o worker a importar o módulo tasks.py ao iniciar.
# ==============================================================================
celery_app = Celery("llm_router", include=["app.tasks"])

celery_app.conf.update(
    broker_url=BROKER_URL,
    result_backend=RESULT_BACKEND,
    broker_connection_retry_on_startup=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    worker_prefetch_multiplier=4,  # Optimized for high-capacity environment (64GB RAM, 8+ cores)
    task_acks_late=True,
    timezone="America/Fortaleza",
    enable_utc=True,
    # Opcional: Define rotas padrão se necessário, mas o queue no decorator já resolve
    task_routes={
        "app.tasks.task_process_feedback": {"queue": "feedback_queue"},
        "app.tasks.task_execute_eval_run": {"queue": "feedback_queue"},
        "app.tasks.task_execute_query_job": {"queue": "celery"},
    }
)
