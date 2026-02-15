"""Módulo principal: descreve responsabilidades e integrações deste arquivo."""

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
    }
)
