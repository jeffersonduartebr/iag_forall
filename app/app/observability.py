"""
observability.py
----------------------------------------------------
Gerencia métricas Prometheus (modo multiprocess ou single-process)
e logging estruturado via Structlog, com um único CollectorRegistry
compartilhado por toda a aplicação.

Principais pontos:
- NÃO apaga arquivos do diretório multiprocess (evita perda de métricas).
- Exporta helpers para renderizar /metrics com o registry correto.
- Mantém as métricas já utilizadas no projeto.
"""

from __future__ import annotations

import os
import logging
import structlog
from typing import Optional

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
    multiprocess,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST

# ============================================================
# ⚙️ Preparação do diretório multiprocess (sem limpeza destrutiva)
# ============================================================

def _ensure_prometheus_dir() -> Optional[str]:
    """
    Garante que o diretório PROMETHEUS_MULTIPROC_DIR exista.
    NÃO remove arquivos existentes (evita perda de métricas entre workers).
    Retorna o caminho do diretório quando definido/ok, senão None.
    """
    prom_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not prom_dir:
        # Operando em modo single-process
        print("[observability] PROMETHEUS_MULTIPROC_DIR não definido — modo single-process.")
        return None

    try:
        os.makedirs(prom_dir, exist_ok=True)
        print(f"[observability] Diretório multiprocess ativo: {prom_dir}")
        return prom_dir
    except Exception as e:
        # Se não conseguir criar o diretório, segue em modo single-process
        print(f"[observability] Aviso: falha ao preparar diretório multiprocess ({prom_dir}): {e}")
        return None


_prom_dir = _ensure_prometheus_dir()

# ============================================================
# 🧠 Inicialização do CollectorRegistry (único e compartilhado)
# ============================================================

def _build_registry() -> CollectorRegistry:
    """
    Cria o CollectorRegistry global.
    Se PROMETHEUS_MULTIPROC_DIR estiver definido, ativa MultiProcessCollector.
    """
    reg = CollectorRegistry()
    try:
        if _prom_dir:
            multiprocess.MultiProcessCollector(reg)
            print("[observability] Prometheus em modo multiprocess.")
        else:
            print("[observability] Prometheus em modo single-process.")
    except Exception as e:
        # Fallback de segurança para evitar crash em bootstrap
        print(f"[observability] Falha ao inicializar MultiProcessCollector: {e}")
        reg = CollectorRegistry()
    return reg


# Registry único exportado para o restante do sistema
registry: CollectorRegistry = _build_registry()

# ============================================================
# 📤 Helper para expor /metrics com o registry correto
# ============================================================

def render_metrics_response():
    """
    Retorna (body_bytes, content_type_str) para ser usado em endpoints /metrics.

    Exemplo (FastAPI):
        from fastapi import Response
        from app.observability import render_metrics_response

        @app.get("/metrics")
        def metrics():
            body, ctype = render_metrics_response()
            return Response(content=body, media_type=ctype)
    """
    data = generate_latest(registry)
    return data, CONTENT_TYPE_LATEST

# ============================================================
# 📈 Definição das métricas (todas registradas no `registry` global)
# ============================================================

# 🔹 API
API_REQUESTS = Counter(
    "api_requests_total",
    "Total de requisições recebidas pela API",
    registry=registry,
)
API_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latência das requisições da API (s)",
    registry=registry,
)

# 🔹 Roteamento e candidatos
ROUTER_CHOSEN = Counter(
    "router_chosen_model_total",
    "Modelo escolhido pelo roteador",
    ["model"],
    registry=registry,
)
FALLBACK_USED = Counter(
    "router_fallback_used_total",
    "Fallback usado entre modelos",
    ["first_model", "second_model"],
    registry=registry,
)
CANDIDATE_COST = Histogram(
    "candidate_estimated_cost_usd",
    "Custo estimado por resposta (USD)",
    registry=registry,
)
CANDIDATE_LAT = Histogram(
    "candidate_latency_seconds",
    "Latência por resposta (s)",
    registry=registry,
)
JUDGE_SCORE = Histogram(
    "judge_score",
    "Valores de pontuação dos juízes",
    ["judge_id"],
    registry=registry,
)
ROUTER_HISTORY_ENTRIES = Gauge(
    "router_history_entries_total",
    "Número de modelos com histórico EMA",
    registry=registry,
)

# 🔹 Bandit (aprendizado adaptativo)
BANDIT_SELECT = Counter(
    "bandit_select_total",
    "Seleções do bandit por modelo",
    ["model"],
    registry=registry,
)
BANDIT_UPDATE = Counter(
    "bandit_update_total",
    "Atualizações de bandit por modelo",
    ["model"],
    registry=registry,
)
BANDIT_REWARD = Histogram(
    "bandit_reward",
    "Valores observados de recompensa do bandit",
    registry=registry,
)

# 🔹 Custos e qualidade
ROUTER_MODEL_COST = Counter(
    "router_model_cost_usd_total",
    "Custo acumulado estimado por modelo (USD)",
    ["model"],
    registry=registry,
)
ROUTER_COST_SAVINGS = Counter(
    "router_cost_savings_usd_total",
    "Custo total economizado (USD)",
    registry=registry,
)
ROUTER_COST_PER_QUERY = Gauge(
    "router_cost_per_query_usd",
    "Custo médio por consulta (USD)",
    registry=registry,
)
ROUTER_QUALITY_AVG = Gauge(
    "router_model_quality_avg",
    "Qualidade média ponderada por modelo (0–10)",
    ["model"],
    registry=registry,
)
ROUTER_LOCAL_USAGE_RATIO = Gauge(
    "router_local_model_usage_ratio",
    "Proporção de requisições atendidas por modelos locais (Ollama)",
    registry=registry,
)

# ============================================================
# 🪵 Logging estruturado com Structlog (idempotente)
# ============================================================

_logger_configured = False
logger = structlog.get_logger("observability")

def setup_logging(level: int = logging.INFO):
    """
    Configura o Structlog para JSON com timestamp e nível de log.
    Idempotente: só configura uma vez.
    """
    global _logger_configured
    if _logger_configured:
        return

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _logger_configured = True
    logger.info("[observability] Logging configurado com sucesso.")
