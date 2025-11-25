"""
observability.py
----------------------------------------------------
Gerencia métricas Prometheus (modo multiprocess ou single-process)
e logging estruturado via Structlog, com suporte a UTF-8.

Agora inclui TODAS as métricas do sistema (API, Router, Providers, Bandit).
"""
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import json
import logging
import structlog
from datetime import datetime
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
# ⚙️ Preparação do diretório multiprocess
# ============================================================

def _ensure_prometheus_dir() -> Optional[str]:
    """
    Garante que o diretório PROMETHEUS_MULTIPROC_DIR exista.
    NÃO remove arquivos existentes para evitar perda de dados entre workers.
    """
    prom_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not prom_dir:
        # print("[observability] PROMETHEUS_MULTIPROC_DIR não definido — modo single-process.")
        return None

    try:
        os.makedirs(prom_dir, exist_ok=True)
        return prom_dir
    except Exception as e:
        print(f"[observability] Aviso: falha ao preparar diretório multiprocess ({prom_dir}): {e}")
        return None


_prom_dir = _ensure_prometheus_dir()


# ============================================================
# 🧠 Inicialização do CollectorRegistry (Global)
# ============================================================

def _build_registry() -> CollectorRegistry:
    """Cria o CollectorRegistry global."""
    reg = CollectorRegistry()
    try:
        if _prom_dir:
            multiprocess.MultiProcessCollector(reg)
        else:
            pass 
    except Exception as e:
        print(f"[observability] Falha ao inicializar MultiProcessCollector: {e}")
        reg = CollectorRegistry()
    return reg


# Registry único exportado para o restante do sistema
registry: CollectorRegistry = _build_registry()


# ============================================================
# 📤 Helper para expor /metrics
# ============================================================

def render_metrics_response():
    """
    Retorna (body_bytes, content_type_str) para o endpoint /metrics.
    """
    data = generate_latest(registry)
    return data, CONTENT_TYPE_LATEST


# ============================================================
# 📈 DEFINIÇÃO DAS MÉTRICAS (Centralizadas)
# ============================================================

# ------------------------------------------------------------
# 1. API & HTTP
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 2. ROTEAMENTO (Router Core)
# ------------------------------------------------------------
ROUTER_CHOSEN = Counter(
    "router_chosen_model_total",
    "Modelo escolhido pelo roteador",
    ["model"],
    registry=registry,
)
ROUTER_MODEL_COST = Counter(
    "router_model_cost_usd_total",
    "Custo acumulado estimado por modelo (USD)",
    ["model"],
    registry=registry,
)
ROUTER_COST_SAVINGS = Counter(
    "router_cost_savings_usd_total",
    "Custo total economizado (USD) vs Baseline",
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
    "Proporção de uso de modelos locais (Ollama)",
    registry=registry,
)
ROUTER_HISTORY_ENTRIES = Gauge(
    "router_history_entries_total",
    "Número de modelos com histórico EMA ativo",
    registry=registry,
)
FALLBACK_USED = Counter(
    "router_fallback_used_total",
    "Fallback usado entre modelos",
    ["first_model", "second_model"],
    registry=registry,
)

# ------------------------------------------------------------
# 3. CANDIDATOS & JUÍZES
# ------------------------------------------------------------
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
    "Valores de pontuação atribuídos pelos juízes",
    ["judge_id"],
    registry=registry,
)

# ------------------------------------------------------------
# 4. BANDITS (Aprendizado por Reforço)
# ------------------------------------------------------------
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
    "Recompensa observada pelo bandit",
    registry=registry,
)

# ------------------------------------------------------------
# 5. PROVEDORES (LLM Providers - Async)
# (Estas eram as que estavam faltando!)
# ------------------------------------------------------------
PROV_REQ = Counter(
    "providers_model_requests_total",
    "Total de chamadas ao provedor por modelo",
    ["model"],
    registry=registry,
)
PROV_ERR = Counter(
    "providers_model_errors_total",
    "Total de erros por modelo",
    ["model"],
    registry=registry,
)
PROV_OK = Counter(
    "providers_model_success_total",
    "Chamadas bem-sucedidas por modelo",
    ["model"],
    registry=registry,
)
PROV_LAT = Histogram(
    "providers_latency_seconds",
    "Latência da chamada ao provedor (s)",
    ["model"],
    registry=registry,
)
PROV_COST = Histogram(
    "providers_cost_usd",
    "Custo da chamada ao provedor (USD)",
    ["model"],
    registry=registry,
)
PROV_LAST_TS = Gauge(
    "providers_last_call_timestamp",
    "Timestamp da última chamada ao provedor",
    ["model"],
    registry=registry,
)


# ============================================================
# 🪵 Logging estruturado (Structlog + JSON)
# ============================================================

_logger_configured = False
logger = structlog.get_logger("observability")


class JsonUTF8Renderer:
    """Renderizador JSON que mantém acentuação legível."""

    def __call__(self, logger, name, event_dict):
        try:
            return json.dumps(event_dict, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Falha ao serializar log: {e}", **event_dict})


def setup_logging(level: int = logging.INFO):
    """
    Configura o Structlog para JSON com timestamp.
    Idempotente.
    """
    global _logger_configured
    if _logger_configured:
        return

    # Tenta forçar UTF-8 no stdout
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        else:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            JsonUTF8Renderer(),
        ]
    )

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    _logger_configured = True
    logger.info("[observability] Logging configurado.")


def json_log(level: str, event: str, **fields):
    """
    Helper manual para logs estruturados em UTF-8.
    """
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="microseconds") + "Z",
        "level": level,
        "event": event,
        **fields,
    }
    msg = json.dumps(record, ensure_ascii=False)
    log = logging.getLogger("observability")
    getattr(log, level, log.info)(msg)