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
# 5. L1 SEMANTIC CACHE
# ------------------------------------------------------------
L1_CACHE_HITS = Counter(
    "l1_cache_hits_total",
    "Total L1 (in-memory) cache hits",
    registry=registry,
)
L1_CACHE_MISSES = Counter(
    "l1_cache_misses_total",
    "Total L1 (in-memory) cache misses",
    registry=registry,
)
L1_CACHE_SIZE = Gauge(
    "l1_cache_size_current",
    "Current number of entries in L1 cache",
    registry=registry,
)

# ------------------------------------------------------------
# 6. PROVEDORES (LLM Providers - Async)
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

# ------------------------------------------------------------
# 7. CIRCUIT BREAKERS
# ------------------------------------------------------------
CIRCUIT_BREAKER_STATE = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half-open, 2=open)",
    ["model"],
    registry=registry,
)
CIRCUIT_BREAKER_FAILURES = Counter(
    "circuit_breaker_failures_total",
    "Total failures recorded by circuit breakers",
    ["model"],
    registry=registry,
)
CIRCUIT_BREAKER_TRIPS = Counter(
    "circuit_breaker_trips_total",
    "Total times circuit breaker tripped (opened)",
    ["model"],
    registry=registry,
)

# ------------------------------------------------------------
# 8. RATE LIMITING
# ------------------------------------------------------------
RATE_LIMIT_EXCEEDED = Counter(
    "rate_limit_exceeded_total",
    "Total requests rejected due to rate limiting",
    ["client_ip"],
    registry=registry,
)

# ------------------------------------------------------------
# 9. REQUEST DEDUPLICATION
# ------------------------------------------------------------
REQUESTS_DEDUPLICATED = Counter(
    "requests_deduplicated_total",
    "Total requests deduplicated (avoided duplicate processing)",
    registry=registry,
)

# ------------------------------------------------------------
# 10. HEALTH CHECK METRICS
# ------------------------------------------------------------
HEALTH_CHECK_DURATION = Histogram(
    "health_check_duration_seconds",
    "Duration of health check execution",
    ["component"],
    registry=registry,
)
COMPONENT_HEALTH = Gauge(
    "component_health_status",
    "Health status of system component (1=healthy, 0=unhealthy)",
    ["component"],
    registry=registry,
)

# ------------------------------------------------------------
# 11. COST TRACKING (Detailed)
# ------------------------------------------------------------
TOTAL_COST_USD = Counter(
    "total_cost_usd",
    "Total cost incurred across all requests (USD)",
    registry=registry,
)
COST_BY_PROVIDER = Counter(
    "cost_by_provider_usd_total",
    "Total cost by provider (USD)",
    ["provider"],
    registry=registry,
)
TOKENS_INPUT_TOTAL = Counter(
    "tokens_input_total",
    "Total input tokens processed",
    ["model"],
    registry=registry,
)
TOKENS_OUTPUT_TOTAL = Counter(
    "tokens_output_total",
    "Total output tokens generated",
    ["model"],
    registry=registry,
)

# ------------------------------------------------------------
# 12. NSGA-II CONVERGENCE MONITORING
# ------------------------------------------------------------
NSGA_CONVERGENCE_SCORE = Gauge(
    "nsga_convergence_score",
    "NSGA-II optimization convergence score (higher is better)",
    ["modality"],
    registry=registry,
)
NSGA_OPTIMIZATION_HEALTH = Gauge(
    "nsga_optimization_health",
    "NSGA-II optimization health (1=healthy, 0=degraded, -1=stuck)",
    ["modality"],
    registry=registry,
)
NSGA_EFFICIENCY_VARIANCE = Gauge(
    "nsga_efficiency_variance",
    "Variance in NSGA-II efficiency over recent runs",
    ["modality"],
    registry=registry,
)

# ------------------------------------------------------------
# 13. CASCADE FAILURE DETECTION
# ------------------------------------------------------------
CASCADE_SEVERITY_LEVEL = Gauge(
    "cascade_severity_level",
    "Cascade failure severity (0=normal, 1=warning, 2=critical, 3=emergency)",
    registry=registry,
)
CASCADE_FAILED_MODEL_RATIO = Gauge(
    "cascade_failed_model_ratio",
    "Ratio of models with open circuit breakers",
    registry=registry,
)

# ------------------------------------------------------------
# 14. QUERY DRIFT DETECTION
# ------------------------------------------------------------
QUERY_DRIFT_SCORE = Gauge(
    "query_drift_score",
    "Cosine distance from baseline query distribution (higher = more drift)",
    registry=registry,
)
QUERY_DRIFT_DETECTED = Counter(
    "query_drift_detected_total",
    "Number of times query drift was detected",
    registry=registry,
)

# ------------------------------------------------------------
# 15. USER FEEDBACK
# ------------------------------------------------------------
USER_FEEDBACK_RECEIVED = Counter(
    "user_feedback_received_total",
    "Total user feedback submissions",
    ["feedback_type"],
    registry=registry,
)
USER_FEEDBACK_QUALITY_AVG = Gauge(
    "user_feedback_quality_avg",
    "Average quality score from user feedback",
    registry=registry,
)

# ------------------------------------------------------------
# 16. A/B TESTING
# ------------------------------------------------------------
AB_EXPERIMENT_ASSIGNMENTS = Counter(
    "ab_experiment_assignments_total",
    "Total A/B experiment variant assignments",
    ["experiment_id", "variant"],
    registry=registry,
)

# ------------------------------------------------------------
# 17. RISK TUNING
# ------------------------------------------------------------
RISK_FACTOR_CURRENT = Gauge(
    "risk_factor_current",
    "Current adaptive risk factor value",
    ["factor_type"],
    registry=registry,
)

# ------------------------------------------------------------
# 18. PREDICTOR VALIDATION (Phase 5 - Autonomous Behavior)
# ------------------------------------------------------------
PREDICTOR_BRIER_SCORE = Gauge(
    "predictor_brier_score",
    "Brier score for online predictor calibration (lower is better, 0.25 = random)",
    ["model"],
    registry=registry,
)
PREDICTOR_ACCURACY = Gauge(
    "predictor_accuracy",
    "Binary classification accuracy of online predictor",
    ["model"],
    registry=registry,
)
PREDICTOR_PREDICTIONS_TOTAL = Counter(
    "predictor_predictions_total",
    "Total predictions made by online predictor",
    ["model"],
    registry=registry,
)
PREDICTOR_CALIBRATION_TEMP = Gauge(
    "predictor_calibration_temp",
    "Temperature scaling factor for predictor calibration",
    ["model"],
    registry=registry,
)

# ------------------------------------------------------------
# 19. ADAPTIVE CACHE THRESHOLD (Phase 5 - Autonomous Behavior)
# ------------------------------------------------------------
CACHE_THRESHOLD_CURRENT = Gauge(
    "cache_threshold_current",
    "Current semantic cache similarity threshold value",
    registry=registry,
)
CACHE_HIT_RATE = Gauge(
    "cache_hit_rate",
    "Observed cache hit rate (hits / total)",
    registry=registry,
)
CACHE_THRESHOLD_ADJUSTMENTS = Counter(
    "cache_threshold_adjustments_total",
    "Number of times cache threshold was adjusted",
    registry=registry,
)

# ------------------------------------------------------------
# 20. UQ CALIBRATION (Phase 5 - Autonomous Behavior)
# ------------------------------------------------------------
UQ_VS_ERROR_CORRELATION = Gauge(
    "uq_vs_error_correlation",
    "Correlation coefficient between uncertainty score and actual errors",
    registry=registry,
)
UQ_HIGH_AVG_QUALITY = Gauge(
    "uq_high_avg_quality",
    "Average quality score for high-uncertainty queries",
    registry=registry,
)
UQ_LOW_AVG_QUALITY = Gauge(
    "uq_low_avg_quality",
    "Average quality score for low-uncertainty queries",
    registry=registry,
)

# ------------------------------------------------------------
# 21. JUDGE CALIBRATION (Phase 5 - Autonomous Behavior)
# ------------------------------------------------------------
JUDGE_CALIBRATION_SCORE = Gauge(
    "judge_calibration_score",
    "Judge prediction accuracy (how well high scores correlate with caching)",
    ["judge_model"],
    registry=registry,
)
JUDGE_CACHE_AGREEMENT = Gauge(
    "judge_cache_agreement",
    "Rate at which high judge scores lead to successful caching",
    ["judge_model"],
    registry=registry,
)
JUDGE_CALIBRATION_UPDATES = Counter(
    "judge_calibration_updates_total",
    "Number of judge calibration updates",
    registry=registry,
)

# ------------------------------------------------------------
# 22. PERFORMANCE METRICS (Quick Wins Optimization)
# ------------------------------------------------------------
JUDGE_CACHE_HITS = Counter(
    "judge_cache_hits_total",
    "Total judge verdict cache hits",
    registry=registry,
)
JUDGE_CACHE_MISSES = Counter(
    "judge_cache_misses_total",
    "Total judge verdict cache misses",
    registry=registry,
)
JUDGE_CACHE_HIT_RATE = Gauge(
    "judge_cache_hit_rate",
    "Judge verdict cache hit rate",
    registry=registry,
)
EMBEDDING_CACHE_HITS = Counter(
    "embedding_cache_hits_total",
    "Total embedding L1 cache hits",
    registry=registry,
)
EMBEDDING_CACHE_MISSES = Counter(
    "embedding_cache_misses_total",
    "Total embedding L1 cache misses",
    registry=registry,
)
EMBEDDING_CACHE_HIT_RATE = Gauge(
    "embedding_cache_hit_rate",
    "Embedding L1 cache hit rate",
    registry=registry,
)
CENTROID_LOOKUP_DURATION = Histogram(
    "centroid_lookup_duration_seconds",
    "Duration of centroid lookup operations",
    registry=registry,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
EMA_HISTORY_SIZE = Gauge(
    "ema_history_size",
    "Current size of EMA history cache",
    registry=registry,
)
EMA_HISTORY_EVICTIONS = Counter(
    "ema_history_evictions_total",
    "Total EMA history entries evicted due to TTL or LRU",
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


def _add_correlation_id(logger, method_name, event_dict):
    """Add correlation ID to log events if available."""
    try:
        from .correlation import get_correlation_id
        correlation_id = get_correlation_id()
        if correlation_id:
            event_dict["correlation_id"] = correlation_id
    except Exception:
        pass  # Don't break logging if correlation module has issues
    return event_dict


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
            _add_correlation_id,  # Inject correlation ID into all logs
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