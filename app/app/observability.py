import structlog
import os

import os
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
    multiprocess
)


if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
    for f in os.listdir(os.environ["PROMETHEUS_MULTIPROC_DIR"]):
        os.remove(os.path.join(os.environ["PROMETHEUS_MULTIPROC_DIR"], f))


logger = structlog.get_logger()

API_REQUESTS = Counter("api_requests_total", "Total API requests")
API_LATENCY = Histogram("api_request_latency_seconds", "Latency of API requests (s)")

ROUTER_CHOSEN = Counter("router_chosen_model_total", "Chosen model by router", ["model"])
FALLBACK_USED = Counter("router_fallback_used_total", "Fallback used", ["first_model","second_model"])
CANDIDATE_COST = Histogram("candidate_estimated_cost_usd", "Estimated cost per answer (USD)")
CANDIDATE_LAT = Histogram("candidate_latency_seconds", "Latency per answer (s)")
JUDGE_SCORE = Histogram("judge_score", "Judge score values", ["judge_id"])

ROUTER_HISTORY_ENTRIES = Gauge("router_history_entries_total", "Number of models with EMA history entry")

BANDIT_SELECT = Counter("bandit_select_total","Bandit selections per model",["model"])
BANDIT_UPDATE = Counter("bandit_update_total","Bandit updates per model",["model"])
BANDIT_REWARD = Histogram("bandit_reward","Observed reward values")


# 💰 Custos e economia
ROUTER_MODEL_COST = Counter(
    "router_model_cost_usd_total",
    "Custo acumulado estimado por modelo (USD)",
    ["model"]
)

ROUTER_COST_SAVINGS = Counter(
    "router_cost_savings_usd_total",
    "Custo total economizado (USD) pelo uso de modelos locais."
)

ROUTER_COST_PER_QUERY = Gauge(
    "router_cost_per_query_usd",
    "Custo médio por consulta (USD)."
)

# ⚖️ Qualidade e desempenho
ROUTER_QUALITY_AVG = Gauge(
    "router_model_quality_avg",
    "Qualidade média ponderada por modelo (0–10)",
    ["model"]
)

# ⚙️ Distribuição local vs remoto
ROUTER_LOCAL_USAGE_RATIO = Gauge(
    "router_local_model_usage_ratio",
    "Proporção de requisições atendidas por modelos locais (Ollama)."
)

def setup_logging():
    structlog.configure(processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ])
