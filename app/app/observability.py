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

def setup_logging():
    structlog.configure(processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ])
