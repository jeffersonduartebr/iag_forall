# Objective: Shadow-only Prometheus metrics, kept apart from the system's cost and efficiency metrics (R10).
from __future__ import annotations

from prometheus_client import Counter, Gauge

from app.observability import registry

SHADOW_CALLS = Counter(
    "aristo_shadow_calls_total", "Chamadas em sombra por modelo e status", ["model", "status"], registry=registry
)
SHADOW_COST = Counter(
    "aristo_shadow_cost_usd_total", "Custo das chamadas em sombra (candidatas e juízes), USD", ["model"], registry=registry
)
SHADOW_SKIPPED = Counter(
    "aristo_shadow_skipped_total", "Requisições ou candidatas em sombra cortadas, por motivo", ["motivo"], registry=registry
)
SHADOW_BUDGET_REMAINING = Gauge(
    "aristo_shadow_budget_remaining_usd",
    "Orçamento diário restante da sombra (USD)",
    registry=registry,
    multiprocess_mode="liveall",
)
SHADOW_BUDGET_EXHAUSTED_HOUR = Gauge(
    "aristo_shadow_budget_exhausted_hour",
    "Hora local (fuso SHADOW_BUDGET_TZ, fração decimal) em que o orçamento do dia esgotou; -1 se não esgotou",
    registry=registry,
    multiprocess_mode="liveall",
)
