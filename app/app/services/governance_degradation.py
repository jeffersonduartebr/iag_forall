# Objective: A database blip must not make the router unavailable.
"""Graceful degradation of the governance pre-checks on ``/query``.

Both checks that run before any model is chosen — the tenant budget and the
active policy — read MariaDB. Their exceptions propagated straight out of the
handler, and the project has no exception handler at all, so a database outage
turned every request into ``500 {"detail":"Internal Server Error"}``.

The contrast is what makes this worth fixing: the LLM call needs no database.
The request failed on a governance pre-check, not on the work it asked for.

The two are treated differently on purpose.

The **policy** is advisory: it selects among configured behaviours, and its
absence means "use the defaults". Degrading is the obvious answer.

The **budget** is a spending control, so degrading has a real cost — a tenant
could exceed its limit during the outage. Failing closed has a cost too, and a
larger one: the whole API becomes unavailable for a request path that does not
need the database. It degrades by default, loudly, and
``TENANT_BUDGET_FAIL_CLOSED=1`` reverses that for deployments where an overrun
is worse than an outage. Note that ``record_tenant_usage`` already fails open,
so failing the *check* closed while the *write* fails open would only make the
accounting incoherent.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from prometheus_client import Counter

from app.observability import registry
from app.settings_dynamic import settings

logger = logging.getLogger(__name__)

#: Definida aqui, junto ao seu único consumidor, e não em `observability`:
#: esse módulo está acima do limite de SLOC e continuar a empilhar métricas
#: nele é o que o mantém lá. O registry é o mesmo, portanto /metrics não muda.
GOVERNANCE_DEGRADED_TOTAL = Counter(
    "governance_degraded_total",
    "Governance pre-checks served in degraded mode because their backend failed",
    ["check"],
    registry=registry,
)


def _fail_closed() -> bool:
    try:
        return str(settings.get("TENANT_BUDGET_FAIL_CLOSED", "0")).strip() == "1"
    except Exception:
        return False


def resolve_budget(value: Any, tenant_id: Optional[str]) -> Any:
    """The budget result, or a degraded pass when the read failed."""
    if not isinstance(value, BaseException):
        return value
    if _fail_closed():
        raise value
    logger.error(
        "[governance] Verificação de orçamento indisponível para tenant=%s; a servir sem controlo de gasto: %s",
        tenant_id,
        value,
    )
    GOVERNANCE_DEGRADED_TOTAL.labels(check="tenant_budget").inc()
    return None


def resolve_policy(value: Any) -> Any:
    """The active policy, or ``None`` when the read failed.

    A missing policy means the defaults apply, which is what a deployment
    without any configured policy already does.
    """
    if not isinstance(value, BaseException):
        return value
    logger.warning("[governance] Política ativa indisponível; a usar os defaults: %s", value)
    GOVERNANCE_DEGRADED_TOTAL.labels(check="active_policy").inc()
    return None
