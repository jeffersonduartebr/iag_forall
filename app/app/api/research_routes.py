# Objective: Tenant-scoped research summary for the Caso 1 dashboards (aggregates only, never any text).
"""``GET /research/summary``: métricas de pesquisa do tenant do chamador.

Admits a JWT or key whose roles include ``instrument`` (the formative tool's token) or ``platform_admin``, and it
always answers for the caller's own tenant: there is no tenant parameter, so one tenant cannot read another's.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError

from ..services.pesquisa import leitura
from ..services.pesquisa.roteamento import resumo_regime, resumo_totais
from ..services.pesquisa.sombra import resumo_sombra
from .auth import AuthContext, resolve_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["Research"])

RESEARCH_ROLES = frozenset({"instrument", "platform_admin"})
_INICIO, _FIM = dt.date(2000, 1, 1), dt.date(9999, 12, 31)


def require_research_caller(
    authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)
) -> AuthContext:
    """Authenticated, with a research role and a tenant (the data is always that tenant's)."""
    auth = resolve_auth(authorization=authorization, x_api_key=x_api_key, required=True)
    if not set(auth.roles) & RESEARCH_ROLES:
        raise HTTPException(status_code=403, detail="Exige papel 'instrument' ou 'platform_admin'.")
    if not (auth.tenant_id or "").strip():
        raise HTTPException(status_code=403, detail="Token sem tenant_id: o resumo é sempre de um tenant.")
    return auth


@router.get("/summary")
def research_summary(
    since: Optional[dt.date] = Query(None, description="Primeiro dia (inclusive), AAAA-MM-DD."),
    until: Optional[dt.date] = Query(None, description="Último dia (inclusive), AAAA-MM-DD; padrão: sem limite."),
    auth: AuthContext = Depends(require_research_caller),
) -> Dict[str, Any]:
    """Shadow-evaluation, exploration-regime and cost/latency aggregates of the caller's tenant."""
    desde = since or _INICIO
    ate = until + dt.timedelta(days=1) if until else _FIM
    if desde >= ate:
        raise HTTPException(status_code=422, detail="'since' deve ser anterior ou igual a 'until'.")
    tenant = str(auth.tenant_id).strip()
    try:
        sombra, query_log = leitura.ler(tenant, desde, ate)
    except SQLAlchemyError as exc:
        logger.error("[research] leitura falhou: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Banco indisponível.") from exc
    return {
        "tenant": tenant,
        "since": desde.isoformat(),
        "until": until.isoformat() if until else None,
        "sombra": resumo_sombra(sombra),
        "regime": resumo_regime(query_log),
        "totais": resumo_totais(query_log),
    }
