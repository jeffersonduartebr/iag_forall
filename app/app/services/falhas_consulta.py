# Objective: A request that ends in an error still leaves a record (request_failures), keyed by correlation id.
"""Failed requests, recorded.

Before this table a timeout, an exhausted fallback chain, a guardrail block or a budget rejection left no row
anywhere: only a Prometheus counter, so a participant's failed attempts were invisible to the analysis. Recording
never masks the original error: any failure here is logged and swallowed.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)

_INSERT = """
INSERT INTO request_failures
 (correlation_id, tenant_id, participant, episode_id, route_path, status_code, category, model, detail_json,
  query_text, modality, latency_s)
VALUES (:cid, :tenant, :participant, :episode, :route, :status, :category, :model, :detail, :query, :modality, :lat)
"""


def classificar(exc: BaseException) -> tuple[int, Optional[str], Optional[str], Any]:
    """``(status_code, category, model, detail)`` of the error the client received."""
    if isinstance(exc, HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        categoria = detail.get("category") or detail.get("error_category") or detail.get("code")
        return exc.status_code, str(categoria) if categoria else None, detail.get("model"), detail
    return 500, type(exc).__name__, getattr(exc, "model", None), {"message": str(exc)[:500]}


def registrar(req: Any, exc: BaseException, *, correlation_id: Optional[str], route_path: Optional[str],
              inicio: float) -> None:
    """Record one failed request; never raises."""
    try:
        from sqlalchemy import text

        from ..db import get_engine

        status, categoria, modelo, detalhe = classificar(exc)
        params = {
            "cid": correlation_id,
            "tenant": getattr(req, "tenant_id", None),
            "participant": getattr(req, "user_key", None),
            "episode": getattr(req, "episode_id", None),
            "route": (route_path or "")[:64] or None,
            "status": int(status),
            "category": (categoria or "")[:64] or None,
            "model": modelo,
            "detail": json.dumps(detalhe, ensure_ascii=False, default=str)[:8000],
            "query": getattr(req, "query", None),
            "modality": getattr(req, "modality", None),
            "lat": round(time.time() - inicio, 3),
        }
        # PostgreSQL TEXT rejeita NUL; a tabela é criada pelo alembic_pg (sem DDL aqui).
        params = {chave: valor.replace("\x00", "") if isinstance(valor, str) else valor
                  for chave, valor in params.items()}
        with get_engine().begin() as conn:
            conn.execute(text(_INSERT), params)
    except Exception as erro:
        logger.error("[falhas] request_failures NÃO foi escrito (%s): %s", correlation_id, erro)
