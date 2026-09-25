# Objective: Read one tenant's shadow rows and query_log rows for a date window (numeric columns only, no text).
"""Leitura do banco para o resumo de pesquisa: sempre filtrada pelo tenant do chamador, nunca colunas de texto."""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Tuple

from sqlalchemy import text

from app.db import get_engine

_SOMBRA = (
    "SELECT request_id, estrato, p_nominal, executada, motivo_corte, modelo, papel, regime_entrega, p_atribuicao, "
    "escore_agregado, painel_uniforme, teria_abstido, custo_usd, latencia_s, status, criado_em "
    "FROM shadow_evaluations WHERE tenant = :tenant AND criado_em >= :desde AND criado_em < :ate "
    "ORDER BY criado_em, id"
)
_QUERY_LOG = (
    "SELECT created_at, chosen_model, estimated_cost_usd, latency_s, abstained, judge_sampled, decision_json "
    "FROM query_log WHERE tenant_id = :tenant AND created_at >= :desde AND created_at < :ate ORDER BY created_at, id"
)


def _normalizar(linha: Dict[str, Any]) -> Dict[str, Any]:
    """The stratum is grouped by its text: a driver that decodes JSON must not change the grouping key."""
    if isinstance(linha.get("estrato"), dict):
        linha["estrato"] = json.dumps(linha["estrato"], ensure_ascii=False, sort_keys=True)
    return linha


def ler(tenant: str, desde: dt.date, ate: dt.date) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(shadow rows, query_log rows)`` of ``tenant`` with ``desde <= created < ate``."""
    params = {"tenant": tenant, "desde": desde.isoformat(), "ate": ate.isoformat()}
    with get_engine().connect() as conn:
        sombra = [_normalizar(dict(r._mapping)) for r in conn.execute(text(_SOMBRA), params)]
        query_log = [dict(r._mapping) for r in conn.execute(text(_QUERY_LOG), params)]
    return sombra, query_log
