# Objective: Persist shadow rows and suspension intervals (Alembic 0009); never any answer text (R7, R8, R9).
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import text

from app.db import get_engine

logger = logging.getLogger(__name__)

COLUNAS = (
    "request_id", "episode_id", "participante", "tenant", "caso", "estrato", "p_nominal", "executada",
    "motivo_corte", "modelo", "provedor", "regiao", "versao_modelo", "papel", "regime_entrega", "p_atribuicao",
    "escores_juizes", "escore_agregado", "painel", "painel_uniforme", "teria_abstido", "custo_usd",
    "latencia_s", "tokens_entrada", "tokens_saida", "sha256_texto", "status", "frozen_run_id",
    "criado_em", "concluido_em",
)
_JSON = ("estrato", "escores_juizes", "painel")


def _linha(dados: Dict[str, Any]) -> Dict[str, Any]:
    linha = {c: dados.get(c) for c in COLUNAS}
    for c in _JSON:
        if linha[c] is not None and not isinstance(linha[c], str):
            linha[c] = json.dumps(linha[c], ensure_ascii=False, sort_keys=True)
    return linha


def gravar(linhas: Iterable[Dict[str, Any]]) -> int:
    """Insert shadow rows; returns how many were written (0 when the database is down: logged, not raised)."""
    lote = [_linha(d) for d in linhas]
    if not lote:
        return 0
    sql = text(
        f"INSERT INTO shadow_evaluations ({', '.join(COLUNAS)}) VALUES ({', '.join(':' + c for c in COLUNAS)})"
    )
    try:
        with get_engine().begin() as conn:
            conn.execute(sql, lote)
        return len(lote)
    except Exception as exc:
        logger.error("[sombra] %d linha(s) não gravada(s): %s", len(lote), exc)
        return 0


def abrir_suspensao(chave: str, motivo: str, tenant: Optional[str], inicio: str) -> None:
    """Record the start of a suspended interval (idempotent per key: the key encodes motivo, tenant and period)."""
    sql = text(
        "INSERT IGNORE INTO shadow_suspensions (chave, motivo, tenant, inicio) VALUES (:chave, :motivo, :tenant, :inicio)"
    )
    _executar(sql, {"chave": chave, "motivo": motivo, "tenant": tenant, "inicio": inicio})


def fechar_suspensao(chave: str, fim: str) -> None:
    _executar(text("UPDATE shadow_suspensions SET fim = :fim WHERE chave = :chave AND fim IS NULL"), {"chave": chave, "fim": fim})


def _executar(sql: Any, params: Dict[str, Any]) -> None:
    try:
        with get_engine().begin() as conn:
            conn.execute(sql, params)
    except Exception as exc:
        logger.error("[sombra] suspensão não registrada: %s", exc)
