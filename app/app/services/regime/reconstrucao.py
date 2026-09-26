# Objective: Rebuild a participant's exploration window from query_log when Redis lost it.
"""The window (``janela``) is the per-participant guarantee of the protocol's 15% exploration cap. It lives in
Redis; if Redis is wiped, a participant would look new and could be explored beyond the cap. Every field-phase
decision is also in ``query_log.decision_json.regime`` (episode and whether it explored), so a missing window is
replayed from there once. A marker key keeps a participant with no history from triggering the query again.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Tuple

from . import janela

logger = logging.getLogger(__name__)
_MARCA = "regime:reconstruido:"
_SQL = """
    SELECT decision_json FROM query_log
    WHERE participant = :p AND decision_json LIKE '%"fase": "campo"%'
    ORDER BY id
"""


def _decisoes(participante: str) -> List[Tuple[str, bool]]:
    from sqlalchemy import text

    from app.db import get_engine

    with get_engine().connect() as conn:
        brutos = conn.execute(text(_SQL), {"p": participante}).scalars().all()
    saida = []
    for bruto in brutos:
        regime = (json.loads(bruto or "{}") or {}).get("regime") or {}
        if regime.get("fase") == "campo" and regime.get("episodio"):
            saida.append((str(regime["episodio"]), bool(regime.get("explorou"))))
    return saida


def garantir(rds: Any, participante: str, tamanho: int) -> None:
    """Replay the participant's recorded decisions into an empty window (once per participant)."""
    try:
        if rds.exists(f"{janela.PREFIXO}episodios:{participante}") or not rds.set(
            f"{_MARCA}{participante}", "1", nx=True, ex=janela.TTL_S
        ):
            return
        decisoes = _decisoes(participante)
    except Exception as exc:
        logger.warning("[regime] janela de %s não reconstruída: %s", participante, type(exc).__name__)
        return
    for episodio, explorou in decisoes:
        janela.registrar(rds, participante, episodio, tamanho, explorou)
    if decisoes:
        logger.warning("[regime] janela de %s reconstruída do query_log: %d decisão(ões)", participante, len(decisoes))
