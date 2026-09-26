# Objective: Durable copy (MariaDB ``learned_state``) of learned documents that live in Redis.
"""A key -> JSON table for learned state that has no table of its own (semantic centroids, the exploration
blocklist). Redis stays the working copy; this is what a wiped or lost Redis is refilled from. Failures are logged
loudly and never raised: the caller's hot path must not depend on the database being up.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

_GRAVAR = text("""
    INSERT INTO learned_state (chave, valor) VALUES (:chave, :valor)
    ON DUPLICATE KEY UPDATE valor = VALUES(valor), atualizado_em = CURRENT_TIMESTAMP
""")
_LER = text("SELECT valor FROM learned_state WHERE chave = :chave")


def _engine():
    from app.db import get_engine

    return get_engine()


def gravar(chave: str, valor: Any) -> bool:
    try:
        with _engine().begin() as conn:
            conn.execute(_GRAVAR, {"chave": chave, "valor": json.dumps(valor, ensure_ascii=False)})
        return True
    except Exception as exc:
        logger.error("[estado] learned_state NÃO gravado (%s): %s", chave, exc)
        return False


def ler(chave: str) -> Optional[Any]:
    try:
        with _engine().connect() as conn:
            bruto = conn.execute(_LER, {"chave": chave}).scalar()
        return json.loads(bruto) if bruto else None
    except Exception as exc:
        logger.warning("[estado] learned_state ilegível (%s): %s", chave, exc)
        return None
