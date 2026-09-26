# Objective: The semantic centroids survive a Redis wipe: copied to ``learned_state`` and restored from it.
"""Why this matters more than for a cache: the bandit's ``cluster:<id>`` statistics in MariaDB are keyed by the
centroid id. Centroids recreated from scratch reuse ids 0, 1, ... for other groups of questions, and the durable
statistics of the old groups would silently be applied to the new ones. So: every new centroid is persisted at once,
movements at most every ``INTERVALO_S``; an empty Redis is refilled from the copy; and when there is no copy but the
bandit already has ``cluster:*`` rows, new ids start after the largest one instead of reusing them.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, List, Optional

from . import estado_duravel

logger = logging.getLogger(__name__)
CHAVE = "meta:bandit:centroids"
INTERVALO_S = 60.0
_estado = {"ultimo": 0.0, "n": -1}
_trava = threading.Lock()


def salvar(serial: List[dict]) -> None:
    """Persist the centroid document: at once when the set changed, else at most every ``INTERVALO_S``."""
    with _trava:
        agora = time.monotonic()
        if len(serial) == _estado["n"] and agora - _estado["ultimo"] < INTERVALO_S:
            return
        if estado_duravel.gravar(CHAVE, serial):
            _estado.update(ultimo=agora, n=len(serial))


def restaurar(rds: Any, chave_redis: str) -> Optional[str]:
    """The durable document written back into an empty Redis (SET NX: never over a live one)."""
    doc = estado_duravel.ler(CHAVE)
    if not doc:
        return None
    bruto = json.dumps(doc)
    try:
        rds.set(chave_redis, bruto, nx=True)
        logger.warning("[centroids] Redis sem centróides: %d restaurado(s) do banco", len(doc))
        return rds.get(chave_redis) or bruto
    except Exception as exc:
        logger.warning("[centroids] restauração falhou: %s", exc)
        return bruto


def primeiro_id() -> int:
    """First id for a centroid set starting from nothing: after every ``cluster:<id>`` the bandit already knows."""
    try:
        from sqlalchemy import text

        from app.db import get_engine

        with get_engine().connect() as conn:
            rotulos = conn.execute(
                text("SELECT DISTINCT context_label FROM bandit_context_stats WHERE context_label LIKE '%cluster:%'")
            ).scalars().all()
        sufixos = [r.rpartition("cluster:")[2] for r in rotulos if "cluster:" in r]
        ids = [int(x) for x in sufixos if x.isdigit()]
        return max(ids) + 1 if ids else 0
    except Exception as exc:
        logger.warning("[centroids] ids do bandit ilegíveis; começando em 0: %s", exc)
        return 0
