# Objective: Durable EMA per (modality, model): written on every update, read back when Redis is empty.
"""``ema_history`` is the durable copy of the shared EMA (``ema_store``, Redis).

Before, the feedback worker only queued the rows (``EMABatchQueue``) and flushed them at 500 distinct keys, which
never happened, and the periodic flusher ran only in the API: the table stayed stale and the NSGA-II, the adaptive
timeouts and any restart read old or no data. Now each update is one upsert (a few per minute), and a Redis that
lost its EMA is refilled from here instead of restarting every model from its first new sample.

Rows are keyed by (model, modality, semantics, policy_namespace): the study period (``BANDIT_POLICY_NAMESPACE``)
has its own EMA in Redis, and now in the table too, instead of collapsing onto one row.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

_UPSERT = text("""
    INSERT INTO ema_history
        (modality, model, semantics, policy_namespace, ema_latency, ema_quality, ema_cost, ema_alignment, updates)
    VALUES (:mod, :m, :sem, :ns, :lat, :q, :c, :align, :u)
    ON CONFLICT (model, modality, semantics, policy_namespace) DO UPDATE SET
        ema_latency = EXCLUDED.ema_latency, ema_quality = EXCLUDED.ema_quality, ema_cost = EXCLUDED.ema_cost,
        ema_alignment = EXCLUDED.ema_alignment, updates = EXCLUDED.updates, updated_at = CURRENT_TIMESTAMP
""")
_LOG = text("""
    INSERT INTO ema_history_log
        (modality, model, semantics, ema_latency, ema_cost, ema_quality, ema_alignment, update_num)
    VALUES (:mod, :m, :sem, :lat, :c, :q, :align, :u)
""")
_LER = """
    SELECT model, ema_latency, ema_quality, ema_cost, ema_alignment, updates FROM ema_history
    WHERE modality = :mod AND semantics = :sem AND policy_namespace = :ns
"""


def _escopo() -> Dict[str, str]:
    from .quality_semantics import current_semantics, policy_namespace

    return {"sem": current_semantics(), "ns": policy_namespace() or ""}


def _engine():
    from app.db import get_engine

    return get_engine()


def linha(modality: str, model: str, record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mod": modality, "m": model, **_escopo(),
        "lat": float(record["ema_latency"]), "q": float(record["ema_quality"]), "c": float(record["ema_cost"]),
        "align": float(record.get("ema_alignment", 1.0)), "u": int(record.get("updates", 1)),
    }


def gravar(modality: str, model: str, record: Dict[str, Any]) -> bool:
    """Upsert one EMA (and every 10th update into the log). Never raises: a failure is logged loudly."""
    try:
        row = linha(modality, model, record)
        with _engine().begin() as conn:
            conn.execute(_UPSERT, row)
            if row["u"] % 10 == 0:
                conn.execute(_LOG, row)
        return True
    except Exception as exc:
        logger.error("[ema] ema_history NÃO gravado (%s/%s): %s", modality, model, exc)
        return False


def carregar(modality: str, model: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """``model -> EMA entry`` of the active semantics and period ({} on any failure)."""
    try:
        sql, params = _LER, {"mod": modality, **_escopo()}
        if model is not None:
            sql, params = sql + " AND model = :m", {**params, "m": model}
        with _engine().connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
    except Exception as exc:
        logger.warning("[ema] ema_history ilegível (%s): %s", modality, exc)
        return {}
    return {
        r["model"]: {
            "ema_latency": float(r["ema_latency"]), "ema_quality": float(r["ema_quality"]),
            "ema_cost": float(r["ema_cost"]), "ema_alignment": float(r["ema_alignment"]),
            "updates": int(r["updates"] or 0),
        }
        for r in rows
    }
