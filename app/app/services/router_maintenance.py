# -*- coding: utf-8 -*-
# Objective: Service-layer helpers for router maintenance.
"""Maintenance helpers for router background services."""

from __future__ import annotations

import threading
from typing import Any, Callable, List, Optional

from sqlalchemy import text

#: Uma vez por dia. As duas tabelas com retenção são grandes e a limpeza é
#: um DELETE por intervalo, não vale a pena corrê-la mais vezes.
DAILY_SECONDS = 86400


def create_background_threads(
    cleanup_old_query_logs: Callable[[], None],
    cleanup_ema_history: Callable[[], None],
    ema_batch_flusher: Callable[[], None],
    cleanup_ema_history_log: Callable[[], None],
    update_db_pool_metrics: Callable[[], None],
) -> List[threading.Thread]:
    """Create background threads.

This function encapsulates the construction and persistence steps for the new resource."""
    return [
        threading.Thread(target=cleanup_old_query_logs, daemon=True, name="router-cleanup-query-log"),
        threading.Thread(target=cleanup_ema_history, daemon=True, name="router-cleanup-ema-cache"),
        threading.Thread(target=ema_batch_flusher, daemon=True, name="router-ema-batch-flusher"),
        threading.Thread(target=cleanup_ema_history_log, daemon=True, name="router-cleanup-ema-log"),
        threading.Thread(target=update_db_pool_metrics, daemon=True, name="router-db-pool-metrics"),
    ]


def retention_loop(
    *,
    stop_event: Any,
    table: str,
    days: int,
    engine_factory: Callable[[], Any],
    logger: Any,
    before: Optional[Callable[[], None]] = None,
    on_deleted: Optional[Callable[[int], None]] = None,
    interval: int = DAILY_SECONDS,
) -> None:
    """Delete rows older than ``days`` from ``table``, once per interval.

    ``days <= 0`` disables the deletion entirely — the value for an experiment
    environment, where the point of the table is that it keeps everything.

    The two retention loops this replaces were near-identical except for how
    loudly they failed: one logged the row count and the error, the other had a
    bare ``except Exception: pass``. The silent one guarded ``query_log``, the
    table the whole empirical analysis rests on, so it could fail every day for
    a year with no symptom but a filling disk.
    """
    while not stop_event.is_set():
        try:
            if days > 0:
                if before is not None:
                    before()
                with engine_factory().begin() as conn:
                    deleted = conn.execute(
                        text(f"DELETE FROM {table} WHERE created_at < (NOW() - INTERVAL :d DAY)"),  # noqa: S608
                        {"d": days},
                    ).rowcount
                if deleted:
                    logger.info(f"[{table} cleanup] {deleted} linhas acima de {days} dias")
                    if on_deleted is not None:
                        on_deleted(deleted)
        except Exception as exc:
            logger.warning(f"[{table} cleanup] falhou: {exc}")
        stop_event.wait(interval)
