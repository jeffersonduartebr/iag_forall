# -*- coding: utf-8 -*-
"""Maintenance helpers for router background services."""

from __future__ import annotations

import threading
from typing import Callable, List


def create_background_threads(
    cleanup_old_query_logs: Callable[[], None],
    cleanup_ema_history: Callable[[], None],
    ema_batch_flusher: Callable[[], None],
    cleanup_ema_history_log: Callable[[], None],
    update_db_pool_metrics: Callable[[], None],
) -> List[threading.Thread]:
    """Executa create background threads."""
    return [
        threading.Thread(target=cleanup_old_query_logs, daemon=True, name="router-cleanup-query-log"),
        threading.Thread(target=cleanup_ema_history, daemon=True, name="router-cleanup-ema-cache"),
        threading.Thread(target=ema_batch_flusher, daemon=True, name="router-ema-batch-flusher"),
        threading.Thread(target=cleanup_ema_history_log, daemon=True, name="router-cleanup-ema-log"),
        threading.Thread(target=update_db_pool_metrics, daemon=True, name="router-db-pool-metrics"),
    ]
