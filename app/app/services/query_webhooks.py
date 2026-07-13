# Objective: Deliver webhook callbacks when async query jobs complete.
"""HTTP webhook notifications for queued query jobs."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def _post_webhook(url: str, payload: Dict[str, Any]) -> None:
    try:
        from ..providers_async import get_http_client

        client = await get_http_client()
        await client.post(url, json=payload, timeout=10.0)
    except Exception as exc:
        logger.warning("[webhook] delivery failed url=%s err=%s", url, exc)


def schedule_query_job_webhook(
    *,
    webhook_url: Optional[str],
    job_id: str,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> None:
    """Fire-and-forget webhook for completed/failed async jobs."""
    url = (webhook_url or "").strip()
    if not url:
        return
    payload = {
        "job_id": job_id,
        "status": status,
        "result": result,
        "error": error,
    }
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_post_webhook(url, payload))
    except RuntimeError:
        asyncio.run(_post_webhook(url, payload))
