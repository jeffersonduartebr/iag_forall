# Objective: A retried request must return the first answer, not buy a second one.
"""Idempotency for ``/query``, split out of ``query_http``.

The feature never worked. The read computed its key with ``model=""`` and the
write computed it with the model that had just answered — a value only known
*after* the response exists — so the two keys could never match. A client
retrying with the same ``Idempotency-Key`` executed the query again and paid
for it again, exactly as if the header had not been sent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Request

from app.schemas import QueryRequest
from app.utils.redis_distributed import (
    compute_idempotency_key,
    redis_idempotency_get,
    redis_idempotency_set,
)


async def _resolve_idempotency(
    req: QueryRequest,
    request: Request | None,
) -> Optional[Dict[str, Any]]:
    """Return a cached idempotent response when Redis has one."""
    if request is None:
        return None
    header_key = getattr(request.state, "idempotency_key", None)
    if not header_key:
        return None
    composite = _composite_key(req, header_key)
    cached = await redis_idempotency_get(composite)
    if cached and cached.get("status") == "completed":
        return cached.get("body")
    return None


def _composite_key(req: QueryRequest, header_key: str) -> str:
    """The key both sides must agree on.

    O modelo escolhido fica deliberadamente de fora: só se conhece depois de a
    resposta existir, e incluí-lo tornaria a chave de escrita diferente da de
    leitura — que era exactamente o defeito.
    """
    digest = compute_idempotency_key(
        tenant_id=req.tenant_id, query=req.query, modality=req.modality, model=""
    )
    return f"{header_key}:{digest}"


async def _store_idempotency(
    req: QueryRequest,
    request: Request | None,
    body: Dict[str, Any],
) -> None:
    if request is None:
        return
    header_key = getattr(request.state, "idempotency_key", None)
    if not header_key:
        return
    # `model=""`, igual ao lado da leitura. Gravar com o modelo escolhido —
    # que só se conhece DEPOIS de responder — produzia uma chave diferente da
    # que a leitura procura, por isso um reenvio nunca encontrava nada e a
    # idempotência não funcionava de todo.
    composite = _composite_key(req, header_key)
    await redis_idempotency_set(composite, {"status": "completed", "body": body}, ttl_s=300)
