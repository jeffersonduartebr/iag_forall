# Objective: Scope RAG retrieval and deletion to a tenant plus client-declared metadata.
"""Scoped RAG: the ``where`` clause behind ``QueryRequest.rag_filter`` and ``POST /rag/delete``.

A scope is a flat set of metadata equalities (e.g. ``{"disciplina": "bd", "periodo": "2026.2"}``)
AND-ed with the caller's tenant. Dense search applies it natively; BM25 hits carry no metadata, so
they are kept only when Chroma confirms the same id matches the scope. A scope that cannot be checked
returns no sparse hits: leaking another scope's document is worse than losing keyword recall.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..sparse_index import sparse_index
from ..vectorstore import _collection_for_modality, get_chroma_client

logger = logging.getLogger("rag_scope")

TENANT_KEY = "tenant_id"


def scope_where(rag_filter: Optional[Dict[str, Any]], tenant_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Chroma ``where`` for one scope, or ``None`` when the request declared no scope.

    The tenant always wins over a ``tenant_id`` key sent by the client.
    """
    if not rag_filter:
        return None
    conditions = {key: value for key, value in rag_filter.items() if key != TENANT_KEY}
    if tenant_id:
        conditions[TENANT_KEY] = tenant_id
    clauses = [{key: value} for key, value in sorted(conditions.items())]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _ids_in_scope_sync(collection_modality: str, ids: List[str], where: Dict[str, Any]) -> set:
    col = get_chroma_client().get_or_create_collection(name=_collection_for_modality(collection_modality))
    return set(col.get(ids=ids, where=where, include=[]).get("ids") or [])


async def restrict_to_scope(hits: Any, where: Optional[Dict[str, Any]], collection_modality: str) -> Any:
    """Drop sparse hits outside the scope (``hits`` is ``(ids, id->text, id->meta)``)."""
    ids, docs_map, meta_map = hits
    if where is None or not ids:
        return hits
    try:
        allowed = await asyncio.to_thread(_ids_in_scope_sync, collection_modality, list(ids), where)
    except Exception as exc:
        logger.warning("[rag_scope] could not verify BM25 hits against the scope, dropping them: %s", exc)
        allowed = set()
    kept = [doc_id for doc_id in ids if doc_id in allowed]
    return kept, {i: docs_map[i] for i in kept if i in docs_map}, {i: meta_map[i] for i in kept if i in meta_map}


def _delete_scope_sync(where: Dict[str, Any]) -> List[str]:
    col = get_chroma_client().get_or_create_collection(name=_collection_for_modality("text"))
    ids = list(col.get(where=where, include=[]).get("ids") or [])
    if ids:
        col.delete(ids=ids)
        sparse_index.remove_documents(ids)
    return ids


async def delete_scope(where: Dict[str, Any]) -> List[str]:
    """Delete every text chunk matching the scope from Chroma and BM25; returns the removed ids."""
    return await asyncio.to_thread(_delete_scope_sync, where)
