# -*- coding: utf-8 -*-
"""
rag_healthcheck.py
----------------------------------------------------
Validador automático de saúde do pipeline RAG.

Checa:
1) Embeddings (embed_text) — síncrono
2) Chroma client up (get_or_create_collection_async)
3) Insert + Query no Chroma (async)
4) Redis opcional funcionando para cache semântico (best-effort)
5) Latências básicas de cada etapa

Expõe:
- async def rag_healthcheck(): -> dict
- def rag_healthcheck_sync(timeout_s=10): -> dict (wrapper para contextos síncronos)
"""

from __future__ import annotations

import time
import asyncio
from typing import Any, Dict

from app.embeddings import embed_text           # ⚠️ SÍNCRONO
from app.vectorstore import (
    get_or_create_collection_async,
    insert_embedding,
    query_embedding,
)
from app.utils.redis_client import get_redis


COLLECTION = "rag_healthcheck"
DOC_TEXT = "Documento de verificação do pipeline RAG (healthcheck)."
QUERY_TEXT = "Verificação do pipeline RAG e conectividade com a base vetorial."


async def rag_healthcheck() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "ok": False,
        "steps": {},
    }

    t0 = time.time()

    # 1) Embedding síncrono
    t = time.time()
    try:
        q_vec = embed_text(QUERY_TEXT)  # SÍNCRONO — sem await
        d_vec = embed_text(DOC_TEXT)    # idem
        report["steps"]["embeddings"] = {"ok": True, "latency_s": round(time.time() - t, 3)}
    except Exception as e:
        report["steps"]["embeddings"] = {"ok": False, "error": str(e)}
        return _finalize(report, t0)

    # 2) Chroma: garantir coleção
    t = time.time()
    try:
        col = await get_or_create_collection_async(COLLECTION, metadata={"source": "healthcheck"})
        report["steps"]["vectorstore_collection"] = {"ok": bool(col), "latency_s": round(time.time() - t, 3)}
        if not col:
            return _finalize(report, t0)
    except Exception as e:
        report["steps"]["vectorstore_collection"] = {"ok": False, "error": str(e)}
        return _finalize(report, t0)

    # 3) Inserção + Query
    #    Usa IDs com timestamp para evitar conflitos
    t = time.time()
    try:
        doc_id = f"hc:{int(time.time()*1000)}"
        await insert_embedding(COLLECTION, doc_id, DOC_TEXT, d_vec.tolist(), metadata={"kind": "hc"})
        report["steps"]["vectorstore_insert"] = {"ok": True, "latency_s": round(time.time() - t, 3)}
    except Exception as e:
        report["steps"]["vectorstore_insert"] = {"ok": False, "error": str(e)}
        return _finalize(report, t0)

    t = time.time()
    try:
        res = await query_embedding(COLLECTION, q_vec.tolist(), n_results=1)
        ok = bool(res and "documents" in res and res["documents"])
        report["steps"]["vectorstore_query"] = {"ok": ok, "latency_s": round(time.time() - t, 3)}
        if not ok:
            return _finalize(report, t0)
    except Exception as e:
        report["steps"]["vectorstore_query"] = {"ok": False, "error": str(e)}
        return _finalize(report, t0)

    # 4) Redis (opcional)
    t = time.time()
    try:
        r = get_redis()
        if r:
            k = "rag_hc:ping"
            r.setex(k, 5, "pong")
            val = r.get(k)
            report["steps"]["redis"] = {"ok": val == b"pong", "latency_s": round(time.time() - t, 3)}
        else:
            report["steps"]["redis"] = {"ok": False, "warning": "Redis não configurado"}
    except Exception as e:
        report["steps"]["redis"] = {"ok": False, "error": str(e)}

    # Tudo ok se todos os steps 'ok' True (exceto redis opcional)
    mandatory = ["embeddings", "vectorstore_collection", "vectorstore_insert", "vectorstore_query"]
    report["ok"] = all(report["steps"].get(s, {}).get("ok") for s in mandatory)
    return _finalize(report, t0)


def rag_healthcheck_sync(timeout_s: int = 10) -> Dict[str, Any]:
    """
    Wrapper síncrono — útil para scripts, CLIs e endpoints WSGI.
    """
    async def _runner():
        return await rag_healthcheck()

    return asyncio.run(asyncio.wait_for(_runner(), timeout=timeout_s))


def _finalize(report: Dict[str, Any], t0: float) -> Dict[str, Any]:
    report["latency_total_s"] = round(time.time() - t0, 3)
    return report
