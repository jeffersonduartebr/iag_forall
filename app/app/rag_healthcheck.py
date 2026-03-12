# -*- coding: utf-8 -*-
# Objective: Application runtime code for rag healthcheck.
"""
rag_healthcheck.py
----------------------------------------------------
Validador automático de saúde do pipeline RAG (multimodal-aware).

Checa:
1) Embeddings de texto (embed_text) — assíncrono
2) Embeddings multimodais (embed_multimodal) — assíncrono (best-effort)
3) Chroma client up (get_or_create_collection_async)
4) Insert + Query no Chroma (async)
5) Redis opcional funcionando para cache semântico (best-effort)
6) Latências básicas de cada etapa

Expõe:
- async def rag_healthcheck(): -> dict
- def rag_healthcheck_sync(timeout_s=10): -> dict (wrapper para contextos síncronos)
"""

from __future__ import annotations

import time
import asyncio
from typing import Any, Dict

from app.embeddings import (
    embed_text,         # assíncrono
    embed_multimodal,   # assíncrono (texto + imagem opcional)
)
from app.vectorstore import (
    get_or_create_collection_async,
    insert_embedding,
    query_embedding,
)
from app.utils.redis_client import get_redis

# Coleção dedicada ao healthcheck
COLLECTION = "rag_healthcheck"
DOC_TEXT = "Documento de verificação do pipeline RAG (healthcheck)."
QUERY_TEXT = "Verificação do pipeline RAG e conectividade com a base vetorial."


async def rag_healthcheck() -> Dict[str, Any]:
    """Execute the rag healthcheck routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    report: Dict[str, Any] = {
        "ok": False,
        "steps": {},
    }

    t0 = time.time()

    # 1) Embeddings de TEXTO (obrigatório)
    t = time.time()
    try:
        q_vec = await embed_text(QUERY_TEXT)
        d_vec = await embed_text(DOC_TEXT)
        report["steps"]["embeddings_text"] = {
            "ok": True,
            "latency_s": round(time.time() - t, 3),
        }
    except Exception as e:
        report["steps"]["embeddings_text"] = {"ok": False, "error": str(e)}
        return _finalize(report, t0)

    # 2) Embeddings MULTIMODAIS (best-effort, não bloqueia o ok geral)
    t = time.time()
    try:
        _ = await embed_multimodal(QUERY_TEXT, image_b64=None)
        report["steps"]["embeddings_multimodal"] = {
            "ok": True,
            "latency_s": round(time.time() - t, 3),
        }
    except Exception as e:
        # Não derruba o healthcheck inteiro, apenas registra o erro
        report["steps"]["embeddings_multimodal"] = {
            "ok": False,
            "error": str(e),
        }

    # 3) Chroma: garantir coleção
    t = time.time()
    try:
        col = await get_or_create_collection_async(
            COLLECTION,
            metadata={"source": "healthcheck"},
        )
        report["steps"]["vectorstore_collection"] = {
            "ok": bool(col),
            "latency_s": round(time.time() - t, 3),
        }
        if not col:
            return _finalize(report, t0)
    except Exception as e:
        report["steps"]["vectorstore_collection"] = {"ok": False, "error": str(e)}
        return _finalize(report, t0)

    # 4) Inserção no vectorstore
    t = time.time()
    try:
        doc_id = f"hc:{int(time.time() * 1000)}"
        # d_vec pode ser np.ndarray ou list — o vectorstore já faz a blindagem
        await insert_embedding(
            COLLECTION,
            doc_id,
            DOC_TEXT,
            d_vec,
            metadata={"kind": "hc"},
        )
        report["steps"]["vectorstore_insert"] = {
            "ok": True,
            "latency_s": round(time.time() - t, 3),
        }
    except Exception as e:
        report["steps"]["vectorstore_insert"] = {"ok": False, "error": str(e)}
        return _finalize(report, t0)

    # 5) Query no vectorstore
    t = time.time()
    try:
        res = await query_embedding(COLLECTION, q_vec, n_results=1)
        ok = bool(res and "documents" in res and res["documents"])
        report["steps"]["vectorstore_query"] = {
            "ok": ok,
            "latency_s": round(time.time() - t, 3),
        }
        if not ok:
            return _finalize(report, t0)
    except Exception as e:
        report["steps"]["vectorstore_query"] = {"ok": False, "error": str(e)}
        return _finalize(report, t0)

    # 6) Redis (opcional)
    t = time.time()
    try:
        r = get_redis()
        if r:
            k = "rag_hc:ping"
            r.setex(k, 5, "pong")
            val = r.get(k)
            # dependendo da config, pode vir str ou bytes
            ok_redis = val in (b"pong", "pong")
            report["steps"]["redis"] = {
                "ok": ok_redis,
                "latency_s": round(time.time() - t, 3),
            }
        else:
            report["steps"]["redis"] = {
                "ok": False,
                "warning": "Redis não configurado",
            }
    except Exception as e:
        report["steps"]["redis"] = {"ok": False, "error": str(e)}

    # Para o ok geral, apenas passos mandatórios de TEXTO e vectorstore contam
    mandatory = [
        "embeddings_text",
        "vectorstore_collection",
        "vectorstore_insert",
        "vectorstore_query",
    ]
    report["ok"] = all(report["steps"].get(s, {}).get("ok") for s in mandatory)
    return _finalize(report, t0)


def rag_healthcheck_sync(timeout_s: int = 10) -> Dict[str, Any]:
    """
    Wrapper síncrono — útil para scripts, CLIs e contexts não-async.
    """
    async def _runner():
        """Execute the runner routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return await rag_healthcheck()

    return asyncio.run(asyncio.wait_for(_runner(), timeout=timeout_s))


def _finalize(report: Dict[str, Any], t0: float) -> Dict[str, Any]:
    """Execute the finalize routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    report["latency_total_s"] = round(time.time() - t0, 3)
    return report
