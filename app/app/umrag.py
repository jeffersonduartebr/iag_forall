# -*- coding: utf-8 -*-
"""
umrag.py — Unified Multimodal RAG
---------------------------------------------------------
Versão simplificada, moderna e 100% compatível com:

 - semantic_cache.py  (cache multimodal completo)
 - router_core
 - rag_local

Responsável por:

 - RAG unificado multimodal (texto / visão / multimodal)
 - Indexação de documentos para o vectorstore
 - Delegação TOTAL do cache ao semantic_cache
 - API estável e retrocompatível

Funções públicas:

 - build_augmented_prompt(...)
 - add_document(...)
 - cache_lookup(...)     → delegado ao semantic_cache
 - cache_store(...)      → delegado ao semantic_cache
 - health()
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, Any, List

import numpy as np

from app.embeddings import embed_text, embed_image, embed_multimodal
from app.vectorstore import (
    add_document as vs_add_document,
    query_embedding as vs_query_embedding,
    health_async as vector_health_async,
)

# delega tudo de cache ao módulo semantic_cache
from app.semantic_cache import (
    check_cache as cache_lookup,
    store_cache as cache_store,
)

from app.settings_dynamic import settings


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================
# 🔧 Configurações
# ============================================================
RAG_TOP_K = int(settings.get("RAG_TOP_K", 3))


# ============================================================
# 🧠 UTILIDADES DE EMBEDDING RAG
# ============================================================

def _unit(x: np.ndarray) -> np.ndarray:
    """Resumo do comportamento desta função.

    Args:
        x: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    x = x.astype(np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    return x if n == 0 else (x / n)


def _embed_for_rag(
    query: str,
    modality: str = "text",
    image_b64: Optional[str] = None,
) -> np.ndarray:

    """Resumo do comportamento desta função.

    Args:
        query: Parâmetro de entrada.
        modality: Parâmetro de entrada.
        image_b64: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    modality = (modality or "text").lower()
    q = (query or "").strip()

    # ----- TEXT -----
    if modality == "text":
        return _unit(np.array(embed_text(q), dtype=np.float32))

    # ----- VISION -----
    if modality == "vision":
        if q:  # prefer texto para visão híbrida
            return _unit(np.array(embed_text(q), dtype=np.float32))
        if image_b64:
            return _unit(np.array(embed_image(image_b64), dtype=np.float32))
        return _unit(np.array(embed_text(q), dtype=np.float32))

    # ----- MULTIMODAL -----
    try:
        emb = embed_multimodal(q, image_b64)
        if isinstance(emb, dict) and "multimodal" in emb:
            return _unit(np.array(emb["multimodal"], dtype=np.float32))
    except Exception as e:
        logger.warning(f"[UM-RAG] embed multimodal falhou, fallback texto: {e}")

    return _unit(np.array(embed_text(q), dtype=np.float32))


# ============================================================
# 🔍 EXTRAIR DOCUMENTOS DO VECTORSTORE
# ============================================================

def _extract_docs_from(res: Dict[str, Any]) -> List[str]:
    """Resumo do comportamento desta função.

    Args:
        res: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    if not res:
        return []
    docs = res.get("documents") or []
    if not docs:
        return []
    if isinstance(docs[0], list):
        return docs[0]
    return docs


# ============================================================
# 📚 RAG MULTIMODAL
# ============================================================

async def build_augmented_prompt(
    query: str,
    modality: str = "text",
    image_b64: Optional[str] = None,
    top_k: Optional[int] = None,
) -> str:
    """
    Estrategia unificada (C):
        1) Texto
        2) Visão
        3) Multimodal
    """
    q = (query or "").strip()
    if not q and not image_b64:
        return ""

    modality = (modality or "text").lower()
    k = top_k or RAG_TOP_K
    context_docs: List[str] = []

    # 1. TEXTO
    try:
        v = await asyncio.to_thread(_embed_for_rag, q, "text", None)
        res = await vs_query_embedding("text", v, n_results=k)
        context_docs = _extract_docs_from(res or {})
    except Exception as e:
        logger.warning(f"[UM-RAG] Falha em RAG-text: {e}")

    # 2. VISÃO (fallback)
    if not context_docs and image_b64:
        try:
            v = await asyncio.to_thread(_embed_for_rag, q, "vision", image_b64)
            res = await vs_query_embedding("vision", v, n_results=k)
            context_docs = _extract_docs_from(res or {})
        except Exception as e:
            logger.warning(f"[UM-RAG] Falha em RAG-vision: {e}")

    # 3. MULTIMODAL (fallback final)
    if not context_docs and q and image_b64:
        try:
            v = await asyncio.to_thread(_embed_for_rag, q, "multimodal", image_b64)
            res = await vs_query_embedding("multimodal", v, n_results=k)
            context_docs = _extract_docs_from(res or {})
        except Exception as e:
            logger.warning(f"[UM-RAG] Falha em RAG-multimodal: {e}")

    # Se nada encontrado
    if not context_docs:
        return q

    context = "\n\n".join(context_docs[:k])
    return (
        "Use o seguinte contexto se for relevante.\n"
        "-----\n"
        f"{context}\n"
        "-----\n"
        f"Usuário: {q}"
    )


# ============================================================
# 🧩 INDEXAÇÃO MULTIMODAL
# ============================================================

async def add_document(
    doc_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    modality: str = "text",
    image_b64: Optional[str] = None,
) -> bool:
    """
    Compatível com chamadas antigas.
    Envia diretamente ao vectorstore.
    """
    try:
        await vs_add_document(
            modality=modality,
            doc_id=doc_id,
            text=text,
            image_b64=image_b64,
            metadata=metadata,
        )
        logger.info(f"[UM-RAG] Documento indexado {doc_id} ({modality}).")
        return True
    except Exception as e:
        logger.error(f"[UM-RAG] Falha ao indexar {doc_id}: {e}")
        return False


# ============================================================
# 🩺 HEALTHCHECK
# ============================================================

async def health() -> Dict[str, Any]:
    """Resumo do comportamento desta função.

    Returns:
        Valor retornado pela função.
    """
    try:
        chroma_ok = await vector_health_async()
    except Exception:
        chroma_ok = False

    return {
        "chroma_ready": chroma_ok,
        "rag_top_k": RAG_TOP_K,
        "cache": "delegated_to_semantic_cache",
    }
