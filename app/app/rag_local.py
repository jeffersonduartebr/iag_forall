# -*- coding: utf-8 -*-
"""
rag_local.py — RAG Multimodal Unificado (versão revisada e corrigida)
---------------------------------------------------------------------

Funções principais:
    • build_augmented_prompt(query, modality, image_b64, k)
    • add_document_local(doc_id, text, metadata, modality, image_b64)
    • health()

Compatível com:
    - vectorstore.py (revisado)
    - embeddings.py multimodal síncrono
    - router_core multimodal
    - semantic_cache multimodal

Características:
    ✔ Detecção automática da modalidade real (text / vision / multimodal)
    ✔ Embeddings via asyncio.to_thread (pois os embeds são síncronos)
    ✔ Recuperação contextual multimodal (via vectorstore)
    ✔ Prompt augmentation seguro
"""

from __future__ import annotations
import logging
import asyncio
from typing import Optional, Dict, Any, List

from .embeddings import (
    embed_text,
    embed_image,
    embed_multimodal,
)

from .vectorstore import (
    query_embedding,
    add_document,
    health_async,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] rag_local: %(message)s"
    )


# ================================================================
# 🔍 DETECÇÃO AUTOMÁTICA DE MODALIDADE
# ================================================================

def _auto_modality(requested: Optional[str], image_b64: Optional[str]) -> str:
    req = (requested or "text").lower().strip()

    if image_b64 and req == "multimodal":
        return "multimodal"
    if image_b64 and req in ("vision", "image"):
        return "vision"
    if image_b64 and req == "text":
        return "vision"

    return "text"


# ================================================================
# 🧠 GERAÇÃO DE EMBEDDING ADEQUADO
# ================================================================

async def _compute_embedding(query: str, modality: str, image_b64: Optional[str]):
    """
    Gera o embedding adequado conforme modalidade real.
    As funções de embedding são síncronas → enviamos ao threadpool.
    """
    try:
        if modality == "text":
            return await asyncio.to_thread(embed_text, query)

        if modality == "vision":
            if not image_b64:
                return await asyncio.to_thread(embed_text, query)
            return await asyncio.to_thread(embed_image, image_b64)

        # multimodal
        emb_dict = await asyncio.to_thread(embed_multimodal, query, image_b64)
        return emb_dict.get("multimodal") or emb_dict.get("text")

    except Exception as e:
        logger.warning(f"[rag_local] Falha ao gerar embedding ({modality}): {e}")
        return None


# ================================================================
# 📚 RAG MULTIMODAL – Construção do Prompt Aumentado
# ================================================================

async def build_augmented_prompt(
    query: str,
    modality: str = "text",
    image_b64: Optional[str] = None,
    k: int = 3,
) -> str:
    query = (query or "").strip()
    if not query and not image_b64:
        return query

    rag_mode = _auto_modality(modality, image_b64)

    emb = await _compute_embedding(query, rag_mode, image_b64)
    if emb is None:
        return query

    try:
        res = await query_embedding(
            modality=rag_mode,
            embedding=emb,
            n_results=k
        )
    except Exception as e:
        logger.warning(f"[rag_local] Erro na consulta RAG ({rag_mode}): {e}")
        return query

    docs = res.get("documents", [[]])
    top_docs: List[str] = docs[0] if docs and isinstance(docs[0], list) else []

    if not top_docs:
        return query

    context = "\n\n".join(top_docs)

    return (
        "Use o contexto abaixo somente se for relevante.\n"
        "------\n"
        f"{context}\n"
        "------\n"
        f"Usuário: {query}"
    )


# ================================================================
# 📝 ADICIONAR DOCUMENTO AO RAG
# ================================================================

async def add_document_local(
    doc_id: str,
    text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    modality: str = "text",
    image_b64: Optional[str] = None,
) -> bool:
    try:
        await add_document(
            modality=modality,
            doc_id=doc_id,
            text=text,
            image_b64=image_b64,
            metadata=metadata,
        )
        return True
    except Exception as e:
        logger.error(f"[rag_local] Falha ao adicionar documento {doc_id}: {e}")
        return False


# ================================================================
# 🩺 HEALTHCHECK
# ================================================================

async def health() -> Dict[str, Any]:
    try:
        chroma_ok = await health_async()
    except Exception:
        chroma_ok = False

    return {
        "vectorstore": chroma_ok,
        "status": "ok" if chroma_ok else "fail"
    }
