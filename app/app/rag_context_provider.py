# -*- coding: utf-8 -*-
"""
rag_context_provider.py
----------------------------------------------------
Fornece contexto RAG para juízes e para o roteador multimodal.

Agora suporta:
✔ Embeddings assíncronos
✔ RAG multimodal (texto + visão)
✔ Coleção dinâmica lida do settings
✔ Vectorstore centralizado (Chroma)
✔ Logs consistentes
"""

from __future__ import annotations

import logging
from typing import Optional

from app.vectorstore import query_embedding
from app.embeddings import (
    embed_text,
    embed_multimodal,   # multimodal-aware
)
from app.settings_dynamic import settings

logger = logging.getLogger(__name__)

# Nome da coleção RAG (compatível com rag_local.py)
RAG_COLLECTION_NAME = settings.get("RAG_COLLECTION_NAME", "docs")

# Número padrão de documentos recuperados
DEFAULT_TOP_K = int(settings.get("RAG_TOP_K", 3))


# =====================================================================
# 🔍 Função principal: recuperar contexto RAG (texto ou multimodal)
# =====================================================================

async def get_rag_context(
    query: str,
    k: Optional[int] = None,
    modality: str = "text",
    image_b64: Optional[str] = None,
) -> str:
    """
    Recupera contexto para enriquecer um prompt — inclusive para juízes multimodais.

    Params:
        query: texto da pergunta do usuário
        k: número de documentos (default configurado no settings)
        modality: "text" | "vision" | "multimodal"
        image_b64: imagem codificada (quando modality != text)

    Retorna:
        string contendo o contexto concatenado
    """

    k = k or DEFAULT_TOP_K
    modality = modality or "text"

    try:
        # ============================================================
        # 1) Gerar embedding correto baseado na modalidade
        # ============================================================
        if modality == "vision":
            if not image_b64:
                logger.warning(
                    "[rag_context_provider] modality='vision' sem imagem fornecida; "
                    "fallback para embed_text()."
                )
                embedding = await embed_text(query)
            else:
                embedding = await embed_multimodal(query, image_b64=image_b64)

        elif modality == "multimodal":
            embedding = await embed_multimodal(query, image_b64=image_b64)

        else:  # TEXT
            embedding = await embed_text(query)

        if embedding is None:
            logger.warning(
                "[rag_context_provider] embedding=NONE — abortando recuperação de contexto."
            )
            return ""

        # ============================================================
        # 2) Consulta ao vectorstore centralizado (Chroma)
        # ============================================================
        results = await query_embedding(
            collection_name=RAG_COLLECTION_NAME,
            embedding=embedding,
            n_results=k,
        )

        if not results or "documents" not in results:
            logger.info(
                "[rag_context_provider] Nenhum documento retornado pelo vectorstore."
            )
            return ""

        # ============================================================
        # 3) Processar documentos recuperados
        # ============================================================
        docs = results.get("documents", [[]])
        top_docs = docs[0] if docs and isinstance(docs[0], list) else []

        if not top_docs:
            logger.info("[rag_context_provider] docs vazios.")
            return ""

        context = "\n\n".join(top_docs)

        logger.info(
            f"[rag_context_provider] {len(top_docs)} trechos recuperados "
            f"para modality='{modality}'."
        )

        return context

    except Exception as e:
        logger.error(f"[rag_context_provider] Falha ao buscar contexto RAG: {e}")
        return ""
