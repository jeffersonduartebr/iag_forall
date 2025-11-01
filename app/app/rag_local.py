# app/rag_local.py
from __future__ import annotations
import os as _os # Mantido caso 'os' seja usado em outro lugar, mas prefixado
import logging
import asyncio
from typing import List, Dict, Any, Optional

# ✅ Importa as funções do módulo centralizado
from app.vectorstore import (
    get_or_create_collection_async,
    insert_embedding,
    query_embedding,
    health_async as chroma_health
)
from app.embeddings import embed_text
# ✅ Importa o módulo de settings centralizado
from app.settings_dynamic import settings

logger = logging.getLogger(__name__)

# --- Configurações Específicas do RAG (Lidas do settings) ---
# CORRIGIDO: Lê da propriedade centralizada
EMBED_MODEL = settings.EMBED_MODEL
# CORRIGIDO: Lê do .get() para configuração dinâmica
TOP_K = int(settings.get("RAG_TOP_K", 3))
RAG_COLLECTION_NAME = "docs" # Nome da coleção para documentos RAG


# ❌ Toda a inicialização do PersistentClient foi REMOVIDA
# ❌ Funções auxiliares (_add_document_sync, _query_sync) REMOVIDAS


# --- Funções públicas assíncronas (Agora usam o 'vectorstore') ---

async def add_document(doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Adiciona um documento RAG ao vector store centralizado."""
    try:
        # 1. Garante que a coleção 'docs' exista
        await get_or_create_collection_async(RAG_COLLECTION_NAME, {"source": "rag_docs"})
        
        # 2. Gera embedding de forma assíncrona
        embedding = await embed_text(text)

        # 3. Adiciona ao Chroma via módulo central
        await insert_embedding(
            collection_name=RAG_COLLECTION_NAME,
            doc_id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {"source": "warmup"}
        )
        
        logger.info(f"[RAG] Documento '{doc_id}' adicionado à coleção '{RAG_COLLECTION_NAME}'.")
        return True
    except Exception as e:
        logger.error(f"[RAG] Falha ao adicionar documento '{doc_id}': {e}")
        return False

async def build_augmented_prompt(query: str) -> str:
    """Monta um prompt RAG consultando o vector store centralizado."""
    if not query:
        return ""
    try:
        # 1. Gera embedding de forma assíncrona
        embedding = await embed_text(query)

        # 2. Consulta o Chroma via módulo central
        res = await query_embedding(
            collection_name=RAG_COLLECTION_NAME,
            embedding=embedding,
            n_results=TOP_K # Usa o TOP_K lido do settings
        )

        # 3. Processa resultados
        docs = (res or {}).get("documents", [[]])
        top_docs: List[str] = docs[0] if docs and isinstance(docs[0], list) else []
        context = "\n\n".join(top_docs[:TOP_K])
        
        if not context.strip():
            return query
        return (
            "Use the following context if relevant.\n"
            "-----\n"
            f"{context}\n"
            "-----\n"
            f"User: {query}"
        )
    except Exception as e:
        logger.error(f"[RAG] Falha na consulta de contexto: {e}")
        return query


async def health() -> Dict[str, Any]:
    """Retorna informações de saúde do módulo RAG."""
    return {
        "chroma_ready": await chroma_health(), # ✅ Verifica a saúde central
        "embed_model": EMBED_MODEL, # ✅ Reporta o modelo lido do settings
        "top_k": TOP_K, # ✅ Reporta o K lido do settings
        "collection_name": RAG_COLLECTION_NAME
    }