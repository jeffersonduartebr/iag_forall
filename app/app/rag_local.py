# app/rag_local.py
from __future__ import annotations
import os as _os
_os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")

import logging
import asyncio  # 👈 Adicionado
from typing import List, Dict, Any, Optional

# ✦ API NOVA do Chroma
import chromadb
from chromadb.api.types import Documents
from chromadb import PersistentClient

from app.embeddings import embed_text  # 👈 Adicionado

logger = logging.getLogger(__name__)

CHROMA_PATH = _os.getenv("CHROMA_PATH", "/data/chroma")
OLLAMA_BASE_URL = _os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBED_MODEL = _os.getenv("EMBED_MODEL", "nomic-embed-text")
TOP_K = int(_os.getenv("RAG_TOP_K", "3"))
EMBED_DIM_DEFAULT = int(_os.getenv("EMBED_DIM", "768"))

# 
# ❌ CLASSE OllamaEmbeddingFunction REMOVIDA
#

# --- Inicialização segura do Chroma (API nova) ---
try:
    _os.makedirs(CHROMA_PATH, exist_ok=True)
    chroma_client = PersistentClient(path=CHROMA_PATH)
    
    # get_or_create_collection SEM embedding_function
    collection = chroma_client.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"},
        # embedding_function=embed_fn, # 👈 Removido
    )
    logger.info("[RAG] Chroma (PersistentClient) e coleção 'docs' inicializados.")
except Exception as e:
    logger.exception(f"[RAG] Erro ao inicializar Chroma: {e}")
    chroma_client = None
    collection = None

# --- Funções auxiliares síncronas para to_thread ---

def _add_document_sync(doc_id: str, text: str, embedding: list, metadata: Optional[Dict[str, Any]] = None):
    """Função síncrona auxiliar para escrita em disco."""
    if collection is None:
        raise RuntimeError("[RAG] Coleção Chroma indisponível.")
    
    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],  # 👈 Passa o embedding manualmente
        metadatas=[metadata or {"source": "warmup"}],
    )
    if hasattr(chroma_client, "persist"):
        chroma_client.persist()

def _query_sync(embedding: list) -> Optional[dict]:
    """Função síncrona auxiliar para consulta de disco."""
    if collection is None:
        logger.error("[RAG] Coleção Chroma indisponível.")
        return None
    
    # 👈 Usa query_embeddings em vez de query_texts
    return collection.query(query_embeddings=[embedding], n_results=TOP_K)

# --- Funções públicas assíncronas ---

async def add_document(doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    # 👈 Convertido para async def
    if collection is None:
        logger.error("[RAG] Coleção Chroma indisponível (falha de inicialização).")
        return False
    try:
        # 1. Gera embedding de forma assíncrona
        embedding = await embed_text(text)

        # 2. Adiciona ao Chroma em um thread separado (I/O de disco)
        await asyncio.to_thread(
            _add_document_sync,
            doc_id, text, embedding, metadata
        )
        
        logger.info(f"[RAG] Documento '{doc_id}' adicionado.")
        return True
    except Exception as e:
        logger.error(f"[RAG] Falha ao adicionar documento '{doc_id}': {e}")
        return False

async def build_augmented_prompt(query: str) -> str:
    # 👈 Convertido para async def
    if not query:
        return ""
    if collection is None:
        logger.warning("[RAG] Coleção indisponível; retornando query sem contexto.")
        return query
    try:
        # 1. Gera embedding de forma assíncrona
        embedding = await embed_text(query)

        # 2. Consulta o Chroma em um thread separado (I/O de disco)
        res = await asyncio.to_thread(_query_sync, embedding)

        # 3. Processa resultados (lógica original)
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

def health() -> Dict[str, Any]:
    # ... (função original está OK) ...
    return {
        "chroma_ready": collection is not None,
        "path": CHROMA_PATH,
        "embed_model": EMBED_MODEL,
        "top_k": TOP_K,
    }