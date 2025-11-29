# -*- coding: utf-8 -*-
"""
vectorstore.py — RAG Multimodal com Versionamento de Coleção
----------------------------------------------------------------------
Gerencia a persistência de embeddings no ChromaDB.
Atualizado para incluir o nome do modelo de embedding no nome da coleção.
Isso evita conflitos de espaço vetorial ao trocar de modelo.
"""

from __future__ import annotations

import os
import logging
import asyncio
import re
from typing import List, Dict, Any, Optional

import chromadb
import numpy as np

from .embeddings import embed_text, embed_image, embed_multimodal
from .settings_dynamic import settings
from .sparse_index import sparse_index # <--- Importar o novo módulo

# ============================================================
# Logging
# ============================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] vectorstore: %(message)s"
    )


# ============================================================
# Configurações & Versionamento
# ============================================================
CHROMA_PATH = settings.get("CHROMA_PATH", "/data/chroma")

# Nomes base das coleções
BASE_TEXT_COLLECTION = "text_embeddings"
BASE_IMAGE_COLLECTION = "image_embeddings"
BASE_MULTIMODAL_COLLECTION = "multimodal_embeddings"
BASE_CACHE_COLLECTION = "semantic_cache_v2"

VALID_MODALITIES = {"text", "vision", "multimodal", "image", "cache"}


def _sanitize_model_name(model_name: str) -> str:
    """
    Transforma 'nomic-ai/nomic-embed-text-v1.5' em 'nomic_ai_nomic_embed_text_v1_5'.
    Usado para sufixar a coleção.
    """
    if not model_name:
        return "default"
    # Remove caracteres especiais e substitui por underscore
    clean = re.sub(r"[^a-zA-Z0-9]", "_", model_name)
    # Remove underscores duplicados e underscores nas pontas
    return re.sub(r"_+", "_", clean).strip("_")


def _get_versioned_collection_name(base_name: str, modality: str) -> str:
    """
    Gera o nome da coleção atrelado ao modelo configurado no settings.
    Ex: text_embeddings_nomic_embed_text_v1_5
    """
    if modality == "text" or modality == "cache":
        model = settings.EMBED_TEXT_MODEL or "default_text"
    elif modality == "vision":
        model = settings.IMAGE_EMBEDDING_MODEL or "clip"
    else:
        model = settings.MULTIMODAL_EMBEDDING_MODEL or "default_mm"
    
    version_suffix = _sanitize_model_name(model)
    return f"{base_name}_{version_suffix}"


# ============================================================
# Conexão com ChromaDB
# ============================================================
def _connect_local():
    """Inicializa cliente persistente do ChromaDB."""
    try:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        logger.info(f"[vectorstore] Inicializando ChromaDB em {CHROMA_PATH}")
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        logger.info("[vectorstore] Chroma PersistentClient inicializado.")
        return client
    except Exception as e:
        logger.error(f"[vectorstore] Falha ao iniciar ChromaDB: {e}")
        raise


chroma_client = _connect_local()


# ============================================================
# Helpers Internos
# ============================================================
def _ensure_list_of_floats(vec):
    """Converte embedding para list[float]."""
    if isinstance(vec, np.ndarray):
        return vec.astype(float).ravel().tolist()
    if isinstance(vec, (list, tuple)):
        return [float(x) for x in np.array(vec, dtype=float).ravel()]
    return [0.0]


def _safe_metadata(meta):
    return meta if isinstance(meta, dict) else {"source": "router"}


def _normalize_modality(modality: Optional[str]) -> str:
    if not modality:
        return "text"
    m = modality.lower().strip()
    if m == "image":
        return "vision"
    return m if m in VALID_MODALITIES else "text"


def _collection_for_modality(modality: str) -> str:
    """Retorna o nome versionado da coleção baseado na modalidade."""
    m = _normalize_modality(modality)
    if m == "cache":
        return _get_versioned_collection_name(BASE_CACHE_COLLECTION, "text")
    if m == "text":
        return _get_versioned_collection_name(BASE_TEXT_COLLECTION, "text")
    if m == "vision":
        return _get_versioned_collection_name(BASE_IMAGE_COLLECTION, "vision")
    return _get_versioned_collection_name(BASE_MULTIMODAL_COLLECTION, "multimodal")


# ============================================================
# Gerenciamento de Coleções (Async Wrapper)
# ============================================================

def _get_or_create_sync(name: str, metadata: Optional[Dict] = None):
    """Wrapper síncrono para o client do Chroma."""
    return chroma_client.get_or_create_collection(name=name, metadata=metadata)


async def get_or_create_collection_async(name: str, metadata: Optional[Dict] = None):
    """
    Cria ou recupera uma coleção de forma assíncrona (threadpool).
    Aplica lógica de versionamento se o nome for um dos padrões base.
    """
    final_name = name
    
    # Se o nome solicitado for um dos bases, aplicamos o versionamento automático
    if name in [BASE_TEXT_COLLECTION, BASE_IMAGE_COLLECTION, BASE_MULTIMODAL_COLLECTION, BASE_CACHE_COLLECTION]:
        mod = "text"
        if "image" in name: mod = "vision"
        elif "multimodal" in name: mod = "multimodal"
        elif "cache" in name: mod = "cache"
        
        final_name = _get_versioned_collection_name(name, mod)
        if final_name != name:
            logger.info(f"[vectorstore] Redirecionando '{name}' -> '{final_name}' (Versionamento)")

    return await asyncio.to_thread(_get_or_create_sync, final_name, metadata)


# ============================================================
# Inicialização Padrão
# ============================================================
def init_vectorstore():
    """Cria coleções versionadas no boot."""
    try:
        # Gera nomes dinâmicos baseados no .env atual
        txt_col = _get_versioned_collection_name(BASE_TEXT_COLLECTION, "text")
        img_col = _get_versioned_collection_name(BASE_IMAGE_COLLECTION, "vision")
        mm_col = _get_versioned_collection_name(BASE_MULTIMODAL_COLLECTION, "multimodal")
        cache_col = _get_versioned_collection_name(BASE_CACHE_COLLECTION, "text")

        for name in (txt_col, img_col, mm_col, cache_col):
            chroma_client.get_or_create_collection(
                name=name,
                metadata={"modality": "auto-versioned", "model_context": name},
            )
        logger.info(f"[vectorstore] Coleções ativas e versionadas: {txt_col}, {img_col}, {mm_col}, {cache_col}")
    except Exception as e:
        logger.error(f"[vectorstore] Falha ao inicializar coleções: {e}")
        raise


# ============================================================
# Inserção
# ============================================================
def _insert_embedding_sync(
    collection_name: str,
    doc_id: str,
    text: Optional[str],
    embedding: List[float],
    metadata: Optional[Dict[str, Any]],
):
    col = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata=_safe_metadata(metadata),
    )

    col.add(
        ids=[str(doc_id)],
        documents=[text or ""],
        embeddings=[_ensure_list_of_floats(embedding)],
        metadatas=[_safe_metadata(metadata)],
    )


async def add_document(
    modality: str,
    doc_id: str,
    text: Optional[str] = None,
    image_b64: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Insere documento multimodal completo na coleção versionada correta E no índice BM25."""
    modality = _normalize_modality(modality)

    # --- 1. Processamento Vetorial (ChromaDB) ---
    if modality == "text" or modality == "cache":
        embedding = await asyncio.to_thread(embed_text, text or "")
    elif modality == "vision":
        embedding = await asyncio.to_thread(embed_image, image_b64 or "")
    else:  # multimodal
        emb = await asyncio.to_thread(embed_multimodal, text or "", image_b64)
        embedding = emb.get("multimodal")

    collection_name = _collection_for_modality(modality)

    await asyncio.to_thread(
        _insert_embedding_sync,
        collection_name,
        doc_id,
        text,
        embedding,
        metadata,
    )

    # --- 2. Processamento Esparso (BM25) ---
    # Apenas para texto ou multimodal que tenha texto
    if text and modality in ("text", "multimodal"):
        # Adiciona ao índice em memória
        sparse_index.add_document(doc_id, text)
        # Commita (em produção, faríamos isso em batch ou periodicamente)
        # Para a tese, commit imediato garante consistência nos testes
        await asyncio.to_thread(sparse_index.commit)

    logger.info(f"[vectorstore] Inserido doc_id={doc_id} em {collection_name} + BM25")
    return True


# ============================================================
# Consulta
# ============================================================
def _query_embedding_sync(collection_name: str, embedding, n_results: int):
    try:
        col = chroma_client.get_or_create_collection(name=collection_name)
        return col.query(
            query_embeddings=[_ensure_list_of_floats(embedding)],
            n_results=n_results,
            include=["documents", "metadatas", "distances", "ids"], # IDs necessários para RRF
        )
    except Exception as e:
        logger.error(f"[vectorstore] Falha na consulta ({collection_name}): {e}")
        return {}


async def query_embedding(modality: str, embedding, n_results: int = 3):
    """Consulta embeddings no Chroma."""
    # Se a modalidade for o nome direto da coleção (ex: knowledge_base), usa direto
    # Caso contrário, resolve o nome versionado
    if modality not in VALID_MODALITIES:
        collection_name = modality
    else:
        collection_name = _collection_for_modality(modality)

    return await asyncio.to_thread(
        _query_embedding_sync,
        collection_name,
        embedding,
        n_results,
    )


# ============================================================
# Health / Reset
# ============================================================
async def reset_collections():
    """Apaga todas as coleções."""
    try:
        await asyncio.to_thread(chroma_client.reset)
        logger.warning("[vectorstore] Todas coleções resetadas.")
    except Exception as e:
        logger.error(f"[vectorstore] Erro ao resetar: {e}")


async def health_async() -> bool:
    try:
        await asyncio.to_thread(chroma_client.heartbeat)
        return True
    except Exception:
        return False