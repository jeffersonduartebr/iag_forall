# -*- coding: utf-8 -*-
"""
vectorstore.py — RAG Multimodal (Texto / Visão / Multimodal)
----------------------------------------------------------------------
Gerencia a persistência de embeddings no ChromaDB.

Funções exportadas:
    - init_vectorstore()
    - get_or_create_collection_async()  <-- ADICIONADO
    - add_document()
    - query_embedding()
    - reset_collections()
    - health_async()
"""

from __future__ import annotations

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union

import chromadb
import numpy as np

from .embeddings import embed_text, embed_image, embed_multimodal
from .settings_dynamic import settings


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
# Configurações
# ============================================================
CHROMA_PATH = settings.get("CHROMA_PATH", "/data/chroma")

TEXT_COLLECTION = "text_embeddings"
IMAGE_COLLECTION = "image_embeddings"
MULTIMODAL_COLLECTION = "multimodal_embeddings"

VALID_MODALITIES = {"text", "vision", "multimodal", "image"}


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
# Helpers
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
    m = _normalize_modality(modality)
    if m == "text":
        return TEXT_COLLECTION
    if m == "vision":
        return IMAGE_COLLECTION
    return MULTIMODAL_COLLECTION


# ============================================================
# Gerenciamento de Coleções (Async Wrapper)
# ============================================================

def _get_or_create_sync(name: str, metadata: Optional[Dict] = None):
    """Wrapper síncrono para o client do Chroma."""
    return chroma_client.get_or_create_collection(name=name, metadata=metadata)

async def get_or_create_collection_async(name: str, metadata: Optional[Dict] = None):
    """
    Cria ou recupera uma coleção de forma assíncrona (threadpool).
    Usado pelo populate_vectorstore.py.
    """
    return await asyncio.to_thread(_get_or_create_sync, name, metadata)


# ============================================================
# Inicialização Padrão
# ============================================================
def init_vectorstore():
    """Cria coleções básicas se não existirem (Síncrono, usado no boot)."""
    try:
        for name in (TEXT_COLLECTION, IMAGE_COLLECTION, MULTIMODAL_COLLECTION):
            chroma_client.get_or_create_collection(
                name=name,
                metadata={"modality": name},
            )
        logger.info("[vectorstore] Coleções base verificadas.")
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
    """Insere documento multimodal completo."""
    modality = _normalize_modality(modality)

    # --- gerar embedding ---
    if modality == "text":
        embedding = await asyncio.to_thread(embed_text, text or "")
    elif modality == "vision":
        embedding = await asyncio.to_thread(embed_image, image_b64 or "")
    else:  # multimodal
        emb = await asyncio.to_thread(embed_multimodal, text or "", image_b64)
        embedding = emb.get("multimodal")

    collection_name = _collection_for_modality(modality)

    # --- inserir ---
    await asyncio.to_thread(
        _insert_embedding_sync,
        collection_name,
        doc_id,
        text,
        embedding,
        metadata,
    )

    logger.info(f"[vectorstore] Inserido doc_id={doc_id} modality={modality}")
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
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"[vectorstore] Falha na consulta ({collection_name}): {e}")
        return {}


async def query_embedding(modality: str, embedding, n_results: int = 3):
    """Consulta embeddings no Chroma."""
    # Se a modalidade for o nome direto da coleção (ex: knowledge_base), usa direto
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