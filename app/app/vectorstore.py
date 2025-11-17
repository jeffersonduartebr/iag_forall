# -*- coding: utf-8 -*-
"""
Vectorstore manager — ChromaDB v2+ (modo local persistente)
-----------------------------------------------------------
Módulo centralizado para todas as interações com o ChromaDB.
Compatível com Chroma >= 0.5.0.

✅ Blindagem global:
   - Qualquer embedding recebido como numpy.ndarray é convertido para list[float]
   - Sanitização de inputs (ids, documents, metadatas)
   - query() sempre recebe list[float] válida
"""

from __future__ import annotations

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union

import chromadb
import numpy as np

# ✅ Importa o módulo de settings centralizado
# (Assumindo que este arquivo está em 'app/vectorstore.py')
from .settings_dynamic import settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] vectorstore: %(message)s")

# ============================================================
# ⚙️ Configurações
# ============================================================
CHROMA_PATH = settings.get("CHROMA_PATH", "/data/chroma")


# ============================================================
# 🔧 Helpers internos
# ============================================================
def _ensure_list_of_floats(vec: Union[np.ndarray, List[float], tuple]) -> List[float]:
    """
    Converte um vetor (np.ndarray|list|tuple) em list[float] achatada.
    """
    if isinstance(vec, np.ndarray):
        return vec.astype(float).ravel().tolist()
    if isinstance(vec, (list, tuple)):
        # Pode haver lista de listas acidentalmente:
        if len(vec) > 0 and isinstance(vec[0], (list, tuple, np.ndarray)):
            # achata um nível
            arr = np.array(vec, dtype=float).ravel()
            return arr.tolist()
        return [float(x) for x in vec]
    # fallback defensivo
    return [float(vec)] if vec is not None else [0.0]


def _safe_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Garante que metadados sejam um dict não-vazio (requisito do Chroma).
    """
    if isinstance(meta, dict) and meta:
        return meta
    return {"source": "router"}


# ============================================================
# 🧠 Inicialização Local Persistente
# ============================================================
def _connect_local():
    """
    (Função Síncrona)
    Inicializa o cliente ChromaDB em modo local persistente.
    """
    try:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        logger.info(f"[vectorstore] Inicializando Chroma local persistente em {CHROMA_PATH}")

        # Novo cliente persistente (Chroma >= 0.5)
        client = chromadb.PersistentClient(path=CHROMA_PATH)

        logger.info("[vectorstore] ✅ Chroma PersistentClient inicializado com sucesso.")
        return client

    except Exception as e:
        logger.error(f"[vectorstore] ❌ Falha ao iniciar Chroma local: {e}")
        raise

# Cliente global (Criado de forma síncrona na inicialização)
# Este é o ÚNICO cliente na aplicação.
chroma_client = _connect_local()


# ============================================================
# 📚 Gerenciamento de Coleções (Async)
# ============================================================
def _get_or_create_collection_sync(name: str, metadata: dict = None):
    """Função síncrona auxiliar."""
    try:
        return chroma_client.get_or_create_collection(
            name=name,
            metadata=_safe_metadata(metadata)
        )
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.warning(f"[vectorstore] Coleção '{name}' já existia, recuperando...")
            return chroma_client.get_collection(name)
        raise

async def get_or_create_collection_async(name: str, metadata: dict = None):
    """(Async) Obtém ou cria uma coleção local persistente."""
    try:
        return await asyncio.to_thread(
            _get_or_create_collection_sync, name, metadata
        )
    except Exception as e:
        logger.error(f"[vectorstore] Falha ao obter/criar coleção '{name}': {e}")
        return None

def _delete_collection_sync(collection_name: str):
    """Função síncrona auxiliar para deletar coleção."""
    chroma_client.delete_collection(name=collection_name)

async def delete_collection(collection_name: str):
    """(Async) Deleta uma coleção do ChromaDB."""
    try:
        await asyncio.to_thread(_delete_collection_sync, collection_name)
        logger.info(f"[vectorstore] Coleção '{collection_name}' deletada.")
        return True
    except Exception as e:
        logger.error(f"[vectorstore] Falha ao deletar '{collection_name}': {e}")
        return False


# ============================================================
# 💾 Inserção e Consulta (Assíncronas via to_thread)
# ============================================================
def _insert_embedding_sync(
    collection_name: str,
    doc_id: str,
    text: str,
    embedding: Union[np.ndarray, List[float], tuple],
    metadata: Optional[Dict[str, Any]] = None
):
    """Função síncrona auxiliar para inserção de embeddings no ChromaDB."""

    collection = chroma_client.get_collection(collection_name)

    emb_list = _ensure_list_of_floats(embedding)
    safe_meta = _safe_metadata(metadata)

    # Sanitização mínima
    safe_id = str(doc_id) if doc_id is not None else str(os.urandom(6).hex())
    safe_text = "" if text is None else str(text)

    collection.add(
        ids=[safe_id],
        documents=[safe_text],
        embeddings=[emb_list],
        metadatas=[safe_meta],
    )

async def insert_embedding(
    collection_name: str,
    doc_id: str,
    text: str,
    embedding: Union[np.ndarray, List[float], tuple],
    metadata: Optional[Dict[str, Any]] = None
):
    """(Async) Insere um documento e seu embedding."""
    try:
        await asyncio.to_thread(
            _insert_embedding_sync,
            collection_name, doc_id, text, embedding, metadata
        )
        logger.debug(f"[vectorstore] Documento '{doc_id}' inserido em '{collection_name}'.")
    except Exception as e:
        logger.error(f"[vectorstore] Falha ao inserir embedding: {e}")

def _query_embedding_sync(
    collection_name: str,
    embedding: Union[np.ndarray, List[float], tuple],
    n_results: int = 3
):
    """Função síncrona auxiliar para ser usada com to_thread."""
    collection = chroma_client.get_collection(collection_name)
    emb_list = _ensure_list_of_floats(embedding)

    # include amplia as infos retornadas (compat com versões variadas)
    return collection.query(
        query_embeddings=[emb_list],
        n_results=n_results,
        include=["documents", "embeddings", "metadatas", "distances"]
    )

async def query_embedding(
    collection_name: str,
    embedding: Union[np.ndarray, List[float], tuple],
    n_results: int = 3
):
    """(Async) Consulta os embeddings mais semelhantes."""
    try:
        results = await asyncio.to_thread(
            _query_embedding_sync,
            collection_name, embedding, n_results
        )
        return results
    except Exception as e:
        logger.error(f"[vectorstore] Erro ao consultar coleção '{collection_name}': {e}")
        return None


# ============================================================
# 🧩 Utilitários (Assíncronos via to_thread)
# ============================================================
async def reset_collections():
    """(Async) Remove todas as coleções (útil para testes)."""
    try:
        await asyncio.to_thread(chroma_client.reset)
        logger.info("[vectorstore] Todas as coleções foram removidas com sucesso.")
    except Exception as e:
        logger.error(f"[vectorstore] Falha ao resetar coleções: {e}")

async def health_async() -> bool:
    """(Async) Verifica se o cliente Chroma está vivo."""
    try:
        await asyncio.to_thread(chroma_client.heartbeat)
        return True
    except Exception:
        return False
