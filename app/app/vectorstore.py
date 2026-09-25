# -*- coding: utf-8 -*-
# Objective: Application runtime code for vectorstore.
"""Manage ChromaDB storage for retrieval and semantic-cache workloads.

This module centralizes vector-store responsibilities for the application:

- mapping modalities to the correct collection names
- versioning collections by embedding model to avoid dimension conflicts
- inserting and querying embeddings through ChromaDB
- auto-healing collections when incompatible dimensions are detected
- coordinating hybrid retrieval with the sparse index

Both the RAG pipeline and the semantic cache rely on this module to provide a
stable persistence and query surface.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, cast

import chromadb
import numpy as np

from .embeddings import embed_image, embed_multimodal, embed_text
from .settings_dynamic import settings
from .sparse_index import sparse_index  # Integração com BM25

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
CHROMA_HOST = (settings.get("CHROMA_HOST", "") or "").strip()
CHROMA_PORT = int(settings.get("CHROMA_PORT", 8000) or 8000)

# Nomes base das coleções
BASE_TEXT_COLLECTION = "text_embeddings"
BASE_IMAGE_COLLECTION = "image_embeddings"
BASE_MULTIMODAL_COLLECTION = "multimodal_embeddings"
BASE_CACHE_COLLECTION = "semantic_cache_v2"

VALID_MODALITIES = {"text", "vision", "multimodal", "image", "cache"}


def _sanitize_model_name(model_name: str) -> str:
    """Normalize a model name so it can be embedded safely into a collection ID."""
    if not model_name:
        return "default"
    # Remove caracteres especiais e substitui por underscore
    clean = re.sub(r"[^a-zA-Z0-9]", "_", model_name)
    # Remove underscores duplicados e underscores nas pontas
    return re.sub(r"_+", "_", clean).strip("_")


TEXT_SCHEMA = "d2"
COSINE = {"hnsw:space": "cosine"}


def _get_versioned_collection_name(base_name: str, modality: str) -> str:
    """Return the collection name bound to the embedding model active for a modality."""
    if modality == "text" or modality == "cache":
        model = settings.EMBED_TEXT_MODEL or "default_text"
    elif modality == "vision":
        model = settings.IMAGE_EMBEDDING_MODEL or "clip"
    else:
        model = settings.MULTIMODAL_EMBEDDING_MODEL or "default_mm"

    version_suffix = _sanitize_model_name(model)
    if base_name == BASE_TEXT_COLLECTION:
        # d2: documentos com prefixo search_document e espaço cosseno. Os vetores antigos (prefixo de consulta,
        # L2) não são comparáveis; a coleção nova começa vazia e o acervo é reingerido.
        return f"{base_name}_{version_suffix}_{TEXT_SCHEMA}"
    return f"{base_name}_{version_suffix}"


# ============================================================
# Conexão com ChromaDB
# ============================================================
def chroma_client_settings():
    """Settings for every Chroma client: telemetry off in all processes.

    ``main.py`` disables telemetry through an env var, but only in the API; Celery
    workers kept it on, and chromadb 0.5.x breaks with current posthog
    ("capture() takes 1 positional argument but 3 were given"). Clients on the same
    path must share identical settings, so health checks use this helper too.
    """
    from chromadb.config import Settings

    return Settings(anonymized_telemetry=False)


def _connect_local():
    """Create the persistent ChromaDB client used by local runtime paths."""
    try:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        logger.info(f"[vectorstore] Inicializando ChromaDB em {CHROMA_PATH}")
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=chroma_client_settings())
        logger.info("[vectorstore] Chroma PersistentClient inicializado.")
        return client
    except Exception as e:
        logger.error(f"[vectorstore] Falha ao iniciar ChromaDB: {e}")
        raise


def _connect_remote():
    """Create a remote ChromaDB HttpClient for shared multi-replica deployments."""
    try:
        import chromadb

        host = CHROMA_HOST
        port = CHROMA_PORT
        logger.info("[vectorstore] Conectando ChromaDB remoto em %s:%s", host, port)
        client = chromadb.HttpClient(host=host, port=port, settings=chroma_client_settings())
        client.heartbeat()
        logger.info("[vectorstore] Chroma HttpClient inicializado.")
        return client
    except Exception as e:
        logger.error(f"[vectorstore] Falha ao conectar ChromaDB remoto: {e}")
        raise


def _connect_client():
    """Select local persistent or remote Chroma client based on configuration."""
    if CHROMA_HOST:
        return _connect_remote()
    return _connect_local()


chroma_client = None
_chroma_lock = threading.Lock()


def get_chroma_client():
    """
    Lazy initializer for Chroma client.
    Avoids heavy side-effects during module import and test collection.
    """
    global chroma_client
    if chroma_client is not None:
        return chroma_client
    with _chroma_lock:
        if chroma_client is None:
            chroma_client = _connect_client()
    return chroma_client


# ============================================================
# Helpers Internos
# ============================================================
def _ensure_list_of_floats(vec):
    """Normalize an embedding value into the float list shape required by Chroma."""
    if isinstance(vec, np.ndarray):
        return vec.astype(float).ravel().tolist()
    if isinstance(vec, (list, tuple)):
        return [float(x) for x in np.array(vec, dtype=float).ravel()]
    return [0.0]


def _safe_metadata(meta):
    """Return metadata in dictionary form, falling back to a minimal default."""
    return meta if isinstance(meta, dict) else {"source": "router"}


def _normalize_modality(modality: Optional[str]) -> str:
    """Normalize a caller-supplied modality into one supported by the vector store."""
    if not modality:
        return "text"
    m = modality.lower().strip()
    if m == "image":
        return "vision"
    return m if m in VALID_MODALITIES else "text"


def _collection_for_modality(modality: str) -> str:
    """Resolve the versioned collection name associated with one modality."""
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
    """Synchronously fetch or create one Chroma collection."""
    return get_chroma_client().get_or_create_collection(name=name, metadata=metadata)


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
    """Create the active versioned collections during application bootstrap."""
    try:
        # Gera nomes dinâmicos baseados no .env atual
        txt_col = _get_versioned_collection_name(BASE_TEXT_COLLECTION, "text")
        img_col = _get_versioned_collection_name(BASE_IMAGE_COLLECTION, "vision")
        mm_col = _get_versioned_collection_name(BASE_MULTIMODAL_COLLECTION, "multimodal")
        cache_col = _get_versioned_collection_name(BASE_CACHE_COLLECTION, "text")

        for name in (txt_col, img_col, mm_col, cache_col):
            meta = {"modality": "auto-versioned", "model_context": name}
            if name == txt_col:
                meta.update(COSINE)  # só a coleção nova: o espaço de uma coleção existente não muda
            get_chroma_client().get_or_create_collection(name=name, metadata=meta)
        logger.info(f"[vectorstore] Coleções ativas e versionadas: {txt_col}, {img_col}, {mm_col}, {cache_col}")
    except Exception as e:
        logger.error(f"[vectorstore] Falha ao inicializar coleções: {e}")
        raise


# ============================================================
# Inserção (Com Auto-Healing)
# ============================================================
def _vetor_utilizavel(embedding) -> bool:
    """A real embedding: more than one dimension and not all zeros (the failure fallbacks are neither)."""
    vec = _ensure_list_of_floats(embedding)
    return len(vec) > 1 and any(vec)


def _insert_embedding_sync(
    collection_name: str,
    doc_id: str,
    text: Optional[str],
    embedding: List[float],
    metadata: Optional[Dict[str, Any]],
) -> bool:
    """Upsert one embedding into Chroma; returns whether the document is actually stored.

    Never deletes a collection. The old "auto-healing" dropped and recreated the whole collection on a
    dimension mismatch — and the usual cause of a mismatch is a failed embedding (``[0.0]``, a zero vector,
    or the 1536-d OpenAI fallback), so one failed ingest could wipe the corpus. Degenerate vectors are now
    refused before they reach Chroma, and a mismatch is reported, not "healed".

    ``upsert`` instead of ``add``: ``add`` silently ignored an id that already existed, so re-ingesting an
    edited material kept the old text while BM25 got the new one.
    """
    if not _vetor_utilizavel(embedding):
        logger.error(f"[vectorstore] doc_id={doc_id}: embedding inválido (falha do modelo); não inserido.")
        return False
    try:
        metadata_colecao = COSINE if collection_name.endswith(f"_{TEXT_SCHEMA}") else None
        col = get_chroma_client().get_or_create_collection(name=collection_name, metadata=metadata_colecao)
        col.upsert(
            ids=[str(doc_id)],
            documents=[text or ""],
            embeddings=[_ensure_list_of_floats(embedding)],
            metadatas=[_safe_metadata(metadata)],
        )
        return True
    except Exception as e:
        msg = str(e).lower()
        if "dimension" in msg and "match" in msg:
            logger.error(
                f"[vectorstore] Dimensão incompatível em '{collection_name}' (modelo de embeddings trocado ou "
                f"falhou): doc_id={doc_id} não inserido. A coleção NÃO foi apagada. {e}"
            )
            return False
        logger.error(f"[vectorstore] Erro na inserção: {e}")
        return False


async def add_document(
    modality: str,
    doc_id: str,
    text: Optional[str] = None,
    image_b64: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    colecao: Optional[str] = None,
) -> bool:
    """Embed and persist one document into the collection that matches the modality.

    ``colecao`` overrides the target collection (e.g. the RAG healthcheck's own); such documents never
    enter the BM25 index of the corpus.

    The helper resolves the correct embedding path, computes the target
    collection name, stores the dense vector in Chroma, and mirrors the text
    payload into the sparse index when the modality participates in hybrid
    retrieval.

    Returns whether the dense vector was actually stored. The sparse (BM25)
    mirror is deliberately skipped when it was not: an entry that is findable
    by keyword but has no vector behind it is worse than no entry at all,
    because it looks like a successful ingest for as long as nobody searches
    semantically.
    """
    modality = _normalize_modality(modality)

    # --- 1. Processamento Vetorial (ChromaDB) ---
    if modality == "text":
        embedding = await asyncio.to_thread(embed_text, text or "", "documento")
    elif modality == "cache":  # o cache semântico guarda consultas: compara consulta com consulta
        embedding = await asyncio.to_thread(embed_text, text or "")
    elif modality == "vision":
        embedding = await asyncio.to_thread(embed_image, image_b64 or "")
    else:  # multimodal
        emb = await asyncio.to_thread(embed_multimodal, text or "", image_b64)
        embedding = cast(List[float], emb.get("multimodal"))

    collection_name = colecao or _collection_for_modality(modality)

    stored = await asyncio.to_thread(
        _insert_embedding_sync,
        collection_name,
        doc_id,
        text,
        embedding,
        metadata,
    )
    if not stored:
        logger.error(f"[vectorstore] doc_id={doc_id} NÃO foi inserido em {collection_name}")
        return False

    # --- 2. Processamento Esparso (BM25) ---
    # Apenas para texto ou multimodal que tenha texto
    if text and modality in ("text", "multimodal") and not colecao:
        # Adiciona ao índice em memória
        sparse_index.add_document(doc_id, text)
        # Commita (em produção, faríamos isso em batch ou periodicamente)
        await asyncio.to_thread(sparse_index.commit)

    logger.info(f"[vectorstore] Inserido doc_id={doc_id} em {collection_name} + BM25")
    return True


# ============================================================
# Consulta (Com Auto-Healing)
# ============================================================
def _query_embedding_sync(collection_name: str, embedding, n_results: int, where: Optional[Dict] = None):
    """Run one synchronous Chroma similarity query with auto-healing behavior."""
    try:
        col = get_chroma_client().get_or_create_collection(name=collection_name)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [_ensure_list_of_floats(embedding)],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return col.query(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        # Uma consulta NUNCA destrói dados. Isto apagava a coleção inteira
        # quando a dimensão não batia — e a causa habitual da dimensão não
        # bater é o modelo de embeddings local ter falhado a carregar e
        # `embed_text` ter devolvido um vetor de zeros do tamanho errado. Ou
        # seja: uma pergunta do utilizador apagava o corpus RAG todo, e o único
        # aviso aparecia depois do estrago. O auto-healing na escrita é
        # defensável, porque aí já se está a mexer na coleção; na leitura não é.
        if "dimension" in msg and "match" in msg:
            logger.error(
                f"[vectorstore] Dimensão incompatível na consulta '{collection_name}': "
                f"a coleção foi escrita com outro modelo de embeddings, ou o modelo actual "
                f"falhou a carregar. A coleção NÃO foi apagada. {e}"
            )
            return {}

        logger.error(f"[vectorstore] Falha na consulta ({collection_name}): {e}")
        return {}


async def query_embedding(modality: str, embedding, n_results: int = 3, where: Optional[Dict] = None):
    """Consulta embeddings no Chroma."""
    if modality not in VALID_MODALITIES:
        collection_name = modality
    else:
        collection_name = _collection_for_modality(modality)

    return await asyncio.to_thread(
        _query_embedding_sync,
        collection_name,
        embedding,
        n_results,
        where,
    )


# ============================================================
# Health / Reset
# ============================================================
async def reset_collections():
    """Apaga todas as coleções."""
    try:
        await asyncio.to_thread(get_chroma_client().reset)
        logger.warning("[vectorstore] Todas coleções resetadas.")
    except Exception as e:
        logger.error(f"[vectorstore] Erro ao resetar: {e}")


async def health_async() -> bool:
    """Executa health async."""
    try:
        await asyncio.to_thread(get_chroma_client().heartbeat)
        return True
    except Exception:
        return False


def reset_vectorstore_runtime_state() -> None:
    """Reset lazy client state (primarily for tests/dev)."""
    global chroma_client
    chroma_client = None
