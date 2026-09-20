# -*- coding: utf-8 -*-
# Objective: Application runtime code for rag local.
"""Hybrid local RAG helpers for text, vision, and multimodal retrieval flows."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import (
    aembed_multimodal,
    aembed_text,
)

# Importamos o call_model para gerar a descrição da imagem (Ponte Visual)
from .providers_async import call_model
from .reranker import rerank_documents  # <--- Importar o novo módulo
from .services.retrieval_assembly import (  # noqa: F401  (reexportados)
    Candidate,
    _augmented_prompt,
    _fuse_candidates,
    _match_items,
    _provenance,
    _snippet,
    _trim_context_to_budget,
    _trim_items,
    reciprocal_rank_fusion,
)
from .settings_dynamic import settings
from .sparse_index import sparse_index  # <--- Importar BM25
from .utils.redis_client import get_redis
from .vectorstore import (
    _collection_for_modality,
    add_document,
    health_async,
    query_embedding,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] rag_local: %(message)s"
    )

# Modelo rápido para descrever imagens para busca (Moondream é ótimo aqui, ou Llama 3.2)
# Se não tiver configurado, usa o primeiro da lista de visão
VISION_HELPER_MODEL = "ollama/moondream:latest"

# Visual Query Cache (Quick Win #5)
VISUAL_CACHE_TTL = 86400  # 24 hours
VISUAL_CACHE_PREFIX = "visual_query:"
_rds = get_redis()

# Metrics (safe import)
try:
    from .observability import (
        RETRIEVAL_CONTEXT_TOKENS,
        RETRIEVAL_DOCUMENTS_RETURNED,
        RETRIEVAL_SCORE,
        VISUAL_QUERY_CACHE_HITS,
        VISUAL_QUERY_CACHE_MISSES,
    )
except ImportError:
    # Optional metrics: disable gracefully when observability is unavailable.
    VISUAL_QUERY_CACHE_HITS = None  # type: ignore[assignment]
    VISUAL_QUERY_CACHE_MISSES = None  # type: ignore[assignment]
    RETRIEVAL_DOCUMENTS_RETURNED = None  # type: ignore[assignment]
    RETRIEVAL_CONTEXT_TOKENS = None  # type: ignore[assignment]
    RETRIEVAL_SCORE = None  # type: ignore[assignment]


def _get_int_setting(key: str, default: int) -> int:
    """Read one integer setting with a defensive fallback for malformed values."""
    try:
        value = settings.get(key, default)
        return int(value if value is not None else default)
    except Exception:
        return int(default)


def _is_enabled(key: str, default: str = "1") -> bool:
    """Read one boolean-like flag from settings using the project's string convention."""
    try:
        return str(settings.get(key, default)).strip() == "1"
    except Exception:
        return str(default).strip() == "1"


def _min_docs_for_grounded_context() -> int:
    """Return the minimum number of useful documents required for strong grounding."""
    return max(1, _get_int_setting("RAG_CONTEXT_QUALITY_MIN_DOCS", 2))


def _knowledge_version(modality: str) -> str:
    """Build one stable knowledge version label for the active retrieval collections."""
    normalized = _auto_modality(modality, None)
    collection_name = _collection_for_modality("text" if normalized == "vision" else normalized)
    sparse_version = getattr(sparse_index, "index_version", None) or getattr(sparse_index, "version", None) or "sparse_v1"
    return f"{collection_name}|{sparse_version}"


def _hash_image(image_b64: str) -> str:
    """Generate a hash for image caching."""
    return hashlib.sha256(image_b64[:10000].encode()).hexdigest()[:32]

# ================================================================
# 🔍 DETECÇÃO AUTOMÁTICA DE MODALIDADE
# ================================================================

def _auto_modality(requested: Optional[str], image_b64: Optional[str]) -> str:
    """Execute the auto modality routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    req = (requested or "text").lower().strip()

    if image_b64 and req == "multimodal":
        return "multimodal"
    if image_b64 and req in ("vision", "image"):
        return "vision"
    if image_b64 and req == "text":
        # Se tem imagem mas pediu texto, tratamos como visão para o RAG aproveitar a imagem
        return "vision"

    return "text"


# ================================================================
# 👁️ GERAÇÃO DE QUERY VISUAL (A PONTE)
# ================================================================

async def _generate_visual_search_query(image_b64: str) -> str:
    """
    Usa um VLM para descrever a imagem e criar uma string de busca textual.
    Isso permite usar a imagem para encontrar documentos de texto no Chroma.

    Quick Win #5: Added Redis cache for visual descriptions.
    """
    # Check cache first
    image_hash = _hash_image(image_b64)
    cache_key = f"{VISUAL_CACHE_PREFIX}{image_hash}"

    if _rds:
        try:
            cached = _rds.get(cache_key)
            if cached:
                result = cached.decode() if isinstance(cached, bytes) else str(cached)
                logger.debug(f"[RAG-Vision] Cache HIT for image hash {image_hash[:8]}")
                if VISUAL_QUERY_CACHE_HITS:
                    VISUAL_QUERY_CACHE_HITS.inc()
                return result
        except Exception as e:
            logger.debug(f"[RAG-Vision] Cache read error: {e}")

    if VISUAL_QUERY_CACHE_MISSES:
        VISUAL_QUERY_CACHE_MISSES.inc()

    try:
        # Tenta pegar um modelo da lista de candidatos se o helper não estiver fixo
        candidates = settings.CANDIDATE_VISION_MODELS_LIST
        model_to_use = list(candidates)[0] if candidates else "ollama/llava:7b"

        # Prompt focado em extração de keywords para busca
        prompt = (
            "Identifique o objeto principal, cenário ou problema nesta imagem. "
            "Gere uma única frase descritiva e técnica para ser usada como termo de busca em um banco de dados. "
            "Não use preâmbulos como 'A imagem mostra...'. Seja direto."
        )

        response, _ = await call_model(
            model=model_to_use,
            prompt=prompt,
            image_b64=image_b64,
            max_tokens=64,
            temperature=0.1
        )

        search_query = response.strip()
        logger.info(f"[RAG-Vision] Query gerada da imagem: '{search_query}'")

        # Cache the result
        if _rds and search_query:
            try:
                _rds.setex(cache_key, VISUAL_CACHE_TTL, search_query)
            except Exception as e:
                logger.debug(f"[RAG-Vision] Cache write error: {e}")

        return search_query

    except Exception as e:
        logger.warning(f"[RAG-Vision] Falha ao descrever imagem: {e}")
        return ""


# ================================================================
# 🧠 GERAÇÃO DE EMBEDDING ADEQUADO
# ================================================================

async def _compute_embedding(query: str, modality: str, image_b64: Optional[str]):
    """
    Gera o embedding adequado.
    """
    try:
        # RAG Clássico (Texto -> Texto)
        if modality == "text":
            return await aembed_text(query)

        # RAG Visual (Imagem -> Texto via Ponte Descritiva)
        if modality == "vision" and image_b64:
            # Se o usuário não mandou texto (só a imagem), geramos a descrição
            if not query or len(query) < 5:
                visual_query = await _generate_visual_search_query(image_b64)
                query_to_embed = visual_query if visual_query else "imagem genérica"
            else:
                # Se o usuário mandou texto junto (ex: "Como conserto isso?"),
                # usamos o texto dele + uma descrição breve da imagem
                visual_desc = await _generate_visual_search_query(image_b64)
                query_to_embed = f"{query} {visual_desc}"

            # Embedamos o TEXTO resultante para buscar no banco de TEXTO
            return await aembed_text(query_to_embed)

        # Multimodal (Conceito avançado de espaço latente compartilhado)
        # Só funciona se o vectorstore suportar embeddings multimodais nativos
        if modality == "multimodal":
            emb_dict = await aembed_multimodal(query, image_b64)
            return emb_dict.get("multimodal") or emb_dict.get("text")

        return await aembed_text(query)

    except Exception as e:
        logger.warning(f"[rag_local] Falha ao gerar embedding ({modality}): {e}")
        return None

# ================================================================
# 🔄 RECIPROCAL RANK FUSION (RRF)
# ================================================================

# ================================================================
# 📚 RAG PRINCIPAL – Construção do Prompt Aumentado
# ================================================================

async def build_augmented_prompt(
    query: str,
    modality: str = "text",
    image_b64: Optional[str] = None,
    k: int = 3,
    retrieval_mode: Optional[str] = None,
    context_token_budget: Optional[int] = None,
    rerank_enabled: Optional[bool] = None,
) -> str:
    """Execute the build augmented prompt routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    bundle = await build_retrieval_bundle(
        query=query,
        modality=modality,
        image_b64=image_b64,
        k=k,
        retrieval_mode=retrieval_mode,
        context_token_budget=context_token_budget,
        rerank_enabled=rerank_enabled,
    )
    return str(bundle.get("augmented_prompt") or bundle.get("query") or "")


SearchHits = Tuple[List[str], Dict[str, str], Dict[str, Dict[str, Any]]]


async def _dense_search(
    query: str, rag_mode: str, image_b64: Optional[str], collection_modality: str, n_results: int
) -> SearchHits:
    """Vector search: ``(ids em ordem, id -> texto, id -> metadados)``; vazio em falha."""
    ids: List[str] = []
    docs_map: Dict[str, str] = {}
    meta_map: Dict[str, Dict[str, Any]] = {}
    emb = await _compute_embedding(query, rag_mode, image_b64)
    if emb is None:
        return ids, docs_map, meta_map
    try:
        res = await query_embedding(modality=collection_modality, embedding=emb, n_results=n_results)
        if res and res.get("ids"):
            ids = res["ids"][0]
            docs = res["documents"][0]
            metadatas = (res.get("metadatas") or [[]])[0]
            for i, doc_id in enumerate(ids):
                docs_map[doc_id] = docs[i]
                meta_map[doc_id] = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {}
            _observe_top_similarity(res, rag_mode)
    except Exception as e:
        logger.warning(f"[rag_local] Erro Vector Search: {e}")
    return ids, docs_map, meta_map


def _observe_top_similarity(res: Dict[str, Any], rag_mode: str) -> None:
    try:
        distances = (res.get("distances") or [[]])[0]
        if distances and RETRIEVAL_SCORE:
            RETRIEVAL_SCORE.labels(modality=rag_mode).observe(max(0.0, min(1.0, 1.0 - float(distances[0]))))
    except Exception:
        pass


def _bm25_hits(query: str, top_k: int) -> SearchHits:
    """BM25 search plus text lookup, run entirely in a worker thread."""
    ids: List[str] = []
    docs_map: Dict[str, str] = {}
    meta_map: Dict[str, Dict[str, Any]] = {}
    for doc_id, _score in sparse_index.search(query, top_k=top_k):
        ids.append(doc_id)
        text = sparse_index.get_text(doc_id)
        if text:
            docs_map[doc_id] = text
            meta_map[doc_id] = {"source": "bm25"}
    return ids, docs_map, meta_map


async def _sparse_search(query: str, top_k: int) -> SearchHits:
    """Sparse (BM25) search; only meaningful with textual query. Empty on failure."""
    if not query:
        return [], {}, {}
    try:
        return await asyncio.to_thread(_bm25_hits, query, top_k)
    except Exception as e:
        logger.warning(f"[rag_local] Erro BM25 Search: {e}")
        return [], {}, {}


def _bundle(
    query: str,
    rag_mode: str,
    retrieval_mode: str,
    *,
    augmented_prompt: str,
    context: str = "",
    citations: Optional[List[Dict[str, Any]]] = None,
    evidence_snippets: Optional[List[Dict[str, Any]]] = None,
    skipped_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """The retrieval bundle contract (grounded iff there are citations)."""
    return {
        "query": query,
        "augmented_prompt": augmented_prompt,
        "context": context,
        "citations": citations or [],
        "evidence_snippets": evidence_snippets or [],
        "grounded": bool(citations),
        "knowledge_version": _knowledge_version(rag_mode),
        "retrieval_mode": retrieval_mode,
        "retrieval_skipped_reason": skipped_reason,
    }


async def _rerank(
    query: str, items: List[Candidate], k: int, retrieval_mode: str, rerank_enabled: Optional[bool]
) -> List[str]:
    """Cross-encoder rerank of the top candidates when enabled and there are enough of them."""
    if rerank_enabled is None:
        rerank_enabled = _is_enabled("RERANK_ENABLED", "1")
    texts = [text for _, text, _ in items]
    pool = texts[: max(k, 5 if retrieval_mode != "light_retrieval" else max(3, k))]
    if rerank_enabled and len(pool) >= max(1, _get_int_setting("RAG_RERANK_MIN_CANDIDATES", 3)):
        return await asyncio.to_thread(rerank_documents, query, pool, k)
    return texts[:k]


def _context_budget(context_token_budget: Optional[int], retrieval_mode: str) -> int:
    if context_token_budget is not None:
        return int(context_token_budget)
    key = "RAG_LIGHT_CONTEXT_TOKEN_BUDGET" if retrieval_mode == "light_retrieval" else "RAG_FULL_CONTEXT_TOKEN_BUDGET"
    return _get_int_setting(key, _get_int_setting("RAG_CONTEXT_TOKEN_BUDGET", 1200))


def _observe_retrieval(rag_mode: str, n_docs: int, context: Optional[str]) -> None:
    """Documents returned and (when a context was assembled) its size in ~tokens."""
    try:
        if RETRIEVAL_DOCUMENTS_RETURNED:
            RETRIEVAL_DOCUMENTS_RETURNED.labels(modality=rag_mode).observe(n_docs)
        if RETRIEVAL_CONTEXT_TOKENS and context is not None:
            RETRIEVAL_CONTEXT_TOKENS.labels(modality=rag_mode).observe(max(0, len(context) // 4))
    except Exception:
        pass


async def build_retrieval_bundle(
    query: str,
    modality: str = "text",
    image_b64: Optional[str] = None,
    k: int = 3,
    retrieval_mode: Optional[str] = None,
    context_token_budget: Optional[int] = None,
    rerank_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return retrieved context plus structured provenance for one query.

    Hybrid retrieval: dense + BM25 in parallel, RRF fusion, optional
    cross-encoder rerank, trim to the token budget, citations/evidence.
    """
    query = (query or "").strip()
    retrieval_mode = (retrieval_mode or "full_retrieval").strip().lower()
    if not query and not image_b64:
        return _bundle(query, modality or "text", retrieval_mode, augmented_prompt="", skipped_reason="empty_query")

    rag_mode = _auto_modality(modality, image_b64)
    collection_modality = "text" if rag_mode == "vision" else rag_mode
    light = retrieval_mode == "light_retrieval"
    dense_k = _get_int_setting("RAG_LIGHT_VECTOR_TOP_K", 6) if light else 20
    sparse_k = _get_int_setting("RAG_LIGHT_SPARSE_TOP_K", 6) if light else 20

    dense, sparse = await asyncio.gather(
        _dense_search(query, rag_mode, image_b64, collection_modality, dense_k),
        _sparse_search(query, sparse_k),
    )
    items = _fuse_candidates(dense, sparse)
    if not items:
        _observe_retrieval(rag_mode, 0, None)
        return _bundle(query, rag_mode, retrieval_mode, augmented_prompt=query, skipped_reason="no_candidates")

    reranked = await _rerank(query, items, k, retrieval_mode, rerank_enabled)
    matched = _match_items(reranked, items)
    budget = _context_budget(context_token_budget, retrieval_mode)
    final_docs = _trim_context_to_budget(reranked, budget)
    provenance_items = _trim_items(matched, budget)
    context = "\n\n".join(final_docs)
    skipped_reason = None
    useful_docs = len([doc for doc in final_docs if str(doc or "").strip()])
    if light and useful_docs < _min_docs_for_grounded_context():
        skipped_reason, context, final_docs, provenance_items = "insufficient_context_quality", "", [], []
    _observe_retrieval(rag_mode, len(final_docs), context)
    logger.info(
        f"[rag_local] Hybrid RAG: {len(final_docs)} docs finais (Vector={len(dense[0])}, BM25={len(sparse[0])})."
    )
    citations, evidence = _provenance(provenance_items or matched[: len(final_docs)])
    return _bundle(
        query,
        rag_mode,
        retrieval_mode,
        augmented_prompt=_augmented_prompt(context, query),
        context=context,
        citations=citations,
        evidence_snippets=evidence,
        skipped_reason=skipped_reason,
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
    """Store one document in the local RAG index.

    Returns whether it was actually stored. The result of ``add_document`` used
    to be discarded and ``True`` returned regardless, so a failed insertion was
    indistinguishable from a successful one.
    """
    try:
        return await add_document(
            modality=modality,
            doc_id=doc_id,
            text=text,
            image_b64=image_b64,
            metadata=metadata,
        )
    except Exception as e:
        logger.error(f"[rag_local] Falha ao adicionar documento {doc_id}: {e}")
        return False


# ================================================================
# 🩺 HEALTHCHECK
# ================================================================

async def health() -> Dict[str, Any]:
    """Execute the health routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    try:
        chroma_ok = await health_async()
    except Exception:
        chroma_ok = False

    return {
        "vectorstore": chroma_ok,
        "status": "ok" if chroma_ok else "fail"
    }
