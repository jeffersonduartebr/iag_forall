# -*- coding: utf-8 -*-
# Objective: Application runtime code for rag local.
"""Hybrid local RAG helpers for text, vision, and multimodal retrieval flows."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import (
    embed_multimodal,
    embed_text,
)

# Importamos o call_model para gerar a descrição da imagem (Ponte Visual)
from .providers_async import call_model
from .reranker import rerank_documents  # <--- Importar o novo módulo
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


def _trim_context_to_budget(documents: List[str], token_budget: int) -> List[str]:
    """Trim retrieved documents so the assembled context stays within a soft token budget."""
    if token_budget <= 0:
        return list(documents)

    budget_chars = token_budget * 4
    total_chars = 0
    trimmed: List[str] = []
    for document in documents:
        if not document:
            continue
        remaining = budget_chars - total_chars
        if remaining <= 0:
            break
        if len(document) <= remaining:
            trimmed.append(document)
            total_chars += len(document)
            continue
        shortened = document[:remaining].rstrip()
        if shortened:
            trimmed.append(shortened)
        break
    return trimmed


def _min_docs_for_grounded_context() -> int:
    """Return the minimum number of useful documents required for strong grounding."""
    return max(1, _get_int_setting("RAG_CONTEXT_QUALITY_MIN_DOCS", 2))


def _snippet(text: str, limit: int = 220) -> str:
    """Return one compact evidence snippet extracted from a retrieved document."""
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


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
            return await asyncio.to_thread(embed_text, query)

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
            return await asyncio.to_thread(embed_text, query_to_embed)

        # Multimodal (Conceito avançado de espaço latente compartilhado)
        # Só funciona se o vectorstore suportar embeddings multimodais nativos
        if modality == "multimodal":
            emb_dict = await asyncio.to_thread(embed_multimodal, query, image_b64)
            return emb_dict.get("multimodal") or emb_dict.get("text")

        return await asyncio.to_thread(embed_text, query)

    except Exception as e:
        logger.warning(f"[rag_local] Falha ao gerar embedding ({modality}): {e}")
        return None

# ================================================================
# 🔄 RECIPROCAL RANK FUSION (RRF)
# ================================================================

def reciprocal_rank_fusion(
    vector_results: List[str], # Lista de Doc IDs
    bm25_results: List[str],   # Lista de Doc IDs
    k: int = 60
) -> List[str]:
    """
    Combina duas listas de resultados usando RRF.
    Score = 1 / (k + rank).
    """
    scores: Dict[str, float] = {}

    # Processa Vetorial
    for rank, doc_id in enumerate(vector_results):
        if doc_id not in scores: scores[doc_id] = 0.0
        scores[doc_id] += 1 / (k + rank + 1)

    # Processa BM25
    for rank, doc_id in enumerate(bm25_results):
        if doc_id not in scores: scores[doc_id] = 0.0
        scores[doc_id] += 1 / (k + rank + 1)

    # Ordena pelo score final
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs]


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


async def build_retrieval_bundle(
    query: str,
    modality: str = "text",
    image_b64: Optional[str] = None,
    k: int = 3,
    retrieval_mode: Optional[str] = None,
    context_token_budget: Optional[int] = None,
    rerank_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return retrieved context plus structured provenance for one query."""
    retrieval_skipped_reason = None
    query = (query or "").strip()
    if not query and not image_b64:
        return {
            "query": query,
            "augmented_prompt": "",
            "context": "",
            "citations": [],
            "evidence_snippets": [],
            "grounded": False,
            "knowledge_version": _knowledge_version(modality or "text"),
            "retrieval_mode": (retrieval_mode or "full_retrieval").strip().lower(),
            "retrieval_skipped_reason": "empty_query",
        }

    rag_mode = _auto_modality(modality, image_b64)
    target_collection_modality = "text" if rag_mode == "vision" else rag_mode
    retrieval_mode = (retrieval_mode or "full_retrieval").strip().lower()
    dense_k = _get_int_setting("RAG_LIGHT_VECTOR_TOP_K", 6) if retrieval_mode == "light_retrieval" else 20
    sparse_k = _get_int_setting("RAG_LIGHT_SPARSE_TOP_K", 6) if retrieval_mode == "light_retrieval" else 20

    # --- 1. Busca Vetorial (Dense Retrieval) ---
    emb = await _compute_embedding(query, rag_mode, image_b64)
    vector_doc_ids = []
    vector_docs_map = {} # ID -> Texto
    vector_meta_map: Dict[str, Dict[str, Any]] = {}

    if emb is not None:
        try:
            # Busca mais candidatos (Top-20) para fusão
            res = await query_embedding(
                modality=target_collection_modality,
                embedding=emb,
                n_results=dense_k
            )
            if res and res.get("ids"):
                ids = res["ids"][0]
                docs = res["documents"][0]
                metadatas = (res.get("metadatas") or [[]])[0]
                vector_doc_ids = ids
                for i, doc_id in enumerate(ids):
                    vector_docs_map[doc_id] = docs[i]
                    vector_meta_map[doc_id] = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {}
                try:
                    distances = (res.get("distances") or [[]])[0]
                    if distances:
                        top_similarity = max(0.0, min(1.0, 1.0 - float(distances[0])))
                        if RETRIEVAL_SCORE:
                            RETRIEVAL_SCORE.labels(modality=rag_mode).observe(top_similarity)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[rag_local] Erro Vector Search: {e}")

    # --- 2. Busca por Palavras-Chave (Sparse Retrieval - BM25) ---
    bm25_doc_ids = []
    bm25_docs_map = {}
    bm25_meta_map: Dict[str, Dict[str, Any]] = {}

    # Só faz sentido BM25 se houver query textual
    if query:
        try:
            # Busca Top-20 no BM25
            bm25_res = await asyncio.to_thread(sparse_index.search, query, top_k=sparse_k)
            bm25_doc_ids = [item[0] for item in bm25_res]
            for doc_id, _ in bm25_res:
                # Recupera o texto do índice esparso
                text = sparse_index.get_text(doc_id)
                if text:
                    bm25_docs_map[doc_id] = text
                    bm25_meta_map[doc_id] = {"source": "bm25"}
        except Exception as e:
            logger.warning(f"[rag_local] Erro BM25 Search: {e}")

    # --- 3. Fusão Híbrida (RRF) ---
    merged_ids = reciprocal_rank_fusion(vector_doc_ids, bm25_doc_ids)

    # Recupera os textos dos IDs vencedores
    candidate_items: List[Tuple[str, str, Dict[str, Any]]] = []
    for doc_id in merged_ids:
        # Tenta pegar do mapa vetorial ou do mapa BM25
        txt = vector_docs_map.get(doc_id) or bm25_docs_map.get(doc_id)
        if txt:
            candidate_items.append((doc_id, txt, vector_meta_map.get(doc_id) or bm25_meta_map.get(doc_id) or {}))

    # Se não achou nada, retorna query original
    if not candidate_items:
        try:
            if RETRIEVAL_DOCUMENTS_RETURNED:
                RETRIEVAL_DOCUMENTS_RETURNED.labels(modality=rag_mode).observe(0)
        except Exception:
            pass
        return {
            "query": query,
            "augmented_prompt": query,
            "context": "",
            "citations": [],
            "evidence_snippets": [],
            "grounded": False,
            "knowledge_version": _knowledge_version(rag_mode),
            "retrieval_mode": retrieval_mode,
            "retrieval_skipped_reason": "no_candidates",
        }

    # --- 4. Re-Ranking (Cross-Encoder) ---
    if rerank_enabled is None:
        rerank_enabled = _is_enabled("RERANK_ENABLED", "1")
    rerank_min_candidates = max(1, _get_int_setting("RAG_RERANK_MIN_CANDIDATES", 3))
    candidate_texts = [item[1] for item in candidate_items]
    rerank_candidates = candidate_texts[: max(k, 5 if retrieval_mode != "light_retrieval" else max(3, k))]

    if rerank_enabled and len(rerank_candidates) >= rerank_min_candidates:
        final_docs = await asyncio.to_thread(rerank_documents, query, rerank_candidates, k)
    else:
        final_docs = candidate_texts[:k]

    final_items: List[Tuple[str, str, Dict[str, Any]]] = []
    used_doc_ids = set()
    for final_doc in final_docs:
        for doc_id, candidate_text, candidate_meta in candidate_items:
            if doc_id in used_doc_ids:
                continue
            if candidate_text == final_doc:
                final_items.append((doc_id, candidate_text, candidate_meta))
                used_doc_ids.add(doc_id)
                break

    if context_token_budget is None:
        budget_key = "RAG_LIGHT_CONTEXT_TOKEN_BUDGET" if retrieval_mode == "light_retrieval" else "RAG_FULL_CONTEXT_TOKEN_BUDGET"
        context_token_budget = _get_int_setting(budget_key, _get_int_setting("RAG_CONTEXT_TOKEN_BUDGET", 1200))
    final_docs = _trim_context_to_budget(final_docs, int(context_token_budget))
    trimmed_items: List[Tuple[str, str, Dict[str, Any]]] = []
    remaining_chars = int(context_token_budget) * 4 if context_token_budget else 0
    for doc_id, candidate_text, candidate_meta in final_items:
        if remaining_chars <= 0:
            break
        if len(candidate_text) <= remaining_chars:
            trimmed_items.append((doc_id, candidate_text, candidate_meta))
            remaining_chars -= len(candidate_text)
            continue
        shortened = candidate_text[:remaining_chars].rstrip()
        if shortened:
            trimmed_items.append((doc_id, shortened, candidate_meta))
        break
    context = "\n\n".join(final_docs)
    min_docs_for_grounded_context = _min_docs_for_grounded_context()
    useful_doc_count = len([doc for doc in final_docs if str(doc or "").strip()])
    if retrieval_mode == "light_retrieval" and useful_doc_count < min_docs_for_grounded_context:
        retrieval_skipped_reason = "insufficient_context_quality"
        context = ""
        final_docs = []
        trimmed_items = []
    try:
        if RETRIEVAL_DOCUMENTS_RETURNED:
            RETRIEVAL_DOCUMENTS_RETURNED.labels(modality=rag_mode).observe(len(final_docs))
        if RETRIEVAL_CONTEXT_TOKENS:
            RETRIEVAL_CONTEXT_TOKENS.labels(modality=rag_mode).observe(max(0, len(context) // 4))
    except Exception:
        pass
    logger.info(f"[rag_local] Hybrid RAG: {len(final_docs)} docs finais (Vector={len(vector_doc_ids)}, BM25={len(bm25_doc_ids)}).")

    citations = []
    evidence_snippets = []
    for rank, (doc_id, doc_text, doc_meta) in enumerate(trimmed_items or final_items[: len(final_docs)], start=1):
        source = str(doc_meta.get("source") or doc_meta.get("title") or doc_meta.get("uri") or "vectorstore")
        citations.append(
            {
                "doc_id": str(doc_id),
                "rank": rank,
                "source": source,
                "snippet": _snippet(doc_text),
                "score": None,
            }
        )
        evidence_snippets.append(
            {
                "doc_id": str(doc_id),
                "rank": rank,
                "source": source,
                "text": _snippet(doc_text, limit=320),
            }
        )

    if context:
        augmented_prompt = (
            "INSTRUÇÃO DE CONTEXTO (RAG):\n"
            "Use as informações técnicas abaixo recuperadas do banco de dados para auxiliar na sua resposta.\n"
            "------ CONTEXTO RECUPERADO (Híbrido + Re-rank) ------\n"
            f"{context}\n"
            "---------------------------------\n"
            f"PERGUNTA DO USUÁRIO: {query}"
        )
    else:
        augmented_prompt = query
    return {
        "query": query,
        "augmented_prompt": augmented_prompt,
        "context": context,
        "citations": citations,
        "evidence_snippets": evidence_snippets,
        "grounded": bool(citations),
        "knowledge_version": _knowledge_version(rag_mode),
        "retrieval_mode": retrieval_mode,
        "retrieval_skipped_reason": retrieval_skipped_reason,
    }


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
    """Execute the add document local routine.

This helper encapsulates one focused step used by the surrounding workflow."""
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
