# -*- coding: utf-8 -*-
# Objective: Application runtime code for rag local.
"""
rag_local.py — RAG Multimodal Unificado (Com suporte Imagem -> Texto)
---------------------------------------------------------------------
Implementa a lógica de recuperação de contexto, incluindo a estratégia
de "Visual Query Generation" para RAG baseado em imagem.

Quick Win #5: Added Redis cache for visual query descriptions.
"""

from __future__ import annotations
import logging
import asyncio
import hashlib
import json
import re
from typing import Optional, Dict, Any, List, Tuple

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

# Importamos o call_model para gerar a descrição da imagem (Ponte Visual)
from .providers_async import call_model
from .settings_dynamic import settings
from .reranker import rerank_documents  # <--- Importar o novo módulo
from .sparse_index import sparse_index  # <--- Importar BM25
from .utils.redis_client import get_redis

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
        VISUAL_QUERY_CACHE_HITS,
        VISUAL_QUERY_CACHE_MISSES,
        RETRIEVAL_DOCUMENTS_RETURNED,
        RETRIEVAL_CONTEXT_TOKENS,
        RETRIEVAL_SCORE,
    )
except ImportError:
    VISUAL_QUERY_CACHE_HITS = None
    VISUAL_QUERY_CACHE_MISSES = None
    RETRIEVAL_DOCUMENTS_RETURNED = None
    RETRIEVAL_CONTEXT_TOKENS = None
    RETRIEVAL_SCORE = None


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
) -> str:
    """Execute the build augmented prompt routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    query = (query or "").strip()
    if not query and not image_b64:
        return ""

    rag_mode = _auto_modality(modality, image_b64)
    target_collection_modality = "text" if rag_mode == "vision" else rag_mode

    # --- 1. Busca Vetorial (Dense Retrieval) ---
    emb = await _compute_embedding(query, rag_mode, image_b64)
    vector_doc_ids = []
    vector_docs_map = {} # ID -> Texto

    if emb is not None:
        try:
            # Busca mais candidatos (Top-20) para fusão
            res = await query_embedding(
                modality=target_collection_modality,
                embedding=emb,
                n_results=20 
            )
            if res and res.get("ids"):
                ids = res["ids"][0]
                docs = res["documents"][0]
                vector_doc_ids = ids
                for i, doc_id in enumerate(ids):
                    vector_docs_map[doc_id] = docs[i]
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
    
    # Só faz sentido BM25 se houver query textual
    if query:
        try:
            # Busca Top-20 no BM25
            bm25_res = await asyncio.to_thread(sparse_index.search, query, top_k=20)
            bm25_doc_ids = [item[0] for item in bm25_res]
            for doc_id, _ in bm25_res:
                # Recupera o texto do índice esparso
                text = sparse_index.get_text(doc_id)
                if text: bm25_docs_map[doc_id] = text
        except Exception as e:
            logger.warning(f"[rag_local] Erro BM25 Search: {e}")

    # --- 3. Fusão Híbrida (RRF) ---
    merged_ids = reciprocal_rank_fusion(vector_doc_ids, bm25_doc_ids)
    
    # Recupera os textos dos IDs vencedores
    candidate_texts = []
    for doc_id in merged_ids:
        # Tenta pegar do mapa vetorial ou do mapa BM25
        txt = vector_docs_map.get(doc_id) or bm25_docs_map.get(doc_id)
        if txt:
            candidate_texts.append(txt)
    
    # Se não achou nada, retorna query original
    if not candidate_texts:
        try:
            if RETRIEVAL_DOCUMENTS_RETURNED:
                RETRIEVAL_DOCUMENTS_RETURNED.labels(modality=rag_mode).observe(0)
        except Exception:
            pass
        return query

    # --- 4. Re-Ranking (Cross-Encoder) ---
    # Pega os Top-N da fusão e reordena com precisão máxima
    if str(settings.get("RERANK_ENABLED", "1")) == "1":
        # Re-rankeia os top 10 candidatos da fusão
        final_docs = await asyncio.to_thread(rerank_documents, query, candidate_texts[:5], k)
    else:
        final_docs = candidate_texts[:k]

    context = "\n\n".join(final_docs)
    try:
        if RETRIEVAL_DOCUMENTS_RETURNED:
            RETRIEVAL_DOCUMENTS_RETURNED.labels(modality=rag_mode).observe(len(final_docs))
        if RETRIEVAL_CONTEXT_TOKENS:
            RETRIEVAL_CONTEXT_TOKENS.labels(modality=rag_mode).observe(max(0, len(context) // 4))
    except Exception:
        pass
    logger.info(f"[rag_local] Hybrid RAG: {len(final_docs)} docs finais (Vector={len(vector_doc_ids)}, BM25={len(bm25_doc_ids)}).")

    return (
        "INSTRUÇÃO DE CONTEXTO (RAG):\n"
        "Use as informações técnicas abaixo recuperadas do banco de dados para auxiliar na sua resposta.\n"
        "------ CONTEXTO RECUPERADO (Híbrido + Re-rank) ------\n"
        f"{context}\n"
        "---------------------------------\n"
        f"PERGUNTA DO USUÁRIO: {query}"
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
