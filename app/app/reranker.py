# -*- coding: utf-8 -*-
# Objective: Application runtime code for reranker.
"""
reranker.py — Módulo de Re-Ranking (Cross-Encoder)
--------------------------------------------------
Implementa o estágio de refinamento do RAG.
Recebe documentos recuperados via busca vetorial (Bi-Encoder) e
reordena usando um Cross-Encoder para maximizar a relevância semântica.
"""

import logging
import threading
import time
from typing import List

# Importação condicional para não quebrar se a lib faltar
try:
    from sentence_transformers import CrossEncoder
    CE_AVAILABLE = True
except ImportError:
    CE_AVAILABLE = False

from .settings_dynamic import settings

logger = logging.getLogger(__name__)

# Singleton para evitar recarga do modelo a cada request
_RERANKER_INSTANCE = None
_LOAD_LOCK = threading.Lock()
_LOAD_FAILED_AT = 0.0
LOAD_RETRY_S = 60.0
# Multilíngue (mMARCO, 14 línguas com português; ~118M parâmetros, viável em CPU). O anterior,
# ms-marco-MiniLM-L-6-v2, só foi treinado em inglês e reordenava mal o acervo em português.
DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

def get_reranker_model():
    """Return reranker model.

This helper centralizes retrieval logic so callers do not have to duplicate lookup behavior."""
    global _RERANKER_INSTANCE, _LOAD_FAILED_AT
    if _RERANKER_INSTANCE is not None or not CE_AVAILABLE:
        return _RERANKER_INSTANCE
    with _LOAD_LOCK:  # uma carga por vez; depois de uma falha, nova tentativa só após LOAD_RETRY_S
        if _RERANKER_INSTANCE is None and time.monotonic() - _LOAD_FAILED_AT >= LOAD_RETRY_S:
            model_name = settings.get("RERANK_MODEL", DEFAULT_RERANK_MODEL)
            logger.info(f"[ReRanker] Carregando Cross-Encoder: {model_name}...")
            try:
                _RERANKER_INSTANCE = CrossEncoder(model_name, device="cpu")
            except Exception as e:
                _LOAD_FAILED_AT = time.monotonic()
                logger.error(f"[ReRanker] Erro ao carregar modelo: {e}")
    return _RERANKER_INSTANCE

def rerank_documents(query: str, documents: List[str], top_k: int = 3) -> List[str]:
    """
    Reordena uma lista de documentos baseada na relevância para a query.
    """
    if not documents or not query:
        return documents[:top_k]

    if not CE_AVAILABLE:
        logger.warning("[ReRanker] sentence-transformers não instalado. Pulando re-rank.")
        return documents[:top_k]

    model = get_reranker_model()

    # Fallback se o modelo não carregar
    if not model:
        return documents[:top_k]

    try:
        start = time.time()
        # Prepara pares (Query, Doc)
        # O Cross-Encoder espera uma lista de pares [Query, Contexto]
        pairs = [[query, doc] for doc in documents]

        # Prediz scores (logits)
        scores = model.predict(pairs)

        # Combina docs com scores e ordena
        scored_docs = list(zip(documents, scores))
        # Ordena decrescente pelo score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        duration = (time.time() - start) * 1000
        top_score = scored_docs[0][1] if scored_docs else 0.0
        logger.info(f"[ReRanker] Reordenado {len(documents)} docs em {duration:.2f}ms. Top score: {top_score:.4f}")

        # Retorna apenas os textos dos top_k
        return [doc for doc, score in scored_docs[:top_k]]

    except Exception as e:
        logger.error(f"[ReRanker] Falha ao reordenar: {e}")
        return documents[:top_k]
