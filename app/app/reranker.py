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
# Modelo leve e rápido treinado no MS MARCO
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def get_reranker_model():
    """Return reranker model.

This helper centralizes retrieval logic so callers do not have to duplicate lookup behavior."""
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None and CE_AVAILABLE:
        model_name = settings.get("RERANK_MODEL", DEFAULT_RERANK_MODEL)
        logger.info(f"[ReRanker] Carregando Cross-Encoder: {model_name}...")
        # device='cpu' é geralmente suficiente para o MiniLM e evita VRAM thrashing com o LLM
        try:
            _RERANKER_INSTANCE = CrossEncoder(model_name, device="cpu")
        except Exception as e:
            logger.error(f"[ReRanker] Erro ao carregar modelo: {e}")
            _RERANKER_INSTANCE = None
            
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
