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

from .services.rag_esquema import CarregadorComRecuo
from .settings_dynamic import settings

logger = logging.getLogger(__name__)

# Um modelo por processo: carga serializada, e uma falha só é tentada de novo após o recuo.
_CARREGADOR = CarregadorComRecuo("reranker")
# Multilíngue (mMARCO, 14 línguas com português; ~118M parâmetros, viável em CPU). O anterior,
# ms-marco-MiniLM-L-6-v2, só foi treinado em inglês e reordenava mal o acervo em português.
DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

def get_reranker_model():
    """Return reranker model.

This helper centralizes retrieval logic so callers do not have to duplicate lookup behavior."""
    if not CE_AVAILABLE:
        return None
    model_name = settings.get("RERANK_MODEL", DEFAULT_RERANK_MODEL)
    return _CARREGADOR.obter(lambda: CrossEncoder(model_name, device="cpu"))


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
