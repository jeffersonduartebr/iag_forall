# app/rag_context_provider.py
import logging
from .vectorstore import search_vectors

logger = logging.getLogger(__name__)

def get_rag_context(query: str, k: int = 3) -> str:
    """
    Retorna os k trechos mais relevantes para o prompt do juiz LLM.
    """
    try:
        results = search_vectors(query, top_k=k)
        if not results:
            return ""
        context = "\n\n".join([r["text"] for r in results])
        logger.info(f"[rag_context_provider] {len(results)} trechos recuperados para o contexto.")
        return context
    except Exception as e:
        logger.error(f"[rag_context_provider] Falha ao buscar contexto RAG: {e}")
        return ""
