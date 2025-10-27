# app/rag.py
import logging
from .settings import settings
from .embeddings import get_embedding
from .vectorstore import chroma_client

logger = logging.getLogger(__name__)

def retrieve_context(query: str, top_k: int | None = None) -> str:
    """
    (Função original simples) - mantém por compatibilidade.
    """
    try:
        k = top_k or getattr(settings, "RAG_TOP_K", 4)
        q_emb = get_embedding(query)
        coll = chroma_client.get_or_create_collection("docs")
        res = coll.query(query_embeddings=[q_emb], n_results=k)
        docs = (res.get("documents") or [[]])[0]
        return "\n\n".join(docs) if docs else ""
    except Exception as e:
        logger.warning(f"[RAG] Falha retrieve_context: {e}")
        return ""

def retrieve_context_adaptive(query: str) -> str:
    """
    Ativa RAG apenas se a similaridade do top-1 exceder o threshold.
    """
    if not getattr(settings, "RAG_ENABLED", True):
        return ""

    try:
        k = getattr(settings, "RAG_TOP_K", 4)
        thr = float(getattr(settings, "RAG_SIM_THRESHOLD", 0.75))

        q_emb = get_embedding(query)
        coll = chroma_client.get_or_create_collection("docs")
        res = coll.query(query_embeddings=[q_emb], n_results=k)

        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        if not dists:
            return ""

        best_sim = 1.0 - float(dists[0])  # se for cos distance
        if best_sim < thr:
            logger.info(f"[RAG] OFF (sim={best_sim:.3f} < thr={thr})")
            return ""

        ctx = "\n\n".join(docs) if docs else ""
        logger.info(f"[RAG] ON (sim={best_sim:.3f}, k={len(docs)})")
        return ctx
    except Exception as e:
        logger.warning(f"[RAG] Falha retrieve_context_adaptive: {e}")
        return ""
