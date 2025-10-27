import logging
import requests
from typing import Optional
from .settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Retrieval-Augmented Generation (RAG)
# ---------------------------------------------------------
# Esta função tenta recuperar contexto relevante de uma base vetorial (ChromaDB)
# com base no texto da consulta. Caso falhe, retorna string vazia.
# ---------------------------------------------------------

def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Recupera contexto textual relevante a partir de um repositório vetorial (ChromaDB).
    Se o serviço não responder, retorna string vazia.
    """
    chroma_url = f"http://{settings.CHROMADB_HOST}:{settings.CHROMADB_PORT}/query"
    payload = {
        "collection": "knowledge",
        "query_texts": [query],
        "n_results": top_k,
    }

    try:
        logger.info(f"[RAG] Consultando ChromaDB em {chroma_url} (top_k={top_k})")
        response = requests.post(chroma_url, json=payload, timeout=5)

        if response.status_code != 200:
            logger.warning(f"[RAG] Erro HTTP {response.status_code} ao consultar ChromaDB")
            return ""

        data = response.json()
        # Chroma retorna estrutura: {"documents": [["texto1", "texto2", ...]]}
        docs = data.get("documents", [])
        if not docs or not isinstance(docs, list):
            logger.warning("[RAG] Nenhum documento retornado ou formato inválido")
            return ""

        flat_docs = []
        for group in docs:
            if isinstance(group, list):
                flat_docs.extend(group)

        # Monta contexto consolidado
        context = "\n---\n".join(flat_docs[:top_k])
        logger.info(f"[RAG] {len(flat_docs)} fragmentos recuperados do ChromaDB")
        return context

    except requests.exceptions.ConnectionError:
        logger.warning("[RAG] Falha de conexão com ChromaDB — RAG desativado temporariamente.")
        return ""
    except requests.exceptions.Timeout:
        logger.warning("[RAG] Timeout ao consultar ChromaDB — ignorando contexto.")
        return ""
    except Exception as e:
        logger.error(f"[RAG] Erro inesperado: {e}")
        return ""
