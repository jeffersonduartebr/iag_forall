# app/rag_local.py
from __future__ import annotations
import os as _os
_os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")

import logging
import requests
from typing import List, Dict, Any, Optional

# ✦ API NOVA do Chroma
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction
from chromadb import PersistentClient

logger = logging.getLogger(__name__)

CHROMA_PATH = _os.getenv("CHROMA_PATH", "/data/chroma")
OLLAMA_BASE_URL = _os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBED_MODEL = _os.getenv("EMBED_MODEL", "nomic-embed-text")
TOP_K = int(_os.getenv("RAG_TOP_K", "3"))
EMBED_DIM_DEFAULT = int(_os.getenv("EMBED_DIM", "768"))

class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, model: str = EMBED_MODEL, base_url: str = OLLAMA_BASE_URL, timeout: int = 60):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def __call__(self, input: List[str]) -> List[List[float]]:
        outputs: List[List[float]] = []
        for text in input:
            try:
                r = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                vec = data.get("embedding")
                if not isinstance(vec, list):
                    raise ValueError("Ollama embeddings: campo 'embedding' ausente/inesperado.")
                outputs.append(vec)
            except Exception as e:
                logger.error(f"[RAG] Falha ao gerar embedding via Ollama: {e}")
                outputs.append([0.0] * EMBED_DIM_DEFAULT)
        return outputs

# --- Inicialização segura do Chroma (API nova) ---
try:
    _os.makedirs(CHROMA_PATH, exist_ok=True)
    chroma_client = PersistentClient(path=CHROMA_PATH)   # <<<< API NOVA
    embed_fn = OllamaEmbeddingFunction()

    # get_or_create_collection com embedding_function
    collection = chroma_client.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"},
        embedding_function=embed_fn,  # assinatura nova aceita objeto chamável
    )
    logger.info("[RAG] Chroma (PersistentClient) e coleção 'docs' inicializados.")
except Exception as e:
    logger.exception(f"[RAG] Erro ao inicializar Chroma: {e}")
    chroma_client = None
    collection = None

def add_document(doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    if collection is None:
        logger.error("[RAG] Coleção Chroma indisponível (falha de inicialização).")
        return False
    try:
        collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {"source": "warmup"}],  # ✅ mínimo: dict não vazio
        )

        # Persistente no PersistentClient é automático, mas manter é ok:
        if hasattr(chroma_client, "persist"):
            chroma_client.persist()
        logger.info(f"[RAG] Documento '{doc_id}' adicionado.")
        return True
    except Exception as e:
        logger.error(f"[RAG] Falha ao adicionar documento '{doc_id}': {e}")
        return False

def build_augmented_prompt(query: str) -> str:
    if not query:
        return ""
    if collection is None:
        logger.warning("[RAG] Coleção indisponível; retornando query sem contexto.")
        return query
    try:
        res = collection.query(query_texts=[query], n_results=TOP_K)
        docs = (res or {}).get("documents", [[]])
        top_docs: List[str] = docs[0] if docs and isinstance(docs[0], list) else []
        context = "\n\n".join(top_docs[:TOP_K])
        if not context.strip():
            return query
        return (
            "Use the following context if relevant.\n"
            "-----\n"
            f"{context}\n"
            "-----\n"
            f"User: {query}"
        )
    except Exception as e:
        logger.error(f"[RAG] Falha na consulta de contexto: {e}")
        return query


def health() -> Dict[str, Any]:
    """
    Retorna informações básicas de saúde do módulo RAG.
    """
    return {
        "chroma_ready": collection is not None,
        "path": CHROMA_PATH,
        "embed_model": EMBED_MODEL,
        "top_k": TOP_K,
    }
