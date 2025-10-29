# app/rag_local.py
from __future__ import annotations

# Desativa telemetria *antes* de importar chromadb
import os as _os
_os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")

import logging
import requests
from typing import List, Dict, Any, Optional

import chromadb

logger = logging.getLogger(__name__)

# ============================
# Configurações
# ============================
CHROMA_PATH = _os.getenv("CHROMA_PATH", "/data/chroma")
OLLAMA_BASE_URL = _os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBED_MODEL = _os.getenv("EMBED_MODEL", "nomic-embed-text")  # modelo local do Ollama
TOP_K = int(_os.getenv("RAG_TOP_K", "3"))

# Dimensão fallback; 'nomic-embed-text' geralmente = 768
EMBED_DIM_DEFAULT = int(_os.getenv("EMBED_DIM", "768"))

# ============================
# EmbeddingFunction compatível (Chroma >= 0.4.16)
# ============================
class OllamaEmbeddingFunction:
    """
    Implementação mínima do protocolo EmbeddingFunction do Chroma (>= 0.4.16).

    Assinatura exigida:
        __call__(self, input: List[str]) -> List[List[float]]
    """

    def __init__(self, model: str = EMBED_MODEL, base_url: str = OLLAMA_BASE_URL, timeout: int = 60):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def __call__(self, input: List[str]) -> List[List[float]]:
        outputs: List[List[float]] = []
        for text in input:
            try:
                # Chamada a /api/embeddings do Ollama
                r = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                vec = data.get("embedding")
                if not isinstance(vec, list):
                    raise ValueError("Resposta do Ollama não contém 'embedding' como lista.")
                outputs.append(vec)
            except Exception as e:
                logger.error(f"[RAG] Falha ao gerar embedding via Ollama: {e}")
                # Fallback: vetor nulo com dimensão configurada
                outputs.append([0.0] * EMBED_DIM_DEFAULT)
        return outputs


# ============================
# Cliente Chroma + Collection (API nova)
# ============================
chroma_client = None
collection = None

try:
    # Novo cliente persistente (Chroma >= 0.5)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    embed_fn = OllamaEmbeddingFunction()
    # Observação: se você mudar o espaço, reabra a coleção do zero
    collection = chroma_client.get_or_create_collection(
        name="docs",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("[RAG] Chroma collection 'docs' pronta (PersistentClient).")
except Exception as e:
    logger.exception(f"[RAG] Erro ao inicializar Chroma: {e}")
    chroma_client = None
    collection = None


# ============================
# API pública
# ============================
def add_document(doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Adiciona um documento à coleção com embedding via Ollama.
    """
    if collection is None:
        logger.error("[RAG] Coleção Chroma indisponível (falha de inicialização).")
        return False

    try:
        collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )
        # Persistência imediata
        try:
            chroma_client.persist()  # no PersistentClient, é idempotente
        except Exception:
            pass
        logger.info(f"[RAG] Documento '{doc_id}' adicionado.")
        return True
    except Exception as e:
        logger.error(f"[RAG] Falha ao adicionar documento '{doc_id}': {e}")
        return False


def build_augmented_prompt(query: str) -> str:
    """
    Consulta a coleção e retorna um prompt com contexto concatenado.
    Se não houver Chroma, retorna apenas o query.
    """
    if not query:
        return ""

    if collection is None:
        logger.warning("[RAG] Coleção indisponível; retornando query sem contexto.")
        return query

    try:
        res = collection.query(
            query_texts=[query],
            n_results=TOP_K,
        )
        docs = (res or {}).get("documents", [[]])
        top_docs: List[str] = docs[0] if docs and isinstance(docs[0], list) else []
        context = "\n\n".join(top_docs[:TOP_K])

        if not context.strip():
            return query

        augmented = (
            "Use the following context if relevant.\n"
            "-----\n"
            f"{context}\n"
            "-----\n"
            f"User: {query}"
        )
        return augmented
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
