"""
Vectorstore manager — ChromaDB v2+ (modo local persistente)
-----------------------------------------------------------
Compatível com Chroma >= 0.5.0.
Usa o novo PersistentClient() sem necessidade de API remota.
"""

import os
import logging
import chromadb

logger = logging.getLogger(__name__)

# ============================================================
# ⚙️ Configurações
# ============================================================
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chromadb-data")


# ============================================================
# 🧠 Inicialização Local Persistente
# ============================================================
def _connect_local():
    """
    Inicializa o cliente ChromaDB em modo local persistente.
    """
    try:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        logger.info(f"[vectorstore] Inicializando Chroma local persistente em {CHROMA_PATH}")

        # Novo cliente persistente (Chroma >= 0.5)
        client = chromadb.PersistentClient(path=CHROMA_PATH)

        logger.info("[vectorstore] ✅ Chroma PersistentClient inicializado com sucesso.")
        return client

    except Exception as e:
        logger.error(f"[vectorstore] ❌ Falha ao iniciar Chroma local: {e}")
        raise


# Cliente global
chroma_client = _connect_local()


# ============================================================
# 📚 Gerenciamento de Coleções
# ============================================================
def get_or_create_collection(name: str, metadata: dict = None):
    """Obtém ou cria uma coleção local persistente."""
    try:
        existing = [c.name for c in chroma_client.list_collections()]
        if name in existing:
            logger.debug(f"[vectorstore] Coleção existente: '{name}'")
            return chroma_client.get_collection(name)
        else:
            logger.info(f"[vectorstore] Criando nova coleção: '{name}'")
            meta = metadata if metadata else {"source": "semantic_cache"}
            return chroma_client.create_collection(name=os.name, metadata=meta)

    except Exception as e:
        logger.error(f"[vectorstore] Erro ao obter/criar coleção '{name}': {e}")
        raise


# ============================================================
# 💾 Inserção e Consulta
# ============================================================
def insert_embedding(collection_name: str, doc_id: str, text: str, embedding: list):
    """Insere um documento e seu embedding."""
    try:
        collection = get_or_create_collection(collection_name)
        collection.add(ids=[doc_id], documents=[text], embeddings=[embedding])
        logger.debug(f"[vectorstore] Documento '{doc_id}' inserido em '{collection_name}'.")
    except Exception as e:
        logger.error(f"[vectorstore] Falha ao inserir embedding: {e}")


def query_embedding(collection_name: str, embedding: list, n_results: int = 3):
    """Consulta os embeddings mais semelhantes."""
    try:
        collection = get_or_create_collection(collection_name)
        results = collection.query(query_embeddings=[embedding], n_results=n_results)
        return results
    except Exception as e:
        logger.error(f"[vectorstore] Erro ao consultar coleção '{collection_name}': {e}")
        return None


# ============================================================
# 🧩 Utilitários
# ============================================================
def reset_collections():
    """Remove todas as coleções (útil para testes)."""
    try:
        chroma_client.reset()
        logger.info("[vectorstore] Todas as coleções foram removidas com sucesso.")
    except Exception as e:
        logger.error(f"[vectorstore] Falha ao resetar coleções: {e}")
