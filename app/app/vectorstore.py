"""
Vectorstore manager — ChromaDB v2+ (modo local persistente)
-----------------------------------------------------------
Compatível com Chroma >= 0.5.0.
Usa o novo PersistentClient() sem necessidade de API remota.
"""

import os
import logging
import chromadb
import asyncio  # 👈 Import para asyncio

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
    (Função Síncrona)
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


# Cliente global (Criado de forma síncrona na inicialização)
chroma_client = _connect_local()


# ============================================================
# 📚 Gerenciamento de Coleções
# ============================================================
def get_or_create_collection(name: str, metadata: dict = None):
    """(Função Síncrona) Obtém ou cria uma coleção local persistente."""
    try:
        for c in chroma_client.list_collections():
            if c.name == name:
                return chroma_client.get_collection(name)
        return chroma_client.create_collection(name=name, metadata=metadata or {"source": "router"})
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.warning(f"[vectorstore] Coleção '{name}' já existia, recuperando...")
            return chroma_client.get_collection(name)
        raise

# ============================================================
# 💾 Inserção e Consulta (Assíncronas via to_thread)
# ============================================================

def _insert_embedding_sync(collection_name: str, doc_id: str, text: str, embedding: list):
    """Função síncrona auxiliar para ser usada com to_thread."""
    collection = get_or_create_collection(collection_name)
    collection.add(ids=[doc_id], documents=[text], embeddings=[embedding])

async def insert_embedding(collection_name: str, doc_id: str, text: str, embedding: list):
    """(Async) Insere um documento e seu embedding."""
    try:
        await asyncio.to_thread(
            _insert_embedding_sync,
            collection_name, doc_id, text, embedding
        )
        logger.debug(f"[vectorstore] Documento '{doc_id}' inserido em '{collection_name}'.")
    except Exception as e:
        logger.error(f"[vectorstore] Falha ao inserir embedding: {e}")


def _query_embedding_sync(collection_name: str, embedding: list, n_results: int = 3):
    """Função síncrona auxiliar para ser usada com to_thread."""
    collection = get_or_create_collection(collection_name)
    return collection.query(query_embeddings=[embedding], n_results=n_results)

async def query_embedding(collection_name: str, embedding: list, n_results: int = 3):
    """(Async) Consulta os embeddings mais semelhantes."""
    try:
        results = await asyncio.to_thread(
            _query_embedding_sync,
            collection_name, embedding, n_results
        )
        return results
    except Exception as e:
        logger.error(f"[vectorstore] Erro ao consultar coleção '{collection_name}': {e}")
        return None

# ============================================================
# 🧩 Utilitários (Assíncronos via to_thread)
# ============================================================
async def reset_collections():
    """(Async) Remove todas as coleções (útil para testes)."""
    try:
        await asyncio.to_thread(chroma_client.reset)
        logger.info("[vectorstore] Todas as coleções foram removidas com sucesso.")
    except Exception as e:
        logger.error(f"[vectorstore] Falha ao resetar coleções: {e}")