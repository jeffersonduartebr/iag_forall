import chromadb
from .settings import settings

# Cliente HTTP para o ChromaDB (compatível com seu docker-compose)
chroma_client = chromadb.HttpClient(
    host=getattr(settings, "CHROMADB_HOST", "chromadb"),
    port=getattr(settings, "CHROMADB_PORT", 8000),
)