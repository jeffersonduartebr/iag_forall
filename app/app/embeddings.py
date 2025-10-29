"""
embeddings.py
-------------------------------------
Gera embeddings locais usando o Ollama (ex: nomic-embed-text)
via LiteLLM, sem necessidade de registrar provedores manualmente.
"""

from litellm import embedding
import os

# Nome do modelo local de embedding no Ollama
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

def get_embedding(text: str) -> list[float]:
    """
    Retorna o vetor de embedding para o texto informado.
    Usa o Ollama local via LiteLLM (api_base definido).
    """
    resp = embedding(
        model=EMBED_MODEL,
        input=[text],
        api_base=OLLAMA_BASE_URL
    )
    return resp["data"][0]["embedding"]
