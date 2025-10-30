"""
embeddings.py
-------------------------------------
Gera embeddings locais usando o Ollama (ex: nomic-embed-text)
via LiteLLM, sem necessidade de provedores externos.
Compatível com o fluxo RAG e os juízes LLM.
"""

from litellm import embedding
import os
import logging

logger = logging.getLogger(__name__)

# Nome do modelo local de embedding no Ollama
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")


def embed_text(text: str) -> list[float]:
    """
    Retorna o vetor de embedding para o texto informado.
    Usa o Ollama local via LiteLLM (api_base definido).
    """
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Texto inválido para geração de embedding.")

        if EMBED_MODEL.lower() == "nomic-embed-text":
            resp = embedding(
                model=f"ollama/{EMBED_MODEL}",
                input=[text],
                api_base=OLLAMA_BASE_URL
            )
        else:
            resp = embedding(
            model=EMBED_MODEL,
            input=[text],
            api_base=OLLAMA_BASE_URL
        )

        vec = resp["data"][0]["embedding"]

        # Normaliza vetor (garante unidade para cálculos de similaridade)
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]

        logger.debug(f"[embeddings] Vetor gerado com {len(vec)} dimensões.")
        return vec

    except Exception as e:
        logger.error(f"[embeddings] Falha ao gerar embedding: {e}")
        return [0.0] * 768  # fallback seguro (tamanho padrão do nomic-embed-text)


# Mantém compatibilidade com versões anteriores
get_embedding = embed_text
