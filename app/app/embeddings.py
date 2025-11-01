"""
embeddings.py
-------------------------------------
Gera embeddings locais usando o Ollama (ex: nomic-embed-text)
via LiteLLM, sem necessidade de provedores externos.
Compatível com o fluxo RAG e os juízes LLM.
"""

from litellm import aembedding
import os
import logging

# ✅ CORRIGIDO: Importa o settings centralizado
from .settings_dynamic import settings

logger = logging.getLogger(__name__)

# ✅ CORRIGIDO: Lê as configurações do settings (Redis > DB > .env)
EMBED_MODEL = settings.EMBED_MODEL
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL


async def embed_text(text: str) -> list[float]:
    """
    Retorna o vetor de embedding para o texto informado.
    Usa o Ollama local via LiteLLM (api_base definido).
    """
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Texto inválido para geração de embedding.")

        if EMBED_MODEL.lower() == "nomic-embed-text":
            resp = await aembedding(
                model=f"ollama/{EMBED_MODEL}",
                input=[text],
                api_base=OLLAMA_BASE_URL
            )
        else:
            resp = await aembedding(
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