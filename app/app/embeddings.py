from litellm import embedding

# Escolha um modelo suportado pelo LiteLLM para embeddings
# Ex.: "text-embedding-3-small" (OpenAI), "nomic-embed-text" (Ollama), etc.
EMBED_MODEL = "text-embedding-3-small"

def get_embedding(text: str) -> list[float]:
    """
    Retorna um vetor (lista de floats) para o texto informado.
    Ajuste EMBED_MODEL conforme seu provedor disponível.
    """
    resp = embedding(model=EMBED_MODEL, input=[text])
    return resp["data"][0]["embedding"]