# providers.py
import logging
import requests
from time import sleep
from litellm import completion
from litellm.exceptions import APIConnectionError, ServiceUnavailableError

logger = logging.getLogger(__name__)

# Base do Ollama
OLLAMA_BASE_URL = "http://ollama:11434"

# Modelos válidos (geração de texto)
TEXT_MODEL_WHITELIST = {
    "ollama/phi4",
    "ollama/deepseek-r1:8b",
    "openai/gpt-5-nano",
    "gemini/gemini-2.0-flash",
}

# Modelos de embedding (proibidos para geração)
EMBED_MODEL_PREFIXES = ("nomic-embed", "text-embedding", "bge-", "e5-")


# -------------------------------------------------------------------
# Garante que o modelo do Ollama esteja disponível localmente
# -------------------------------------------------------------------
def _ensure_ollama_model(model_name: str):
    """Baixa o modelo via Ollama se ainda não existir localmente."""
    if not model_name or not isinstance(model_name, str):
        logger.warning(f"[Ollama] Nome de modelo inválido: {model_name!r}")
        return

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if model_name in models:
                logger.info(f"[Ollama] Modelo '{model_name}' já disponível localmente.")
                return

        logger.info(f"[Ollama] Baixando modelo '{model_name}' ...")
        pull = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull", json={"name": model_name}, timeout=600
        )
        if pull.status_code == 200:
            logger.info(f"[Ollama] Pull concluído para '{model_name}'.")
        else:
            logger.warning(f"[Ollama] Falha ao baixar '{model_name}': {pull.text[:100]}")

    except Exception as e:
        logger.error(f"[Ollama] Erro durante verificação/pull: {e}")


# -------------------------------------------------------------------
# Chamada segura ao modelo via LiteLLM
# -------------------------------------------------------------------
def call_model(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 1024, api_base: str = None):
    """
    Chama modelo com validação e tratamento de falhas.
    Protege contra uso de embeddings e falhas de rede no Ollama.
    """

    # 1️⃣ Verificação básica de parâmetros
    if not model or not isinstance(model, str):
        raise ValueError(f"[providers] Modelo inválido: {model!r}")

    if any(model.startswith(prefix) for prefix in EMBED_MODEL_PREFIXES):
        raise ValueError(f"[providers] Modelo '{model}' é de embedding e não suporta geração.")

    # 2️⃣ Corrige nome para LiteLLM
    is_ollama = model.startswith("ollama/") or ":" in model
    model_clean = model.replace("ollama/", "")

    # 3️⃣ Define base URL para Ollama
    base_url = api_base or OLLAMA_BASE_URL if is_ollama else None

    # 4️⃣ Re-tentativas automáticas
    retries = 3
    delay = 3
    for attempt in range(1, retries + 1):
        try:
            if is_ollama:
                resp = completion(
                    model=f"ollama/{model_clean}",
                    api_base=base_url,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                resp = completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

        except Exception as e:
            # --- Tratamento especial para modelos sem mapeamento de custo ---
            if "isn't mapped yet" in str(e):
                logger.warning(
                    f"[LiteLLM] Modelo '{model}' ainda não está mapeado no cost_calculator. "
                    f"Aplicando custo padrão local definido em bandits.py."
                )
                resp = {
                    "choices": [{"message": {"content": "[Aviso] Modelo sem custo mapeado."}}],
                    "usage": {},
                    "latency": 0.0,
                }
            else:
                raise

        # --- Interpretação de resposta ---
        text = resp["choices"][0]["message"]["content"] if "choices" in resp else "[Sem resposta]"
        meta = {
            "usage": resp.get("usage", {}),
            "latency_s": resp.get("latency", 0.0),
            "provider": "ollama" if is_ollama else "remote",
        }
        return text, meta


    # Fallback final — se todos os retries falharem
    return "[Erro: todas as tentativas falharam]", {"latency_s": 0.0}
