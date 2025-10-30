# providers.py
import logging
import requests
import re
from time import sleep
from litellm import completion
from litellm.exceptions import APIConnectionError, ServiceUnavailableError
import os
logger = logging.getLogger(__name__)

# Base do Ollama
OLLAMA_BASE_URL = "http://ollama:11434"

# Modelos válidos (geração de texto)
TEXT_MODEL_WHITELIST = {
    "ollama/phi4",
    "ollama/deepseek-r1:8b",
    "openai/gpt-5-nano",
    "gemini/gemini-2.0-flash",
    "ollama/gemma3:4b-it-qat",
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
def call_model(model: str, prompt: str, temperature: float = 0.4, max_tokens: int = 1024, api_base: str = None):
    """
    Chama modelo com validação e tratamento de falhas.
    Agora compatível com modelos GPT-5 / GPT-4o-mini / GPT-5-nano,
    que usam 'max_completion_tokens' ao invés de 'max_tokens'.
    """

    # Validação básica
    if not model or not isinstance(model, str):
        raise ValueError(f"[providers] Modelo inválido: {model!r}")

    if any(model.startswith(prefix) for prefix in EMBED_MODEL_PREFIXES):
        raise ValueError(f"[providers] Modelo '{model}' é de embedding e não suporta geração.")

    # Identificação do tipo de modelo
    is_ollama = model.startswith("ollama/") or ":" in model
    model_clean = model.replace("ollama/", "")

    # Define URL base segura (usa o container "ollama" e não localhost)
    base_url = None
    if is_ollama:
        base_url = api_base or os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    logger.debug(f"[providers] Usando Ollama base_url={base_url}")



    # 3️⃣ Verifica se o modelo é da nova geração OpenAI
    use_new_param = bool(re.search(r"gpt-5|gpt-4o-mini|nano", model))

    # 4️⃣ Configura parâmetros da chamada
    if use_new_param:
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1,
        }
    else:
        params = {
            "model": f"ollama/{model_clean}" if is_ollama else model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

    if use_new_param:
        params["max_completion_tokens"] = max_tokens
    else:
        params["max_tokens"] = max_tokens

    # 5️⃣ Re-tentativas com fallback
    retries = 3
    delay = 3
    for attempt in range(1, retries + 1):
        try:
            resp = completion(**params)
            text = resp["choices"][0]["message"]["content"] if "choices" in resp else "[Sem resposta]"
            meta = {
                "usage": resp.get("usage", {}),
                "latency_s": resp.get("latency", 0.0),
                "provider": "ollama" if is_ollama else "remote",
            }
            return text, meta

        except Exception as e:
            # Fallback automático para erro de parâmetro inválido
            if "unsupported_parameter" in str(e) or "max_tokens" in str(e):
                logger.warning(f"[providers] Parâmetro incompatível detectado para {model}. Retentando com chave alternativa.")
                if "max_tokens" in params:
                    params.pop("max_tokens", None)
                    params["max_completion_tokens"] = max_tokens
                else:
                    params.pop("max_completion_tokens", None)
                    params["max_tokens"] = max_tokens
                sleep(1)
                continue

            # Modelos sem custo mapeado
            if "isn't mapped yet" in str(e):
                logger.warning(
                    f"[LiteLLM] Modelo '{model}' ainda não está mapeado no cost_calculator. "
                    f"Aplicando custo padrão local definido em bandits.py."
                )
                return (
                    "[Aviso] Modelo sem custo mapeado.",
                    {"usage": {}, "latency_s": 0.0, "provider": "remote"},
                )

            # Erros genéricos
            logger.error(f"[providers] Falha tentativa {attempt}/{retries} para modelo {model}: {e}")
            if attempt < retries:
                sleep(delay)
                continue
            raise

    # 6️⃣ Fallback final
    logger.error(f"[providers] Todas as tentativas falharam para {model}.")
    return "[Erro: todas as tentativas falharam]", {"latency_s": 0.0, "provider": "remote"}
