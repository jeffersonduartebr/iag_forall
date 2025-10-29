# providers.py
import logging
import requests
from litellm import completion

logger = logging.getLogger(__name__)

# Configuração do Ollama local
OLLAMA_BASE_URL = "http://ollama:11434"


# ---------------------------------------------------------
# Função auxiliar: garantir que o modelo Ollama esteja presente
# ---------------------------------------------------------
def _ensure_ollama_model(model_name: str):
    """
    Garante que o modelo Ollama esteja disponível localmente.
    Se não existir, realiza automaticamente o 'pull' via API REST.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            available = [m["name"] for m in models]
            if model_name in available:
                logger.info(f"[Ollama] Modelo '{model_name}' já presente localmente.")
                return

        # Caso o modelo não esteja presente, faz o pull
        logger.info(f"[Ollama] Modelo '{model_name}' ausente. Iniciando download...")
        pull = requests.post(f"{OLLAMA_BASE_URL}/api/pull", json={"name": model_name}, timeout=600)

        if pull.status_code == 200:
            logger.info(f"[Ollama] Pull concluído para '{model_name}'.")
        else:
            logger.warning(f"[Ollama] Falha ao baixar '{model_name}': {pull.text}")
    except Exception as e:
        logger.error(f"[Ollama] Erro ao verificar/baixar modelo '{model_name}': {e}")


# ---------------------------------------------------------
# Função principal: chamada segura a modelos LLM
# ---------------------------------------------------------
def call_model(
    model: str,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    api_base: str = None
):
    """
    Executa chamada segura a um modelo (Ollama ou remoto via LiteLLM).
    Suporta fallback automático: se o modelo for local e não estiver baixado, faz pull.
    """
    try:
        # Detecta se é um modelo local do Ollama (tem ':' ou corresponde ao modelo local configurado)
        is_ollama = (":" in model) or (api_base and "ollama" in api_base)

        if is_ollama:
            _ensure_ollama_model(model)
            base_url = api_base or OLLAMA_BASE_URL

            resp = completion(
                model=model,  # exemplo: "phi4", "deepseek-r1:8b", "nomic-embed-text"
                messages=[{"role": "user", "content": prompt}],
                api_base=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            # Modelos comerciais (OpenAI, Gemini, etc.)
            resp = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        text = resp["choices"][0]["message"]["content"]

        meta = {
            "usage": resp.get("usage", {}),
            "latency_s": resp.get("latency", 0.0),
            "provider": "ollama" if is_ollama else "remote",
        }

        return text, meta

    except Exception as e:
        logger.error(f"[providers] Falha ao chamar modelo '{model}': {e}")
        return f"[Erro ao chamar modelo {model}]", {"usage": {}, "latency_s": 0.0}
