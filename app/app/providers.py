"""
providers.py
----------------------------------------------------
Gerencia chamadas a modelos LLM (Ollama, OpenAI, Gemini, etc.)
Compatível com múltiplas versões do LiteLLM (novas e legadas).
Inclui métricas Prometheus e fallback seguro para erros de registro.
"""

import logging
import requests
import re
import os
from time import sleep
from prometheus_client import Histogram
import litellm
from litellm import completion
from .settings import settings

logger = logging.getLogger(__name__)

# ============================================================
# ⚙️ Configurações globais
# ============================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# ------------------------------------------------------------
# 🔹 Registro seguro do modelo Gemini (com fallback)
# ------------------------------------------------------------
try:
    gemini_cost = 0.00015  # $0.15 / 1k tokens

    if hasattr(litellm, "register_model"):
        try:
            # Tentativa 1 — versão moderna (kwargs)
            litellm.register_model(
                "gemini-2.0-flash",
                litellm_provider="gemini",
                input_cost_per_token=gemini_cost,
                output_cost_per_token=gemini_cost,
                max_tokens=8192,
                model_info={"base_model": "gemini-2.0-flash"},
            )
            litellm.register_model(
                "gemini/gemini-2.0-flash",
                litellm_provider="gemini",
                input_cost_per_token=gemini_cost,
                output_cost_per_token=gemini_cost,
                max_tokens=8192,
                model_info={"base_model": "gemini-2.0-flash"},
            )
            logger.info("[providers] Modelos Gemini registrados (modo moderno).")
        except TypeError:
            # Tentativa 2 — versão antiga (posicional)
            litellm.register_model(
                "gemini-2.0-flash",
                "gemini",
                gemini_cost,
                gemini_cost,
                8192,
                {"base_model": "gemini-2.0-flash"},
            )
            litellm.register_model(
                "gemini/gemini-2.0-flash",
                "gemini",
                gemini_cost,
                gemini_cost,
                8192,
                {"base_model": "gemini-2.0-flash"},
            )
            logger.info("[providers] Modelos Gemini registrados (modo legacy).")
    else:
        logger.warning("[providers] register_model() ausente; registro será ignorado.")

except Exception as e:
    logger.warning(f"[providers] Falha ao registrar/atualizar modelo no LiteLLM: {e}")

# ============================================================
# 📊 Métricas Prometheus
# ============================================================

MODEL_COST_USD = Histogram(
    "model_cost_usd",
    "Custo estimado por requisição",
    ["model"],
)

TEXT_MODEL_WHITELIST = settings.CANDIDATE_MODELS_LIST

EMBED_MODEL_PREFIXES = ("nomic-embed", "text-embedding", "bge-", "e5-")

# ============================================================
# 🧠 Utilitários do Ollama
# ============================================================

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

# ============================================================
# 🚀 Função principal de chamada a modelos
# ============================================================

def call_model(
    model: str,
    prompt: str,
    temperature: float = 0.4,
    max_tokens: int = 1024,
    api_base: str = OLLAMA_BASE_URL,
):
    """
    Chama um modelo com validação, fallback e métricas.
    Compatível com GPT-5 / GPT-4o-mini / GPT-5-nano / Ollama / Gemini.
    """

    if not model or not isinstance(model, str):
        raise ValueError(f"[providers] Modelo inválido: {model!r}")

    # Impede uso de modelos de embedding para geração
    if any(model.startswith(prefix) for prefix in EMBED_MODEL_PREFIXES):
        raise ValueError(f"[providers] Modelo '{model}' é de embedding e não suporta geração.")

    # Detecta modelo Ollama
    is_ollama = model.startswith("ollama/") or ":" in model
    model_clean = model.replace("ollama/", "")
    base_url = api_base if is_ollama else None
    logger.debug(f"[providers] Usando base_url={base_url}")

    # Detecção para novos parâmetros (OpenAI-like)
    use_new_param = bool(re.search(r"gpt-5|gpt-4o-mini|nano", model))

    params = {
        "model": f"ollama/{model_clean}" if is_ollama else model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1 if use_new_param else temperature,
        "api_base": base_url,
    }

    if use_new_param:
        params["max_completion_tokens"] = max_tokens
    else:
        params["max_tokens"] = max_tokens

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

            # Prometheus — custo estimado
            if "usage" in meta and "total_tokens" in meta["usage"]:
                MODEL_COST_USD.labels(model=model).observe(
                    meta["usage"]["total_tokens"] / 1000.0 * 0.001
                )

            return text, meta

        except Exception as e:
            # Modelos sem custo mapeado
            if "isn't mapped yet" in str(e):
                logger.warning(
                    f"[LiteLLM] Modelo '{model}' não mapeado — aplicando custo padrão."
                )
                MODEL_COST_USD.labels(model=model).observe(0.00015)
                return (
                    "[Aviso: modelo sem custo mapeado — custo estimado localmente.]",
                    {"usage": {}, "latency_s": 6.0, "provider": "gemini", "cost_per_1k": 0.15},
                )

            # Parâmetro incompatível
            if "unsupported_parameter" in str(e) or "max_tokens" in str(e):
                logger.warning(f"[providers] Parâmetro incompatível para {model}, ajustando chave.")
                if "max_tokens" in params:
                    params.pop("max_tokens", None)
                    params["max_completion_tokens"] = max_tokens
                else:
                    params.pop("max_completion_tokens", None)
                    params["max_tokens"] = max_tokens
                sleep(1)
                continue

            # Tentativa falhou
            logger.error(f"[providers] Falha tentativa {attempt}/{retries} para modelo {model}: {e}")
            if attempt < retries:
                sleep(delay)
                continue
            raise

    logger.error(f"[providers] Todas as tentativas falharam para {model}.")
    return "[Erro: todas as tentativas falharam]", {"latency_s": 0.0, "provider": "remote"}
