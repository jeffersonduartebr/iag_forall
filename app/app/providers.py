"""
providers.py
----------------------------------------------------
Chamadas a modelos LLM (Ollama, OpenAI, Gemini, etc.)
- Usa DynamicSettings (Redis+DB+.env)
- Exporte _ensure_ollama_model, call_model() e async_call_model()
"""

import logging
import requests
import re
import os
from time import sleep
from prometheus_client import Histogram
import litellm
from litellm import completion
import asyncio

from .settings_dynamic import settings

logger = logging.getLogger(__name__)

# ============================================================
# ⚙️ Configurações globais
# ============================================================
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL

# ------------------------------------------------------------
# 🔹 Registro seguro do modelo Gemini (com fallback)
# ------------------------------------------------------------
try:
    gemini_cost = 0.00015  # $0.15 / 1k tokens

    if hasattr(litellm, "register_model"):
        try:
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
# 🚀 Função principal de chamada a modelos (síncrona)
# ============================================================
def call_model(
    model: str,
    prompt: str,
    temperature: float = None,
    max_tokens: int = None,
    api_base: str = None,
):
    """
    Chama um modelo com validação, fallback e métricas.
    Compatível com GPT-5 / GPT-4o-mini / GPT-5-nano / Ollama / Gemini.
    """
    if temperature is None:
        temperature = settings.TEMPERATURE_DEFAULT
    if max_tokens is None:
        max_tokens = settings.MAX_TOKENS_DEFAULT
    if api_base is None:
        api_base = settings.OLLAMA_BASE_URL

    if not model or not isinstance(model, str):
        raise ValueError(f"[providers] Modelo inválido: {model!r}")

    if any(model.startswith(prefix) for prefix in EMBED_MODEL_PREFIXES):
        raise ValueError(f"[providers] Modelo '{model}' é de embedding e não suporta geração.")

    is_ollama = model.startswith("ollama/") or ":" in model
    model_clean = model.replace("ollama/", "")
    base_url = api_base if is_ollama else None

    use_new_param = bool(re.search(r"gpt-5|gpt-4o-mini|nano|gpt-5-mini", model))
    params = {
        "model": f"ollama/{model_clean}" if is_ollama else model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1 if use_new_param else float(temperature),
        "api_base": base_url,
    }

    if use_new_param:
        params["max_completion_tokens"] = int(max_tokens)
    else:
        params["max_tokens"] = int(max_tokens)

    retries = 3
    delay = 2
    for attempt in range(1, retries + 1):
        try:
            resp = completion(**params)
            text = resp["choices"][0]["message"]["content"] if "choices" in resp else "[Sem resposta]"
            meta = {
                "usage": resp.get("usage", {}),
                "latency_s": resp.get("latency", 0.0),
                "provider": "ollama" if is_ollama else "remote",
            }

            if "usage" in meta and "total_tokens" in meta["usage"]:
                MODEL_COST_USD.labels(model=model).observe(
                    meta["usage"]["total_tokens"] / 1000.0 * 0.001
                )
            return text, meta

        except Exception as e:
            if "isn't mapped yet" in str(e):
                MODEL_COST_USD.labels(model=model).observe(0.00015)
                return (
                    "[Aviso: modelo sem custo mapeado — custo estimado localmente.]",
                    {"usage": {}, "latency_s": 6.0, "provider": "gemini", "cost_per_1k": 0.15},
                )

            if "unsupported_parameter" in str(e) or "max_tokens" in str(e):
                if "max_tokens" in params:
                    params.pop("max_tokens", None)
                    params["max_completion_tokens"] = int(max_tokens)
                else:
                    params.pop("max_completion_tokens", None)
                    params["max_tokens"] = int(max_tokens)
                sleep(1)
                continue

            logger.error(f"[providers] Falha tentativa {attempt}/{retries} para modelo {model}: {e}")
            if attempt < retries:
                sleep(delay)
                continue
            raise

    return "[Erro: todas as tentativas falharam]", {"latency_s": 0.0, "provider": "remote"}

# ============================================================
# 🔄 Versão assíncrona (para uso em funções async)
# ============================================================
async def async_call_model(
    model: str,
    prompt: str,
    temperature: float = None,
    max_tokens: int = None,
    api_base: str = None,
):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: call_model(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
        ),
    )
