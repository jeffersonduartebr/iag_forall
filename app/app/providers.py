"""
providers.py
----------------------------------------------------
Gerencia chamadas a modelos LLM (Ollama, OpenAI, Gemini, etc.)
Compatível com múltiplas versões do LiteLLM (novas e legadas).
Inclui métricas Prometheus e fallback seguro para erros de registro.
Implementa controle de uso exclusivo de modelo Ollama (um por vez).
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
import asyncio

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
# 🧠 Utilitários do Ollama — uso exclusivo de 1 modelo por vez
# ============================================================

def _ensure_ollama_model(model_name: str):
    """
    Garante que apenas um modelo Ollama esteja carregado por vez.
    Se houver outro modelo ativo, ele será descarregado antes.
    """
    if not model_name or not isinstance(model_name, str):
        logger.warning(f"[Ollama] Nome de modelo inválido: {model_name!r}")
        return

    try:
        # 1️⃣ Verifica modelos locais
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        models_local = []
        if resp.status_code == 200:
            models_local = [m["name"] for m in resp.json().get("models", [])]

        # 2️⃣ Verifica modelos atualmente carregados
        try:
            ps_resp = requests.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=5)
            running_models = []
            if ps_resp.status_code == 200:
                running_models = [m["model"] for m in ps_resp.json().get("models", [])]
        except Exception:
            running_models = []

        # 3️⃣ Descarrega modelos diferentes do solicitado
        for loaded in running_models:
            if loaded != model_name:
                try:
                    unload = requests.post(
                        f"{OLLAMA_BASE_URL}/api/unload",
                        json={"model": loaded},
                        timeout=10
                    )
                    if unload.status_code == 200:
                        logger.info(f"[Ollama] Modelo '{loaded}' descarregado da memória.")
                    else:
                        logger.warning(f"[Ollama] Falha ao descarregar '{loaded}': {unload.text[:80]}")
                except Exception as e:
                    logger.warning(f"[Ollama] Erro ao descarregar modelo '{loaded}': {e}")

        # 4️⃣ Baixa o modelo se necessário
        if model_name not in models_local:
            logger.info(f"[Ollama] Baixando modelo '{model_name}' ...")
            pull = requests.post(
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": model_name},
                timeout=600
            )
            if pull.status_code == 200:
                logger.info(f"[Ollama] Modelo '{model_name}' baixado com sucesso.")
            else:
                logger.warning(f"[Ollama] Falha ao baixar '{model_name}': {pull.text[:100]}")

        # 5️⃣ Aquecimento rápido (pré-carregamento leve)
        try:
            warmup = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model_name, "prompt": "ping"},
                timeout=30
            )
            if warmup.status_code == 200:
                logger.info(f"[Ollama] Modelo '{model_name}' pronto para uso.")
        except Exception as e:
            logger.warning(f"[Ollama] Falha no warmup de '{model_name}': {e}")

    except Exception as e:
        logger.error(f"[Ollama] Erro durante verificação/pull: {e}")

# ============================================================
# 🚀 Função principal de chamada a modelos (síncrona)
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

    if any(model.startswith(prefix) for prefix in EMBED_MODEL_PREFIXES):
        raise ValueError(f"[providers] Modelo '{model}' é de embedding e não suporta geração.")

    is_ollama = model.startswith("ollama/") or ":" in model
    model_clean = model.replace("ollama/", "")
    base_url = api_base if is_ollama else None

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
            if "isn't mapped yet" in str(e):
                MODEL_COST_USD.labels(model=model).observe(0.00015)
                return (
                    "[Aviso: modelo sem custo mapeado — custo estimado localmente.]",
                    {"usage": {}, "latency_s": 6.0, "provider": "gemini", "cost_per_1k": 0.15},
                )

            if "unsupported_parameter" in str(e) or "max_tokens" in str(e):
                if "max_tokens" in params:
                    params.pop("max_tokens", None)
                    params["max_completion_tokens"] = max_tokens
                else:
                    params.pop("max_completion_tokens", None)
                    params["max_tokens"] = max_tokens
                sleep(1)
                continue

            logger.error(f"[providers] Falha tentativa {attempt}/{retries} para modelo {model}: {e}")
            if attempt < retries:
                sleep(delay)
                continue
            raise

    logger.error(f"[providers] Todas as tentativas falharam para {model}.")
    return "[Erro: todas as tentativas falharam]", {"latency_s": 0.0, "provider": "remote"}

# ============================================================
# 🔄 Versão assíncrona para uso com await (juízes e RAG)
# ============================================================

async def async_call_model(
    model: str,
    prompt: str,
    temperature: float = 0.4,
    max_tokens: int = 1024,
    api_base: str = OLLAMA_BASE_URL,
):
    """
    Versão assíncrona de call_model().
    Executa a função síncrona em uma thread separada para não travar o event loop.
    """
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
