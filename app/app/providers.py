# providers.py
# ------------------------------------------------------------
# Camada de abstração para chamadas a modelos LLM.
# Suporta múltiplos provedores: OpenAI, Anthropic, Gemini, Ollama.
# Integra-se com o NSGA-II, judges.py e query_log.
# ------------------------------------------------------------

import os
import time
import logging
from typing import Tuple, Dict

# Bibliotecas dos provedores
import openai
import anthropic
import google.generativeai as genai
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ============================================================
# 🔧 Configurações e credenciais
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ============================================================
# ⚙️ Estimativas auxiliares (custos e qualidade)
# ============================================================
def heuristic_quality_estimate(text: str) -> float:
    """Score heurístico de qualidade textual."""
    if not text or not text.strip():
        return 0.0
    length_factor = min(1.0, len(text) / 500)
    punctuation_bonus = 0.2 if any(p in text for p in ".!?") else 0
    return round(5.0 + 5.0 * length_factor + punctuation_bonus, 2)

def estimate_anthropic_cost(model: str, prompt_len: int, output_len: int) -> float:
    RATES = {
        "claude-3-opus": (0.015, 0.075),
        "claude-3-sonnet": (0.003, 0.015),
        "claude-3-haiku": (0.00025, 0.00125),
        "claude-3-5-sonnet": (0.005, 0.025),
    }
    in_rate, out_rate = RATES.get(model, (0.005, 0.025))
    return round((prompt_len / 1000 * in_rate) + (output_len / 1000 * out_rate), 6)

def estimate_openai_cost(model: str, prompt_len: int, output_len: int) -> float:
    RATES = {
        "gpt-4o": (0.005, 0.015),
        "gpt-4o-mini": (0.0005, 0.002),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-3.5-turbo": (0.001, 0.002),
    }
    in_rate, out_rate = RATES.get(model, (0.005, 0.015))
    return round((prompt_len / 1000 * in_rate) + (output_len / 1000 * out_rate), 6)

def estimate_gemini_cost(model: str, prompt_len: int, output_len: int) -> float:
    # custo aproximado baseado em tokens processados
    base_rate = 0.0005 if "flash" in model else 0.0015
    return round((prompt_len + output_len) / 1000 * base_rate, 6)

def estimate_ollama_cost(model: str, prompt_len: int, output_len: int) -> float:
    # custos locais geralmente desprezíveis, mas simulamos um custo simbólico
    return round((prompt_len + output_len) / 1000 * 0.0001, 6)

# ============================================================
# 🚀 Função principal
# ============================================================
def call_model(
    model: str,
    prompt: str,
    temperature: float = 0.5,
    max_tokens: int = 512,
) -> Tuple[str, Dict[str, float]]:
    """
    Envia prompt ao modelo especificado e retorna:
      - texto gerado
      - metadados padronizados (latência, custo, qualidade)
    """
    start_time = time.time()
    model_lower = model.lower()

    try:
        # ====================================================
        # 🧠 OpenAI
        # ====================================================
        if model_lower.startswith("openai/"):
            model_name = model.split("/", 1)[1]
            client = openai.OpenAI(api_key=OPENAI_API_KEY)

            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            text_out = completion.choices[0].message.content.strip()
            latency = time.time() - start_time
            cost = estimate_openai_cost(model_name, len(prompt), len(text_out))

            meta = {
                "latency": latency,
                "cost_per_1k": cost,
                "quality": heuristic_quality_estimate(text_out),
            }
            return text_out, meta

        # ====================================================
        # 🤖 Ollama (modelos locais)
        # ====================================================
        elif model_lower.startswith("ollama/"):
            model_name = model.split("/", 1)[1]
            payload = {"model": model_name, "prompt": prompt, "stream": False}

            resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()

            data = resp.json()
            text_out = data.get("response", "").strip()
            latency = time.time() - start_time
            cost = estimate_ollama_cost(model_name, len(prompt), len(text_out))

            meta = {
                "latency": latency,
                "cost_per_1k": cost,
                "quality": heuristic_quality_estimate(text_out),
            }
            return text_out, meta

        # ====================================================
        # 🌟 Google Gemini
        # ====================================================
        elif model_lower.startswith("gemini/"):
            model_name = model.split("/", 1)[1]
            model_ref = genai.GenerativeModel(model_name)
            response = model_ref.generate_content(prompt)
            text_out = response.text.strip() if hasattr(response, "text") else str(response)
            latency = time.time() - start_time
            cost = estimate_gemini_cost(model_name, len(prompt), len(text_out))

            meta = {
                "latency": latency,
                "cost_per_1k": cost,
                "quality": heuristic_quality_estimate(text_out),
            }
            return text_out, meta

        # ====================================================
        # 🦉 Anthropic (Claude)
        # ====================================================
        elif model_lower.startswith("anthropic/"):
            model_name = model.split("/", 1)[1]
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

            response = client.messages.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            text_out = response.content[0].text if response.content else ""
            latency = time.time() - start_time
            cost = estimate_anthropic_cost(model_name, len(prompt), len(text_out))

            meta = {
                "latency": latency,
                "cost_per_1k": cost,
                "quality": heuristic_quality_estimate(text_out),
            }
            return text_out, meta

        # ====================================================
        # 🧩 Desconhecido / Default
        # ====================================================
        else:
            logger.warning(f"[providers] Modelo desconhecido: {model}")
            latency = time.time() - start_time
            meta = {"latency": latency, "cost_per_1k": 0.0, "quality": 0.0}
            return f"[Modelo não suportado: {model}]", meta

    except Exception as e:
        logger.error(f"[providers] Falha ao chamar {model}: {e}")
        latency = time.time() - start_time
        meta = {"latency": latency, "cost_per_1k": 0.0, "quality": 0.0}
        return f"[Erro ao processar com {model}: {e}]", meta
# ============================================================

def _ensure_ollama_model(model_name: str):
    """
    Garante que o modelo local (Ollama) está disponível.
    Se não existir, tenta baixar com 'ollama pull'.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if model_name in models:
            logger.info(f"[ollama] Modelo '{model_name}' já disponível.")
            return True

        logger.info(f"[ollama] Baixando modelo '{model_name}' via API...")
        pull = requests.post(f"{base_url}/api/pull", json={"name": model_name}, timeout=600)
        pull.raise_for_status()
        logger.info(f"[ollama] Modelo '{model_name}' baixado com sucesso.")
        return True
    except Exception as e:
        logger.warning(f"[ollama] Falha ao garantir modelo '{model_name}': {e}")
        return False