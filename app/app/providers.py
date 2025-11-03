# -*- coding: utf-8 -*-
# providers.py
# ------------------------------------------------------------
# Camada de abstração para chamadas a modelos LLM.
# Suporta múltiplos provedores: OpenAI, Anthropic, Gemini, Ollama.
# Integra-se com o NSGA-II, judges.py e query_log.
# ------------------------------------------------------------

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Tuple
import requests

# ========= Importações defensivas =========
try:
    from openai import OpenAI as OpenAIClient  # Novo SDK
except Exception:  # pragma: no cover
    OpenAIClient = None  # type: ignore

try:
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ============================================================
# 🔧 Configurações e credenciais
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"))

if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.warning(f"[providers] Falha ao configurar Gemini: {e}")

# ============================================================
# ⚙️ Estimativas auxiliares (custos e qualidade)
# ============================================================
def heuristic_quality_estimate(text: str) -> float:
    """Score heurístico de qualidade textual (0–10)."""
    if not text or not text.strip():
        return 0.0
    length_factor = min(1.0, len(text) / 500)
    punctuation_bonus = 0.2 if any(p in text for p in ".!?") else 0.0
    return round(5.0 + 5.0 * length_factor + punctuation_bonus, 2)


def estimate_anthropic_cost(model: str, prompt_len: int, output_len: int) -> float:
    rates = {
        "claude-3-opus": (0.015, 0.075),
        "claude-3-sonnet": (0.003, 0.015),
        "claude-3-haiku": (0.00025, 0.00125),
        "claude-3-5-sonnet": (0.005, 0.025),
    }
    in_rate, out_rate = rates.get(model, (0.005, 0.025))
    return round((prompt_len / 1000 * in_rate) + (output_len / 1000 * out_rate), 6)


def estimate_openai_cost(model: str, prompt_len: int, output_len: int) -> float:
    rates = {
        "gpt-5": (0.02, 0.06),
        "gpt-5-mini": (0.002, 0.006),
        "gpt-4o": (0.005, 0.015),
        "gpt-4o-mini": (0.0005, 0.002),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-3.5-turbo": (0.001, 0.002),
    }
    in_rate, out_rate = rates.get(model, (0.005, 0.015))
    return round((prompt_len / 1000 * in_rate) + (output_len / 1000 * out_rate), 6)


def estimate_gemini_cost(model: str, prompt_len: int, output_len: int) -> float:
    base_rate = 0.0005 if "flash" in model.lower() else 0.0015
    return round((prompt_len + output_len) / 1000 * base_rate, 6)


def estimate_ollama_cost(model: str, prompt_len: int, output_len: int) -> float:
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
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY não configurada.")
            if not OpenAIClient:
                raise RuntimeError("SDK OpenAI não disponível no ambiente.")

            model_name = model.split("/", 1)[1]
            client = OpenAIClient(api_key=OPENAI_API_KEY)
            text_out = ""

            try:
                # Modelos GPT-5 e Mini → usam max_completion_tokens
                if "gpt-5" in model_name or "mini" in model_name:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                    )
                else:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                text_out = (completion.choices[0].message.content or "").strip()
            except Exception as e_first:
                # fallback: tenta API responses
                try:
                    resp = client.responses.create(
                        model=model_name,
                        input=prompt,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                    if hasattr(resp, "output") and resp.output:
                        text_out = "".join(
                            chunk.text for chunk in resp.output[0].content
                            if getattr(chunk, "type", "") == "output_text"
                        ).strip()
                    else:
                        text_out = str(resp)
                except Exception as e_second:
                    raise RuntimeError(f"OpenAI falhou: {e_first} | {e_second}") from e_second

            latency = time.time() - start_time
            cost = estimate_openai_cost(model_name, len(prompt), len(text_out))
            meta = {
                "latency": latency,
                "cost_per_1k": cost,
                "quality": heuristic_quality_estimate(text_out),
            }
            return text_out, meta

        # ====================================================
        # 🤖 Ollama
        # ====================================================
        elif model_lower.startswith("ollama/"):
            model_name = model.split("/", 1)[1]
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            }
            resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text_out = (data.get("response") or "").strip()
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
            if not GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY não configurada.")
            if not genai:
                raise RuntimeError("SDK do Gemini (google-generativeai) não disponível no ambiente.")

            model_name = model.split("/", 1)[1]
            model_ref = genai.GenerativeModel(model_name)
            response = model_ref.generate_content(prompt)
            text_out = getattr(response, "text", None) or str(response)
            text_out = text_out.strip()
            latency = time.time() - start_time
            cost = estimate_gemini_cost(model_name, len(prompt), len(text_out))
            meta = {
                "latency": latency,
                "cost_per_1k": cost,
                "quality": heuristic_quality_estimate(text_out),
            }
            return text_out, meta

        # ====================================================
        # 🦉 Anthropic
        # ====================================================
        elif model_lower.startswith("anthropic/"):
            if not ANTHROPIC_API_KEY:
                raise RuntimeError("ANTHROPIC_API_KEY não configurada.")
            if not anthropic:
                raise RuntimeError("SDK Anthropic não disponível no ambiente.")

            model_name = model.split("/", 1)[1]
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            text_out = ""
            if getattr(response, "content", None):
                parts = [
                    part.text for part in response.content
                    if hasattr(part, "text") and isinstance(part.text, str)
                ]
                text_out = "".join(parts).strip()

            latency = time.time() - start_time
            cost = estimate_anthropic_cost(model_name, len(prompt), len(text_out))
            meta = {
                "latency": latency,
                "cost_per_1k": cost,
                "quality": heuristic_quality_estimate(text_out),
            }
            return text_out, meta

        # ====================================================
        # 🧩 Modelo desconhecido
        # ====================================================
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
# 🔐 Garantia de modelo Ollama
# ============================================================
def _ensure_ollama_model(model_name: str) -> bool:
    """
    Garante que o modelo local (Ollama) está disponível.
    Se não existir, tenta baixar com a API /api/pull.
    """
    base_url = OLLAMA_HOST or "http://ollama:11434"
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        tags_payload = resp.json() or {}
        models = [m.get("name", "") for m in tags_payload.get("models", [])]
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
