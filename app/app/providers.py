# -*- coding: utf-8 -*-
"""
providers.py
------------------------------------------------------------
Camada de abstração para chamadas a modelos LLM.
- Suporta: OpenAI, Anthropic, Gemini, Ollama
- Exporta métricas Prometheus por modelo
- Fornece helper para servir /metrics diretamente
------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Tuple

import requests

# ========= Importações defensivas (evitam crash no import do módulo) =========
try:
    # Novo SDK OpenAI
    from openai import OpenAI as OpenAIClient  # type: ignore
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

# ========= Prometheus: tenta usar registry central, senão cria local =========
try:
    # Se você já tiver app/observability.py, vamos reaproveitar o registry
    from app.observability import registry as _shared_registry  # type: ignore
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        CollectorRegistry,
        generate_latest,
    )
    REGISTRY: CollectorRegistry = _shared_registry
except Exception:  # pragma: no cover
    from prometheus_client import (  # type: ignore
        Counter,
        Histogram,
        Gauge,
        CollectorRegistry,
        generate_latest,
    )

    REGISTRY = CollectorRegistry()

from prometheus_client.exposition import CONTENT_TYPE_LATEST

# ========= Logger =========
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ============================================================
# 🔧 Configurações e credenciais
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"))

# Configura Gemini se disponível
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as exc:  # pragma: no cover
        logger.warning("[providers] Falha ao configurar Gemini: %s", exc)

# ============================================================
# 📈 Métricas Prometheus (por modelo)
# ============================================================
PROV_REQ = Counter(
    "providers_model_requests_total",
    "Total de chamadas ao provedor por modelo",
    labelnames=["model"],
    registry=REGISTRY,
)
PROV_ERR = Counter(
    "providers_model_errors_total",
    "Total de erros ao chamar o provedor por modelo",
    labelnames=["model"],
    registry=REGISTRY,
)
PROV_OK = Counter(
    "providers_model_success_total",
    "Total de chamadas bem-sucedidas por modelo",
    labelnames=["model"],
    registry=REGISTRY,
)
PROV_LAT = Histogram(
    "providers_model_latency_seconds",
    "Latência observada por modelo (s)",
    labelnames=["model"],
    registry=REGISTRY,
)
PROV_COST = Histogram(
    "providers_model_cost_usd",
    "Custo estimado por chamada (USD) por modelo",
    labelnames=["model"],
    registry=REGISTRY,
)
PROV_QUAL = Histogram(
    "providers_model_quality_score",
    "Qualidade heurística (0–10) por modelo",
    labelnames=["model"],
    registry=REGISTRY,
)
PROV_LAST_TS = Gauge(
    "providers_model_last_call_timestamp",
    "Epoch da última chamada por modelo",
    labelnames=["model"],
    registry=REGISTRY,
)


def render_metrics_response() -> Tuple[bytes, str]:
    """
    Retorna (body_bytes, content_type) para ser usado num endpoint /metrics.
    Exemplo (FastAPI):
        from fastapi import Response
        from app.providers import render_metrics_response

        @app.get("/metrics")
        def metrics():
            body, ctype = render_metrics_response()
            return Response(content=body, media_type=ctype)
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


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
    # custo aproximado simbólico
    base_rate = 0.0005 if "flash" in model.lower() else 0.0015
    return round((prompt_len + output_len) / 1000 * base_rate, 6)


def estimate_ollama_cost(model: str, prompt_len: int, output_len: int) -> float:
    # custos locais geralmente desprezíveis; valor simbólico
    return round((prompt_len + output_len) / 1000 * 0.0001, 6)


# ============================================================
# 🚀 Função principal (com instrumentação de métricas)
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
    model_label = model  # label Prometheus

    # Contabiliza a requisição
    PROV_REQ.labels(model=model_label).inc()

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
                # Heurística:
                # - se SDK possui Responses API e o modelo é da família gpt-5/mini,
                #   use Responses com max_output_tokens
                if hasattr(client, "responses") and (
                    "gpt-5" in model_name.lower() or "mini" in model_name.lower() or "nano" in model_name.lower()
                ):
                    resp = client.responses.create(
                        model=model_name,
                        input=prompt,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                    # SDK novo tem .output_text — se não houver, cai no str(resp)
                    text_out = getattr(resp, "output_text", None) or str(resp)
                    text_out = text_out.strip()
                else:
                    # Chat Completions (modelos que aceitam max_tokens)
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    text_out = (completion.choices[0].message.content or "").strip()
            except Exception as exc_first:
                # Se falhou e Responses existe, tenta Responses como fallback
                if hasattr(client, "responses"):
                    try:
                        resp = client.responses.create(
                            model=model_name,
                            input=prompt,
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                        text_out = getattr(resp, "output_text", None) or str(resp)
                        text_out = text_out.strip()
                    except Exception as exc_second:
                        raise RuntimeError(f"OpenAI falhou: {exc_first} | {exc_second}") from exc_second
                else:
                    raise RuntimeError(f"OpenAI falhou: {exc_first}") from exc_first

            latency = time.time() - start_time
            cost = estimate_openai_cost(model_name, len(prompt), len(text_out))
            quality = heuristic_quality_estimate(text_out)

            # Métricas
            PROV_LAT.labels(model=model_label).observe(latency)
            PROV_COST.labels(model=model_label).observe(cost)
            PROV_QUAL.labels(model=model_label).observe(quality)
            PROV_OK.labels(model=model_label).inc()
            PROV_LAST_TS.labels(model=model_label).set(time.time())

            meta = {"latency": latency, "cost_per_1k": cost, "quality": quality}
            return text_out, meta

        # ====================================================
        # 🤖 Ollama (modelos locais)
        # ====================================================
        if model_lower.startswith("ollama/"):
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
            quality = heuristic_quality_estimate(text_out)

            PROV_LAT.labels(model=model_label).observe(latency)
            PROV_COST.labels(model=model_label).observe(cost)
            PROV_QUAL.labels(model=model_label).observe(quality)
            PROV_OK.labels(model=model_label).inc()
            PROV_LAST_TS.labels(model=model_label).set(time.time())

            meta = {"latency": latency, "cost_per_1k": cost, "quality": quality}
            return text_out, meta

        # ====================================================
        # 🌟 Google Gemini
        # ====================================================
        if model_lower.startswith("gemini/"):
            if not GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY não configurada.")
            if not genai:
                raise RuntimeError("SDK do Gemini (google-generativeai) não disponível.")

            model_name = model.split("/", 1)[1]
            model_ref = genai.GenerativeModel(model_name)
            response = model_ref.generate_content(prompt)

            text_out = getattr(response, "text", None) or str(response)
            text_out = text_out.strip()

            latency = time.time() - start_time
            cost = estimate_gemini_cost(model_name, len(prompt), len(text_out))
            quality = heuristic_quality_estimate(text_out)

            PROV_LAT.labels(model=model_label).observe(latency)
            PROV_COST.labels(model=model_label).observe(cost)
            PROV_QUAL.labels(model=model_label).observe(quality)
            PROV_OK.labels(model=model_label).inc()
            PROV_LAST_TS.labels(model=model_label).set(time.time())

            meta = {"latency": latency, "cost_per_1k": cost, "quality": quality}
            return text_out, meta

        # ====================================================
        # 🦉 Anthropic (Claude)
        # ====================================================
        if model_lower.startswith("anthropic/"):
            if not ANTHROPIC_API_KEY:
                raise RuntimeError("ANTHROPIC_API_KEY não configurada.")
            if not anthropic:
                raise RuntimeError("SDK Anthropic não disponível.")

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
                    part.text
                    for part in response.content
                    if hasattr(part, "text") and isinstance(part.text, str)
                ]
                text_out = "".join(parts).strip()

            latency = time.time() - start_time
            cost = estimate_anthropic_cost(model_name, len(prompt), len(text_out))
            quality = heuristic_quality_estimate(text_out)

            PROV_LAT.labels(model=model_label).observe(latency)
            PROV_COST.labels(model=model_label).observe(cost)
            PROV_QUAL.labels(model=model_label).observe(quality)
            PROV_OK.labels(model=model_label).inc()
            PROV_LAST_TS.labels(model=model_label).set(time.time())

            meta = {"latency": latency, "cost_per_1k": cost, "quality": quality}
            return text_out, meta

        # ====================================================
        # 🧩 Desconhecido / Default
        # ====================================================
        logger.warning("[providers] Modelo desconhecido: %s", model)
        latency = time.time() - start_time

        PROV_LAT.labels(model=model_label).observe(latency)
        PROV_ERR.labels(model=model_label).inc()
        PROV_LAST_TS.labels(model=model_label).set(time.time())

        meta = {"latency": latency, "cost_per_1k": 0.0, "quality": 0.0}
        return f"[Modelo não suportado: {model}]", meta

    except Exception as exc:
        # Telemetria de erro
        latency = time.time() - start_time
        PROV_LAT.labels(model=model_label).observe(latency)
        PROV_ERR.labels(model=model_label).inc()
        PROV_LAST_TS.labels(model=model_label).set(time.time())

        logger.error("[providers] Falha ao chamar %s: %s", model, exc)
        meta = {"latency": latency, "cost_per_1k": 0.0, "quality": 0.0}
        return f"[Erro ao processar com {model}: {exc}]", meta


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
            logger.info("[ollama] Modelo '%s' já disponível.", model_name)
            return True

        logger.info("[ollama] Baixando modelo '%s' via API...", model_name)
        pull = requests.post(f"{base_url}/api/pull", json={"name": model_name}, timeout=600)
        pull.raise_for_status()
        logger.info("[ollama] Modelo '%s' baixado com sucesso.", model_name)
        return True
    except Exception as exc:
        logger.warning("[ollama] Falha ao garantir modelo '%s': %s", model_name, exc)
        return False
