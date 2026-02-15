# -*- coding: utf-8 -*-
"""
providers.py — versão multimodal + UM-RAG compatível (VALIDAÇÃO DE PARÂMETROS)
------------------------------------------------------------
Suporta:
- OpenAI (texto, visão, multimodal via Chat Completions API)
- Anthropic Claude 3.x (texto, visão)
- Google Gemini 1.5 (texto, visão)
- Ollama (texto, visão)

Atualizações recentes:
- ✅ Validação dinâmica de parâmetros para OpenAI:
  - Detecta modelos 'o1-*' e 'gpt-5-*'.
  - Troca 'max_tokens' por 'max_completion_tokens' automaticamente.
  - Remove 'temperature' para modelos 'o1' (que não suportam sampling customizado).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Tuple, Optional, Any
import base64

import requests

# ============ SDKs (import defensivo) ======================

try:
    from openai import OpenAI as OpenAIClient
except Exception:  # pragma: no cover
    OpenAIClient = None

try:
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None

# ============ Observabilidade ==============================

try:
    from app.observability import registry as _shared_registry
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        CollectorRegistry,
        generate_latest,
    )
    REGISTRY = _shared_registry
except Exception:  # fallback independente
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        CollectorRegistry,
        generate_latest,
    )
    REGISTRY = CollectorRegistry()

from prometheus_client.exposition import CONTENT_TYPE_LATEST

# ============ Logger =======================================

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] providers: %(message)s")

# ============ Credenciais / Host ===========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") or os.getenv("OPENAI_APIKEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

if genai and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ============================================================
#  PROMETHEUS METRICS
# ============================================================

PROV_REQ = Counter(
    "providers_model_requests_total",
    "Total de chamadas ao provedor por modelo",
    ["model"],
    registry=REGISTRY,
)
PROV_ERR = Counter(
    "providers_model_errors_total",
    "Total de erros por modelo",
    ["model"],
    registry=REGISTRY,
)
PROV_OK = Counter(
    "providers_model_success_total",
    "Chamadas OK por modelo",
    ["model"],
    registry=REGISTRY,
)
PROV_LAT = Histogram(
    "providers_latency_seconds",
    "Latência por modelo",
    ["model"],
    registry=REGISTRY,
)
PROV_COST = Histogram(
    "providers_cost_usd",
    "Custo por chamada",
    ["model"],
    registry=REGISTRY,
)
PROV_LAST_TS = Gauge(
    "providers_last_call_timestamp",
    "Epoch timestamp da última chamada",
    ["model"],
    registry=REGISTRY,
)

# ============================================================
#  AUXILIARES
# ============================================================

def heuristic_quality_estimate(text: str) -> float:
    """
    Estimativa simples de "qualidade" (0–10) baseada em:
    - tamanho do texto
    - presença de pontuação final (., !, ?)
    Não substitui os JUDGES, apenas fornece um fallback leve.
    """
    if not text or not text.strip():
        return 0.0

    length = len(text)
    # até ~400 chars cresce linearmente, depois satura
    length_factor = min(1.0, length / 400.0)
    punctuation_bonus = 0.3 if any(p in text for p in [".", "!", "?"]) else 0.0

    score = 4.0 + 5.5 * length_factor + punctuation_bonus
    return max(0.0, min(10.0, round(score, 2)))


def _estimate_tokens(text: str) -> int:
    """
    Aproximação bem grosseira: ~4 chars por token.
    Serve apenas para custo estimado quando o provedor não retorna contagem oficial.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _encode_image_vision(image_b64: str) -> Dict[str, Any]:
    """
    Helper genérico para payloads multimodais que usam URL base64.
    """
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_b64}"
        }
    }


# ============================================================
#  CALL MODEL — SUPORTE COMPLETO MULTIMODAL
# ============================================================

def call_model(
    model: str,
    prompt: str,
    modality: str = "text",
    image_b64: Optional[str] = None,
    temperature: float = 0.5,
    max_tokens: int = 512,
) -> Tuple[str, Dict[str, Any]]:
    """
    Chamada unificada para todos os provedores.

    Parâmetros:
        model: string com namespace ("openai/gpt-4o", "gemini/...", "anthropic/...", "ollama/...")
        prompt: texto já com system/user formatado pelo router_core
        modality: "text", "vision" ou "multimodal"
        image_b64: imagem em base64 (opcional)
        temperature, max_tokens: parâmetros padrão de geração

    Retorno:
        text_out: string com output textual principal
        meta: dict com metadados (latência, custo, tokens, etc)
    """
    start = time.time()
    PROV_REQ.labels(model=model).inc()

    # defaults meta
    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = 0

    try:
        # ====================================================
        # OPENAI (Chat Completions API — multimodal)
        # ====================================================
        if model.lower().startswith("openai/"):
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY ausente.")

            model_name = model.split("/", 1)[1]
            if OpenAIClient is None:
                raise RuntimeError("SDK OpenAI não disponível no ambiente.")

            client = OpenAIClient(api_key=OPENAI_API_KEY)

            # Conteúdo multimodal: texto + imagem (se houver)
            content: list[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            if image_b64 and modality in ("vision", "multimodal"):
                content.append(_encode_image_vision(image_b64))

            # --- VALIDAÇÃO DINÂMICA DE PARÂMETROS ---
            # Modelos recentes (o1, gpt-5) exigem 'max_completion_tokens' e têm regras de temperatura
            
            api_kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": content}],
            }

            # Heurística para detectar modelos "Reasoning" ou Nova Geração
            is_o1 = model_name.startswith("o1-")
            is_next_gen = "gpt-5" in model_name or is_o1

            if is_next_gen:
                # Usa o novo nome do parâmetro
                api_kwargs["max_completion_tokens"] = max_tokens
            else:
                # Usa o nome antigo para gpt-4o, gpt-4, gpt-3.5
                api_kwargs["max_tokens"] = max_tokens

            # Tratamento de Temperatura
            if is_o1:
                # Modelos o1 (preview/mini) atualmente rejeitam temperature != 1 (ou o parâmetro todo)
                # O mais seguro é não enviar temperature para o1
                pass 
            else:
                api_kwargs["temperature"] = temperature

            # Chamada da API
            resp = client.chat.completions.create(**api_kwargs)

            # Extração padrão do ChatCompletion
            text_out = resp.choices[0].message.content or ""
            text_out = text_out.strip()

            # Uso de tokens real
            usage = getattr(resp, "usage", None)
            if usage:
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
            else:
                completion_tokens = _estimate_tokens(text_out)

            latency = time.time() - start

            # custo simples: tokens * preço unitário aproximado
            total_tokens = max(1, prompt_tokens + completion_tokens)
            cost = total_tokens * 0.000005  # Exemplo genérico

            quality = heuristic_quality_estimate(text_out)
            
            # Tratamento seguro do payload para o log
            raw_payload = None
            try:
                raw_payload = resp.model_dump() if hasattr(resp, "model_dump") else str(resp)
            except Exception:
                raw_payload = str(resp)

            meta = {
                "latency": latency,
                "cost_per_1k": cost,
                "quality": quality,
                "raw_payload": raw_payload,
                "image_output_b64": None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

            PROV_OK.labels(model=model).inc()
            PROV_LAT.labels(model=model).observe(latency)
            PROV_COST.labels(model=model).observe(cost)
            PROV_LAST_TS.labels(model=model).set(time.time())

            return text_out, meta

        # ====================================================
        # GEMINI (Google Generative AI) — texto/visão
        # ====================================================
        if model.lower().startswith("gemini/"):
            if not GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY ausente.")
            if genai is None:
                raise RuntimeError("SDK Google GenerativeAI não disponível.")

            model_name = model.split("/", 1)[1]
            gmodel = genai.GenerativeModel(model_name)

            parts: list[Any] = [prompt]
            if image_b64 and modality in ("vision", "multimodal"):
                parts.append({
                    "mime_type": "image/jpeg",
                    "data": base64.b64decode(image_b64),
                })

            resp = gmodel.generate_content(
                parts,
                generation_config={
                    "temperature": float(temperature),
                    "max_output_tokens": int(max_tokens),
                }
            )

            text_out = resp.text or ""
            latency = time.time() - start
            completion_tokens = _estimate_tokens(text_out)

            total_tokens = prompt_tokens + completion_tokens
            cost = total_tokens * 0.000004

            quality = heuristic_quality_estimate(text_out)

            meta = {
                "latency": latency,
                "quality": quality,
                "cost_per_1k": cost,
                "raw_payload": str(resp), # Gemini object complexo para serializar
                "image_output_b64": None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

            PROV_OK.labels(model=model).inc()
            PROV_LAT.labels(model=model).observe(latency)
            PROV_COST.labels(model=model).observe(cost)
            PROV_LAST_TS.labels(model=model).set(time.time())

            return text_out, meta

        # ====================================================
        # ANTHROPIC (Claude 3.x) — texto/visão
        # ====================================================
        if model.lower().startswith("anthropic/"):
            if not ANTHROPIC_API_KEY:
                raise RuntimeError("ANTHROPIC_API_KEY ausente.")
            if anthropic is None:
                raise RuntimeError("SDK Anthropic não disponível.")

            model_name = model.split("/", 1)[1]
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

            parts: list[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            if image_b64 and modality in ("vision", "multimodal"):
                parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                })

            resp = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": parts}],
            )

            text_out = ""
            for p in getattr(resp, "content", []):
                if hasattr(p, "text"):
                    text_out += getattr(p, "text", "")
                elif isinstance(p, dict) and p.get("type") == "text":
                    text_out += p.get("text", "")

            text_out = (text_out or "").strip()
            latency = time.time() - start
            
            # Anthropic fornece usage
            usage = getattr(resp, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "input_tokens", prompt_tokens)
                completion_tokens = getattr(usage, "output_tokens", completion_tokens)
            else:
                completion_tokens = _estimate_tokens(text_out)

            total_tokens = prompt_tokens + completion_tokens
            cost = total_tokens * 0.000007 

            quality = heuristic_quality_estimate(text_out)

            meta = {
                "latency": latency,
                "quality": quality,
                "cost_per_1k": cost,
                "raw_payload": str(resp),
                "image_output_b64": None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

            PROV_OK.labels(model=model).inc()
            PROV_LAT.labels(model=model).observe(latency)
            PROV_COST.labels(model=model).observe(cost)
            PROV_LAST_TS.labels(model=model).set(time.time())

            return text_out, meta

        # ====================================================
        # OLLAMA (local) — texto / visão
        # ====================================================
        if model.lower().startswith("ollama/"):

            model_name = model.split("/", 1)[1]

            # payload básico; Ollama trata imagem via campo "images"
            payload: Dict[str, Any] = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": float(temperature),
                    "num_predict": int(max_tokens),
                },
            }

            if image_b64 and modality in ("vision", "multimodal"):
                # ollama espera lista de strings base64
                payload["images"] = [image_b64]

            r = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=240,
            )
            r.raise_for_status()
            data = r.json()

            text_out = data.get("response", "") or ""
            text_out = text_out.strip()
            latency = time.time() - start

            # Ollama retorna eval_count (tokens de saída) e prompt_eval_count (tokens de entrada)
            prompt_tokens = data.get("prompt_eval_count", prompt_tokens)
            completion_tokens = data.get("eval_count", _estimate_tokens(text_out))
            
            total_tokens = prompt_tokens + completion_tokens
            cost = total_tokens * 0.0000001 # Custo elétrico simbólico

            quality = heuristic_quality_estimate(text_out)
            image_out = None
            
            # Verificação defensiva se o Ollama retornar imagem (futuro)
            if isinstance(data.get("images"), list) and data["images"]:
                image_out = data["images"][0]

            meta = {
                "latency": latency,
                "quality": quality,
                "cost_per_1k": cost,
                "raw_payload": data,
                "image_output_b64": image_out,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

            PROV_OK.labels(model=model).inc()
            PROV_LAT.labels(model=model).observe(latency)
            PROV_COST.labels(model=model).observe(cost)
            PROV_LAST_TS.labels(model=model).set(time.time())

            return text_out, meta

        # ====================================================
        # Fallback — modelo não reconhecido
        # ====================================================
        raise RuntimeError(f"Modelo não suportado ou namespace inválido: {model}")

    except Exception as exc:
        latency = time.time() - start
        PROV_ERR.labels(model=model).inc()
        PROV_LAT.labels(model=model).observe(latency)

        logger.error(f"[providers] Erro chamando {model}: {exc}")

        meta_err = {
            "latency": latency,
            "cost_per_1k": 0.0,
            "quality": 0.0,
            "raw_payload": str(exc),
            "image_output_b64": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

        return f"[Erro ao chamar {model}: {exc}]", meta_err


# ============================================================
# GARANTIA OLLAMA
# ============================================================

def _ensure_ollama_model(model_name: str) -> bool:
    """
    Garante que o modelo Ollama esteja presente:
    - Se já existir em /api/tags → OK
    - Se não existir → tenta baixar via /api/pull
    """
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
        tags = resp.json()
        names = [x.get("name") for x in tags.get("models", []) if isinstance(x, dict)]

        # Aceita tanto "gemma3:4b" quanto "gemma3:4b-latest"
        if any(model_name in n for n in names):
            return True

        logger.info(f"[ensure_ollama_model] Baixando '{model_name}'...")
        pull = requests.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"name": model_name},
            timeout=600, # 10 min timeout para download
        )
        pull.raise_for_status()
        logger.info(f"[ensure_ollama_model] Modelo '{model_name}' baixado com sucesso.")
        return True

    except Exception as e:
        logger.warning(f"[ensure_ollama_model] Falha ao garantir modelo '{model_name}': {e}")
        return False