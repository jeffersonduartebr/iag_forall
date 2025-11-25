# -*- coding: utf-8 -*-
"""
embeddings.py — versão MULTIMODAL e Inteligente
------------------------------------------------------------
Oferece:
- Embeddings de texto (Ollama ou OpenAI)
- Embeddings de visão (Ollama)
- Cache em Redis
- Roteamento inteligente baseado no nome do modelo

Correção 404: Evita enviar modelos locais (ex: all-minilm) para a API da OpenAI.
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import logging
from typing import List, Dict, Optional, Any

import numpy as np
import requests
import httpx 

# OpenAI SDK opcional
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

from app.settings_dynamic import settings
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] embeddings: %(message)s"
    )


# ============================================================
# Configs
# ============================================================

# Lê configurações dinâmicas
EMBED_PROVIDER = settings.get("EMBED_PROVIDER", "ollama").lower()
EMBED_MODEL_TEXT = settings.get("EMBED_TEXT_MODEL", "all-minilm:latest")
EMBED_MODEL_VISION = settings.get("EMBED_VISION_MODEL", "llava:7b")

OLLAMA_HOST = settings.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "http://ollama:11434"))
OPENAI_API_KEY = settings.get("OPENAI_API_KEY", "")

EMBED_CACHE_ENABLED = str(settings.get("EMBED_CACHE_ENABLED", "true")).lower() in ("1", "true", "yes")
EMBED_CACHE_TTL_S = int(settings.get("EMBED_CACHE_TTL_S", 86400 * 7)) # 7 dias

_rds = get_redis() if EMBED_CACHE_ENABLED else None
REDIS_KEY_PREFIX = "emb:v3"


# ============================================================
# Utils
# ============================================================

def _hash_text(text: str, model: str, modality: str) -> str:
    payload = f"{model}|{modality}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


def _norm(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec if n == 0 else vec / n


def _save_cache(key: str, payload: Dict[str, Any]):
    if not _rds: return
    try:
        _rds.setex(key, EMBED_CACHE_TTL_S, json.dumps(payload))
    except Exception: pass


def _load_cache(key: str) -> Optional[Dict[str, Any]]:
    if not _rds: return None
    try:
        raw = _rds.get(key)
        return json.loads(raw) if raw else None
    except Exception: return None


# ============================================================
# OLLAMA
# ============================================================

def _ollama_embed_text(text: str, model: str) -> List[float]:
    """Gera embedding via Ollama API."""
    url = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"
    
    # Remove prefixo se existir (ex: ollama/all-minilm -> all-minilm)
    real_model = model.split("/", 1)[1] if "/" in model else model

    payload = {"model": real_model, "prompt": text}
    
    # Tenta até 3 vezes (útil se o Ollama estiver carregando o modelo)
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=60)
            if r.status_code == 404:
                # Modelo não encontrado no Ollama? Tenta baixar ou lançar erro específico
                raise ValueError(f"Modelo '{real_model}' não encontrado no Ollama (404).")
            
            r.raise_for_status()
            data = r.json() or {}
            vec = data.get("embedding")
            
            if not vec:
                raise RuntimeError("Resposta vazia do Ollama.")
            return [float(x) for x in vec]
            
        except ValueError as e:
            # Erro fatal (modelo não existe), não adianta tentar de novo
            raise e
        except Exception as e:
            if attempt == 2: raise
            time.sleep(1)
    
    return []


def _ollama_embed_image(image_b64: str, prompt: str = "") -> List[float]:
    """
    Usa endpoint de generate do Ollama para extrair features visuais (hacky),
    ou endpoint de embeddings se o modelo suportar (ex: nomic-embed-vision).
    Por enquanto, focamos no endpoint embeddings padrão.
    """
    url = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"
    # Nota: Nem todos os modelos Ollama suportam imagem no endpoint embeddings.
    # Modelos multimodais como llava geralmente suportam.
    payload = {
        "model": EMBED_MODEL_VISION,
        "prompt": prompt or "Describe image",
        "images": [image_b64],
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("embedding", [])


# ============================================================
# OPENAI
# ============================================================

def _openai_embed_text(text: str, model: str) -> List[float]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY ausente.")
    
    client = OpenAIClient(api_key=OPENAI_API_KEY)
    # Normaliza nome do modelo
    real_model = model.split("/", 1)[1] if "/" in model else model
    
    # Fallback se o nome for inválido para OpenAI
    if "minilm" in real_model or "llama" in real_model:
        real_model = "text-embedding-3-small"

    resp = client.embeddings.create(model=real_model, input=text)
    return resp.data[0].embedding


# ============================================================
# API PÚBLICA
# ============================================================

def embed_text(text: str) -> List[float]:
    """
    Gera embedding de texto.
    Decide automaticamente entre Ollama e OpenAI baseado no nome do modelo.
    """
    text = (text or "").strip()
    if not text: return [0.0]

    # Determina qual modelo usar
    model_name = EMBED_MODEL_TEXT
    
    # Verifica Cache
    key = f"{REDIS_KEY_PREFIX}:text:{model_name}:{_hash_text(text, model_name, 'text')}"
    if cached := _load_cache(key):
        return cached.get("v", [0.0])

    vec = []
    
    # --- ROTEAMENTO DE EMBEDDING ---
    # Se o nome do modelo sugere OpenAI
    if model_name.startswith("text-embedding") or "openai" in model_name or model_name.startswith("gpt-"):
        try:
            vec = _openai_embed_text(text, model_name)
        except Exception as e:
            logger.error(f"Falha OpenAI embedding: {e}")
            # Não faz fallback para Ollama para evitar misturar espaços vetoriais
            return [0.0]
    
    # Caso contrário, assume Ollama (padrão para all-minilm, nomic, etc)
    else:
        try:
            vec = _ollama_embed_text(text, model_name)
        except Exception as e:
            logger.error(f"Falha Ollama embedding ({model_name}): {e}")
            # Se falhar o local, aqui sim poderíamos tentar um fallback Cloud se configurado,
            # mas misturar vetores quebra o RAG. Melhor retornar erro/zero.
            return [0.0]

    if vec:
        _save_cache(key, {"v": vec})
    
    return vec


def embed_image(image_b64: str, prompt: str = "") -> List[float]:
    image_b64 = (image_b64 or "").strip()
    if not image_b64: return [0.0]

    key = f"{REDIS_KEY_PREFIX}:vision:{EMBED_MODEL_VISION}:{_hash_text(image_b64, EMBED_MODEL_VISION, 'vision')}"
    if cached := _load_cache(key):
        return cached.get("v", [0.0])

    try:
        vec = _ollama_embed_image(image_b64, prompt)
        if vec:
            _save_cache(key, {"v": vec})
        return vec
    except Exception as e:
        logger.error(f"Falha Vision embedding: {e}")
        return [0.0]


def embed_multimodal(text: str, image_b64: Optional[str]) -> Dict[str, List[float]]:
    """
    Retorna { "text": [...], "vision": [...], "multimodal": [...] }
    Combina vetores normalizados.
    """
    text_vec = np.array(embed_text(text))
    
    if not image_b64:
        return {
            "text": text_vec.tolist(),
            "vision": None,
            "multimodal": _norm(text_vec).tolist()
        }

    vis_vec = np.array(embed_image(image_b64))
    
    # Concatenação simples com normalização (late fusion)
    combined = np.concatenate([_norm(text_vec), _norm(vis_vec)])
    
    return {
        "text": text_vec.tolist(),
        "vision": vis_vec.tolist(),
        "multimodal": _norm(combined).tolist()
    }