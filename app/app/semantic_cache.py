# -*- coding: utf-8 -*-
"""
semantic_cache.py — Cache Multimodal com TTL e Async Embeddings
---------------------------------------------------------------
"""

from __future__ import annotations
import base64
import hashlib
import logging
import asyncio
import numpy as np
from typing import Optional, Dict, Any
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from .settings_dynamic import settings
from .embeddings import embed_text, embed_image, embed_multimodal

logger = logging.getLogger(__name__)
if not logger.handlers: logging.basicConfig(level=logging.INFO)

DB_URL = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:3306/{settings.DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

CACHE_TTL_DAYS = int(settings.get("CACHE_TTL_DAYS", 7))

def _compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _image_hash(b64: Optional[str]) -> str:
    if not b64: return ""
    try:
        return hashlib.sha256(base64.b64decode(b64)).hexdigest()
    except: return ""

def _normalize_modality(mod: str) -> str:
    mod = mod.lower().strip()
    if mod in ("vision", "image"): return "vision"
    if mod == "multimodal": return "multimodal"
    return "text"

def _cosine(a, b) -> float:
    if a is None or b is None: return 0.0
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

@lru_cache(maxsize=512)
def _lru_get(key: str): return None
def _lru_store(key: str, val: Any): 
    try: _lru_get.cache_clear()
    except: pass

async def _make_embedding(query: str, modality: str, image_b64: Optional[str]):
    modality = _normalize_modality(modality)
    try:
        if modality == "text":
            return await asyncio.to_thread(embed_text, query)
        if modality == "vision":
            if not image_b64: return await asyncio.to_thread(embed_text, query)
            return await asyncio.to_thread(embed_image, image_b64)
        emb = await asyncio.to_thread(embed_multimodal, query, image_b64)
        return emb.get("multimodal") or emb.get("text")
    except Exception as e:
        logger.warning(f"[semantic_cache] Embed fail: {e}")
        return None

async def check_cache(query: str, modality: str="text", image_b64: str=None, threshold: float=0.90) -> Optional[Dict]:
    modality = _normalize_modality(modality)
    full_hash = f"{modality}:{_compute_sha256(query)}:{_image_hash(image_b64)}"

    if cached := _lru_get(full_hash):
        return cached

    q_emb = await _make_embedding(query, modality, image_b64)
    if q_emb is None: return None
    q_vec = np.array(q_emb, dtype=float)

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT id, query_text, answer, model_used, modality, embedding, image_output_b64
                    FROM semantic_cache
                    WHERE modality = :m AND created_at > NOW() - INTERVAL :ttl DAY
                    ORDER BY created_at DESC LIMIT 25
                """),
                {"m": modality, "ttl": CACHE_TTL_DAYS}
            ).mappings().all()
    except SQLAlchemyError:
        return None

    best, best_sim = None, 0.0
    for row in rows:
        try:
            sim = _cosine(q_vec, np.frombuffer(row["embedding"], dtype=float))
            if sim > best_sim: best, best_sim = row, sim
        except: continue

    if best and best_sim >= threshold:
        res = {
            "text": best["answer"], "similarity": float(best_sim),
            "model_used": best["model_used"], "image_output_b64": best["image_output_b64"]
        }
        _lru_store(full_hash, res)
        return res
    return None

async def store_cache(query: str, answer: str, modality: str="text", image_b64: str=None, model_used: str=None):
    modality = _normalize_modality(modality)
    full_hash = f"{modality}:{_compute_sha256(query)}:{_image_hash(image_b64)}"
    
    emb = await _make_embedding(query, modality, image_b64)
    if emb is None: return

    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO semantic_cache (query_hash, modality, query_text, answer, model_used, embedding, image_output_b64)
                    VALUES (:qh, :m, :qt, :ans, :mod, :emb, :img)
                    ON DUPLICATE KEY UPDATE answer=VALUES(answer), created_at=CURRENT_TIMESTAMP
                """),
                {"qh": full_hash, "m": modality, "qt": query, "ans": answer, 
                 "mod": model_used or "unknown", "emb": np.array(emb, dtype=float).tobytes(), "img": image_b64}
            )
    except SQLAlchemyError as e:
        logger.warning(f"[semantic_cache] Store fail: {e}")