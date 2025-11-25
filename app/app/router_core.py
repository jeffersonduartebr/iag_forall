# -*- coding: utf-8 -*-
"""
router_core.py — Multimodal + UM-RAG + Meta-bandit + Background Tasks (BLINDADO)
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import time
import asyncio
import threading
from typing import Dict, Tuple, Any, Optional

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# --- Módulos Internos ---
from .providers_async import call_model, _ensure_ollama_model
from .router_strategy import choose_top2_models
from .semantic_cache import check_cache, store_cache
from .settings_dynamic import settings
from .bandits import select_model, bandit_update, compute_reward
from .judges import judge_answer
from .metrics_collector import update_model_metrics
from .rag_local import build_augmented_prompt
from .embeddings import embed_text
from .query_service import insert_query_log, ensure_query_log

# --- Novos Módulos de Precisão ---
from .utils.pricing import get_model_cost

from .observability import (
    ROUTER_MODEL_COST,
    ROUTER_QUALITY_AVG,
    ROUTER_COST_SAVINGS,
    ROUTER_LOCAL_USAGE_RATIO,
    ROUTER_COST_PER_QUERY,
    ROUTER_HISTORY_ENTRIES,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] router: %(message)s",
    )

# ============================================================
# DB / Engine
# ============================================================

DB_URL = (
    f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}"
    f"@{settings.DB_HOST}:3306/{settings.DB_NAME}"
)
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

BLOCKED_PREFIXES = ("nomic-embed", "text-embedding", "bge-", "e5-")
EMA_HISTORY: Dict[Tuple[str, str], Dict[str, Any]] = {}
LOG_RETENTION_DAYS = int(settings.get("QUERY_LOG_RETENTION_DAYS", 7))


# ============================================================
# EMA & Manutenção (Background)
# ============================================================

def _load_ema_from_db() -> None:
    global EMA_HISTORY
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT modality, model, ema_latency, ema_quality, ema_cost, ema_alignment FROM ema_history")
            ).mappings().all()

        EMA_HISTORY = {}
        for r in rows:
            key = (r["modality"], r["model"])
            EMA_HISTORY[key] = {
                "ema_latency": float(r["ema_latency"]),
                "ema_quality": float(r["ema_quality"]),
                "ema_cost": float(r["ema_cost"]),
                "ema_alignment": float(r["ema_alignment"]),
                "updates": 0,
            }
        logger.info(f"[EMA] Carregado: {len(EMA_HISTORY)} modelos.")
    except Exception:
        logger.warning("[EMA] Banco vazio ou erro ao carregar histórico.")

def _persist_ema(modality: str, model: str, record: Dict[str, Any]) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO ema_history (modality, model, ema_latency, ema_quality, ema_cost, ema_alignment)
                    VALUES (:mod, :m, :lat, :q, :c, :align)
                    ON DUPLICATE KEY UPDATE
                        ema_latency = :lat, ema_quality = :q, ema_cost = :c, 
                        ema_alignment = :align, updated_at = CURRENT_TIMESTAMP
                """),
                {
                    "mod": modality, "m": model,
                    "lat": record["ema_latency"], "q": record["ema_quality"],
                    "c": record["ema_cost"], "align": record.get("ema_alignment", 1.0)
                }
            )
    except Exception as e:
        logger.warning(f"[EMA] Erro de persistência: {e}")

def _cleanup_old_query_logs() -> None:
    while True:
        try:
            ensure_query_log()
            with engine.begin() as conn:
                conn.execute(
                    text("DELETE FROM query_log WHERE created_at < (NOW() - INTERVAL :d DAY)"),
                    {"d": LOG_RETENTION_DAYS},
                )
        except Exception:
            pass
        time.sleep(86400)

_load_ema_from_db()
threading.Thread(target=_cleanup_old_query_logs, daemon=True).start()


# ============================================================
# 🚀 FAST PATH: Roteamento e Resposta Imediata
# ============================================================

async def route_and_answer(
    query: str,
    system_prompt: str = "",
    use_rag: bool = False,
    max_tokens: int | None = None,
    temperature: float | None = None,
    modality: str = "text",
    image_b64: str | None = None,
    rag_modality: str = "text",
    use_cache: bool = True,
) -> Dict[str, Any]:
    
    start_time = time.time()
    modality = modality or "text"
    
    if bool(image_b64) and modality == "text":
        modality = "vision"

    max_tokens = max_tokens or settings.MAX_TOKENS_DEFAULT
    temperature = temperature or settings.TEMPERATURE_DEFAULT

    # 2. Cache Lookup
    if use_cache:
        try:
            cached = await check_cache(query, modality=modality, image_b64=image_b64)
            if cached:
                logger.info(f"[router] Cache HIT ({cached.get('similarity', 0):.2f})")
                return {
                    "model": "semantic_cache",
                    "modality": modality,
                    "answer": cached.get("text", ""),
                    "image_output_b64": cached.get("image_output_b64"),
                    "latency_s": round(time.time() - start_time, 3),
                    "cost_per_1k": 0.0,
                    "metadata": {"cached": True},
                    "route": {"chosen_model": "semantic_cache", "objectives": {}, "pareto_front": [], "explanation": "Cache"},
                    "candidates": []
                }
        except Exception as e: 
            logger.warning(f"[router] Cache error: {e}")

    # 3. Seleção
    all_candidates = (
        settings.CANDIDATE_MODELS_LIST +
        settings.CANDIDATE_VISION_MODELS_LIST +
        settings.CANDIDATE_MULTIMODAL_MODELS_LIST
    )
    valid_models = list(set([
        m for m in all_candidates 
        if isinstance(m, str) and not any(m.startswith(p) for p in BLOCKED_PREFIXES)
    ]))

    if not valid_models:
        valid_models = ["ollama/phi4:latest"]

    top2 = choose_top2_models(valid_models, 0.0, query, modality)
    chosen = select_model(top2, query, modality)
    
    logger.info(f"[router] Model: {chosen} | Modality: {modality}")

    if chosen.startswith("ollama/"):
        asyncio.create_task(asyncio.to_thread(_ensure_ollama_model, chosen.replace("ollama/", "")))

    # 6. RAG
    final_prompt = query
    if use_rag:
        try:
            rag_mode = rag_modality if not (image_b64 and modality != "text") else modality
            aug = await build_augmented_prompt(query, modality=rag_mode, image_b64=image_b64)
            final_prompt = f"{system_prompt}\n\nUsuário: {aug}".strip()
        except Exception as e:
            logger.warning(f"[router] RAG fail: {e}")

    # 7. Inferência (BLINDADA)
    try:
        out, meta = await call_model(
            model=chosen,
            prompt=final_prompt,
            modality=modality,
            image_b64=image_b64,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        logger.error(f"[router] Inference Error: {e}")
        out = f"Erro ao processar com {chosen}: {str(e)}"
        meta = {}

    latency_s = round(time.time() - start_time, 3)
    
    # 8. Cálculo de Custo (BLINDADO)
    p_tok = 0
    c_tok = 0
    total_cost = 0.0

    try:
        if isinstance(meta, dict):
            p_tok = int(meta.get("prompt_tokens", 0) or 0)
            c_tok = int(meta.get("completion_tokens", 0) or 0)
            
            if meta.get("cost_per_1k") is not None and float(meta.get("cost_per_1k", 0)) > 0:
                total_cost = float(meta["cost_per_1k"])
            else:
                # Tenta calcular via tabela de preços, se falhar usa 0.0
                try:
                    total_cost = get_model_cost(chosen, p_tok, c_tok)
                except Exception as pe:
                    logger.warning(f"[router] Pricing error: {pe}")
                    total_cost = 0.0
    except Exception as e:
        logger.warning(f"[router] Metadata error: {e}")

    # Defesas contra metadados corrompidos
    meta_safe = meta if isinstance(meta, dict) else {}
    
    return {
        "answer": out if isinstance(out, str) else str(out),
        "model": chosen,
        "modality": modality,
        "image_output_b64": meta_safe.get("image_output_b64"),
        "latency_s": latency_s,
        "cost_per_1k": total_cost,
        "metadata": {
            "raw_payload": meta_safe.get("raw_payload"),
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok
        },
        "route": {
            "chosen_model": chosen,
            "modality_selected": modality,
            "is_multimodal_route": bool(image_b64),
            "objectives": {"latency": latency_s, "cost": total_cost},
            "pareto_front": [],
            "explanation": f"Selected {chosen}"
        },
        "candidates": []
    }


# ============================================================
# 🐢 SLOW PATH: Background Tasks (Feedback Loop)
# ============================================================

async def process_background_feedback(
    query: str,
    answer: str,
    chosen_model: str,
    modality: str,
    latency_s: float,
    cost_val: float,
    image_b64: Optional[str] = None,
    raw_payload: Optional[Any] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0
):
    try:
        # 1. Avaliação
        if len(answer) < 5 or "Erro" in answer:
            final_quality = 1.0
        else:
            try:
                judge_scores = await judge_answer(query, answer)
                valid_scores = [s["score"] for s in judge_scores if "score" in s]
                q_mean = float(np.mean(valid_scores)) if valid_scores else 5.0
                final_quality = round(q_mean * 10.0, 2)
            except Exception:
                final_quality = 5.0

        # 2. Reward
        try:
            reward = compute_reward(chosen_model, final_quality, latency_s, cost_val)
        except Exception:
            reward = 0.0

        # 3. Bandit Update
        try:
            bandit_update(model=chosen_model, query=query, reward=reward, modality=modality)
        except Exception as e:
            logger.warning(f"[Background] Bandit fail: {e}")

        # 4. EMA Update
        try:
            global EMA_HISTORY
            alpha = 0.2
            key = (modality, chosen_model)
            
            if key not in EMA_HISTORY:
                EMA_HISTORY[key] = {
                    "ema_latency": latency_s, "ema_quality": final_quality,
                    "ema_cost": cost_val, "ema_alignment": 1.0, "updates": 1
                }
            else:
                prev = EMA_HISTORY[key]
                EMA_HISTORY[key] = {
                    "ema_latency": alpha * latency_s + (1-alpha) * prev["ema_latency"],
                    "ema_quality": alpha * final_quality + (1-alpha) * prev["ema_quality"],
                    "ema_cost": alpha * cost_val + (1-alpha) * prev["ema_cost"],
                    "ema_alignment": prev.get("ema_alignment", 1.0),
                    "updates": prev.get("updates", 0) + 1
                }
            
            # Async DB write
            asyncio.create_task(asyncio.to_thread(_persist_ema, modality, chosen_model, EMA_HISTORY[key]))
        except Exception:
            pass

        # 5. Cache
        if final_quality >= 7.0:
            try:
                await store_cache(
                    query=query, answer=answer, modality=modality, 
                    image_b64=image_b64, model_used=chosen_model
                )
            except Exception:
                pass

        # 6. Metrics
        try:
            ROUTER_QUALITY_AVG.labels(model=chosen_model).set(final_quality)
            if "ollama" in chosen_model:
                ROUTER_LOCAL_USAGE_RATIO.set(1.0)
        except Exception:
            pass

        # 7. Log Definitivo
        try:
            q_emb = await asyncio.to_thread(embed_text, query)
            insert_query_log(
                query_text=query,
                model=chosen_model,
                modality=modality,
                image_provided=bool(image_b64),
                answer=answer,
                image_output_b64=None,
                latency_s=latency_s,
                cost_per_1k=cost_val,
                quality=final_quality,
                reward=reward,
                context_label="async_processed",
                raw_payload=raw_payload,
                query_embedding=q_emb,
                answer_embedding=None
            )
        except Exception as e:
            logger.warning(f"[Background] Log fail: {e}")
        
        logger.info(f"[Background] OK | {chosen_model} | Q:{final_quality} | R:{reward:.2f}")

    except Exception as e:
        logger.exception(f"[Background] Critical fail: {e}")