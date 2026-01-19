# -*- coding: utf-8 -*-
"""
router_core.py — Multimodal + UM-RAG + Meta-bandit + UQ + Online Learning
-------------------------------------------------------------------------
O Core do sistema de roteamento.

ATUALIZAÇÃO (Tese):
- Integração com Online Machine Learning (River) para predição de erro.
- Amostragem Híbrida: Monte Carlo (Decaimento) + Active Learning (Predição de Erro).
"""

from __future__ import annotations

import logging
import time
import json
import asyncio
import threading
import uuid
import math
import random
from typing import Dict, Tuple, Any, Optional

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# --- Módulos Internos ---
from .providers_async import call_model, _ensure_ollama_model
from .router_strategy import choose_top2_models
from .semantic_cache import check_cache, store_cache
from .settings_dynamic import settings
from .bandits import select_model, bandit_update, compute_reward, _get_ctx_stats
from .judges import judge_answer
from .metrics_collector import update_model_metrics
from .rag_local import build_augmented_prompt
from .embeddings import embed_text
from .query_service import insert_query_log, ensure_query_log
from .online_predictor import get_predictor  # <--- NOVO IMPORT

# --- Novos Módulos de Inteligência e Precisão ---
from .utils.pricing import get_model_cost
from .utils.uncertainty import get_uncertainty_score
from .utils.redis_client import get_redis

from .observability import (
    ROUTER_MODEL_COST,
    ROUTER_QUALITY_AVG,
    ROUTER_COST_SAVINGS,
    ROUTER_LOCAL_USAGE_RATIO,
    ROUTER_COST_PER_QUERY,
    ROUTER_HISTORY_ENTRIES,
    FEEDBACK_PROCESSING_LATENCY,
    EMA_BATCH_QUEUE_SIZE,
    EMA_BATCH_FLUSHES,
    EMA_LOG_CLEANUP_ROWS,
)
from .settings_dynamic import update_db_pool_metrics

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] router: %(message)s",
    )

# ============================================================
# DB / Engine & Estado Global
# ============================================================

DB_URL = (
    f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}"
    f"@{settings.DB_HOST}:3306/{settings.DB_NAME}"
)
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

BLOCKED_PREFIXES = ("nomic-embed", "text-embedding", "bge-", "e5-")
LOG_RETENTION_DAYS = int(settings.get("QUERY_LOG_RETENTION_DAYS", 7))
rds = get_redis()

# ============================================================
# EMA History com TTL e LRU Eviction
# ============================================================
EMA_MAX_ENTRIES = 1000
EMA_TTL_SECONDS = 86400  # 24 horas

class EMAHistoryCache:
    """Cache LRU com TTL para histórico EMA."""

    def __init__(self, maxsize: int = EMA_MAX_ENTRIES, ttl_s: int = EMA_TTL_SECONDS):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._data: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._access_order: list = []

    def get(self, key: Tuple[str, str]) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            if key not in self._data:
                return None
            entry = self._data[key]
            last_update = entry.get("_last_update", 0)
            if self.ttl_s > 0 and (now - last_update) > self.ttl_s:
                del self._data[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return entry

    def set(self, key: Tuple[str, str], value: Dict[str, Any]) -> None:
        now = time.time()
        value["_last_update"] = now
        with self._lock:
            if key in self._data:
                if key in self._access_order:
                    self._access_order.remove(key)
            self._data[key] = value
            self._access_order.append(key)
            # Evict oldest entries if over capacity
            while len(self._data) > self.maxsize and self._access_order:
                oldest = self._access_order.pop(0)
                self._data.pop(oldest, None)

    def __contains__(self, key: Tuple[str, str]) -> bool:
        return key in self._data

    def items(self):
        with self._lock:
            return list(self._data.items())

    def cleanup_expired(self) -> int:
        """Remove entradas expiradas. Retorna número de itens removidos."""
        now = time.time()
        removed = 0
        with self._lock:
            expired_keys = []
            for key, entry in self._data.items():
                last_update = entry.get("_last_update", 0)
                if self.ttl_s > 0 and (now - last_update) > self.ttl_s:
                    expired_keys.append(key)
            for key in expired_keys:
                del self._data[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                removed += 1
        return removed

    def size(self) -> int:
        return len(self._data)

EMA_HISTORY = EMAHistoryCache()

# ============================================================
# 🚀 BATCH EMA UPDATES (Quick Win #1)
# ============================================================
EMA_BATCH_INTERVAL_S = 30  # Flush every 30 seconds
EMA_BATCH_MAX_SIZE = 100   # Force flush if queue gets too large

class EMABatchQueue:
    """Queue for batching EMA updates to reduce DB writes."""

    def __init__(self, max_size: int = EMA_BATCH_MAX_SIZE, flush_interval: int = EMA_BATCH_INTERVAL_S):
        self.max_size = max_size
        self.flush_interval = flush_interval
        self._lock = threading.Lock()
        self._queue: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (modality, model) -> latest record
        self._last_flush = time.time()

    def add(self, modality: str, model: str, record: Dict[str, Any]) -> None:
        """Add or update an EMA record in the queue."""
        key = (modality, model)
        with self._lock:
            self._queue[key] = record.copy()
            EMA_BATCH_QUEUE_SIZE.set(len(self._queue))

            # Force flush if queue is too large
            if len(self._queue) >= self.max_size:
                self._flush_locked()

    def _flush_locked(self) -> int:
        """Flush all queued updates to DB. Must hold lock."""
        if not self._queue:
            return 0

        items = list(self._queue.items())
        self._queue.clear()
        EMA_BATCH_QUEUE_SIZE.set(0)
        self._last_flush = time.time()

        # Release lock before DB operations
        return self._persist_batch(items)

    def _persist_batch(self, items: list) -> int:
        """Persist batch of EMA updates to database."""
        if not items:
            return 0

        count = 0
        try:
            with engine.begin() as conn:
                for (modality, model), record in items:
                    try:
                        # Upsert EMA history
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
                        # Log history (sampling - only log every 10th update to reduce writes further)
                        if record.get("updates", 1) % 10 == 0:
                            conn.execute(
                                text("""
                                    INSERT INTO ema_history_log (modality, model, ema_latency, ema_cost, ema_quality, ema_alignment, update_num)
                                    VALUES (:mod, :m, :lat, :c, :q, :align, :u)
                                """),
                                {
                                    "mod": modality, "m": model,
                                    "lat": record["ema_latency"], "c": record["ema_cost"],
                                    "q": record["ema_quality"], "align": record.get("ema_alignment", 1.0),
                                    "u": record.get("updates", 1)
                                }
                            )
                        count += 1
                    except SQLAlchemyError as e:
                        logger.warning(f"[EMA Batch] Error persisting {modality}/{model}: {e}")

            EMA_BATCH_FLUSHES.inc()
            logger.debug(f"[EMA Batch] Flushed {count} updates to DB")
        except SQLAlchemyError as e:
            logger.warning(f"[EMA Batch] Batch persist error: {e}")

        return count

    def flush(self) -> int:
        """Public flush method."""
        with self._lock:
            return self._flush_locked()

    def should_flush(self) -> bool:
        """Check if it's time to flush based on interval."""
        return (time.time() - self._last_flush) >= self.flush_interval

    def size(self) -> int:
        with self._lock:
            return len(self._queue)


EMA_BATCH = EMABatchQueue()


def _ema_batch_flusher() -> None:
    """Background thread that periodically flushes EMA batch queue."""
    while True:
        try:
            if EMA_BATCH.should_flush():
                flushed = EMA_BATCH.flush()
                if flushed > 0:
                    logger.info(f"[EMA Batch] Periodic flush: {flushed} updates")
        except Exception as e:
            logger.warning(f"[EMA Batch] Flusher error: {e}")
        time.sleep(10)  # Check every 10 seconds


# ============================================================
# 🧹 EMA HISTORY LOG RETENTION (Quick Win #2)
# ============================================================
EMA_LOG_RETENTION_DAYS = 30  # Keep logs for 30 days


def _cleanup_ema_history_log() -> None:
    """Cleanup old ema_history_log entries (runs daily)."""
    while True:
        try:
            with engine.begin() as conn:
                result = conn.execute(
                    text("DELETE FROM ema_history_log WHERE created_at < (NOW() - INTERVAL :d DAY)"),
                    {"d": EMA_LOG_RETENTION_DAYS},
                )
                deleted = result.rowcount if result else 0
                if deleted > 0:
                    logger.info(f"[EMA Log Cleanup] Removed {deleted} rows older than {EMA_LOG_RETENTION_DAYS} days")
                    EMA_LOG_CLEANUP_ROWS.inc(deleted)
        except Exception as e:
            logger.warning(f"[EMA Log Cleanup] Error: {e}")
        time.sleep(86400)  # Run once per day


def _update_db_pool_metrics() -> None:
    """Background thread to update DB pool metrics (Quick Win #10)."""
    while True:
        try:
            update_db_pool_metrics()
        except Exception as e:
            logger.debug(f"[DB Pool Metrics] Error: {e}")
        time.sleep(30)  # Update every 30 seconds


# ============================================================
# 🧠 RECUPERAÇÃO DINÂMICA DE PESOS (NSGA-II)
# ============================================================

def get_dynamic_strategy_weights(modality: str) -> Dict[str, float]:
    """
    Recupera os pesos da estratégia (Objetivos) diretamente do Settings Dinâmico.
    """
    return {
        "w_quality": settings.NSGA_W_QUALITY,
        "w_latency": settings.NSGA_W_LATENCY,
        "w_cost": settings.NSGA_W_COST
    }


# ============================================================
# EMA & Manutenção (Background)
# ============================================================

def _load_ema_from_db() -> None:
    """Carrega histórico EMA do banco para memória."""
    global EMA_HISTORY
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT modality, model, ema_latency, ema_quality, ema_cost, ema_alignment FROM ema_history")
            ).mappings().all()

        for r in rows:
            key = (r["modality"], r["model"])
            EMA_HISTORY.set(key, {
                "ema_latency": float(r["ema_latency"]),
                "ema_quality": float(r["ema_quality"]),
                "ema_cost": float(r["ema_cost"]),
                "ema_alignment": float(r["ema_alignment"]),
                "updates": 0,
            })
        logger.info(f"[EMA] Carregado: {EMA_HISTORY.size()} modelos.")
    except Exception:
        logger.warning("[EMA] Banco vazio ou erro ao carregar histórico (primeira execução?).")

def _persist_ema(modality: str, model: str, record: Dict[str, Any]) -> None:
    """Queue EMA update for batch persistence (Quick Win #1)."""
    EMA_BATCH.add(modality, model, record)

def _cleanup_old_query_logs() -> None:
    """Limpeza periódica de logs antigos."""
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
        time.sleep(86400)  # Roda uma vez por dia


def _cleanup_ema_history() -> None:
    """Limpeza periódica de entradas EMA expiradas."""
    while True:
        try:
            removed = EMA_HISTORY.cleanup_expired()
            if removed > 0:
                logger.info(f"[EMA] Cleanup: {removed} entradas expiradas removidas. Tamanho atual: {EMA_HISTORY.size()}")
            # Atualiza métrica
            ROUTER_HISTORY_ENTRIES.set(EMA_HISTORY.size())
        except Exception as e:
            logger.warning(f"[EMA] Erro no cleanup: {e}")
        time.sleep(3600)  # Roda a cada hora


# Inicialização
_load_ema_from_db()
threading.Thread(target=_cleanup_old_query_logs, daemon=True).start()
threading.Thread(target=_cleanup_ema_history, daemon=True).start()
threading.Thread(target=_ema_batch_flusher, daemon=True).start()       # Quick Win #1: Batch EMA
threading.Thread(target=_cleanup_ema_history_log, daemon=True).start() # Quick Win #2: Log Retention
threading.Thread(target=_update_db_pool_metrics, daemon=True).start()  # Quick Win #10: DB Pool Metrics


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
    timeout_seconds: int | None = None,
) -> Dict[str, Any]:
    """
    Executa o fluxo crítico de resposta.

    Args:
        query: User query text
        system_prompt: Optional system prompt
        use_rag: Whether to use RAG augmentation
        max_tokens: Max tokens for response
        temperature: Temperature for generation
        modality: text, vision, or multimodal
        image_b64: Base64 encoded image for vision
        rag_modality: RAG search modality
        use_cache: Whether to use semantic cache
        timeout_seconds: Optional request timeout override

    Returns:
        Dict with answer, model, cost, latency, and metadata
    """
    start_time = time.time()
    modality = modality or "text"

    # Apply timeout if specified
    default_timeout = int(settings.get("REQUEST_TIMEOUT_SECONDS", 120))
    effective_timeout = timeout_seconds or default_timeout
    
    # 1. Heurística de imagem
    if bool(image_b64) and modality == "text":
        modality = "vision"

    # Defaults
    max_tokens = max_tokens or settings.MAX_TOKENS_DEFAULT
    temperature = temperature or settings.TEMPERATURE_DEFAULT

    # ============================================================
    # 2. Cache Lookup (Leitura Rápida)
    # ============================================================
    if use_cache:
        cached = None
        try:
            cached = await check_cache(query, modality=modality, image_b64=image_b64)
        except Exception: 
            pass # Falha no cache não deve parar o fluxo

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
                "route": {
                    "chosen_model": "semantic_cache", 
                    "objectives": {"latency": 0, "cost": 0, "uncertainty": 0}, 
                    "pareto_front": [], 
                    "explanation": "Cache"
                },
                "candidates": []
            }

    # ============================================================
    # 3. Análise de Incerteza (UQ)
    # ============================================================
    uncertainty_score = 0.5 # Valor neutro padrão
    try:
        # Calcula quão "nova" é a query em relação ao conhecimento do bandit
        uncertainty_score = await asyncio.to_thread(get_uncertainty_score, query, modality)
    except Exception as e:
        logger.warning(f"[router] UQ fail: {e}")

    # ============================================================
    # 4. Seleção de Candidatos e Estratégia
    # ============================================================
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
        valid_models = ["ollama/phi4:latest"] # Fallback final

    # Pesos da estratégia (Dinâmico via Redis/NSGA)
    current_weights = get_dynamic_strategy_weights(modality)

    # Estratégia + Bandit (Considerando Incerteza e Pesos)
    top2 = choose_top2_models(
        candidates=valid_models,
        weights=current_weights,
        query_text=query, 
        modality=modality,
        uncertainty_score=uncertainty_score
    )
    
    chosen = select_model(top2, query, modality)
    
    logger.info(f"[router] Model: {chosen} | UQ: {uncertainty_score:.2f} | W: {current_weights}")

    # Background Ollama Ensure (Fire and forget para modelos locais)
    if chosen.startswith("ollama/"):
        asyncio.create_task(asyncio.to_thread(_ensure_ollama_model, chosen.replace("ollama/", "")))

    # ============================================================
    # 5. RAG Multimodal
    # ============================================================
    final_prompt = query
    if use_rag:
        try:
            rag_mode = rag_modality if not (image_b64 and modality != "text") else modality
            aug = await build_augmented_prompt(query, modality=rag_mode, image_b64=image_b64)
            final_prompt = f"{system_prompt}\n\nUsuário: {aug}".strip()
        except Exception as e:
            logger.warning(f"[router] RAG fail: {e}")

    # ============================================================
    # 6. Inferência (Provider Call)
    # ============================================================
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
    
    # Extrai load_time do meta (se existir)
    load_time_s = meta.get("load_time", 0.0) if isinstance(meta, dict) else 0.0

    # ============================================================
    # 7. Cálculo Preciso de Custo
    # ============================================================
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
                # Tenta calcular via tabela de preços dinâmica
                try:
                    total_cost = get_model_cost(chosen, p_tok, c_tok)
                except Exception as pe:
                    logger.warning(f"[router] Pricing error: {pe}")
                    total_cost = 0.0
    except Exception as e:
        logger.warning(f"[router] Metadata error: {e}")

    # Defesas contra metadados corrompidos
    meta_safe = meta if isinstance(meta, dict) else {}
    
    # Monta retorno para o usuário
    return {
        "answer": out if isinstance(out, str) else str(out),
        "model": chosen,
        "modality": modality,
        "image_output_b64": meta_safe.get("image_output_b64"),
        "latency_s": latency_s,
        "load_time_s": load_time_s,
        "cost_per_1k": total_cost,
        "metadata": {
            "raw_payload": meta_safe.get("raw_payload"),
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "load_time": load_time_s,
            "uncertainty_score": uncertainty_score,
        },
        "route": {
            "chosen_model": chosen,
            "modality_selected": modality,
            "is_multimodal_route": bool(image_b64),
            "objectives": {
                "latency": latency_s, 
                "cost": total_cost,
                "uncertainty": uncertainty_score
            },
            "pareto_front": [],
            "explanation": f"Selected {chosen} (UQ={uncertainty_score:.2f})"
        },
        "candidates": [] # Simplificado para performance
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
    """
    Executado após a resposta ser enviada ao usuário.
    Responsável por: Juízes, Reward, Bandit Update, Online Learning, Cache Store.

    ATUALIZADO: Risk-Aware Hybrid Sampling com Online Learning.
    Quick Win #9: Added latency tracking metric.
    """
    feedback_start = time.time()
    try:
        # 1. Recupera estatísticas e Preditor Online
        stats = _get_ctx_stats("global")
        model_stats = stats.get(chosen_model, {})
        n_samples = model_stats.get("count", 0)
        
        # --- ONLINE LEARNING PREDICTION ---
        # Instancia o preditor para este modelo
        predictor = get_predictor(chosen_model)
        
        # Gera embedding se necessário (geralmente já temos, mas aqui garantimos)
        query_embedding = await asyncio.to_thread(embed_text, query)
        
        # Prediz a probabilidade de erro (0.0 a 1.0)
        predicted_error_prob = predictor.predict_error_probability(query_embedding)
        # ----------------------------------

        # 2. Cálculo da Probabilidade de Julgamento (Híbrida)
        # Base: Decaimento de Monte Carlo
        if n_samples < 5:
            base_prob = 1.0 
        else:
            base_prob = 1.0 / math.sqrt(n_samples)

        # Fator de Risco: Se o modelo online diz que vai errar, aumentamos a chance de auditar
        # Usamos o MAX entre a base e a predição de erro
        prob_judge = max(base_prob, predicted_error_prob)

        # Fator de Confiança SOTA
        if "gpt-5" in chosen_model or "gpt-4" in chosen_model or "claude" in chosen_model:
            prob_judge *= 0.1 
        
        # Piso mínimo de segurança
        prob_judge = max(settings.JUDGE_MIN_SAMPLE_RATE, prob_judge)

        # 3. O Sorteio
        should_judge = random.random() < prob_judge
        
        final_quality = 0.0
        reward = 0.0

        if should_judge:
            try:
                logger.info(f"[Background] 🎲 Sampling Judge for {chosen_model} (p={prob_judge:.2f}, pred_err={predicted_error_prob:.2f})")
                judge_scores = await judge_answer(query, answer)
                valid_scores = [s["score"] for s in judge_scores if "score" in s]
                q_mean = float(np.mean(valid_scores)) if valid_scores else 5.0
                final_quality = round(q_mean * 10.0, 2)

                # --- ONLINE LEARNING UPDATE ---
                # Se julgamos, temos o rótulo real (ou aproximado pelo juiz)
                # Consideramos "Correto" se nota >= 7.0
                is_correct_label = final_quality >= 7.0
                predictor.learn(query_embedding, is_correct_label)

                # --- PREDICTOR VALIDATION (Phase 5) ---
                # Record prediction vs actual outcome for calibration
                actual_error = not is_correct_label
                predictor.record_outcome(predicted_error_prob, actual_error)

                predictor.save()  # Persiste o aprendizado e validação
                # ------------------------------

            except Exception:
                final_quality = 5.0
        else:
            # Se não julgamos, usamos a média histórica
            # logger.info(f"[Background] ⏩ Skipping Judge for {chosen_model}")
            final_quality = model_stats.get("mean", 0.5) * 10.0

        # 4. Reward (NSGA-II weights)
        try:
            reward = compute_reward(chosen_model, final_quality, latency_s, cost_val)
        except Exception:
            reward = 0.0

        # 5. Bandit Update
        try:
            bandit_update(model=chosen_model, query=query, reward=reward, modality=modality)
        except Exception as e:
            logger.warning(f"[Background] Bandit fail: {e}")

        # 6. EMA Update
        try:
            alpha = 0.2
            key = (modality, chosen_model)
            prev = EMA_HISTORY.get(key)

            if prev is None:
                new_entry = {
                    "ema_latency": latency_s, "ema_quality": final_quality,
                    "ema_cost": cost_val, "ema_alignment": 1.0, "updates": 1
                }
            else:
                new_entry = {
                    "ema_latency": alpha * latency_s + (1-alpha) * prev["ema_latency"],
                    "ema_quality": alpha * final_quality + (1-alpha) * prev["ema_quality"],
                    "ema_cost": alpha * cost_val + (1-alpha) * prev["ema_cost"],
                    "ema_alignment": prev.get("ema_alignment", 1.0),
                    "updates": prev.get("updates", 0) + 1
                }

            EMA_HISTORY.set(key, new_entry)
            asyncio.create_task(asyncio.to_thread(_persist_ema, modality, chosen_model, new_entry))
        except Exception:
            pass

        # 7. Cache (Apenas se for bom)
        if final_quality >= 7.0:
            try:
                await store_cache(
                    query=query, answer=answer, modality=modality, 
                    image_b64=image_b64, model_used=chosen_model
                )
            except Exception:
                pass

        # 8. Metrics
        try:
            ROUTER_QUALITY_AVG.labels(model=chosen_model).set(final_quality)
            if "ollama" in chosen_model:
                ROUTER_LOCAL_USAGE_RATIO.set(1.0)
        except Exception:
            pass

        # 9. Log Definitivo
        try:
            # Embedding já gerado acima
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
                query_embedding=query_embedding,
                answer_embedding=None
            )
        except Exception as e:
            logger.warning(f"[Background] Log fail: {e}")

        # 10. Record feedback processing latency (Quick Win #9)
        feedback_duration = time.time() - feedback_start
        FEEDBACK_PROCESSING_LATENCY.observe(feedback_duration)

    except Exception as e:
        # Still record latency even on failure
        feedback_duration = time.time() - feedback_start
        FEEDBACK_PROCESSING_LATENCY.observe(feedback_duration)
        logger.exception(f"[Background] Critical fail: {e}")