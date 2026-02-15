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
import random
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pybreaker
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# --- Módulos Internos ---
from .providers_async import call_model, _ensure_ollama_model, ProviderCallError
from .router_strategy import choose_top2_models
from .semantic_cache import check_cache, store_cache
from .settings_dynamic import settings
from .db import get_engine
from .bandits import select_model, bandit_update, compute_reward, _get_ctx_stats
from .judges import judge_answer
from .metrics_collector import update_model_metrics
from .rag_local import build_augmented_prompt
from .embeddings import embed_text
from .query_service import insert_query_log, ensure_query_log
from .online_predictor import get_predictor  # <--- NOVO IMPORT
from .reliability import get_request_deduplicator, execute_with_fallback  # Request deduplication
from .services.router_services import (
    normalize_modality,
    build_final_prompt,
    parse_meta_cost,
    should_enable_dedup,
    compute_judge_probability,
)
from .services.router_maintenance import create_background_threads

# --- Novos Módulos de Inteligência e Precisão ---
from .utils.pricing import get_model_cost
from .utils.uncertainty import get_uncertainty_score
from .utils.redis_client import get_redis_async_safe, ensure_redis_connected

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
    DEPENDENCY_CIRCUIT_STATE,
    DEPENDENCY_FAILURES,
    ROUTER_ROUTE_COST,
)
from .settings_dynamic import update_db_pool_metrics

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] router: %(message)s",
    )

# ============================================================
# DB / Engine & Estado Global (using centralized engine)
# ============================================================
def _get_db_engine():
    """Get database engine from centralized module."""
    return get_engine()

def _safe_setting_int(key: str, default: int) -> int:
    """Executa safe setting int."""
    try:
        return int(settings.get(key, default))
    except Exception:
        return int(default)


def _safe_setting_float(key: str, default: float) -> float:
    """Executa safe setting float."""
    try:
        return float(settings.get(key, default))
    except Exception:
        return float(default)


def _safe_setting_bool(key: str, default: bool) -> bool:
    """Executa safe setting bool."""
    raw_default = "1" if default else "0"
    try:
        getter = getattr(settings, "get", None)
        if callable(getter):
            raw = str(getter(key, raw_default)).strip()
        else:
            raw = str(getattr(settings, key, raw_default)).strip()
    except Exception:
        raw = raw_default
    return raw in ("1", "true", "True")

BLOCKED_PREFIXES = ("nomic-embed", "text-embedding", "bge-", "e5-")
LOG_RETENTION_DAYS = _safe_setting_int("QUERY_LOG_RETENTION_DAYS", 7)
def _get_rds():
    """Executa get rds."""
    return get_redis_async_safe() or ensure_redis_connected(max_wait_s=0.0, min_retry_interval_s=2.0)


_dep_cache_breaker = pybreaker.CircuitBreaker(fail_max=10, reset_timeout=30, name="dep_cache")
_dep_uq_breaker = pybreaker.CircuitBreaker(fail_max=10, reset_timeout=30, name="dep_uq")


def _record_dependency_breaker_metrics() -> None:
    """Executa record dependency breaker metrics."""
    mapping = {"closed": 0, "half-open": 1, "open": 2}
    try:
        DEPENDENCY_CIRCUIT_STATE.labels(dependency="cache").set(mapping.get(_dep_cache_breaker.current_state, 0))
        DEPENDENCY_CIRCUIT_STATE.labels(dependency="uq").set(mapping.get(_dep_uq_breaker.current_state, 0))
    except Exception:
        pass


def _error_budget_window() -> Tuple[int, float, int]:
    """Executa error budget window."""
    window_s = _safe_setting_int("ERROR_BUDGET_WINDOW_S", 300)
    threshold = _safe_setting_float("ERROR_BUDGET_THRESHOLD", 0.20)
    min_requests = _safe_setting_int("ERROR_BUDGET_MIN_REQUESTS", 20)
    return max(10, window_s), max(0.0, min(1.0, threshold)), max(1, min_requests)


def _record_request_outcome(success: bool) -> None:
    """Executa record request outcome."""
    rds = _get_rds()
    if not rds or not _safe_setting_bool("ERROR_BUDGET_ENABLED", True):
        return
    try:
        now = int(time.time())
        bucket = now // 10
        w, _, _ = _error_budget_window()
        key = f"router:error_budget:{bucket}"
        pipe = rds.pipeline()
        pipe.hincrby(key, "total", 1)
        if not success:
            pipe.hincrby(key, "errors", 1)
        pipe.expire(key, w + 60)
        pipe.execute()
    except Exception:
        pass


def _is_error_budget_exceeded() -> bool:
    """Executa is error budget exceeded."""
    rds = _get_rds()
    if not rds or not _safe_setting_bool("ERROR_BUDGET_ENABLED", True):
        return False
    try:
        w, threshold, min_requests = _error_budget_window()
        now = int(time.time())
        total = 0
        errors = 0
        for offset in range((w // 10) + 1):
            bucket = (now // 10) - offset
            key = f"router:error_budget:{bucket}"
            raw = rds.hgetall(key) or {}
            t = int((raw.get(b"total") or raw.get("total") or 0))
            e = int((raw.get(b"errors") or raw.get("errors") or 0))
            total += t
            errors += e
        if total < min_requests:
            return False
        rate = errors / max(1, total)
        return rate >= threshold
    except Exception:
        return False

# ============================================================
# EMA History com TTL e LRU Eviction
# ============================================================
EMA_MAX_ENTRIES = 50000  # Optimized for high-capacity environment (64GB RAM) - ~5MB memory
EMA_TTL_SECONDS = 86400  # 24 horas

class EMAHistoryCache:
    """Cache LRU com TTL para histórico EMA."""

    def __init__(self, maxsize: int = EMA_MAX_ENTRIES, ttl_s: int = EMA_TTL_SECONDS):
        """Inicializa estado interno necessário para uso da classe."""
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._data: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._access_order: list = []

    def get(self, key: Tuple[str, str]) -> Optional[Dict[str, Any]]:
        """Executa get."""
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
        """Executa set."""
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
        """Executa contains."""
        return key in self._data

    def items(self):
        """Executa items."""
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
        """Executa size."""
        return len(self._data)

EMA_HISTORY = EMAHistoryCache()

# ============================================================
# 🚀 BATCH EMA UPDATES (Quick Win #1)
# ============================================================
EMA_BATCH_INTERVAL_S = 60  # Flush every 60 seconds - optimized for high-capacity
EMA_BATCH_MAX_SIZE = 500   # Force flush if queue gets too large - reduces DB writes by ~80%

class EMABatchQueue:
    """Queue for batching EMA updates to reduce DB writes."""

    def __init__(self, max_size: int = EMA_BATCH_MAX_SIZE, flush_interval: int = EMA_BATCH_INTERVAL_S):
        """Inicializa estado interno necessário para uso da classe."""
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
            with _get_db_engine().begin() as conn:
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
        """Executa size."""
        with self._lock:
            return len(self._queue)


EMA_BATCH = EMABatchQueue()
_bg_stop_event = threading.Event()
_bg_threads: list[threading.Thread] = []
_bg_started = False


def _ema_batch_flusher() -> None:
    """Background thread that periodically flushes EMA batch queue."""
    while not _bg_stop_event.is_set():
        try:
            if EMA_BATCH.should_flush():
                flushed = EMA_BATCH.flush()
                if flushed > 0:
                    logger.info(f"[EMA Batch] Periodic flush: {flushed} updates")
        except Exception as e:
            logger.warning(f"[EMA Batch] Flusher error: {e}")
        _bg_stop_event.wait(10)  # Check every 10 seconds


# ============================================================
# 🧹 EMA HISTORY LOG RETENTION (Quick Win #2)
# ============================================================
EMA_LOG_RETENTION_DAYS = 30  # Keep logs for 30 days


def _cleanup_ema_history_log() -> None:
    """Cleanup old ema_history_log entries (runs daily)."""
    while not _bg_stop_event.is_set():
        try:
            with _get_db_engine().begin() as conn:
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
        _bg_stop_event.wait(86400)  # Run once per day


def _update_db_pool_metrics() -> None:
    """Background thread to update DB pool metrics (Quick Win #10)."""
    while not _bg_stop_event.is_set():
        try:
            update_db_pool_metrics()
        except Exception as e:
            logger.debug(f"[DB Pool Metrics] Error: {e}")
        _bg_stop_event.wait(30)  # Update every 30 seconds


# ============================================================
# 🧠 RECUPERAÇÃO DINÂMICA DE PESOS (NSGA-II)
# ============================================================

def get_dynamic_strategy_weights(modality: str) -> Dict[str, float]:
    """
    Recupera os pesos da estratégia (Objetivos) diretamente do Settings Dinâmico.
    """
    def _safe_attr(name: str, default: float) -> float:
        """Executa safe attr."""
        try:
            return float(getattr(settings, name))
        except Exception:
            return float(default)

    return {
        "w_quality": _safe_attr("NSGA_W_QUALITY", 1.0),
        "w_latency": _safe_attr("NSGA_W_LATENCY", 0.5),
        "w_cost": _safe_attr("NSGA_W_COST", 100.0),
    }


# ============================================================
# EMA & Manutenção (Background)
# ============================================================

def _load_ema_from_db() -> None:
    """Carrega histórico EMA do banco para memória."""
    global EMA_HISTORY
    try:
        with _get_db_engine().connect() as conn:
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
    while not _bg_stop_event.is_set():
        try:
            ensure_query_log()
            with _get_db_engine().begin() as conn:
                conn.execute(
                    text("DELETE FROM query_log WHERE created_at < (NOW() - INTERVAL :d DAY)"),
                    {"d": LOG_RETENTION_DAYS},
                )
        except Exception:
            pass
        _bg_stop_event.wait(86400)  # Roda uma vez por dia


def _cleanup_ema_history() -> None:
    """Limpeza periódica de entradas EMA expiradas."""
    while not _bg_stop_event.is_set():
        try:
            removed = EMA_HISTORY.cleanup_expired()
            if removed > 0:
                logger.info(f"[EMA] Cleanup: {removed} entradas expiradas removidas. Tamanho atual: {EMA_HISTORY.size()}")
            # Atualiza métrica
            ROUTER_HISTORY_ENTRIES.set(EMA_HISTORY.size())
        except Exception as e:
            logger.warning(f"[EMA] Erro no cleanup: {e}")
        _bg_stop_event.wait(3600)  # Roda a cada hora


def start_background_services() -> None:
    """Start router maintenance background services exactly once."""
    global _bg_started
    if _bg_started:
        return
    _bg_started = True
    _bg_stop_event.clear()
    _load_ema_from_db()
    _bg_threads.extend(
        create_background_threads(
            cleanup_old_query_logs=_cleanup_old_query_logs,
            cleanup_ema_history=_cleanup_ema_history,
            ema_batch_flusher=_ema_batch_flusher,
            cleanup_ema_history_log=_cleanup_ema_history_log,
            update_db_pool_metrics=_update_db_pool_metrics,
        )
    )
    for thread in _bg_threads:
        thread.start()
    logger.info("[router] Background services started")


def stop_background_services() -> None:
    """Stop router maintenance background services and flush pending EMA batch."""
    global _bg_started
    if not _bg_started:
        return
    _bg_stop_event.set()
    try:
        EMA_BATCH.flush()
    except Exception as e:
        logger.warning(f"[router] Failed to flush EMA batch at shutdown: {e}")
    for thread in _bg_threads:
        thread.join(timeout=1.0)
    _bg_threads.clear()
    _bg_started = False
    logger.info("[router] Background services stopped")


# ============================================================
# 🚀 FAST PATH: Roteamento e Resposta Imediata
# ============================================================

async def _route_and_answer_internal(
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
    """
    Internal implementation of route_and_answer.
    Separated to allow wrapping with timeout.
    """
    start_time = time.time()
    modality = normalize_modality(modality, image_b64)

    # Defaults
    max_tokens = max_tokens or settings.MAX_TOKENS_DEFAULT
    temperature = temperature or settings.TEMPERATURE_DEFAULT

    # ============================================================
    # 2. Cache Lookup (Leitura Rápida)
    # ============================================================
    if use_cache:
        cached = None
        try:
            if _dep_cache_breaker.current_state != "open":
                try:
                    cached = await _dep_cache_breaker.call_async(
                        check_cache,
                        query,
                        modality=modality,
                        image_b64=image_b64,
                    )
                except Exception:
                    # Preserve cache-path behavior even if breaker wrapper fails.
                    cached = await check_cache(query, modality=modality, image_b64=image_b64)
            _record_dependency_breaker_metrics()
        except Exception:
            try:
                DEPENDENCY_FAILURES.labels(dependency="cache").inc()
            except Exception:
                pass
            _record_dependency_breaker_metrics()

        if cached:
            logger.info(f"[router] Cache HIT ({cached.get('similarity', 0):.2f})")
            try:
                ROUTER_ROUTE_COST.labels(route_type="cache").inc(0.0)
            except Exception:
                pass
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
                    "explanation": "Cache",
                    "fallback": {"used": False, "models_tried": ["semantic_cache"], "errors": []},
                },
                "candidates": []
            }

    # ============================================================
    # 3. Análise de Incerteza (UQ)
    # ============================================================
    uncertainty_score = 0.5 # Valor neutro padrão
    try:
        if _dep_uq_breaker.current_state != "open":
            uncertainty_score = _dep_uq_breaker.call(get_uncertainty_score, query, modality)
        _record_dependency_breaker_metrics()
    except Exception as e:
        try:
            DEPENDENCY_FAILURES.labels(dependency="uq").inc()
        except Exception:
            pass
        _record_dependency_breaker_metrics()
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
    elif _is_error_budget_exceeded():
        local_models = [m for m in valid_models if m.startswith("ollama/")]
        if local_models:
            valid_models = local_models
            logger.warning("[router] Error budget exceeded; forcing local-only candidate set")

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
    # 5. RAG Multimodal e System Prompt
    # ============================================================
    if use_rag:
        try:
            rag_mode = rag_modality if not (image_b64 and modality != "text") else modality
            aug = await build_augmented_prompt(query, modality=rag_mode, image_b64=image_b64)
            final_prompt = build_final_prompt(query=query, system_prompt=system_prompt, use_rag=True, rag_text=aug)
        except Exception as e:
            logger.warning(f"[router] RAG fail: {e}")
            final_prompt = build_final_prompt(query=query, system_prompt=system_prompt, use_rag=True, rag_text=None)
    else:
        final_prompt = build_final_prompt(query=query, system_prompt=system_prompt, use_rag=False, rag_text=None)

    # ============================================================
    # 6. Inferência (Provider Call + Fallback Chain)
    # ============================================================
    use_fallback_chain = _safe_setting_bool("REQUEST_FALLBACK_ENABLED", False)
    fallback_used = False
    fallback_models_tried = [chosen]
    fallback_errors = []

    if use_fallback_chain:
        max_fallbacks = _safe_setting_int("REQUEST_MAX_FALLBACKS", 2)

        async def _execute_provider(model_name: str):
            """Executa a responsabilidade descrita por este método.

            Args:
                model_name: Parâmetro de entrada.

            Returns:
                Valor produzido pela execução.
            """
            return await call_model(
                model=model_name,
                prompt=final_prompt,
                modality=modality,
                image_b64=image_b64,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        fallback_result = await execute_with_fallback(
            primary_model=chosen,
            execute_fn=_execute_provider,
            max_fallbacks=max_fallbacks,
        )
        if not fallback_result.success:
            if fallback_result.errors:
                last = fallback_result.errors[-1]
                raise ProviderCallError(
                    model=fallback_result.model_used,
                    message=last.get("error", "All fallback models failed"),
                    category=last.get("category", "provider_unavailable"),
                    retryable=True,
                )
            raise ProviderCallError(
                model=chosen,
                message="All fallback models failed",
                category="provider_unavailable",
                retryable=True,
            )

        out, meta = fallback_result.result
        chosen = fallback_result.model_used
        fallback_used = len(fallback_result.models_tried) > 1
        fallback_models_tried = fallback_result.models_tried
        fallback_errors = fallback_result.errors
    else:
        out, meta = await call_model(
            model=chosen,
            prompt=final_prompt,
            modality=modality,
            image_b64=image_b64,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    latency_s = round(time.time() - start_time, 3)
    
    # ============================================================
    # 7. Cálculo Preciso de Custo
    # ============================================================
    try:
        p_tok, c_tok, total_cost, load_time_s, meta_safe = parse_meta_cost(
            meta=meta,
            chosen_model=chosen,
            cost_lookup=get_model_cost,
        )
    except Exception as e:
        logger.warning(f"[router] Metadata error: {e}")
        p_tok, c_tok, total_cost, load_time_s, meta_safe = 0, 0, 0.0, 0.0, {}
    
    # Monta retorno para o usuário
    route_type = "fallback" if fallback_used else "direct"
    try:
        ROUTER_ROUTE_COST.labels(route_type=route_type).inc(float(total_cost))
    except Exception:
        pass

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
            "explanation": f"Selected {chosen} (UQ={uncertainty_score:.2f})",
            "fallback": {
                "used": fallback_used,
                "models_tried": fallback_models_tried,
                "errors": fallback_errors,
            },
        },
        "candidates": [] # Simplificado para performance
    }


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
    deduplicate: bool = True,
) -> Dict[str, Any]:
    """
    Executa o fluxo crítico de resposta com timeout global e deduplicação.

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
        deduplicate: Whether to deduplicate identical in-flight requests

    Returns:
        Dict with answer, model, cost, latency, and metadata

    Raises:
        asyncio.TimeoutError: If request exceeds timeout
    """
    # Calculate effective timeout
    default_timeout = _safe_setting_int("REQUEST_TIMEOUT_SECONDS", 120)
    effective_timeout = timeout_seconds or default_timeout

    async def _execute_request():
        """Inner function for request execution."""
        return await _route_and_answer_internal(
            query=query,
            system_prompt=system_prompt,
            use_rag=use_rag,
            max_tokens=max_tokens,
            temperature=temperature,
            modality=modality,
            image_b64=image_b64,
            rag_modality=rag_modality,
            use_cache=use_cache,
        )

    try:
        max_retries = _safe_setting_int("REQUEST_MAX_RETRIES", 1)
        # Check if deduplication is enabled
        dedup_enabled = should_enable_dedup(settings.get, deduplicate)

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                if dedup_enabled:
                    deduplicator = get_request_deduplicator()
                    result = await asyncio.wait_for(
                        deduplicator.deduplicate(
                            query=query,
                            model="auto",
                            execute_fn=_execute_request,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            system_prompt=system_prompt,
                        ),
                        timeout=effective_timeout,
                    )
                else:
                    result = await asyncio.wait_for(_execute_request(), timeout=effective_timeout)

                _record_request_outcome(success=True)
                return result
            except asyncio.TimeoutError:
                _record_request_outcome(success=False)
                logger.error(f"[router] Request timeout after {effective_timeout}s for query: {query[:60]}...")
                raise
            except Exception as e:
                last_exc = e
                _record_request_outcome(success=False)
                if attempt >= max_retries:
                    raise
                await asyncio.sleep(min(1.5, 0.25 * (2 ** attempt)))

        if last_exc:
            raise last_exc
    except asyncio.TimeoutError:
        raise


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

        prob_judge = compute_judge_probability(
            n_samples=n_samples,
            predicted_error_prob=predicted_error_prob,
            chosen_model=chosen_model,
            min_sample_rate=settings.JUDGE_MIN_SAMPLE_RATE,
        )

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
        except Exception as e:
            logger.warning(f"[Background] EMA update failed: {e}")

        # 7. Cache (Apenas se for bom)
        if final_quality >= 7.0:
            try:
                await store_cache(
                    query=query, answer=answer, modality=modality, 
                    image_b64=image_b64, model_used=chosen_model
                )
            except Exception as e:
                logger.warning(f"[Background] Cache store failed: {e}")

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


def reset_router_runtime_state() -> None:
    """
    Reset global runtime state for tests/dev.
    """
    global EMA_HISTORY, EMA_BATCH, _bg_started
    EMA_HISTORY = EMAHistoryCache()
    EMA_BATCH = EMABatchQueue()
    _bg_stop_event.set()
    _bg_threads.clear()
    _bg_started = False
