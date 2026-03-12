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
from .services.router_feedback import process_background_feedback_impl
from .services.router_execution import route_and_answer_internal_impl
from .services.router_resilience import (
    dep_cache_breaker as _dep_cache_breaker,
    dep_uq_breaker as _dep_uq_breaker,
    get_router_redis,
    is_error_budget_exceeded as _is_error_budget_exceeded_impl,
    record_dependency_breaker_metrics as _record_dependency_breaker_metrics_impl,
    record_request_outcome as _record_request_outcome_impl,
    safe_setting_bool as _safe_setting_bool_impl,
    safe_setting_float as _safe_setting_float_impl,
    safe_setting_int as _safe_setting_int_impl,
)
from .services.router_state import (
    EMABatchQueue as BaseEMABatchQueue,
    EMAHistoryCache,
)

# --- Novos Módulos de Inteligência e Precisão ---
from .utils.pricing import get_model_cost
from .utils.uncertainty import get_uncertainty_score

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


def _settings_getter(key: str, default: Any) -> Any:
    """Read settings through get() when available, else fallback to attributes."""
    getter = getattr(settings, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(settings, key, default)


def _safe_setting_int(key: str, default: int) -> int:
    """Executa safe setting int."""
    return _safe_setting_int_impl(_settings_getter, key, default)


def _safe_setting_float(key: str, default: float) -> float:
    """Executa safe setting float."""
    return _safe_setting_float_impl(_settings_getter, key, default)


def _safe_setting_bool(key: str, default: bool) -> bool:
    """Executa safe setting bool."""
    return _safe_setting_bool_impl(_settings_getter, key, default)

BLOCKED_PREFIXES = ("nomic-embed", "text-embedding", "bge-", "e5-")
LOG_RETENTION_DAYS = _safe_setting_int("QUERY_LOG_RETENTION_DAYS", 7)
def _get_rds():
    """Executa get rds."""
    return get_router_redis()


def _record_dependency_breaker_metrics() -> None:
    """Executa record dependency breaker metrics."""
    _record_dependency_breaker_metrics_impl()


def _error_budget_window() -> Tuple[int, float, int]:
    """Executa error budget window."""
    from .services.router_resilience import error_budget_window

    return error_budget_window(_settings_getter)


def _record_request_outcome(success: bool) -> None:
    """Executa record request outcome."""
    _record_request_outcome_impl(settings_getter=_settings_getter, success=success)


def _is_error_budget_exceeded() -> bool:
    """Executa is error budget exceeded."""
    return _is_error_budget_exceeded_impl(settings_getter=_settings_getter)

# ============================================================
# EMA History com TTL e LRU Eviction
# ============================================================
EMA_HISTORY = EMAHistoryCache()

# ============================================================
# 🚀 BATCH EMA UPDATES (Quick Win #1)
# ============================================================
class EMABatchQueue(BaseEMABatchQueue):
    """Queue for batching EMA updates to reduce DB writes."""

    def _on_queue_size_changed(self, size: int) -> None:
        EMA_BATCH_QUEUE_SIZE.set(size)

    def _persist_batch(self, items: list) -> int:
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
    return await route_and_answer_internal_impl(
        deps={
            "asyncio": asyncio,
            "settings": settings,
            "normalize_modality": normalize_modality,
            "_dep_cache_breaker": _dep_cache_breaker,
            "_dep_uq_breaker": _dep_uq_breaker,
            "check_cache": check_cache,
            "_record_dependency_breaker_metrics": _record_dependency_breaker_metrics,
            "DEPENDENCY_FAILURES": DEPENDENCY_FAILURES,
            "logger": logger,
            "ROUTER_ROUTE_COST": ROUTER_ROUTE_COST,
            "get_uncertainty_score": get_uncertainty_score,
            "BLOCKED_PREFIXES": BLOCKED_PREFIXES,
            "_is_error_budget_exceeded": _is_error_budget_exceeded,
            "get_dynamic_strategy_weights": get_dynamic_strategy_weights,
            "choose_top2_models": choose_top2_models,
            "select_model": select_model,
            "_ensure_ollama_model": _ensure_ollama_model,
            "build_augmented_prompt": build_augmented_prompt,
            "build_final_prompt": build_final_prompt,
            "_safe_setting_bool": _safe_setting_bool,
            "_safe_setting_int": _safe_setting_int,
            "call_model": call_model,
            "execute_with_fallback": execute_with_fallback,
            "ProviderCallError": ProviderCallError,
            "parse_meta_cost": parse_meta_cost,
            "get_model_cost": get_model_cost,
        },
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
    await process_background_feedback_impl(
        deps={
            "_get_ctx_stats": _get_ctx_stats,
            "get_predictor": get_predictor,
            "asyncio": asyncio,
            "random": random,
            "embed_text": embed_text,
            "compute_judge_probability": compute_judge_probability,
            "settings": settings,
            "logger": logger,
            "judge_answer": judge_answer,
            "compute_reward": compute_reward,
            "bandit_update": bandit_update,
            "_persist_ema": _persist_ema,
            "store_cache": store_cache,
            "insert_query_log": insert_query_log,
            "ROUTER_QUALITY_AVG": ROUTER_QUALITY_AVG,
            "ROUTER_LOCAL_USAGE_RATIO": ROUTER_LOCAL_USAGE_RATIO,
            "FEEDBACK_PROCESSING_LATENCY": FEEDBACK_PROCESSING_LATENCY,
        },
        state={"EMA_HISTORY": EMA_HISTORY},
        query=query,
        answer=answer,
        chosen_model=chosen_model,
        modality=modality,
        latency_s=latency_s,
        cost_val=cost_val,
        image_b64=image_b64,
        raw_payload=raw_payload,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


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
