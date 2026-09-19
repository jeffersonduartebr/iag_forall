# -*- coding: utf-8 -*-
# Objective: Application runtime code for bandits.
"""Implement the contextual bandit layer used for model selection.

This module combines three concerns that shape router behavior over time:

- contextual model selection with epsilon-greedy, UCB1, and Thompson sampling
- semantic clustering so similar queries share learning signal
- reward computation and persistence across Redis and MariaDB

The public entry points are ``select_model()``, ``bandit_update()``, and
``compute_reward()``. The rest of the module supports those APIs with centroid
maintenance, context aggregation, and policy-combination helpers.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random  # noqa: F401  (usado via patch em app.bandits.random.random nos testes)
from typing import Any, Dict, List, Tuple

import numpy as np

from app.db import get_engine
from app.observability import (
    BANDIT_REWARD,
    BANDIT_SELECT,
    BANDIT_UPDATE,
)
from app.services import bandit_policy
from app.services.bandit_centroids import (
    reset_centroid_matrix_cache,
)
from app.services.bandit_stats_store import (
    cold_contexts,
    db_fallback,
    load_stats_from_db,
    parse_redis_stats,
    sanitize_model_stats,
    serialize_model_stats,
    upsert_stats_db,
)

# Centróides semânticos (Redis, lock, aprendizado online): services.centroid_store.
from app.services.centroid_store import (  # noqa: F401  (reexportados: pontos de uso e testes)
    CENTROIDS_DIM,
    CENTROIDS_K,
    CENTROIDS_LEARN_RATE,
    CENTROIDS_MIN_SIM_CREATE,
    R_CENTROIDS,
    R_CENTROIDS_LOCK,
    R_CENTROIDS_META,
    CentroidMatrixCache,
    _acquire_lock,
    _centroid_matrix_cache,
    _cosine,
    _ensure_dim,
    _load_centroids,
    _nearest_centroid_label,
    _nearest_centroid_vec,
    _new_centroid_id,
    _release_lock,
    _save_centroids,
    _unit,
    centroids_online_update,
)

# Recompensa acoplada ao NSGA-II: implementação em app.services.reward, reexportada
# aqui para os chamadores atuais (router_core, user_feedback, governança, tools).
from app.services.reward import compute_reward  # noqa: F401
from app.settings_dynamic import settings
from app.utils.background import spawn
from app.utils.executors import run_background
from app.utils.redis_async_ops import redis_get_str, redis_hgetall_map
from app.utils.redis_client import ensure_redis_connected, get_redis_async_safe

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] bandit: %(message)s",
    )

# ============================================================
# DB (using centralized engine)
# ============================================================
def _get_db_engine():
    """Get database engine from centralized module."""
    return get_engine()

# ============================================================
# Redis
# ============================================================
def _get_rds():
    """Return the Redis client used for bandit state and centroid storage."""
    return get_redis_async_safe() or ensure_redis_connected(max_wait_s=0.0, min_retry_interval_s=2.0)

# Redis Keys (ajustadas)
R_CTX_PREFIX = "meta:bandit:ctx"                  # por contexto → stats
R_META_STRATEGY = "meta:bandit:strategy"          # qual estratégia meta-bandit venceu
R_CLUSTERING_MODEL = "meta:bandit:cluster:model"  # clustering automático
R_CLUSTERING_LOCK = "meta:bandit:cluster:lock"

# ============================================================
# Hyperparams
# ============================================================
def _safe_setting_float(key: str, default: float) -> float:
    """Read one float setting defensively, falling back on parse errors."""
    try:
        return float(settings.get(key, default))
    except Exception:
        return float(default)


def _safe_setting_int(key: str, default: int) -> int:
    """Read one integer setting defensively, falling back on parse errors."""
    try:
        return int(settings.get(key, default))
    except Exception:
        return int(default)


DEFAULT_EPSILON = _safe_setting_float("BANDIT_EPSILON", 0.12)
CENTROIDS_MIN_RECORDS_FOR_TRAIN = _safe_setting_int("CENTROIDS_MIN_RECORDS_FOR_TRAIN", 50)
# Atualizações de centróide em voo (pool de segundo plano); acima disso são descartadas.
CENTROID_LEARNING_MAX_INFLIGHT = _safe_setting_int("CENTROID_LEARNING_MAX_INFLIGHT", 8)

# Meta-Bandit: estratégias
META_STRATEGIES = ["epsilon_greedy", "ucb1", "thompson"]

# Sanitização das estatísticas: ver services.bandit_stats_store.sanitize_model_stats.
_sanitize_model_stats = sanitize_model_stats


# ============================================================
# Contextos automáticos via clustering dinâmico
# ============================================================

def _auto_context_labels(query: str, modality: str = "text") -> List[str]:
    """
    Gera contextos automáticos:
      - 'cluster:<id>' baseado em centróide mais próximo / atualizado online
      - 'mod:<modality>'
      - 'global'
    """
    labels: List[str] = []

    # 1) cluster automático (atualiza centróides online)
    try:
        cid = centroids_online_update(query)
        if cid is not None:
            labels.append(f"cluster:{cid}")
    except Exception:
        pass

    # 2) modalidade (text|vision|multimodal)
    mod = (modality or "text").strip().lower()
    labels.append(f"mod:{mod}")

    # 3) global
    labels.append("global")

    # remove duplicatas preservando ordem
    seen = set()
    out: List[str] = []
    for c in labels:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ============================================================
# Stats de contexto em Redis + DB
# ============================================================

def _ctx_key(ctx: str) -> str:
    """Build the Redis hash key used to store one context's model statistics."""
    return f"{R_CTX_PREFIX}:{ctx}"


def _get_ctx_stats_from_db(ctx: str) -> Dict[str, Dict[str, float]]:
    """Load contextual bandit stats from MariaDB."""
    return load_stats_from_db(_get_db_engine, ctx)


_parse_ctx_stats_from_redis = parse_redis_stats


def _ctx_stats_db_fallback(ctx: str) -> Dict[str, Dict[str, float]]:
    """DB read for a context missing from Redis (write-back + short negative cache)."""
    return db_fallback(ctx, _get_ctx_stats_from_db, _set_ctx_stats)


def _get_ctx_stats(ctx: str) -> Dict[str, Dict[str, float]]:
    """
    Lê stats de um contexto:
    {
      "modelA": {"mean":..., "count":..., "var":..., "M2":..., "alpha":..., "beta":...},
      ...
    }
    """
    stats: Dict[str, Dict[str, float]] = {}
    rds = _get_rds()
    if rds:
        try:
            stats = parse_redis_stats(rds.hgetall(_ctx_key(ctx)))
        except Exception as e:
            logger.warning(f"[bandit] Falha Redis ctx={ctx}: {e}")
    return stats or _ctx_stats_db_fallback(ctx)


async def _get_ctx_stats_async(ctx: str) -> Dict[str, Dict[str, float]]:
    """Load contextual stats using async Redis with DB fallback in a worker thread."""
    stats = _parse_ctx_stats_from_redis(await redis_hgetall_map(_ctx_key(ctx)))
    if stats:
        return stats
    return await asyncio.to_thread(_ctx_stats_db_fallback, ctx)


def _set_ctx_stats(ctx: str, stats: Dict[str, Dict[str, float]]) -> None:
    """Persist one context's bandit statistics to Redis."""
    cold_contexts.forget(ctx)
    rds = _get_rds()
    if not rds:
        return
    try:
        pipe = rds.pipeline()
        for model, s in stats.items():
            pipe.hset(_ctx_key(ctx), model, serialize_model_stats(s))
        pipe.execute()
    except Exception as e:
        logger.warning(f"[bandit] Falha ao salvar ctx={ctx} no Redis: {e}")


def _batch_upsert_ctx_db(updates: list[tuple[str, str, Dict[str, float]]]) -> None:
    """Persist multiple contextual statistics in one DB transaction."""
    upsert_stats_db(_get_db_engine, updates)


# ============================================================
# Estratégias básicas: ε-greedy, UCB1, Thompson Sampling
# ============================================================

def _dynamic_epsilon(ctx_stats: Dict[str, Dict[str, float]]) -> float:
    """Compute the exploration rate for one context from current statistics."""
    try:
        from app.services.frozen_policy import frozen_bandit_epsilon

        eps = frozen_bandit_epsilon(DEFAULT_EPSILON)
    except Exception:
        eps = DEFAULT_EPSILON
    return bandit_policy.dynamic_epsilon(ctx_stats, eps)


def _choose_epsilon_greedy(
    models: List[str], ctx_stats: Dict[str, Dict[str, float]]
) -> str:
    """Choose a model using the epsilon-greedy policy implementation."""
    try:
        from app.services.frozen_policy import frozen_bandit_epsilon

        eps = frozen_bandit_epsilon(DEFAULT_EPSILON)
    except Exception:
        eps = DEFAULT_EPSILON
    return bandit_policy.choose_epsilon_greedy(models, ctx_stats, eps)


def _choose_ucb1(models: List[str], ctx_stats: Dict[str, Dict[str, float]]) -> str:
    """Choose a model using the UCB1 policy implementation."""
    return bandit_policy.choose_ucb1(models, ctx_stats)


def _choose_thompson(models: List[str], ctx_stats: Dict[str, Dict[str, float]]) -> str:
    """Choose a model using the Thompson sampling policy implementation."""
    return bandit_policy.choose_thompson(models, ctx_stats)


# ============================================================
# Meta-Bandit: combinação híbrida ε-greedy + UCB1 + TS
# ============================================================

def _meta_choose_strategy() -> str:
    """
    Estratégia meta:
      - Se houver override em Redis/settings → usa.
      - Caso contrário, combinação leve favorecendo UCB1 em estáveis,
        TS em dados ricos e ε-greedy em dados escassos.
    """
    rds = _get_rds()
    # override via Redis
    if rds:
        try:
            raw = rds.get(R_META_STRATEGY)
            if raw:
                val = raw.decode() if isinstance(raw, bytes) else raw
                val = val.strip().lower()
                if val in META_STRATEGIES:
                    return val
        except Exception:
            pass

    # fallback simples: usar UCB1 como default
    return "ucb1"


async def _meta_choose_strategy_async() -> str:
    """Resolve the meta-bandit strategy using async Redis."""
    raw = await redis_get_str(R_META_STRATEGY)
    if raw:
        val = raw.strip().lower()
        if val in META_STRATEGIES:
            return val
    return "ucb1"


def _schedule_centroid_learning(query: str) -> None:
    """Queue centroid online learning without blocking model selection.

    Runs on the small background pool (not the default executor shared with
    the hot path). Under saturation new updates are dropped and counted in
    ``background_tasks_dropped_total{name="centroid_learning"}``.
    """
    try:
        spawn(
            run_background(centroids_online_update, query),
            name="centroid_learning",
            limit=CENTROID_LEARNING_MAX_INFLIGHT,
        )
    except RuntimeError:
        centroids_online_update(query)


async def _auto_context_labels_async(query: str, modality: str = "text") -> List[str]:
    """Build context labels using read-only centroid lookup and background learning."""
    labels: List[str] = []
    try:
        nearest = await asyncio.to_thread(_nearest_centroid_label, query)
        if nearest and nearest.startswith("semctx:"):
            labels.append(f"cluster:{nearest.split(':', 1)[1]}")
    except Exception:
        pass
    _schedule_centroid_learning(query)

    mod = (modality or "text").strip().lower()
    labels.append(f"mod:{mod}")
    labels.append("global")

    seen = set()
    out: List[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


async def _meta_combine_choices_async(
    models: List[str],
    ctx_stats: Dict[str, Dict[str, float]],
) -> Tuple[str, Dict[str, str]]:
    """Combine bandit policies using async strategy resolution."""
    return bandit_policy.meta_combine_choices(
        models=models,
        ctx_stats=ctx_stats,
        default_epsilon=DEFAULT_EPSILON,
        preferred_strategy=await _meta_choose_strategy_async(),
    )


def _meta_combine_choices(
    models: List[str],
    ctx_stats: Dict[str, Dict[str, float]],
) -> Tuple[str, Dict[str, str]]:
    """Execute the meta combine choices routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    return bandit_policy.meta_combine_choices(
        models=models,
        ctx_stats=ctx_stats,
        default_epsilon=DEFAULT_EPSILON,
        preferred_strategy=_meta_choose_strategy(),
    )


# ============================================================
# API principal de seleção
# ============================================================

def select_model(
    valid_models: List[str],
    query: str,
    modality: str = "text",
) -> str:
    """
    Select one model for the current query using the hybrid meta-bandit.

    The selection path derives a semantic context, loads contextual statistics,
    executes the component policies, combines their suggestions, and emits the
    chosen model as the router-facing decision.
    """
    if not valid_models:
        logger.warning("[bandit] Lista de modelos vazia; retornando default.")
        return "ollama/gemma3:4b"

    # Contextos automáticos por clustering + modalidade
    contexts = _auto_context_labels(query, modality)
    main_ctx = contexts[0] if contexts else "global"

    # Stats do contexto principal
    ctx_stats = _get_ctx_stats(main_ctx)

    # Meta-bandit híbrido
    chosen, debug_choices = _meta_combine_choices(valid_models, ctx_stats)

    logger.info(
        "[bandit] ctx=%s | models=%s | chosen=%s | eps=%s | ucb1=%s | ts=%s",
        main_ctx,
        valid_models,
        chosen,
        debug_choices["epsilon_greedy"],
        debug_choices["ucb1"],
        debug_choices["thompson"],
    )

    try:
        BANDIT_SELECT.labels(model=chosen).inc()
    except Exception:
        pass

    return chosen


async def select_model_async(
    valid_models: List[str],
    query: str,
    modality: str = "text",
) -> str:
    """Async model selection using non-blocking Redis reads on the hot path."""
    if not valid_models:
        logger.warning("[bandit] Lista de modelos vazia; retornando default.")
        return "ollama/gemma3:4b"

    contexts = await _auto_context_labels_async(query, modality)
    main_ctx = contexts[0] if contexts else "global"
    ctx_stats = await _get_ctx_stats_async(main_ctx)
    chosen, debug_choices = await _meta_combine_choices_async(valid_models, ctx_stats)

    logger.info(
        "[bandit] ctx=%s | models=%s | chosen=%s | eps=%s | ucb1=%s | ts=%s",
        main_ctx,
        valid_models,
        chosen,
        debug_choices["epsilon_greedy"],
        debug_choices["ucb1"],
        debug_choices["thompson"],
    )

    try:
        BANDIT_SELECT.labels(model=chosen).inc()
    except Exception:
        pass

    return chosen

# ============================================================

def bandit_update(
    model: str,
    query: str,
    reward: float,
    modality: str = "text",
) -> None:
    """
    Update contextual bandit statistics after observing one reward value.

    The update touches every derived context for the query, keeps Welford moments
    for stable variance estimates, and refreshes the Beta parameters used by
    Thompson sampling.
    """
    # normaliza reward
    try:
        r = float(reward)
    except Exception:
        r = 0.0
    r = max(0.0, min(1.0, r))

    # reforça centróides com a query atual (não crítico se falhar)
    try:
        centroids_online_update(query)
    except Exception:
        pass

    contexts = _auto_context_labels(query, modality)
    db_updates: list[tuple[str, str, Dict[str, float]]] = []
    for ctx in contexts:
        try:
            stats = _get_ctx_stats(ctx)
            cur = stats.get(
                model,
                {
                    "mean": 0.0,
                    "count": 0,
                    "var": 0.0,
                    "M2": 0.0,
                    "alpha": 1.0,
                    "beta": 1.0,
                },
            )
            cur = _sanitize_model_stats(cur)

            mean = float(cur.get("mean", 0.0))
            count = int(cur.get("count", 0))
            M2 = float(cur.get("M2", 0.0))

            # ---- Welford para mean/var ----
            count += 1
            delta = r - mean
            mean += delta / max(1, count)
            delta2 = r - mean
            M2 += delta * delta2
            var = M2 / (count - 1) if count > 1 else 0.0

            # ---- Beta para TS ----
            alpha = float(cur.get("alpha", 1.0)) + r
            beta = float(cur.get("beta", 1.0)) + (1.0 - r)
            if alpha <= 0 or not math.isfinite(alpha):
                alpha = 1.0
            if beta <= 0 or not math.isfinite(beta):
                beta = 1.0

            stats[model] = {
                "mean": mean,
                "count": count,
                "var": var,
                "M2": M2,
                "alpha": alpha,
                "beta": beta,
            }
            stats[model] = _sanitize_model_stats(stats[model])

            # Redis + DB (batched)
            _set_ctx_stats(ctx, stats)
            db_updates.append((ctx, model, stats[model]))

        except Exception as e:
            logger.warning(
                f"[bandit] Falha ao atualizar contexto ctx={ctx}, model={model}: {e}"
            )

    if db_updates:
        _batch_upsert_ctx_db(db_updates)

    try:
        BANDIT_UPDATE.labels(model=model).inc()
        BANDIT_REWARD.observe(float(r))
    except Exception:
        pass

    # --- HOOK PARA GET_SNAPSHOT (USADO PELO UQ/ROUTER_STRATEGY) ---
    # O método sample_metrics_from_snapshot é chamado pelo router_strategy.
    # Ele espera que o snapshot (stats) esteja disponível.
    # O Redis já foi atualizado acima via _set_ctx_stats, então está ok.


# ============================================================
# Helpers para UQ e Router Strategy (Compatibilidade)
# ============================================================

def get_snapshot() -> Dict[str, Any]:
    """
    Helper para router_strategy.py.
    Recupera um snapshot de todos os modelos do Redis.
    Para simplificar, podemos iterar sobre o contexto 'global' ou 'mod:*'.
    """
    # Como temos múltiplos contextos, o snapshot ideal seria o agregado.
    # Para o UQ/Strategy, usamos o contexto global por padrão.
    return _get_ctx_stats("global")

def sample_metrics_from_snapshot(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """
    Helper para router_strategy.py.
    Amostra qualidade (Thompson Sampling) do snapshot.
    """
    samples = {}
    for model, stats in snapshot.items():
        alpha = stats.get("alpha", 1.0)
        beta = stats.get("beta", 1.0)
        # Amostra da Beta distribution e projeta para escala 0-10 (como 'quality')
        val = np.random.beta(alpha, beta)
        samples[f"{model}::text"] = val * 10.0 # Assume text por padrão se chave simples
        samples[model] = val * 10.0
    return samples


def reset_bandits_runtime_state() -> None:
    """Reset in-memory runtime caches/singletons (test/dev utility)."""
    _centroid_matrix_cache.update([])
    cold_contexts.clear()
    reset_centroid_matrix_cache()
