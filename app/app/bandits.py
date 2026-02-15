# -*- coding: utf-8 -*-
"""
bandits.py — Meta-Bandit Multimodal Completo + UQ Support
==========================================================
Compatível com:
- router_core.py
- settings_dynamic.py
- embeddings multimodais
- judges
- RAG multimodal
- query_log multimodal

Inclui:
- ε-greedy + UCB1 + Thompson Sampling (Meta-Bandit)
- Contextos automáticos via clustering dinâmico
- Centróides semânticos vetorizados (NumPy)
- Reinicialização automática de centróides degenerados
- NSGA-II weight-aware reward
- Toda persistência em Redis + MariaDB
- NOVO: Persistência de centróides para Quantificação de Incerteza (UQ)

APIs públicas:
- select_model(valid_models, query, modality="text")
- bandit_update(model, query, reward, modality="text")
- compute_reward(model, quality, latency_s, cost_per_1k)
"""

from __future__ import annotations
import os
import json
import math
import time
import random
import logging
import threading
from typing import Dict, List, Tuple, Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.settings_dynamic import settings
from app.db import get_engine
from app.utils.redis_client import get_redis_async_safe, ensure_redis_connected
from app.embeddings import embed_text
from app.services import bandit_policy
from app.services.bandit_centroids import (
    normalize_centroid_vec,
    nearest_centroid_from_array,
)
from app.observability import (
    BANDIT_SELECT,
    BANDIT_UPDATE,
    BANDIT_REWARD,
)

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
    """Executa get rds."""
    return get_redis_async_safe() or ensure_redis_connected(max_wait_s=0.0, min_retry_interval_s=2.0)

# Redis Keys (ajustadas)
R_CTX_PREFIX = "meta:bandit:ctx"                  # por contexto → stats
R_META_STRATEGY = "meta:bandit:strategy"          # qual estratégia meta-bandit venceu
R_CENTROIDS = "meta:bandit:centroids"             # lista completa de centróides (UQ)
R_CENTROIDS_META = "meta:bandit:centroids:meta"
R_CENTROIDS_LOCK = "meta:bandit:centroids:lock"
R_CLUSTERING_MODEL = "meta:bandit:cluster:model"  # clustering automático
R_CLUSTERING_LOCK = "meta:bandit:cluster:lock"

# ============================================================
# Hyperparams
# ============================================================
def _safe_setting_float(key: str, default: float) -> float:
    """Executa safe setting float."""
    try:
        return float(settings.get(key, default))
    except Exception:
        return float(default)


def _safe_setting_int(key: str, default: int) -> int:
    """Executa safe setting int."""
    try:
        return int(settings.get(key, default))
    except Exception:
        return int(default)


DEFAULT_EPSILON = _safe_setting_float("BANDIT_EPSILON", 0.12)
CENTROIDS_K = _safe_setting_int("CENTROIDS_K", 20)
CENTROIDS_DIM = _safe_setting_int("CENTROIDS_DIM", 768)
CENTROIDS_LEARN_RATE = _safe_setting_float("CENTROIDS_LEARN_RATE", 0.15)
CENTROIDS_MIN_SIM_CREATE = _safe_setting_float("CENTROIDS_MIN_SIM_CREATE", 0.35)
CENTROIDS_MIN_RECORDS_FOR_TRAIN = _safe_setting_int("CENTROIDS_MIN_RECORDS_FOR_TRAIN", 50)

# Meta-Bandit: estratégias
META_STRATEGIES = ["epsilon_greedy", "ucb1", "thompson"]

# ============================================================
# Utils NumPy
# ============================================================
def _unit(v: np.ndarray) -> np.ndarray:
    """Executa unit."""
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Executa cosine."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _ensure_dim(v: np.ndarray) -> np.ndarray:
    """Executa ensure dim."""
    return normalize_centroid_vec(v, CENTROIDS_DIM)


def _sanitize_model_stats(raw: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Centralized sanitization for per-model stats used by selection/update."""
    s = raw or {}

    def _f(key: str, default: float) -> float:
        """Executa f."""
        try:
            v = float(s.get(key, default))
            return v if math.isfinite(v) else default
        except Exception:
            return default

    def _i(key: str, default: int) -> int:
        """Executa i."""
        try:
            v = int(s.get(key, default))
            return max(0, v)
        except Exception:
            return default

    out = {
        "mean": max(0.0, min(1.0, _f("mean", _f("avg", 0.0)))),
        "count": _i("count", 0),
        "var": max(0.0, _f("var", 0.0)),
        "M2": max(0.0, _f("M2", 0.0)),
        "alpha": max(1e-6, _f("alpha", 1.0)),
        "beta": max(1e-6, _f("beta", 1.0)),
    }
    return out


# ============================================================
# Pre-computed Centroid Matrix Cache (Performance Optimization)
# ============================================================
class CentroidMatrixCache:
    """Cache para matriz de centróides pré-computada."""

    def __init__(self):
        """Inicializa estado interno necessário para uso da classe."""
        self._lock = threading.Lock()
        self._matrix: Optional[np.ndarray] = None  # (K, D)
        self._ids: List[int] = []
        self._version: int = 0
        self._last_update: float = 0.0

    def update(self, cents: List[dict]) -> None:
        """Atualiza a matriz cache com os centróides atuais."""
        if not cents:
            with self._lock:
                self._matrix = None
                self._ids = []
                self._version += 1
            return

        with self._lock:
            self._matrix = np.stack([c["vec"] for c in cents], axis=0).astype(np.float32)
            self._ids = [c["id"] for c in cents]
            self._version += 1
            self._last_update = time.time()

    def nearest(self, v: np.ndarray) -> Tuple[Optional[int], float, int]:
        """
        Busca o centróide mais próximo usando a matriz pré-computada.
        Retorna: (índice no array original, similaridade, centroid_id)
        """
        with self._lock:
            if self._matrix is None or len(self._ids) == 0:
                return None, 0.0, -1

            # Produto escalar vetorizado (unit vectors → cosine)
            sims = self._matrix @ v
            idx = int(np.argmax(sims))
            return idx, float(sims[idx]), self._ids[idx]

    def is_stale(self, max_age_s: float = 60.0) -> bool:
        """Verifica se o cache está desatualizado."""
        return time.time() - self._last_update > max_age_s


_centroid_matrix_cache = CentroidMatrixCache()

# ============================================================
# Centróides semânticos (NumPy vetorizado) + clustering dinâmico
# ============================================================

def _acquire_lock(key: str, ttl: int = 10) -> bool:
    """Executa acquire lock."""
    rds = _get_rds()
    if not rds:
        return False
    try:
        return bool(rds.set(key, "1", nx=True, ex=ttl))
    except Exception:
        return False


def _release_lock(key: str) -> None:
    """Executa release lock."""
    rds = _get_rds()
    if not rds:
        return
    try:
        rds.delete(key)
    except Exception:
        pass


def _load_centroids(update_matrix_cache: bool = True) -> List[dict]:
    """
    Carrega centróides de Redis:
    [
      {"id": int, "vec": np.ndarray(D,), "count": int, "last": int}
    ]
    """
    rds = _get_rds()
    if not rds:
        return []
    try:
        raw = rds.get(R_CENTROIDS)
        if not raw:
            return []
        arr = json.loads(raw)
        cents: List[dict] = []
        for it in arr:
            if not isinstance(it, dict):
                continue
            if "id" not in it or "vec" not in it:
                continue
            vec = np.array(it["vec"], dtype=np.float32)
            vec = _ensure_dim(vec)
            cnt = int(it.get("count", 0))
            last = int(it.get("last", int(time.time())))
            cents.append({"id": int(it["id"]), "vec": vec, "count": cnt, "last": last})

        # Update the pre-computed matrix cache
        if update_matrix_cache and cents:
            _centroid_matrix_cache.update(cents)

        return cents
    except Exception as e:
        logger.warning(f"[centroids] Falha ao carregar: {e}")
        return []


def _save_centroids(cents: List[dict]) -> None:
    """
    Persiste centróides com reinicialização automática de degenerados.
    """
    rds = _get_rds()
    if not rds:
        return
    serial = []
    normalized_cents = []
    now_ts = int(time.time())

    for it in cents:
        vec = np.array(it["vec"], dtype=np.float32).reshape(-1)
        # reinicialização de degenerados
        if not np.isfinite(vec).all() or np.linalg.norm(vec) < 1e-4:
            vec = np.random.normal(size=(CENTROIDS_DIM,)).astype(np.float32)
            vec = _unit(vec)
            cnt = 0
        else:
            vec = _unit(vec)
            cnt = int(it.get("count", 0))

        serial.append(
            {
                "id": int(it["id"]),
                "vec": vec.tolist(),
                "count": cnt,
                "last": int(it.get("last", now_ts)),
            }
        )
        normalized_cents.append({"id": int(it["id"]), "vec": vec, "count": cnt, "last": int(it.get("last", now_ts))})

    try:
        pipe = rds.pipeline()
        pipe.set(R_CENTROIDS, json.dumps(serial))
        pipe.hset(
            R_CENTROIDS_META,
            mapping={
                "updated_at": str(now_ts),
                "k": str(CENTROIDS_K),
                "dim": str(CENTROIDS_DIM),
                "count": str(len(serial)),
            },
        )
        pipe.execute()

        # Update the pre-computed matrix cache
        _centroid_matrix_cache.update(normalized_cents)

    except Exception as e:
        logger.warning(f"[centroids] Falha ao salvar: {e}")


def _new_centroid_id(cents: List[dict]) -> int:
    """Executa new centroid id."""
    used = {c["id"] for c in cents}
    cid = 0
    while cid in used:
        cid += 1
    return cid


def _nearest_centroid_vec(
    v: np.ndarray, cents: List[dict], use_cache: bool = True
) -> Tuple[Optional[int], float]:
    """
    Versão vetorizada: empilha centróides e faz produto escalar.
    Assumimos vetores unitários.

    Se use_cache=True e o cache da matriz está atualizado, usa o cache.
    """
    if not cents:
        return None, 0.0

    # Try to use pre-computed matrix cache for faster lookup
    if use_cache and not _centroid_matrix_cache.is_stale(max_age_s=120.0):
        idx, sim, _ = _centroid_matrix_cache.nearest(v)
        if idx is not None:
            return idx, sim

    # Fallback: compute on-the-fly
    return nearest_centroid_from_array(v, cents)


def centroids_online_update(query_text: str) -> Optional[int]:
    """
    Atualização ONLINE:
      - embed_text(query_text) → v (np.ndarray normalizado)
      - se não há centróides: cria o primeiro
      - senão: acha centróide mais próximo
           * se sim < CENTROIDS_MIN_SIM_CREATE e len < K → cria novo
           * senão: update exponencial c := (1-α)c + αv ; renorma; count++
    Retorna ID do centróide associado ou None em caso de falha.
    """
    try:
        v = embed_text(query_text)
        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype=np.float32)
        v = _ensure_dim(v)
    except Exception as e:
        logger.debug(f"[centroids] Falha ao gerar embedding: {e}")
        return None

    if not _acquire_lock(R_CENTROIDS_LOCK, ttl=5):
        return None

    try:
        cents = _load_centroids()
        if not cents:
            cid = 0
            cents = [{"id": cid, "vec": v, "count": 1, "last": int(time.time())}]
            _save_centroids(cents)
            return cid

        idx, sim = _nearest_centroid_vec(v, cents)

        if (idx is None) or (sim < CENTROIDS_MIN_SIM_CREATE and len(cents) < CENTROIDS_K):
            cid = _new_centroid_id(cents)
            cents.append(
                {"id": cid, "vec": v, "count": 1, "last": int(time.time())}
            )
            _save_centroids(cents)
            return cid

        # Update exponencial no centróide existente
        c = cents[idx]
        new_vec = (1.0 - CENTROIDS_LEARN_RATE) * c["vec"] + CENTROIDS_LEARN_RATE * v
        new_vec = _unit(new_vec.astype(np.float32))
        c["vec"] = new_vec
        c["count"] = int(c.get("count", 0)) + 1
        c["last"] = int(time.time())
        _save_centroids(cents)
        return c["id"]
    finally:
        _release_lock(R_CENTROIDS_LOCK)


def _nearest_centroid_label(query_text: str) -> Optional[str]:
    """
    Somente leitura: retorna 'semctx:<id>' do centróide mais próximo.
    NÃO cria centróides novos (usa embedding atual).
    Usado pelo router_core para logging.
    """
    try:
        v = embed_text(query_text)
        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype=np.float32)
        v = _ensure_dim(v)
    except Exception:
        return None

    cents = _load_centroids()
    if not cents:
        return None

    idx, sim = _nearest_centroid_vec(v, cents)
    if idx is None:
        return None
    cid = int(cents[idx]["id"])
    return f"semctx:{cid}"

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
    """Executa ctx key."""
    return f"{R_CTX_PREFIX}:{ctx}"


def _get_ctx_stats(ctx: str) -> Dict[str, Dict[str, float]]:
    """
    Lê stats de um contexto:
    {
      "modelA": {"mean":..., "count":..., "var":..., "M2":..., "alpha":..., "beta":...},
      ...
    }
    """
    stats: Dict[str, Dict[str, float]] = {}

    # Redis
    rds = _get_rds()
    if rds:
        try:
            raw = rds.hgetall(_ctx_key(ctx))
            for k, v in raw.items():
                model = k.decode() if isinstance(k, bytes) else k
                try:
                    obj = json.loads(v)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                stats[model] = _sanitize_model_stats(obj)
        except Exception as e:
            logger.warning(f"[bandit] Falha Redis ctx={ctx}: {e}")

    # Se vazio, tenta DB
    if not stats:
        try:
            with _get_db_engine().connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT model, avg_reward, count, var, M2
                        FROM bandit_context_stats
                        WHERE context_label = :ctx
                        """
                    ),
                    {"ctx": ctx},
                ).fetchall()
            for row in rows:
                model = row[0]
                stats[model] = _sanitize_model_stats(
                    {
                        "mean": float(row[1] or 0.0),
                        "count": int(row[2] or 0),
                        "var": float(row[3] or 0.0),
                        "M2": float(row[4] or 0.0),
                        "alpha": 1.0 + float(row[1] or 0.0) * float(row[2] or 0.0),
                        "beta": 1.0 + max(
                            0.0,
                            float(row[2] or 0.0)
                            - float(row[1] or 0.0) * float(row[2] or 0.0),
                        ),
                    }
                )
        except Exception as e:
            logger.warning(f"[bandit] Falha DB ctx={ctx}: {e}")

    return stats


def _set_ctx_stats(ctx: str, stats: Dict[str, Dict[str, float]]) -> None:
    """Executa set ctx stats."""
    rds = _get_rds()
    if not rds:
        return
    try:
        key = _ctx_key(ctx)
        pipe = rds.pipeline()
        for model, s in stats.items():
            payload = json.dumps(
                {
                    "mean": float(s.get("mean", 0.0)),
                    "count": int(s.get("count", 0)),
                    "var": float(s.get("var", 0.0)),
                    "M2": float(s.get("M2", 0.0)),
                    "alpha": float(s.get("alpha", 1.0)),
                    "beta": float(s.get("beta", 1.0)),
                }
            )
            pipe.hset(key, model, payload)
        pipe.execute()
    except Exception as e:
        logger.warning(f"[bandit] Falha ao salvar ctx={ctx} no Redis: {e}")


def _upsert_ctx_db(ctx: str, model: str, s: Dict[str, float]) -> None:
    """Executa upsert ctx db."""
    try:
        with _get_db_engine().begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO bandit_context_stats
                      (context_label, model, avg_reward, count, var, M2)
                    VALUES (:ctx, :model, :avg, :count, :var, :M2)
                    ON DUPLICATE KEY UPDATE
                      avg_reward = :avg,
                      count = :count,
                      var = :var,
                      M2 = :M2,
                      last_update = CURRENT_TIMESTAMP
                    """
                ),
                {
                    "ctx": ctx,
                    "model": model,
                    "avg": float(s.get("mean", 0.0)),
                    "count": int(s.get("count", 0)),
                    "var": float(s.get("var", 0.0)),
                    "M2": float(s.get("M2", 0.0)),
                },
            )
    except Exception as e:
        logger.warning(
            f"[bandit] Falha no upsert DB ctx={ctx}, model={model}: {e}"
        )
# ============================================================


# ============================================================
# Estratégias básicas: ε-greedy, UCB1, Thompson Sampling
# ============================================================

def _dynamic_epsilon(ctx_stats: Dict[str, Dict[str, float]]) -> float:
    """Executa dynamic epsilon."""
    return bandit_policy.dynamic_epsilon(ctx_stats, DEFAULT_EPSILON)


def _choose_epsilon_greedy(
    models: List[str], ctx_stats: Dict[str, Dict[str, float]]
) -> str:
    """Executa choose epsilon greedy."""
    return bandit_policy.choose_epsilon_greedy(models, ctx_stats, DEFAULT_EPSILON)


def _choose_ucb1(models: List[str], ctx_stats: Dict[str, Dict[str, float]]) -> str:
    """Executa choose ucb1."""
    return bandit_policy.choose_ucb1(models, ctx_stats)


def _choose_thompson(models: List[str], ctx_stats: Dict[str, Dict[str, float]]) -> str:
    """Executa choose thompson."""
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


def _meta_combine_choices(
    models: List[str],
    ctx_stats: Dict[str, Dict[str, float]],
) -> Tuple[str, Dict[str, str]]:
    """
    Executa as três estratégias e combina por:
      1) voto majoritário; se empate total, prioriza meta-stratégia.
    Retorna (modelo_escolhido, {"eps":..., "ucb1":..., "ts":...}).
    """
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
    Seleção de modelo via Meta-Bandit híbrido.

    Compatível com router_core:
      select_model(top2, query, modality=modality)
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

# ============================================================

# ============================================================
# Atualização do bandit com recompensa observada
# ============================================================

def bandit_update(
    model: str,
    query: str,
    reward: float,
    modality: str = "text",
) -> None:
    """
    Atualiza estatísticas Welford (mean/var) + Beta(α,β) por contexto e modelo.
    Compatível com router_core.bandit_update(model, query, reward, modality=...).
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

            # Redis + DB
            _set_ctx_stats(ctx, stats)
            _upsert_ctx_db(ctx, model, stats[model])

        except Exception as e:
            logger.warning(
                f"[bandit] Falha ao atualizar contexto ctx={ctx}, model={model}: {e}"
            )

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
# Recompensa NSGA-II aware
# ============================================================

def _load_nsga_weights() -> Tuple[float, float, float]:
    """
    Lê pesos NSGA-II de Redis (nsga:weights) ou usa defaults.
    Formato esperado:
      {"quality": 0.55, "latency": 0.30, "cost": 0.15}
    """
    w_q, w_l, w_c = 0.55, 0.30, 0.15
    rds = _get_rds()
    if not rds:
        return w_q, w_l, w_c
    try:
        raw = rds.get("nsga:weights")
        if not raw:
            return w_q, w_l, w_c
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return w_q, w_l, w_c
        w_q = float(obj.get("quality", w_q))
        w_l = float(obj.get("latency", w_l))
        w_c = float(obj.get("cost", w_c))
        total = w_q + w_l + w_c
        if total <= 0:
            return 0.55, 0.30, 0.15
        w_q, w_l, w_c = w_q / total, w_l / total, w_c / total
        return w_q, w_l, w_c
    except Exception as e:
        logger.warning(f"[bandit] Falha ao carregar nsga:weights: {e}")
        return w_q, w_l, w_c


def compute_reward(
    model: str,
    quality: float,
    latency_s: float,
    cost_per_1k: Optional[float] = None,
) -> float:
    """
    Converte métricas em recompensa [0..1] com pesos NSGA-II dinâmicos.

    - quality  ∈ [0..10] → normalizado [0..1]
    - latência: logística com x0=20s (quanto menor melhor)
    - custo: penaliza custos acima de baseline; bonifica próximos ao local
    """
    try:
        w_q, w_l, w_c = _load_nsga_weights()

        # qualidade
        q = max(0.0, min(10.0, float(quality))) / 10.0

        # latência (logística invertida)
        L, k, x0 = 1.0, 0.12, 20.0
        lat_score = L / (1.0 + math.exp(k * (latency_s - x0)))
        lat_score = max(0.0, min(1.0, lat_score))

        # custo
        cost_score = 1.0
        if cost_per_1k is not None:
            baseline = 0.12
            ratio = float(cost_per_1k) / baseline if baseline > 0 else 1.0
            # >1 significa mais caro que baseline
            cost_score = 1.0 / (1.0 + max(0.0, ratio - 1.0))
            cost_score = max(0.3, min(1.0, cost_score))

        reward = w_q * q + w_l * lat_score + w_c * cost_score
        reward = max(0.0, min(1.0, float(reward)))
        return reward
    except Exception as e:
        logger.warning(f"[bandit] Falha em compute_reward: {e}")
        return 0.0

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
