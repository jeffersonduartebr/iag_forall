# -*- coding: utf-8 -*-
"""
bandits.py — Meta-Bandit Multimodal Completo
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
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.settings_dynamic import settings
from app.utils.redis_client import get_redis
from app.embeddings import embed_text
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
# DB
# ============================================================
DB_HOST = settings.DB_HOST
DB_USER = settings.DB_USER
DB_PASS = settings.DB_PASS
DB_NAME = settings.DB_NAME

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

# ============================================================
# Redis
# ============================================================
rds = get_redis()

# Redis Keys (ajustadas)
R_CTX_PREFIX = "meta:bandit:ctx"                  # por contexto → stats
R_META_STRATEGY = "meta:bandit:strategy"          # qual estratégia meta-bandit venceu
R_CENTROIDS = "meta:bandit:centroids"             # lista completa de centróides
R_CENTROIDS_META = "meta:bandit:centroids:meta"
R_CENTROIDS_LOCK = "meta:bandit:centroids:lock"
R_CLUSTERING_MODEL = "meta:bandit:cluster:model"  # clustering automático
R_CLUSTERING_LOCK = "meta:bandit:cluster:lock"

# ============================================================
# Hyperparams
# ============================================================
DEFAULT_EPSILON = float(settings.get("BANDIT_EPSILON", 0.12))
CENTROIDS_K = int(settings.get("CENTROIDS_K", 20))
CENTROIDS_DIM = int(settings.get("CENTROIDS_DIM", 768))
CENTROIDS_LEARN_RATE = float(settings.get("CENTROIDS_LEARN_RATE", 0.15))
CENTROIDS_MIN_SIM_CREATE = float(settings.get("CENTROIDS_MIN_SIM_CREATE", 0.35))
CENTROIDS_MIN_RECORDS_FOR_TRAIN = int(settings.get("CENTROIDS_MIN_RECORDS_FOR_TRAIN", 50))

# Meta-Bandit: estratégias
META_STRATEGIES = ["epsilon_greedy", "ucb1", "thompson"]

# ============================================================
# Utils NumPy
# ============================================================
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _ensure_dim(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32).reshape(-1)
    if len(v) != CENTROIDS_DIM:
        # fallback seguro: pad/trim
        if len(v) > CENTROIDS_DIM:
            v = v[:CENTROIDS_DIM]
        else:
            pad = CENTROIDS_DIM - len(v)
            v = np.concatenate([v, np.zeros(pad, dtype=np.float32)])
    return _unit(v)

# ============================================================
# DDL — tabela “bandit_context_stats”
# ============================================================

DDL = """
CREATE TABLE IF NOT EXISTS bandit_context_stats (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    context_label VARCHAR(255) NOT NULL,
    model VARCHAR(255) NOT NULL,
    avg_reward FLOAT DEFAULT 0,
    count INT DEFAULT 0,
    var FLOAT DEFAULT 0,
    M2 FLOAT DEFAULT 0,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_ctx_model (context_label, model)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

def _init_tables():
    try:
        with engine.begin() as conn:
            conn.execute(text(DDL))
        logger.info("[bandit] Tabela bandit_context_stats criada/verificada.")
    except Exception as e:
        logger.warning(f"[bandit] Falha ao criar tabelas: {e}")

_init_tables()


# ============================================================
# Centróides semânticos (NumPy vetorizado) + clustering dinâmico
# ============================================================

def _acquire_lock(key: str, ttl: int = 10) -> bool:
    if not rds:
        return False
    try:
        return bool(rds.set(key, "1", nx=True, ex=ttl))
    except Exception:
        return False


def _release_lock(key: str) -> None:
    if not rds:
        return
    try:
        rds.delete(key)
    except Exception:
        pass


def _load_centroids() -> List[dict]:
    """
    Carrega centróides de Redis:
    [
      {"id": int, "vec": np.ndarray(D,), "count": int, "last": int}
    ]
    """
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
        return cents
    except Exception as e:
        logger.warning(f"[centroids] Falha ao carregar: {e}")
        return []


def _save_centroids(cents: List[dict]) -> None:
    """
    Persiste centróides com reinicialização automática de degenerados.
    """
    if not rds:
        return
    serial = []
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
    except Exception as e:
        logger.warning(f"[centroids] Falha ao salvar: {e}")


def _new_centroid_id(cents: List[dict]) -> int:
    used = {c["id"] for c in cents}
    cid = 0
    while cid in used:
        cid += 1
    return cid


def _nearest_centroid_vec(
    v: np.ndarray, cents: List[dict]
) -> Tuple[Optional[int], float]:
    """
    Versão vetorizada: empilha centróides e faz produto escalar.
    Assumimos vetores unitários.
    """
    if not cents:
        return None, 0.0

    C = np.stack([c["vec"] for c in cents], axis=0)  # (K, D)
    sims = C @ v  # produto escalar (unit vectors → cosine)
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])


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
                stats[model] = {
                    "mean": float(obj.get("mean", obj.get("avg", 0.0))),
                    "count": int(obj.get("count", 0)),
                    "var": float(obj.get("var", 0.0)),
                    "M2": float(obj.get("M2", 0.0)),
                    "alpha": float(obj.get("alpha", 1.0)),
                    "beta": float(obj.get("beta", 1.0)),
                }
        except Exception as e:
            logger.warning(f"[bandit] Falha Redis ctx={ctx}: {e}")

    # Se vazio, tenta DB
    if not stats:
        try:
            with engine.connect() as conn:
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
                stats[model] = {
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
        except Exception as e:
            logger.warning(f"[bandit] Falha DB ctx={ctx}: {e}")

    return stats


def _set_ctx_stats(ctx: str, stats: Dict[str, Dict[str, float]]) -> None:
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
    try:
        with engine.begin() as conn:
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
    eps = DEFAULT_EPSILON
    if not ctx_stats:
        return min(1.0, eps + 0.15)

    counts = [s.get("count", 0) for s in ctx_stats.values()]
    if counts and min(counts) < 3:
        eps += 0.10

    vars_ = [s.get("var", 0.0) for s in ctx_stats.values()]
    if vars_ and float(np.mean(vars_)) > 0.05:
        eps += 0.08

    return max(0.0, min(1.0, eps))


def _choose_epsilon_greedy(
    models: List[str], ctx_stats: Dict[str, Dict[str, float]]
) -> str:
    eps = _dynamic_epsilon(ctx_stats)
    if random.random() < eps:
        # exploração: escolhe modelo com maior variância / menor count
        scored = []
        for m in models:
            s = ctx_stats.get(m, {})
            cnt = s.get("count", 0)
            var = s.get("var", 0.0)
            expl = (1.0 / (1.0 + cnt)) + 0.5 * var
            scored.append((m, expl))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
    else:
        # aproveitamento: maior média
        scored = []
        for m in models:
            s = ctx_stats.get(m, {})
            mean = s.get("mean", 0.0)
            scored.append((m, mean))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]


def _choose_ucb1(models: List[str], ctx_stats: Dict[str, Dict[str, float]]) -> str:
    total = sum(s.get("count", 0) for s in ctx_stats.values())
    if total <= 0:
        total = 1
    C = 1.4
    scores = []
    for m in models:
        s = ctx_stats.get(m, {})
        mean = s.get("mean", 0.0)
        cnt = s.get("count", 0)
        if cnt <= 0:
            bonus = float("inf")  # força serem escolhidos ao menos uma vez
        else:
            bonus = C * math.sqrt(math.log(total + 1.0) / cnt)
        scores.append((m, mean + bonus))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]


def _choose_thompson(models: List[str], ctx_stats: Dict[str, Dict[str, float]]) -> str:
    cand = []
    for m in models:
        s = ctx_stats.get(m, {})
        alpha = float(s.get("alpha", 1.0))
        beta = float(s.get("beta", 1.0))
        # proteção contra degeneração
        if alpha <= 0 or beta <= 0 or not math.isfinite(alpha) or not math.isfinite(beta):
            alpha, beta = 1.0, 1.0
        sample = np.random.beta(alpha, beta)
        cand.append((m, float(sample)))
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[0][0]


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
    if not models:
        raise RuntimeError("Nenhum modelo recebido em _meta_combine_choices.")

    eps_choice = _choose_epsilon_greedy(models, ctx_stats)
    ucb_choice = _choose_ucb1(models, ctx_stats)
    ts_choice = _choose_thompson(models, ctx_stats)

    votes: Dict[str, int] = {}
    for m in (eps_choice, ucb_choice, ts_choice):
        votes[m] = votes.get(m, 0) + 1

    # modelo com maior número de votos
    best_model = None
    best_votes = -1
    for m, v in votes.items():
        if v > best_votes:
            best_model = m
            best_votes = v

    if best_votes >= 2:
        chosen = best_model  # maioria absoluta
    else:
        # todos diferentes → consulta meta-stratégia default
        strat = _meta_choose_strategy()
        if strat == "epsilon_greedy":
            chosen = eps_choice
        elif strat == "thompson":
            chosen = ts_choice
        else:  # ucb1
            chosen = ucb_choice

    return chosen, {
        "epsilon_greedy": eps_choice,
        "ucb1": ucb_choice,
        "thompson": ts_choice,
    }


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
