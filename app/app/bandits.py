# -*- coding: utf-8 -*-
"""
bandits.py
----------------------------------------------------
Bandit contextual dinâmico + centróides semânticos online:

- Contextos granulares (12+ categorias) via análise de texto.
- Centróides semânticos ONLINE (incremental) em Redis:
    * Embeddings via embeddings.embed_text (np.ndarray normalizado).
    * Atribui query ao centróide mais próximo (cosine).
    * Se sim < MIN_SIM_CREATE e #centroides < K → cria novo centróide.
    * Atualização incremental (exponencial) c_{t+1} = (1-α) c_t + α x, com renormalização.
    * Mantém contagem por centróide.
    * Persistência em Redis (chaves bandit:centroids:*).

- Rotina horária (thread) para manutenção:
    * Só ativa se houver pelo menos 50 linhas em query_log (DB).
    * Se há mais que K centróides → mescla pares mais similares até K.
    * Atualiza metadados em Redis.

- Bandit ε-greedy contextual:
    * Estatísticas por (contexto, modelo) em Redis e MariaDB.
    * Welford para variância (armazenada no Redis).
    * Integra a etiqueta "semctx:<id>" (centróide mais próximo) à lista de contextos.

APIs públicas:
- select_model(valid_models: list[str], query: str) -> str
- bandit_update(model: str, query: str, reward: float) -> None
- compute_reward(model: str, quality: float, latency_s: float, cost_per_1k: float | None = None) -> float

Este módulo NÃO altera embeddings.py. Usa:
    from app.embeddings import embed_text, embed_many
"""

from __future__ import annotations

import os
import json
import math
import time
import random
import logging
import threading
import statistics
from typing import Dict, List, Tuple, Optional

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.settings_dynamic import settings
from app.utils.redis_client import get_redis
from app.observability import BANDIT_SELECT, BANDIT_UPDATE, BANDIT_REWARD
from app.embeddings import embed_text  # <- compatível com seu embeddings.py

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] bandit: %(message)s")

# ============================================================
# 🔧 Conexões / Config
# ============================================================

DB_HOST = settings.get("DB_HOST", os.getenv("DB_HOST", "mariadb"))
DB_USER = settings.get("DB_USER", os.getenv("DB_USER", "router_user"))
DB_PASS = settings.get("DB_PASS", os.getenv("DB_PASS", "router_pass"))
DB_NAME = settings.get("DB_NAME", os.getenv("DB_NAME", "routerdb"))
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

rds = get_redis()

# Chaves Redis
R_KEY_BANDIT_CTXT_PREFIX = "bandit:ctx"       # bandit:ctx:<ctx> -> hash {model: json(stats)}
R_KEY_BANDIT_CTXT_META   = "bandit:ctx:meta"
R_KEY_CENTROIDS_V        = "bandit:centroids:v"        # versão de schema (p/ futura evolução)
R_KEY_CENTROIDS          = "bandit:centroids:data"     # lista de centróides [{"id":int,"vec":[...],"count":int}]
R_KEY_CENTROIDS_META     = "bandit:centroids:meta"     # {"updated_at": ts, "k":K, "dim":D}
R_KEY_CENTROIDS_LOCK     = "bandit:centroids:lock"     # simples lock

# Defaults inteligentes
DEFAULT_EPSILON = 0.12
MIN_OBS_FOR_EXPLOIT = 3
EPSILON_BOOST_UNDEREXP = 0.10
CONTEXT_VARIANCE_BOOST = 0.08

# Centróides (com defaults se não existirem no settings_dynamic)
CENTROIDS_DIM = int(settings.get("CENTROIDS_DIM", 768))
CENTROIDS_K = int(settings.get("CENTROIDS_K", 20))
CENTROIDS_MIN_SIM_CREATE = float(settings.get("CENTROIDS_MIN_SIM_CREATE", 0.35))
CENTROIDS_LEARN_RATE = float(settings.get("CENTROIDS_LEARN_RATE", 0.15))  # α do update exponencial
CENTROIDS_HOURLY_REFRESH_ENABLED = str(settings.get("CENTROIDS_HOURLY_REFRESH_ENABLED", "true")).lower() in ("1","true","yes","on")
CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH = int(settings.get("CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH", 50))

# ============================================================
# 🧱 DDL: Estatísticas contextuais do bandit
# ============================================================

DDL_BANDIT_CONTEXT = """
CREATE TABLE IF NOT EXISTS bandit_context_stats (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    context_type VARCHAR(50) NOT NULL,
    model VARCHAR(255) NOT NULL,
    avg_reward FLOAT DEFAULT 0,
    count INT DEFAULT 0,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_ctx_model (context_type, model)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

def _init_tables():
    try:
        with engine.begin() as conn:
            conn.execute(text(DDL_BANDIT_CONTEXT))
        logger.info("[bandit] Tabela bandit_context_stats verificada/criada.")
    except SQLAlchemyError as e:
        logger.warning(f"[bandit] Falha ao criar tabela bandit_context_stats: {e}")

_init_tables()

# ============================================================
# 🔢 Utilidades numéricas
# ============================================================

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1 or b.ndim != 1:
        a = a.reshape(-1)
        b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x if n == 0 else (x / n)

# ============================================================
# 🧭 Contextos
# ============================================================

def _token_count(text: str) -> int:
    return len((text or "").split())

def detect_contexts(query: str) -> List[str]:
    """
    Contextos estruturais/domínio + etiqueta semântica de centróide (se houver).
    """
    q = (query or "").lower()
    toks = _token_count(q)

    contexts: List[str] = []

    # Estruturais
    if toks < 40:   contexts.append("short")
    if toks > 250:  contexts.append("long")

    # Técnicos
    if any(x in q for x in ["def ", "class ", "import ", "public ", "{", "}", "console.log", "async ", "await ", "=> "]):
        contexts.append("code")
    if any(x in q for x in ["select ", "from ", "json", "csv", "tabela", "dataframe", "parquet", "sql "]):
        contexts.append("data")
    if any(x in q for x in ["http", "https", "port ", "socket", "tcp", "udp", "dns", "bind9", "api "]):
        contexts.append("network")
    if any(x in q for x in [" if ", " then ", " else ", " while ", " loop ", " state ", " fsm "]):
        contexts.append("logic")
    if any(x in q for x in ["∑", "√", " integral", " derivada", " teorema", " álgebra", "matriz", "vetor"]):
        contexts.append("math")

    # Acadêmico/científico/negócios/legal
    if any(x in q for x in ["referência", "metodologia", "introdução", "revisão", "abnt", "doi", "citação"]):
        contexts.append("academic")
    if any(x in q for x in ["experimento", "hipótese", "teoria", "modelo", "observação"]):
        contexts.append("scientific")
    if any(x in q for x in ["custo", "lucro", "capex", "opex", "negócio", "vendas", "mercado"]):
        contexts.append("business")
    if any(x in q for x in ["lei", "artigo", "parágrafo", "constituição", "jurídico"]):
        contexts.append("legal")

    # Domínios usuais do seu projeto
    if any(x in q for x in ["aviário", "granjas", "amônia", "irrigação", "bomba submersa", "gotejador"]):
        contexts.append("agribusiness")
    if any(x in q for x in ["moodle", "udl", "rubrica", "docência", "ifrn", "ppgite"]):
        contexts.append("education")

    # Fallback
    if not contexts:
        contexts.append("generic")

    # Anexa rótulo semântico de centróide (se existir)
    sem_label = _nearest_centroid_label(query)
    if sem_label:
        contexts.append(sem_label)

    # Remover duplicatas mantendo ordem
    seen = set()
    ordered = []
    for c in contexts:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered

# ============================================================
# 🧠 Centróides semânticos (ONLINE) em Redis
# ============================================================

def _load_centroids() -> List[dict]:
    try:
        if not rds:
            return []
        payload = rds.get(R_KEY_CENTROIDS)
        if not payload:
            return []
        arr = json.loads(payload)
        # validação leve
        out = []
        for it in arr:
            if isinstance(it, dict) and "id" in it and "vec" in it and "count" in it:
                # converte vetor para np.array normalizado
                v = np.array([float(x) for x in it["vec"]], dtype=np.float32)
                out.append({"id": int(it["id"]), "vec": _unit(v), "count": int(it["count"])})
        return out
    except Exception as e:
        logger.warning(f"[centroids] Falha ao carregar do Redis: {e}")
        return []

def _save_centroids(items: List[dict]) -> None:
    try:
        if not rds:
            return
        serial = []
        for it in items:
            serial.append({
                "id": int(it["id"]),
                "vec": [float(x) for x in _unit(it["vec"]).tolist()],
                "count": int(it["count"])
            })
        pipe = rds.pipeline()
        pipe.set(R_KEY_CENTROIDS, json.dumps(serial))
        pipe.hset(R_KEY_CENTROIDS_META, mapping={
            "updated_at": str(int(time.time())),
            "k": str(CENTROIDS_K),
            "dim": str(CENTROIDS_DIM),
        })
        pipe.set(R_KEY_CENTROIDS_V, "1")
        pipe.execute()
    except Exception as e:
        logger.warning(f"[centroids] Falha ao salvar no Redis: {e}")

def _acquire_lock(name: str, ttl: int = 10) -> bool:
    if not rds:
        return False
    try:
        return bool(rds.set(name, "1", nx=True, ex=ttl))
    except Exception:
        return False

def _release_lock(name: str) -> None:
    if not rds:
        return
    try:
        rds.delete(name)
    except Exception:
        pass

def _ensure_dim(v: np.ndarray) -> np.ndarray:
    # não forçamos dimensão; assumimos que embeddings já vêm com a dimensão correta do modelo
    return _unit(v.astype(np.float32).reshape(-1))

def _nearest_centroid(v: np.ndarray, centroids: List[dict]) -> Tuple[Optional[int], float]:
    if not centroids:
        return None, 0.0
    best_idx = None
    best_sim = -1.0
    for idx, it in enumerate(centroids):
        sim = _cosine_sim(v, it["vec"])  # ambos unit norm
        if sim > best_sim:
            best_sim = sim
            best_idx = idx
    return best_idx, float(best_sim)

def _new_centroid_id(centroids: List[dict]) -> int:
    used = {int(it["id"]) for it in centroids}
    cid = 0
    while cid in used:
        cid += 1
    return cid

def centroids_online_update(query_text: str) -> Optional[int]:
    """
    Atualização ON-LINE de centróides:
      - calcula embedding via embed_text(query_text) -> np.ndarray (normalizado).
      - atribui ao centróide mais próximo (cosine).
      - se sim < MIN_SIM_CREATE e len < K -> cria novo centróide (inicial = x).
      - senão atualiza centróide com c = (1-α)c + α x; re-normaliza.
    Retorna id do centróide atribuído (ou None).
    """
    try:
        v = _ensure_dim(embed_text(query_text))
    except Exception as e:
        logger.debug(f"[centroids] Falha ao embedar query para update online: {e}")
        return None

    # lock simples para evitar corridas
    if not _acquire_lock(R_KEY_CENTROIDS_LOCK, ttl=5):
        # sem lock, apenas desiste silenciosamente
        return None

    try:
        cents = _load_centroids()
        idx, sim = _nearest_centroid(v, cents)
        if idx is None or sim < CENTROIDS_MIN_SIM_CREATE:
            if len(cents) < CENTROIDS_K:
                # cria novo centróide
                cid = _new_centroid_id(cents)
                cents.append({"id": cid, "vec": v.copy(), "count": 1})
                _save_centroids(cents)
                return cid
            # sem vaga para criar; atualiza o mais próximo mesmo assim
            if idx is None:
                return None

        # atualização exponencial
        c = cents[idx]
        cvec = c["vec"]
        new_vec = _unit((1.0 - CENTROIDS_LEARN_RATE) * cvec + CENTROIDS_LEARN_RATE * v)
        c["vec"] = new_vec
        c["count"] = int(c.get("count", 0)) + 1
        _save_centroids(cents)
        return c["id"]
    finally:
        _release_lock(R_KEY_CENTROIDS_LOCK)

def _nearest_centroid_label(query_text: str) -> Optional[str]:
    """
    Retorna etiqueta 'semctx:<id>' do centróide mais próximo se existir algum.
    Não cria centróides; somente leitura.
    """
    try:
        v = _ensure_dim(embed_text(query_text))
    except Exception:
        return None
    cents = _load_centroids()
    idx, sim = _nearest_centroid(v, cents)
    if idx is None:
        return None
    cid = cents[idx]["id"]
    return f"semctx:{cid}"

# ============================================================
# 🧹 Tarefa horária de manutenção (merge até K; só com ≥ 50 logs)
# ============================================================

def _count_query_log_rows() -> int:
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT COUNT(*) AS c FROM query_log")).fetchone()
            return int(row[0] if row else 0)
    except Exception:
        return 0

def _merge_closest_pairs(cents: List[dict], target_k: int) -> List[dict]:
    """
    Estratégia simples: enquanto len(cents) > target_k, mescla o par mais similar:
      c_ij = unit( (w_i * c_i + w_j * c_j) / (w_i + w_j) ), count = w_i + w_j
    """
    cents = list(cents)
    while len(cents) > target_k and len(cents) >= 2:
        best = (-1.0, None, None)  # (sim, i, j)
        for i in range(len(cents)):
            for j in range(i + 1, len(cents)):
                sim = _cosine_sim(cents[i]["vec"], cents[j]["vec"])
                if sim > best[0]:
                    best = (sim, i, j)
        _, i, j = best
        if i is None or j is None:
            break
        ci, cj = cents[i], cents[j]
        wi, wj = float(ci["count"]), float(cj["count"])
        new_vec = _unit((wi * ci["vec"] + wj * cj["vec"]) / max(1.0, wi + wj))
        new_id = min(ci["id"], cj["id"])
        new_count = int(wi + wj)
        # remove j>i para manter índices
        for idx in sorted([i, j], reverse=True):
            del cents[idx]
        cents.append({"id": new_id, "vec": new_vec, "count": new_count})
    return cents

def _hourly_centroids_maintenance():
    while True:
        try:
            if not CENTROIDS_HOURLY_REFRESH_ENABLED:
                time.sleep(3600)
                continue

            total = _count_query_log_rows()
            if total < CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH:
                logger.info(f"[centroids] Manutenção: ignorada (query_log={total} < {CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH}).")
                time.sleep(3600)
                continue

            if not _acquire_lock(R_KEY_CENTROIDS_LOCK, ttl=30):
                time.sleep(3600)
                continue

            try:
                cents = _load_centroids()
                if len(cents) > CENTROIDS_K:
                    cents = _merge_closest_pairs(cents, CENTROIDS_K)
                    _save_centroids(cents)
                    logger.info(f"[centroids] Mesclados até K={CENTROIDS_K}. Total atual: {len(cents)}")
                else:
                    # Nada para fazer além de marcar meta
                    if rds:
                        rds.hset(R_KEY_CENTROIDS_META, mapping={"touched_at": str(int(time.time()))})
            finally:
                _release_lock(R_KEY_CENTROIDS_LOCK)

        except Exception as e:
            logger.warning(f"[centroids] Manutenção horária falhou: {e}")

        time.sleep(3600)  # uma vez por hora

# dispara thread de manutenção
threading.Thread(target=_hourly_centroids_maintenance, daemon=True).start()

# ============================================================
# 📥 Leitura de modelos
# ============================================================

def load_candidate_models() -> List[str]:
    try:
        models = settings.CANDIDATE_MODELS_LIST
        if models and isinstance(models, list):
            return models
    except Exception as e:
        logger.warning(f"[bandit] Falha ao ler modelos do settings_dynamic: {e}")
    logger.warning("[bandit] Nenhuma lista de modelos encontrada; retornando lista vazia.")
    return []

# ============================================================
# 🧮 Estatísticas por (contexto, modelo)
# ============================================================

def _redis_ctx_key(ctx: str) -> str:
    return f"{R_KEY_BANDIT_CTXT_PREFIX}:{ctx}"

def _get_ctx_stats_from_redis(ctx: str) -> Dict[str, Dict[str, float]]:
    try:
        if not rds:
            return {}
        key = _redis_ctx_key(ctx)
        if not rds.exists(key):
            return {}
        raw = rds.hgetall(key)
        stats: Dict[str, Dict[str, float]] = {}
        for model, payload in raw.items():
            try:
                item = json.loads(payload)
                if isinstance(item, dict):
                    stats[model.decode() if isinstance(model, bytes) else model] = {
                        "avg": float(item.get("avg", 0.0)),
                        "count": int(item.get("count", 0)),
                        "var": float(item.get("var", 0.0)),
                        "mean": float(item.get("mean", item.get("avg", 0.0))),
                        "M2": float(item.get("M2", 0.0)),
                    }
            except Exception:
                continue
        return stats
    except Exception as e:
        logger.warning(f"[bandit] Falha ao ler contexto {ctx} do Redis: {e}")
        return {}

def _set_ctx_stats_to_redis(ctx: str, stats: Dict[str, Dict[str, float]]) -> None:
    try:
        if not rds:
            return
        key = _redis_ctx_key(ctx)
        pipe = rds.pipeline()
        for model, s in stats.items():
            payload = json.dumps({
                "avg": float(s.get("avg", s.get("mean", 0.0))),
                "count": int(s.get("count", 0)),
                "var": float(s.get("var", 0.0)),
                "mean": float(s.get("mean", s.get("avg", 0.0))),
                "M2": float(s.get("M2", 0.0)),
            })
            pipe.hset(key, model, payload)
        pipe.execute()
    except Exception as e:
        logger.warning(f"[bandit] Falha ao persistir stats do contexto {ctx} no Redis: {e}")

def _load_ctx_from_db(ctx: str) -> Dict[str, Dict[str, float]]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT model, avg_reward, count FROM bandit_context_stats WHERE context_type = :ctx"),
                {"ctx": ctx}
            ).fetchall()
        out = {}
        for row in rows:
            model = row[0]
            out[model] = {"avg": float(row[1] or 0.0), "count": int(row[2] or 0), "var": 0.0, "mean": float(row[1] or 0.0), "M2": 0.0}
        return out
    except SQLAlchemyError as e:
        logger.warning(f"[bandit] Falha ao carregar stats de {ctx} do DB: {e}")
        return {}

def _upsert_ctx_model_to_db(ctx: str, model: str, avg: float, count: int) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO bandit_context_stats (context_type, model, avg_reward, count)
                    VALUES (:ctx, :model, :avg, :count)
                    ON DUPLICATE KEY UPDATE
                        avg_reward = :avg,
                        count = :count,
                        last_update = CURRENT_TIMESTAMP
                """),
                {"ctx": ctx, "model": model, "avg": float(avg), "count": int(count)}
            )
    except SQLAlchemyError as e:
        logger.warning(f"[bandit] Falha no upsert de ({ctx}, {model}) no DB: {e}")

# ============================================================
# 🧠 Seleção ε-greedy contextual
# ============================================================

def _dynamic_epsilon(ctx_stats: Dict[str, Dict[str, float]]) -> float:
    try:
        eps = float(settings.get("BANDIT_EPSILON", DEFAULT_EPSILON))
    except Exception:
        eps = DEFAULT_EPSILON

    if not ctx_stats:
        return min(1.0, eps + 0.1)

    counts = [s.get("count", 0) for s in ctx_stats.values()]
    if counts and min(counts) < MIN_OBS_FOR_EXPLOIT:
        eps += EPSILON_BOOST_UNDEREXP

    variances = [s.get("var", 0.0) for s in ctx_stats.values()]
    if variances and statistics.mean(variances) > 0.05:
        eps += CONTEXT_VARIANCE_BOOST

    return max(0.0, min(1.0, eps))

def _score_for_exploit(stats: Dict[str, float]) -> float:
    cnt = stats.get("count", 0)
    var = stats.get("var", 0.0)
    return (1.0 / (1.0 + cnt)) + (0.5 * var)

def select_model(valid_models: List[str], query: str) -> str:
    if not valid_models:
        valid_models = load_candidate_models()
    if not valid_models:
        logger.warning("[bandit] Sem modelos válidos; fallback 'ollama/gemma3:4b'.")
        return "ollama/gemma3:4b"

    # Atualização online de centróides com a query atual (não bloqueante)
    try:
        centroids_online_update(query)
    except Exception:
        pass

    contexts = detect_contexts(query)

    # Coletar estatísticas de cada contexto
    per_ctx_stats = []
    for ctx in contexts:
        stats = _get_ctx_stats_from_redis(ctx)
        if not stats:
            stats = _load_ctx_from_db(ctx)
            if stats:
                _set_ctx_stats_to_redis(ctx, stats)
        per_ctx_stats.append((ctx, stats))

    # Agregar por modelo: média dos avg por contexto
    agg: Dict[str, Dict[str, float]] = {m: {"avg": 0.0, "count": 0, "var": 0.0} for m in valid_models}
    for m in valid_models:
        avgs, counts, vars_ = [], [], []
        for _, s in per_ctx_stats:
            if m in s:
                avgs.append(s[m].get("avg", 0.0))
                counts.append(s[m].get("count", 0))
                vars_.append(s[m].get("var", 0.0))
        if avgs:
            agg[m]["avg"] = float(sum(avgs) / len(avgs))
        if counts:
            agg[m]["count"] = int(sum(counts))
        if vars_:
            agg[m]["var"] = float(sum(vars_) / max(1, len(vars_)))

    # Epsilon dinâmico com base no contexto principal
    main_ctx = contexts[0] if contexts else "generic"
    eps = _dynamic_epsilon(_get_ctx_stats_from_redis(main_ctx))

    # Explora vs. Explora
    if random.random() < eps:
        scored = [(m, _score_for_exploit(agg[m])) for m in valid_models]
        scored.sort(key=lambda x: x[1], reverse=True)
        chosen = scored[0][0]
        logger.info(f"[bandit] Exploração (ε={eps:.2f}) → {chosen} | ctx={contexts}")
    else:
        scored = [(m, agg[m]["avg"]) for m in valid_models]
        scored.sort(key=lambda x: x[1], reverse=True)
        chosen = scored[0][0]
        logger.info(f"[bandit] Aproveitamento (ε={eps:.2f}) → {chosen} | ctx={contexts}")

    try:
        BANDIT_SELECT.labels(model=chosen).inc()
    except Exception:
        pass

    return chosen

# ============================================================
# 📝 Atualização do bandit com recompensa observada
# ============================================================

def bandit_update(model: str, query: str, reward: float) -> None:
    """
    Atualiza (contexto, modelo) com Welford + Redis + DB.
    Também realiza update online dos centróides com a query corrente.
    """
    # Atualiza centróides (online) — robusto a falhas
    try:
        centroids_online_update(query)
    except Exception:
        pass

    contexts = detect_contexts(query)
    for ctx in contexts:
        try:
            # Estado atual
            stats = _get_ctx_stats_from_redis(ctx)
            cur = stats.get(model, {"avg": 0.0, "count": 0, "var": 0.0, "mean": 0.0, "M2": 0.0})

            mean = float(cur.get("mean", cur.get("avg", 0.0)))
            M2 = float(cur.get("M2", 0.0))
            count = int(cur.get("count", 0))

            # Welford
            count += 1
            delta = reward - mean
            mean += delta / count
            delta2 = reward - mean
            M2 += delta * delta2
            var = (M2 / (count - 1)) if count > 1 else 0.0

            # Update local
            stats[model] = {"avg": mean, "count": count, "var": var, "mean": mean, "M2": M2}

            # Redis
            _set_ctx_stats_to_redis(ctx, stats)

            # DB
            _upsert_ctx_model_to_db(ctx, model, mean, count)

        except Exception as e:
            logger.warning(f"[bandit] Falha ao atualizar bandit para ({ctx}, {model}): {e}")

    try:
        BANDIT_UPDATE.labels(model=model).inc()
        BANDIT_REWARD.observe(float(reward))
    except Exception:
        pass

# ============================================================
# 🎯 Recompensa
# ============================================================

def compute_reward(model: str, quality: float, latency_s: float, cost_per_1k: float | None = None) -> float:
    """
    Converte métricas em recompensa [0..1].
    - quality: [0..10] → [0..1]
    - latência: logística com x0=20s
    - custo: penaliza acima do baseline; bonifica locais por natureza
    """
    try:
        q = max(0.0, min(10.0, float(quality))) / 10.0

        # penalização latência
        L, k, x0 = 1.0, 0.12, 20.0
        lat_factor = L / (1.0 + math.exp(k * (latency_s - x0)))
        lat_factor = max(0.0, min(1.0, lat_factor))

        cost_factor = 1.0
        if cost_per_1k is not None:
            baseline = 0.12
            ratio = float(cost_per_1k) / baseline if baseline > 0 else 1.0
            cost_factor = 1.0 / (1.0 + max(0.0, ratio - 1.0))
            cost_factor = max(0.3, min(1.0, cost_factor))

        reward = 0.55 * q + 0.30 * lat_factor + 0.15 * cost_factor
        return float(max(0.0, min(1.0, reward)))
    except Exception as e:
        logger.warning(f"[bandit] Falha ao calcular reward: {e}")
        return 0.0
# ============================================================