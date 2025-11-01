# bandits.py
"""
bandits.py
----------------------------------------------------
Bandit contextual dinâmico com persistência:
- Contextos granulares detectados automaticamente (12+ categorias).
- Leitura da lista de modelos via settings_dynamic (Redis → DB → .env).
- Exploração ε-greedy adaptativa por contexto (dinâmica, não-estática).
- Estatísticas por (contexto, modelo) armazenadas em Redis e MariaDB.
- Métricas Prometheus e logs estruturados.

APIs públicas (compatíveis com o restante do projeto):
- select_model(valid_models: list[str], query: str) -> str
- bandit_update(model: str, query: str, reward: float) -> None
- compute_reward(model: str, quality: float, latency_s: float, cost_per_1k: float | None = None) -> float
"""

from __future__ import annotations

import os
import json
import math
import time
import random
import logging
import statistics
from typing import Dict, List, Tuple, Optional

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.settings_dynamic import settings # <-- CORRIGIDO
from app.utils.redis_client import get_redis
from app.observability import BANDIT_SELECT, BANDIT_UPDATE, BANDIT_REWARD

logger = logging.getLogger(__name__)

# ============================================================
# 🔧 Conexões e configurações globais
# ============================================================

# --- Banco ---
# Lendo do settings para consistência, com fallback para os.getenv
DB_HOST = settings.get("DB_HOST", os.getenv("DB_HOST", "mariadb"))
DB_USER = settings.get("DB_USER", os.getenv("DB_USER", "router_user"))
DB_PASS = settings.get("DB_PASS", os.getenv("DB_PASS", "router_pass"))
DB_NAME = settings.get("DB_NAME", os.getenv("DB_NAME", "routerdb"))
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

# --- Redis ---
rds = get_redis()

# --- Settings dinâmicos (instância global 'settings') ---

# Chaves Redis (para estatísticas, não para config)
R_KEY_BANDIT_CTXT_PREFIX = "bandit:ctx"              # bandit:ctx:<ctx> -> hash {model: json(stats)}
R_KEY_BANDIT_CTXT_META = "bandit:ctx:meta"           # info geral por contexto

# Defaults
DEFAULT_EPSILON = 0.12
MIN_OBS_FOR_EXPLOIT = 3        # abaixo disso, força mais exploração
EPSILON_BOOST_UNDEREXP = 0.10  # incremento dinâmico se pouco explorado
CONTEXT_VARIANCE_BOOST = 0.08  # se variância de um contexto-modelo for alta

# ============================================================
# 🧱 DDL: Tabela para estatísticas contextuais do bandit
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
);
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
# 🧭 Detecção de contexto (granular e extensível)
# ============================================================

def _token_count(text: str) -> int:
    return len((text or "").split())

# 12+ categorias; múltiplos contextos podem ser atribuídos
def detect_contexts(query: str) -> List[str]:
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
    if any(x in q for x in ["if ", "then ", "else ", " while ", "loop ", "state ", "fsm "]):
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

    # Remover duplicatas mantendo ordem
    seen = set()
    ordered = []
    for c in contexts:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered

# ============================================================
# 📥 Leitura dos modelos (Redis → DB → .env)
# ============================================================

def load_candidate_models() -> List[str]:
    """
    Busca lista de modelos candidate via settings (Redis → DB → .env).
    """
    try:
        # CORRIGIDO: Lê diretamente da propriedade centralizada
        models = settings.CANDIDATE_MODELS_LIST
        if models and isinstance(models, list):
            return models
    except Exception as e:
        logger.warning(f"[bandit] Falha ao ler modelos do settings_dynamic: {e}")

    # Fallback final (vazio)
    logger.warning("[bandit] Nenhuma lista de modelos encontrada; retornando lista vazia.")
    return []

# ============================================================
# 🧮 Estatísticas por (contexto, modelo)
# ============================================================

def _redis_ctx_key(ctx: str) -> str:
    return f"{R_KEY_BANDIT_CTXT_PREFIX}:{ctx}"

def _get_ctx_stats_from_redis(ctx: str) -> Dict[str, Dict[str, float]]:
    """
    Retorna dicionário {model: {"avg": float, "count": int, "var": float}} para o contexto ctx.
    """
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
                    }
            except Exception:
                continue
        return stats
    except Exception as e:
        logger.warning(f"[bandit] Falha ao ler contexto {ctx} do Redis: {e}")
        return {}

def _set_ctx_stats_to_redis(ctx: str, stats: Dict[str, Dict[str, float]]) -> None:
    """
    Persiste no Redis um hash com {model: json(stats)}.
    """
    try:
        if not rds:
            return
        key = _redis_ctx_key(ctx)
        pipe = rds.pipeline()
        for model, s in stats.items():
            payload = json.dumps({"avg": s.get("avg", 0.0),
                                  "count": int(s.get("count", 0)),
                                  "var": float(s.get("var", 0.0))})
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
            out[model] = {"avg": float(row[1] or 0.0), "count": int(row[2] or 0), "var": 0.0}
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
# 🧠 Seleção (ε-greedy contextual dinâmica)
# ============================================================

def _dynamic_epsilon(ctx_stats: Dict[str, Dict[str, float]]) -> float:
    """
    Ajusta epsilon dinamicamente:
    - O valor base é lido de settings (Redis > DB > .env)
    - Se existem modelos com count baixo, aumenta ε
    - Se variância média é alta, aumenta ε
    """
    try:
        # CORRIGIDO: Lê o valor base do settings, que já checa Redis > DB > .env
        eps = float(settings.get("BANDIT_EPSILON", DEFAULT_EPSILON))
    except Exception:
        eps = DEFAULT_EPSILON
    
    if not ctx_stats:
        return eps + 0.1  # nada conhecido -> explore mais

    counts = [s.get("count", 0) for s in ctx_stats.values()]
    if counts and min(counts) < MIN_OBS_FOR_EXPLOIT:
        eps += EPSILON_BOOST_UNDEREXP

    variances = [s.get("var", 0.0) for s in ctx_stats.values() if "var" in s]
    if variances and statistics.mean(variances) > 0.05:  # limiar heurístico
        eps += CONTEXT_VARIANCE_BOOST

    # REMOVIDO: A leitura manual do Redis (R_KEY_BANDIT_EPS) foi removida
    # pois settings.get() já faz isso de forma centralizada.

    return max(0.0, min(1.0, eps))

def _score_for_exploit(stats: Dict[str, float]) -> float:
    """
    Score de exploração (na verdade, de exploração direcionada / aprendizado):
    - Prefere modelos com poucos dados (count baixo)
    - Prefere modelos com variância alta (potencial de descoberta)
    """
    cnt = stats.get("count", 0)
    var = stats.get("var", 0.0)
    # Heurística:  peso maior para baixa contagem e variância
    return (1.0 / (1.0 + cnt)) + (0.5 * var)

def select_model(valid_models: List[str], query: str) -> str:
    """
    Seleciona um modelo com base no contexto da query e histórico.
    Estratégia:
      - Detecta contextos (lista).
      - Combina estatísticas dos contextos (média) para cada modelo.
      - ε-greedy dinâmica: com prob ε -> explora, senão -> explota.
    """
    if not valid_models:
        # fallback: tenta dinâmico
        valid_models = load_candidate_models()
    if not valid_models:
        # fallback final
        logger.warning("[bandit] Sem modelos válidos; retornando 'ollama/gemma3:4b-it-qat'.")
        return "ollama/gemma3:4b-it-qat"

    contexts = detect_contexts(query)
    per_ctx_stats = []
    for ctx in contexts:
        stats = _get_ctx_stats_from_redis(ctx)
        if not stats:
            # Tentar popular via DB (primeira vez)
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

    # Epsilon dinâmico com base no contexto principal (primeiro da lista)
    main_ctx = contexts[0] if contexts else "generic"
    eps = _dynamic_epsilon(_get_ctx_stats_from_redis(main_ctx))

    # Decide exploração vs. exploração
    if random.random() < eps:
        # Exploração: escolhe o modelo que pode mais ensinar (poucos dados/variância)
        scored = [(m, _score_for_exploit(agg[m])) for m in valid_models]
        scored.sort(key=lambda x: x[1], reverse=True)
        chosen = scored[0][0]
        logger.info(f"[bandit] Exploração (ε={eps:.2f}) → {chosen} | ctx={contexts}")
    else:
        # Exploitation: maior média agregada
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
    Atualiza as estatísticas por contexto para o modelo escolhido.
    Mantém média incremental e uma variância (Welford).
    Persiste no Redis e faz upsert no DB.
    """
    contexts = detect_contexts(query)
    for ctx in contexts:
        try:
            # 1) Puxa estado atual do Redis (ou DB)
            stats = _get_ctx_stats_from_redis(ctx)
            cur = stats.get(model, {"avg": 0.0, "count": 0, "var": 0.0, "mean": 0.0, "M2": 0.0})

            # Reconstruir Welford se precisarmos manter var robusta no Redis
            mean = cur.get("mean", cur.get("avg", 0.0))
            M2 = float(cur.get("M2", 0.0))
            count = int(cur.get("count", 0))

            # 2) Atualiza via Welford
            count += 1
            delta = reward - mean
            mean += delta / count
            delta2 = reward - mean
            M2 += delta * delta2
            var = (M2 / (count - 1)) if count > 1 else 0.0

            # 3) Persiste no dicionário local
            stats[model] = {
                "avg": float(mean),
                "count": int(count),
                "var": float(var),
                "mean": float(mean),
                "M2": float(M2),
            }

            # 4) Redis
            _set_ctx_stats_to_redis(ctx, stats)

            # 5) DB (upsert)
            _upsert_ctx_model_to_db(ctx, model, mean, count)

        except Exception as e:
            logger.warning(f"[bandit] Falha ao atualizar bandit para ({ctx}, {model}): {e}")

    try:
        BANDIT_UPDATE.labels(model=model).inc()
        BANDIT_REWARD.observe(float(reward))
    except Exception:
        pass

# ============================================================
# 🎯 Cálculo de recompensa
# ============================================================

def compute_reward(model: str, quality: float, latency_s: float, cost_per_1k: float | None = None) -> float:
    """
    Converte métricas em recompensa [0..1].
    - quality: esperado em [0..10] → normalizado para [0..1]
    - latency_s: penaliza >= 20s
    - cost_per_1k: se informado, penaliza custos acima do baseline
    """
    try:
        q = max(0.0, min(10.0, float(quality))) / 10.0

        # Penalização de latência (curva logística)
        # 0s => 1.0 ; 10s => ~0.73 ; 20s => 0.5 ; 40s => ~0.27
        L = 1.0
        k = 0.12
        x0 = 20.0
        lat_factor = L / (1.0 + math.exp(k * (latency_s - x0)))
        lat_factor = max(0.0, min(1.0, lat_factor))

        # Penalização de custo (opcional)
        cost_factor = 1.0
        if cost_per_1k is not None:
            baseline = 0.12  # baseline remoto (p.ex., GPT-5 ~12¢ / 1k)
            ratio = float(cost_per_1k) / baseline if baseline > 0 else 1.0
            # quanto mais barato, mais próximo de 1; se mais caro, cai
            cost_factor = 1.0 / (1.0 + max(0.0, ratio - 1.0))  # 0.5 se 2x mais caro, etc.
            cost_factor = max(0.3, min(1.0, cost_factor))     # piso 0.3

        # Combinação ponderada
        # Dê um pouco mais de peso a qualidade, depois latência, depois custo
        reward = 0.55 * q + 0.30 * lat_factor + 0.15 * cost_factor
        return float(max(0.0, min(1.0, reward)))
    except Exception as e:
        logger.warning(f"[bandit] Falha ao calcular reward: {e}")
        return 0.0