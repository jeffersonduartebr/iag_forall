# nsga_weights_updater.py
# ------------------------------------------------------------
# Serviço NSGA-II com fallback dinâmico entre modelos.
# - Lê modelos do settings_dynamic (Redis -> DB -> .env)
# - Troca automaticamente de modelo em caso de quota, erro ou falha
# - Registra eventos de fallback em banco (nsga_fallback_log)
# ------------------------------------------------------------

import os
import json
import time
import random
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import redis
from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest
from deap import base, creator, tools, algorithms

from .settings_dynamic import settings

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("nsga-updater")

# ------------------------------------------------------------
# Config / Conexões (Lidas do settings)
# ------------------------------------------------------------
DB_HOST = settings.DB_HOST
DB_USER = settings.DB_USER
DB_PASS = settings.DB_PASS
DB_NAME = settings.DB_NAME
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

REDIS_KEY_WEIGHTS = "nsga:weights"
REDIS_KEY_CANDIDATES = "nsga:candidate_models"

UPDATE_INTERVAL_S = int(settings.get("NSGA_UPDATE_INTERVAL_S", "300"))
QUERY_LOG_LOOKBACK_MINUTES = int(settings.get("NSGA_LOOKBACK_MINUTES", "180"))
QUERY_LOG_MAX_ROWS = int(settings.get("NSGA_LOOKBACK_MAXROWS", "2000"))

# ------------------------------------------------------------
# Redis client helper
# ------------------------------------------------------------
def get_redis(max_wait_s: int = 0):
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    pwd = os.getenv("REDIS_PASSWORD") or os.getenv("REDIS_PASS") or "SenhaForte"
    dbn = int(os.getenv("REDIS_DB", "0"))
    deadline = time.time() + max_wait_s
    while True:
        try:
            r = redis.Redis(host=host, port=port, password=pwd, db=dbn)
            r.ping()
            logger.info(f"[nsga] Conectado ao Redis {host}:{port} (db={dbn})")
            return r
        except Exception as e:
            if max_wait_s <= 0 or time.time() > deadline:
                logger.warning(f"[nsga] Não foi possível conectar ao Redis {host}:{port}: {e}")
                return None
            time.sleep(1)

redis_client = get_redis(max_wait_s=0)

# ------------------------------------------------------------
# Prometheus metrics
# ------------------------------------------------------------
registry = CollectorRegistry()
NSGA_RUNS = Counter("nsga_runs_total", "Total de execuções NSGA-II realizadas", registry=registry)
NSGA_LAST_RUN_TS = Gauge("nsga_last_run_timestamp", "Epoch da última execução do NSGA", registry=registry)
NSGA_POP_FITNESS_AVG = Gauge("nsga_population_fitness_avg", "Fitness médio da população (proxy)", registry=registry)
NSGA_BEST_LATENCY = Gauge("nsga_best_latency_seconds", "Latência esperada (s)", registry=registry)
NSGA_BEST_COST = Gauge("nsga_best_cost_per_1k", "Custo esperado ($/1k toks)", registry=registry)
NSGA_BEST_QUALITY = Gauge("nsga_best_quality", "Qualidade esperada (0-10)", registry=registry)
NSGA_BEST_ALIGNMENT = Gauge("nsga_best_query_alignment", "Alinhamento esperado (0-1)", registry=registry)
WEIGHT_BY_MODEL = Gauge("nsga_model_weight", "Peso/Probabilidade do modelo", ["model"], registry=registry)
MODEL_OBS_LAT = Gauge("nsga_model_obs_latency", "Latência média observada (s)", ["model"], registry=registry)
MODEL_OBS_COST = Gauge("nsga_model_obs_cost_per_1k", "Custo médio observado", ["model"], registry=registry)
MODEL_OBS_QUAL = Gauge("nsga_model_obs_quality", "Qualidade média observada", ["model"], registry=registry)
MODEL_OBS_SR = Gauge("nsga_model_obs_success_rate", "Taxa de sucesso observada", ["model"], registry=registry)
MODEL_ALIGNMENT = Gauge("nsga_model_alignment", "Score de alinhamento (0-1)", ["model"], registry=registry)

# ------------------------------------------------------------
# Tabela de fallback (auditoria)
# ------------------------------------------------------------
def ensure_fallback_table():
    ddl = """
    CREATE TABLE IF NOT EXISTS nsga_fallback_log (
        id INT AUTO_INCREMENT PRIMARY KEY,
        old_model VARCHAR(255),
        new_model VARCHAR(255),
        event_type VARCHAR(64),
        message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

def log_model_fallback(old_model: str, new_model: str, event_type: str, message: str):
    ensure_fallback_table()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO nsga_fallback_log (old_model, new_model, event_type, message) "
                "VALUES (:old, :new, :etype, :msg)"
            ),
            {"old": old_model, "new": new_model, "etype": event_type, "msg": (message or "")[:500]},
        )

def select_next_model(current_model: str, event_type: str, message: str = "") -> str:
    """Seleciona dinamicamente o próximo modelo em caso de falha."""
    try:
        models = load_candidate_models()
        if not models:
            logger.warning("[nsga-fallback] Lista de modelos vazia.")
            return current_model

        if current_model not in models:
            next_model = random.choice(models)
        else:
            idx = models.index(current_model)
            next_model = models[(idx + 1) % len(models)]

        log_model_fallback(current_model, next_model, event_type, message)
        logger.info(f"[nsga-fallback] {current_model} → {next_model} ({event_type})")
        return next_model
    except Exception as e:
        logger.error(f"[nsga-fallback] Erro ao trocar modelo: {e}")
        return current_model

# ------------------------------------------------------------
# Tabelas base
# ------------------------------------------------------------
def ensure_tables():
    ddl_weights = """
    CREATE TABLE IF NOT EXISTS nsga_weights (
        model VARCHAR(255) PRIMARY KEY,
        weight FLOAT NOT NULL,
        computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """
    ddl_qlogs = """
    CREATE TABLE IF NOT EXISTS query_log (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        query_text TEXT NOT NULL,
        chosen_model VARCHAR(255) NOT NULL,
        answer TEXT,
        quality FLOAT,
        latency_s FLOAT,
        cost_per_1k FLOAT,
        reward FLOAT,
        context_label VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB;
    """
    with engine.begin() as conn:
        conn.execute(text(ddl_weights))
        conn.execute(text(ddl_qlogs))

# ------------------------------------------------------------
# Carregamento de modelos (Redis -> settings)
# ------------------------------------------------------------
def _decode_redis_value(val):
    if val is None:
        return None
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except Exception:
            return str(val)
    return val

def load_candidate_models() -> List[str]:
    """
    Ordem de precedência:
      1) Redis key nsga:candidate_models (JSON list)
      2) settings.CANDIDATE_MODELS_LIST (lista Python)
    """
    # 1) Redis
    try:
        if redis_client:
            raw = redis_client.get(REDIS_KEY_CANDIDATES)
            if raw:
                txt = _decode_redis_value(raw)
                models = json.loads(txt)
                if isinstance(models, list) and all(isinstance(x, str) for x in models):
                    logger.info(f"[nsga] Modelos carregados do Redis ({len(models)}): {models}")
                    return models
                logger.warning("[nsga] Valor de nsga:candidate_models não é uma lista válida.")
    except Exception as e:
        logger.warning(f"[nsga] Falha ao ler/parsear modelos do Redis: {e}")

    # 2) settings
    try:
        models = getattr(settings, "CANDIDATE_MODELS_LIST", []) or []
        if models:
            logger.info(f"[nsga] Modelos carregados do settings ({len(models)}): {models}")
            return list(models)
    except Exception as e:
        logger.warning(f"[nsga] Falha ao ler CANDIDATE_MODELS_LIST do settings: {e}")

    logger.warning("[nsga] Nenhum modelo disponível (Redis e settings vazios).")
    return []

# ------------------------------------------------------------
# Carregamento de logs e métricas
# ------------------------------------------------------------
def load_recent_query_logs(minutes_back: int, max_rows: int) -> List[Dict[str, Any]]:
    try:
        with engine.connect() as conn:
            since = datetime.utcnow() - timedelta(minutes=minutes_back)
            rs = conn.execute(
                text(
                    """
                    SELECT created_at as ts, query_text as query, chosen_model as model,
                           latency_s, quality, cost_per_1k, (reward > 0.3) as success
                    FROM query_log
                    WHERE created_at >= :since
                    ORDER BY id DESC
                    LIMIT :lim
                    """
                ),
                {"since": since, "lim": max_rows},
            )
            rows = rs.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            m = r._mapping
            out.append(
                {
                    "ts": m["ts"],
                    "query": m["query"] or "",
                    "model": m["model"] or "",
                    "latency_s": float(m["latency_s"]) if m["latency_s"] is not None else 8.0,
                    "quality": float(m["quality"]) if m["quality"] is not None else 5.0,
                    "cost_per_1k": float(m["cost_per_1k"]) if m["cost_per_1k"] is not None else 0.02,
                    "success": bool(m["success"]) if m["success"] is not None else True,
                }
            )
        return out
    except Exception as e:
        logger.warning(f"[nsga] Falha ao ler query_log: {e}")
        return []

# ------------------------------------------------------------
# NSGA-II
# ------------------------------------------------------------
def run_nsga(models: List[str], pm_obs: Dict[str, Dict[str, float]], align: Dict[str, float]) -> Dict[str, float]:
    """Executa o algoritmo NSGA-II e retorna novos pesos."""
    n = len(models)
    if n == 1:
        return {models[0]: 1.0}

    try:
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, +1.0, +1.0))
    except Exception:
        pass
    try:
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    except Exception:
        pass

    toolbox = base.Toolbox()
    toolbox.register("attr_w", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_w, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        # normaliza pesos
        w = [max(0.0, min(1.0, x)) for x in ind]
        s = sum(w) or 1.0
        w = [x / s for x in w]
        lat = sum(w[i] * pm_obs[models[i]]["latency"] for i in range(n))
        cst = sum(w[i] * pm_obs[models[i]]["cost_per_1k"] for i in range(n))
        qlt = sum(w[i] * pm_obs[models[i]]["quality"] for i in range(n))
        aln = sum(w[i] * align[models[i]] for i in range(n))
        NSGA_POP_FITNESS_AVG.set((max(0.0, 10.0 - lat) + max(0.0, 10.0 - cst) + qlt + 10.0 * aln) / 4.0)
        return lat, cst, qlt, aln

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0 / n)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=40)
    hof = tools.ParetoFront()
    algorithms.eaMuPlusLambda(pop, toolbox, 40, 40, 0.9, 0.1, 30, stats=None, halloffame=hof, verbose=False)

    best = None
    best_score = -1e9
    for ind in hof:
        w = [max(0.0, min(1.0, x)) for x in ind]
        s = sum(w) or 1.0
        w = [x / s for x in w]
        lat = sum(w[i] * pm_obs[models[i]]["latency"] for i in range(n))
        cst = sum(w[i] * pm_obs[models[i]]["cost_per_1k"] for i in range(n))
        qlt = sum(w[i] * pm_obs[models[i]]["quality"] for i in range(n))
        aln = sum(w[i] * align[models[i]] for i in range(n))
        score = -lat - 0.5 * cst + 0.8 * qlt + 5.0 * aln
        if score > best_score:
            best = (w, (lat, cst, qlt, aln))
            best_score = score

    if not best:
        return {m: 1.0 / n for m in models}

    w, (lat, cst, qlt, aln) = best
    NSGA_BEST_LATENCY.set(lat)
    NSGA_BEST_COST.set(cst)
    NSGA_BEST_QUALITY.set(qlt)
    NSGA_BEST_ALIGNMENT.set(aln)
    return {models[i]: float(w[i]) for i in range(n)}

# ------------------------------------------------------------
# Persistência e iteração principal
# ------------------------------------------------------------
def persist_weights(weights: Dict[str, float]):
    try:
        with engine.begin() as conn:
            for m, w in weights.items():
                conn.execute(
                    text(
                        """
                        INSERT INTO nsga_weights (model, weight, computed_at)
                        VALUES (:m, :w, NOW())
                        ON DUPLICATE KEY UPDATE
                            weight = :w,
                            computed_at = NOW()
                        """
                    ),
                    {"m": m, "w": w},
                )
        if redis_client:
            redis_client.set(REDIS_KEY_WEIGHTS, json.dumps(weights))
        for m, w in weights.items():
            WEIGHT_BY_MODEL.labels(model=m).set(w)
    except Exception as e:
        logger.warning(f"[nsga] Falha ao gravar pesos: {e}")

def one_iteration():
    try:
        ensure_tables()

        # 🔹 Carrega modelos (Redis -> settings)
        models = load_candidate_models()
        if not models:
            logger.warning("[nsga] Sem modelos; interrompendo iteração.")
            return

        # 🔹 Carrega logs recentes (se não houver, ainda calculamos um mix baseado em defaults)
        logs = load_recent_query_logs(QUERY_LOG_LOOKBACK_MINUTES, QUERY_LOG_MAX_ROWS)
        if not logs:
            logger.warning("[nsga] Nenhum log recente encontrado — usando métricas default conservadoras.")

        # 🔹 Métricas observadas por modelo (placeholder simples; substitua por médias reais quando disponíveis)
        pm_obs = {
            m: {
                "latency": random.uniform(1.0, 5.0),
                "cost_per_1k": (0.001 if "ollama" in m else random.uniform(0.005, 0.03)),
                "quality": random.uniform(5.5, 9.0),
                "success_rate": random.uniform(0.85, 0.99),
            }
            for m in models
        }

        # 🔹 Alinhamento (placeholder)
        align = {m: random.uniform(0.3, 1.0) for m in models}

        # 🔹 NSGA-II
        weights = run_nsga(models, pm_obs, align)
        persist_weights(weights)

        NSGA_RUNS.inc()
        NSGA_LAST_RUN_TS.set(time.time())
        logger.info(f"[nsga] Pesos atualizados: {weights}")
    except Exception as e:
        logger.exception(f"[nsga] Falha na iteração NSGA: {e}")
        try:
            fallback_model = select_next_model("nsga:current_model", "iteration_error", str(e))
            logger.info(f"[nsga] Fallback dinâmico ativado → {fallback_model}")
        except Exception as e2:
            logger.error(f"[nsga] Falha ao aplicar fallback: {e2}")

def nsga_loop():
    one_iteration()
    while True:
        time.sleep(UPDATE_INTERVAL_S)
        one_iteration()

# ------------------------------------------------------------
# API / Health / Metrics
# ------------------------------------------------------------
app = FastAPI(title="NSGA Weights Updater")

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(registry).decode("utf-8"))

@app.get("/health")
def health():
    # DB
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    # Redis
    redis_ok = False
    try:
        if redis_client:
            redis_client.ping()
            redis_ok = True
    except Exception:
        redis_ok = False

    return JSONResponse({
        "status": "ok" if (db_ok and redis_ok) else "degraded",
        "db": db_ok,
        "redis": redis_ok,
        "last_run_epoch": NSGA_LAST_RUN_TS._value.get() if hasattr(NSGA_LAST_RUN_TS, "_value") else None,
    })

if __name__ == "__main__":
    t = threading.Thread(target=nsga_loop, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=9999, log_level="info")
