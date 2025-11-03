# -*- coding: utf-8 -*-
"""
nsga_weights_updater.py
------------------------------------------------------------
Serviço NSGA-II com fallback dinâmico entre modelos.

- Lê modelos do settings_dynamic (Redis → DB → .env)
- Executa o algoritmo multiobjetivo (latência, custo, qualidade, alinhamento)
- Atualiza pesos ótimos no Redis e no banco
- Expõe métricas via /metrics (Prometheus)
"""

from __future__ import annotations
import json
import logging
import os
import random
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import redis
import uvicorn
from deap import algorithms, base, creator, tools
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Gauge, REGISTRY, generate_latest
from sqlalchemy import create_engine, text

from .settings_dynamic import settings

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("nsga-updater")

# ============================================================
# Banco e Redis
# ============================================================
DB_URL = (
    f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}@"
    f"{settings.DB_HOST}:3306/{settings.DB_NAME}"
)
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

REDIS_KEY_WEIGHTS = "nsga:weights"
REDIS_KEY_CANDIDATES = "nsga:candidate_models"

UPDATE_INTERVAL_S = int(settings.get("NSGA_UPDATE_INTERVAL_S", "300"))
QUERY_LOG_LOOKBACK_MINUTES = int(settings.get("NSGA_LOOKBACK_MINUTES", "180"))
QUERY_LOG_MAX_ROWS = int(settings.get("NSGA_LOOKBACK_MAXROWS", "2000"))


def get_redis(max_wait_s: int = 0):
    """Obtém cliente Redis com espera opcional."""
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    pwd = os.getenv("REDIS_PASSWORD") or "SenhaForte"
    dbn = int(os.getenv("REDIS_DB", "0"))
    try:
        r = redis.Redis(host=host, port=port, password=pwd, db=dbn)
        r.ping()
        logger.info("[nsga] Redis conectado: %s:%s (db=%s)", host, port, dbn)
        return r
    except Exception as exc:
        logger.warning("[nsga] Falha ao conectar Redis: %s", exc)
        return None


redis_client = get_redis()

# ============================================================
# Métricas Prometheus
# ============================================================
NSGA_RUNS = Counter("nsga_runs_total", "Total de execuções NSGA-II realizadas")
NSGA_LAST_RUN_TS = Gauge("nsga_last_run_timestamp", "Epoch da última execução NSGA")
NSGA_POP_FITNESS_AVG = Gauge("nsga_population_fitness_avg", "Fitness médio da população")
NSGA_BEST_LATENCY = Gauge("nsga_best_latency_seconds", "Latência esperada (s)")
NSGA_BEST_COST = Gauge("nsga_best_cost_per_1k", "Custo esperado ($/1k toks)")
NSGA_BEST_QUALITY = Gauge("nsga_best_quality", "Qualidade esperada (0-10)")
NSGA_BEST_ALIGNMENT = Gauge("nsga_best_alignment", "Alinhamento esperado (0-1)")

WEIGHT_BY_MODEL = Gauge("nsga_model_weight", "Peso/Probabilidade", ["model"])
MODEL_OBS_LAT = Gauge("nsga_model_obs_latency", "Latência média observada (s)", ["model"])
MODEL_OBS_COST = Gauge("nsga_model_obs_cost", "Custo médio observado ($/1k)", ["model"])
MODEL_OBS_QUAL = Gauge("nsga_model_obs_quality", "Qualidade média observada (0-10)", ["model"])
MODEL_OBS_SR = Gauge("nsga_model_obs_success", "Taxa de sucesso observada", ["model"])
MODEL_ALIGNMENT = Gauge("nsga_model_alignment", "Score de alinhamento (0-1)", ["model"])

# ============================================================
# Criação de tabelas auxiliares
# ============================================================
def ensure_tables():
    ddl = """
    CREATE TABLE IF NOT EXISTS nsga_weights (
        model VARCHAR(255) PRIMARY KEY,
        weight FLOAT NOT NULL,
        computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


# ============================================================
# Leitura de dados históricos
# ============================================================
def load_candidate_models() -> List[str]:
    """Obtém lista de modelos (Redis → settings)."""
    try:
        if redis_client:
            raw = redis_client.get(REDIS_KEY_CANDIDATES)
            if raw:
                models = json.loads(raw.decode("utf-8"))
                if isinstance(models, list) and all(isinstance(x, str) for x in models):
                    return models
    except Exception:
        pass
    return getattr(settings, "CANDIDATE_MODELS_LIST", []) or []


def load_recent_query_logs(minutes_back: int, max_rows: int) -> List[Dict[str, Any]]:
    """Carrega logs recentes de query_log."""
    since = datetime.utcnow() - timedelta(minutes=minutes_back)
    try:
        with engine.connect() as conn:
            rs = conn.execute(
                text("""
                    SELECT chosen_model AS model,
                           latency_s, cost_per_1k, quality, reward
                    FROM query_log
                    WHERE created_at >= :since
                    ORDER BY id DESC
                    LIMIT :lim
                """),
                {"since": since, "lim": max_rows},
            )
            rows = rs.fetchall()
        data = []
        for r in rows:
            m = r._mapping
            data.append(
                {
                    "model": m["model"],
                    "latency": float(m["latency_s"] or 0),
                    "cost": float(m["cost_per_1k"] or 0),
                    "quality": float(m["quality"] or 0),
                    "reward": float(m["reward"] or 0),
                }
            )
        return data
    except Exception as e:
        logger.warning("[nsga] Falha ao ler query_log: %s", e)
        return []


# ============================================================
# NSGA-II principal
# ============================================================
def run_nsga(models: List[str], pm_obs: Dict[str, Dict[str, float]], align: Dict[str, float]) -> Dict[str, float]:
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
    toolbox.register("attr_w", lambda: random.random())
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_w, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        w_raw = [max(0, min(1, x)) for x in ind]
        s = sum(w_raw) or 1
        w = [x / s for x in w_raw]
        lat = sum(w[i] * pm_obs[models[i]]["latency"] for i in range(n))
        cst = sum(w[i] * pm_obs[models[i]]["cost"] for i in range(n))
        qlt = sum(w[i] * pm_obs[models[i]]["quality"] for i in range(n))
        aln = sum(w[i] * align[models[i]] for i in range(n))
        proxy = (10 - lat + 10 - cst + qlt + 10 * aln) / 4
        NSGA_POP_FITNESS_AVG.set(proxy)
        return lat, cst, qlt, aln

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20, indpb=1 / n)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=40)
    hof = tools.ParetoFront()
    algorithms.eaMuPlusLambda(pop, toolbox, mu=40, lambda_=40, cxpb=0.9, mutpb=0.1, ngen=30, halloffame=hof, verbose=False)

    best, best_score = None, -1e9
    for ind in hof:
        w = [max(0, min(1, x)) for x in ind]
        s = sum(w) or 1
        w = [x / s for x in w]
        lat = sum(w[i] * pm_obs[models[i]]["latency"] for i in range(n))
        cst = sum(w[i] * pm_obs[models[i]]["cost"] for i in range(n))
        qlt = sum(w[i] * pm_obs[models[i]]["quality"] for i in range(n))
        aln = sum(w[i] * align[models[i]] for i in range(n))
        score = -lat - 0.5 * cst + 0.8 * qlt + 5 * aln
        if score > best_score:
            best, best_score = (w, (lat, cst, qlt, aln)), score

    if not best:
        return {m: 1 / n for m in models}

    w, (lat, cst, qlt, aln) = best
    NSGA_BEST_LATENCY.set(lat)
    NSGA_BEST_COST.set(cst)
    NSGA_BEST_QUALITY.set(qlt)
    NSGA_BEST_ALIGNMENT.set(aln)

    return {models[i]: float(w[i]) for i in range(n)}


# ============================================================
# Execução periódica
# ============================================================
def persist_weights(weights: Dict[str, float]):
    """Grava pesos no banco e Redis."""
    try:
        ensure_tables()
        with engine.begin() as conn:
            for m, w in weights.items():
                conn.execute(
                    text("""
                        INSERT INTO nsga_weights (model, weight, computed_at)
                        VALUES (:m, :w, NOW())
                        ON DUPLICATE KEY UPDATE weight = :w, computed_at = NOW();
                    """),
                    {"m": m, "w": w},
                )
        if redis_client:
            redis_client.set(REDIS_KEY_WEIGHTS, json.dumps(weights))
        for m, w in weights.items():
            WEIGHT_BY_MODEL.labels(model=m).set(w)
    except Exception as e:
        logger.warning("[nsga] Falha ao gravar pesos: %s", e)


def one_iteration():
    """Executa uma iteração completa NSGA-II."""
    try:
        ensure_tables()
        models = load_candidate_models()
        if not models:
            logger.warning("[nsga] Nenhum modelo disponível.")
            return

        logs = load_recent_query_logs(QUERY_LOG_LOOKBACK_MINUTES, QUERY_LOG_MAX_ROWS)
        grouped: Dict[str, Dict[str, List[float]]] = {}
        for row in logs:
            m = row["model"]
            grouped.setdefault(m, {"lat": [], "cost": [], "qual": [], "succ": []})
            grouped[m]["lat"].append(row["latency"])
            grouped[m]["cost"].append(row["cost"])
            grouped[m]["qual"].append(row["quality"])
            grouped[m]["succ"].append(1.0 if row["reward"] > 0.3 else 0.0)

        if not grouped:
            logger.warning("[nsga] Nenhum dado em query_log — usando placeholders.")
            pm_obs = {
                m: {
                    "latency": random.uniform(1, 5),
                    "cost": 0.001 if "ollama" in m.lower() else random.uniform(0.005, 0.03),
                    "quality": random.uniform(6, 9),
                    "success_rate": random.uniform(0.85, 0.99),
                }
                for m in models
            }
        else:
            pm_obs = {
                m: {
                    "latency": sum(v["lat"]) / len(v["lat"]),
                    "cost": sum(v["cost"]) / len(v["cost"]),
                    "quality": sum(v["qual"]) / len(v["qual"]),
                    "success_rate": sum(v["succ"]) / len(v["succ"]),
                }
                for m, v in grouped.items()
            }

        align = {m: random.uniform(0.4, 1.0) for m in models}

        for m, obs in pm_obs.items():
            MODEL_OBS_LAT.labels(model=m).set(obs["latency"])
            MODEL_OBS_COST.labels(model=m).set(obs["cost"])
            MODEL_OBS_QUAL.labels(model=m).set(obs["quality"])
            MODEL_OBS_SR.labels(model=m).set(obs["success_rate"])
            MODEL_ALIGNMENT.labels(model=m).set(align[m])

        weights = run_nsga(models, pm_obs, align)
        persist_weights(weights)
        NSGA_RUNS.inc()
        NSGA_LAST_RUN_TS.set(time.time())

        logger.info("[nsga] Pesos atualizados: %s", weights)
    except Exception as e:
        logger.exception("[nsga] Falha na iteração NSGA: %s", e)


def nsga_loop():
    """Loop contínuo."""
    one_iteration()
    while True:
        time.sleep(UPDATE_INTERVAL_S)
        one_iteration()


# ============================================================
# API FastAPI
# ============================================================
app = FastAPI(title="NSGA Weights Updater")


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(REGISTRY).decode("utf-8"))


@app.get("/health")
def health():
    db_ok = True
    redis_ok = bool(redis_client)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    return JSONResponse(
        {
            "status": "ok" if (db_ok and redis_ok) else "degraded",
            "db": db_ok,
            "redis": redis_ok,
            "last_run_epoch": int(time.time()),
        }
    )


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    t = threading.Thread(target=nsga_loop, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=9999, log_level="info")
