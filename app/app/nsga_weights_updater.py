# -*- coding: utf-8 -*-
"""
nsga_weights_updater.py
------------------------------------------------------------
Serviço NSGA-II com fallback dinâmico entre modelos.

- Lê modelos do settings_dynamic (Redis → DB → .env)
- Executa o algoritmo multiobjetivo (latência, custo, qualidade, alinhamento)
- Atualiza pesos ótimos no Redis e no banco
- Expõe métricas via /metrics (Prometheus)
- Aguarda Redis e MariaDB automaticamente (sem prestart.sh)

Parâmetros NSGA-II (mapeamento → variável):
- N_pop  → tamanho da população  (env: NSGA_POP, body JSON: N_pop)
- N_gen  → nº de gerações        (env: NSGA_GEN, body JSON: N_gen)
- cxpb   → prob. de cruzamento   (env: NSGA_CXPB, body JSON: cxpb)
- mutpb  → prob. de mutação      (env: NSGA_MUTPB, body JSON: mutpb)
- eta_c  → parâmetro SBX (cx)    (env: NSGA_ETA_C, body JSON: eta_c)
- eta_m  → parâmetro PolyMut     (env: NSGA_ETA_M, body JSON: eta_m)
"""
from __future__ import annotations
import json
import logging
import os
import random
import threading
import time
import socket
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pymysql
import redis
import uvicorn
from deap import algorithms, base, creator, tools
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Gauge, REGISTRY, generate_latest
from sqlalchemy import create_engine, text

from app.settings_dynamic import settings

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("nsga-updater")

# ============================================================
# 🕒 Aguarda Redis e MariaDB antes de iniciar
# ============================================================
def wait_for_service(host: str, port: int, name: str, timeout: int = 60, delay: int = 2) -> None:
    start = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=2):
                logger.info(f"✅ {name} disponível em {host}:{port}")
                return
        except OSError:
            elapsed = int(time.time() - start)
            if elapsed > timeout:
                logger.warning(f"⚠️ Timeout ao aguardar {name} ({host}:{port})")
                return
            logger.info(f"⏳ Aguardando {name}... ({elapsed}s)")
            time.sleep(delay)

def wait_for_redis(host: str, port: int, password: str, retries: int = 20, delay: int = 3) -> None:
    client = redis.Redis(host=host, port=port, password=password, socket_connect_timeout=2)
    for i in range(1, retries + 1):
        try:
            if client.ping():
                logger.info(f"✅ Redis disponível ({host}:{port})")
                return
        except redis.exceptions.ConnectionError:
            logger.info(f"⏳ Tentativa {i}/{retries}: aguardando Redis...")
        time.sleep(delay)
    logger.warning("⚠️ Redis não respondeu após múltiplas tentativas.")

def wait_for_mariadb(host: str, user: str, password: str, dbname: str, retries: int = 20, delay: int = 3) -> None:
    for i in range(1, retries + 1):
        try:
            conn = pymysql.connect(host=host, user=user, password=password, database=dbname, connect_timeout=2)
            conn.close()
            logger.info(f"✅ MariaDB disponível ({host})")
            return
        except pymysql.err.OperationalError:
            logger.info(f"⏳ Tentativa {i}/{retries}: aguardando MariaDB...")
        time.sleep(delay)
    logger.warning("⚠️ MariaDB não respondeu após múltiplas tentativas.")

# Executa as verificações iniciais
logger.info("🚀 Inicializando NSGA-II com verificação de dependências...")
redis_host = os.getenv("REDIS_HOST", "redis")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_pass = os.getenv("REDIS_PASSWORD", "SenhaForte")
db_host = os.getenv("DB_HOST", "mariadb")
db_user = os.getenv("DB_USER", "router_user")
db_pass = os.getenv("DB_PASS", "router_pass")
db_name = os.getenv("DB_NAME", "routerdb")

wait_for_service(db_host, 3306, "MariaDB TCP")
wait_for_service(redis_host, redis_port, "Redis TCP")
wait_for_redis(redis_host, redis_port, redis_pass)
wait_for_mariadb(db_host, db_user, db_pass, db_name)
logger.info("✅ Todas as dependências disponíveis. Iniciando NSGA-II.")

# ============================================================
# Banco e Redis
# ============================================================
DB_URL = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:3306/{settings.DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

REDIS_KEY_WEIGHTS = "nsga:weights"
REDIS_KEY_CANDIDATES = "nsga:candidate_models"

UPDATE_INTERVAL_S = int(settings.get("NSGA_UPDATE_INTERVAL_S", "300"))
QUERY_LOG_LOOKBACK_MINUTES = int(settings.get("NSGA_LOOKBACK_MINUTES", "180"))
QUERY_LOG_MAX_ROWS = int(settings.get("NSGA_LOOKBACK_MAXROWS", "2000"))

def get_redis(max_wait_s: int = 0):
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
# NSGA-II principal (parametrizado)
# ============================================================
def run_nsga(
    models: List[str],
    pm_obs: Dict[str, Dict[str, float]],
    align: Dict[str, float],
    *,
    N_pop: int = 40,       # tamanho da população
    N_gen: int = 30,       # nº gerações
    cxpb: float = 0.9,     # prob. cruzamento
    mutpb: float = 0.1,    # prob. mutação
    eta_c: float = 20.0,   # eta do SBX
    eta_m: float = 20.0    # eta do Polynomial Mutation
) -> Tuple[Dict[str, float], Dict[str, float]]:
    n = len(models)
    if n == 1:
        return {models[0]: 1.0}, {"latency": pm_obs[models[0]]["latency"], "cost": pm_obs[models[0]]["cost"], "quality": pm_obs[models[0]]["quality"], "alignment": align[models[0]]}

    # Limites defensivos
    N_pop = max(4, int(N_pop))
    N_gen = max(1, int(N_gen))
    cxpb = min(max(float(cxpb), 0.0), 1.0)
    mutpb = min(max(float(mutpb), 0.0), 1.0)
    eta_c = max(1.0, float(eta_c))
    eta_m = max(1.0, float(eta_m))

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
        s = sum(w_raw) or 1.0
        w = [x / s for x in w_raw]
        lat = sum(w[i] * pm_obs[models[i]]["latency"] for i in range(n))
        cst = sum(w[i] * pm_obs[models[i]]["cost"] for i in range(n))
        qlt = sum(w[i] * pm_obs[models[i]]["quality"] for i in range(n))
        aln = sum(w[i] * align[models[i]] for i in range(n))
        proxy = (10 - lat + 10 - cst + qlt + 10 * aln) / 4
        NSGA_POP_FITNESS_AVG.set(proxy)
        return lat, cst, qlt, aln

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=eta_c)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=eta_m, indpb=1 / n)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=N_pop)
    hof = tools.ParetoFront()
    algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=N_pop, lambda_=N_pop,
        cxpb=cxpb, mutpb=mutpb,
        ngen=N_gen, halloffame=hof, verbose=False
    )

    best, best_score = None, -1e9
    best_tuple = (0.0, 0.0, 0.0, 0.0)
    for ind in hof:
        w = [max(0, min(1, x)) for x in ind]
        s = sum(w) or 1.0
        w = [x / s for x in w]
        lat = sum(w[i] * pm_obs[models[i]]["latency"] for i in range(n))
        cst = sum(w[i] * pm_obs[models[i]]["cost"] for i in range(n))
        qlt = sum(w[i] * pm_obs[models[i]]["quality"] for i in range(n))
        aln = sum(w[i] * align[models[i]] for i in range(n))
        score = -lat - 0.5 * cst + 0.8 * qlt + 5.0 * aln
        if score > best_score:
            best, best_score = ([float(x) for x in w], score)
            best_tuple = (lat, cst, qlt, aln)

    if not best:
        weights = {m: 1.0 / n for m in models}
        return weights, {"latency": 0.0, "cost": 0.0, "quality": 0.0, "alignment": 0.0}

    lat, cst, qlt, aln = best_tuple
    NSGA_BEST_LATENCY.set(lat)
    NSGA_BEST_COST.set(cst)
    NSGA_BEST_QUALITY.set(qlt)
    NSGA_BEST_ALIGNMENT.set(aln)

    weights = {models[i]: float(best[i]) for i in range(n)}
    summary = {"latency": float(lat), "cost": float(cst), "quality": float(qlt), "alignment": float(aln)}
    return weights, summary

# ============================================================
# Persistência e ciclo
# ============================================================
def persist_weights(weights: Dict[str, float]):
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

def aggregate_obs(models: List[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
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
                "latency": sum(v["lat"]) / max(1, len(v["lat"])),
                "cost":    sum(v["cost"]) / max(1, len(v["cost"])),
                "quality": sum(v["qual"]) / max(1, len(v["qual"])),
                "success_rate": sum(v["succ"]) / max(1, len(v["succ"])),
            }
            for m, v in grouped.items()
            if m in models
        }
        # completa algum modelo sem dado
        for m in models:
            if m not in pm_obs:
                pm_obs[m] = {
                    "latency": 3.0, "cost": 0.005 if "ollama" in m.lower() else 0.02,
                    "quality": 7.5, "success_rate": 0.9
                }

    align = {m: random.uniform(0.4, 1.0) for m in models}  # placeholder de alinhamento
    # Exporta métricas observadas
    for m, obs in pm_obs.items():
        MODEL_OBS_LAT.labels(model=m).set(obs["latency"])
        MODEL_OBS_COST.labels(model=m).set(obs["cost"])
        MODEL_OBS_QUAL.labels(model=m).set(obs["quality"])
        MODEL_OBS_SR .labels(model=m).set(obs["success_rate"])
        MODEL_ALIGNMENT.labels(model=m).set(align[m])
    return pm_obs, align

def one_iteration_with_params(N_pop:int, N_gen:int, cxpb:float, mutpb:float, eta_c:float, eta_m:float) -> Dict[str, Any]:
    models = load_candidate_models()
    if not models:
        raise RuntimeError("Nenhum modelo disponível.")
    pm_obs, align = aggregate_obs(models)
    if cxpb + mutpb > 1.0:
        mutpb = 1.0 - cxpb
    weights, summary = run_nsga(
        models, pm_obs, align,
        N_pop=N_pop, N_gen=N_gen, cxpb=cxpb, mutpb=mutpb, eta_c=eta_c, eta_m=eta_m
    )
    persist_weights(weights)
    NSGA_RUNS.inc()
    NSGA_LAST_RUN_TS.set(time.time())
    # Eficiência composta (qualidade / (latência * custo))
    eff = 0.0
    lat = max(1e-6, summary["latency"])
    cst = max(1e-9, summary["cost"])
    qlt = max(0.0, summary["quality"])
    eff = float(qlt / (lat * cst))
    return {
        "weights": weights,
        "summary": summary,
        "efficiency": eff,
        "params": dict(N_pop=N_pop, N_gen=N_gen, cxpb=cxpb, mutpb=mutpb, eta_c=eta_c, eta_m=eta_m),
    }

def one_iteration_defaults() -> Dict[str, Any]:
    def _getf(name:str, default:float)->float:
        try:
            return float(os.getenv(name, default))
        except Exception:
            return float(default)
    def _geti(name:str, default:int)->int:
        try:
            return int(os.getenv(name, default))
        except Exception:
            return int(default)

    N_pop = _geti("NSGA_POP", 40)
    N_gen = _geti("NSGA_GEN", 30)
    cxpb  = _getf("NSGA_CXPB", 0.9)
    mutpb = _getf("NSGA_MUTPB", 0.1)
    eta_c = _getf("NSGA_ETA_C", 20.0)
    eta_m = _getf("NSGA_ETA_M", 20.0)
    return one_iteration_with_params(N_pop, N_gen, cxpb, mutpb, eta_c, eta_m)

def nsga_loop():
    one_iteration_defaults()
    while True:
        time.sleep(UPDATE_INTERVAL_S)
        one_iteration_defaults()

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

@app.post("/run")
def run_once(payload: Dict[str, Any] = Body(default=None)):
    """
    Executa UMA rodada do NSGA-II com parâmetros informados no body:
    {
      "N_pop": 40, "N_gen": 30, "cxpb": 0.9, "mutpb": 0.1, "eta_c": 20.0, "eta_m": 20.0
    }
    Caso não informado, usa valores das variáveis de ambiente.
    """
    try:
        if payload and isinstance(payload, dict):
            N_pop = int(payload.get("N_pop", os.getenv("NSGA_POP", 40)))
            N_gen = int(payload.get("N_gen", os.getenv("NSGA_GEN", 30)))
            cxpb  = float(payload.get("cxpb",  os.getenv("NSGA_CXPB", 0.9)))
            mutpb = float(payload.get("mutpb", os.getenv("NSGA_MUTPB", 0.1)))
            eta_c = float(payload.get("eta_c", os.getenv("NSGA_ETA_C", 20.0)))
            eta_m = float(payload.get("eta_m", os.getenv("NSGA_ETA_M", 20.0)))
            result = one_iteration_with_params(N_pop, N_gen, cxpb, mutpb, eta_c, eta_m)
        else:
            result = one_iteration_defaults()
        return JSONResponse(result)
    except Exception as e:
        logger.exception("[nsga] Falha em /run: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    t = threading.Thread(target=nsga_loop, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=9999, log_level="info")
