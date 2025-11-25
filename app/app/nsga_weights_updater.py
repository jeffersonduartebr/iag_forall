# -*- coding: utf-8 -*-
"""
nsga_weights_updater.py — MULTIMODAL (CORRIGIDO)
------------------------------------------------------------
Extensão multimodal do NSGA-II para calcular pesos ótimos.

Correção:
- Mapeamento correto das variáveis de ambiente para cada modalidade.
- Fallback de segurança para evitar RuntimeError se a lista estiver vazia.
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
from fastapi import FastAPI, Body, Path
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import create_engine, text
from prometheus_client import (
    Counter, Gauge, REGISTRY, generate_latest
)

from app.settings_dynamic import settings

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("nsga-updater-mm")

# ============================================================
# Wait for services
# ============================================================
def wait_for_service(host: str, port: int, name: str,
                     timeout: int = 60, delay: int = 2) -> None:
    start = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=2):
                logger.info(f"✅ {name} disponível em {host}:{port}")
                return
        except OSError:
            elapsed = int(time.time() - start)
            if elapsed > timeout:
                logger.warning(f"⚠️ Timeout esperando {name}")
                return
            logger.info(f"⏳ Aguardando {name}... ({elapsed}s)")
            time.sleep(delay)

def wait_for_redis(host: str, port: int, password: str,
                   retries: int = 20, delay: int = 3) -> None:
    c = redis.Redis(host=host, port=port, password=password,
                    socket_connect_timeout=2)
    for i in range(1, retries + 1):
        try:
            if c.ping():
                logger.info(f"✅ Redis disponível")
                return
        except redis.exceptions.ConnectionError:
            logger.info(f"⏳ Tentativa {i}/{retries}: aguardando Redis...")
        time.sleep(delay)
    logger.warning("⚠️ Redis não respondeu após múltiplas tentativas.")

def wait_for_mariadb(host: str, user: str, password: str,
                     dbname: str, retries: int = 20, delay: int = 3) -> None:
    for i in range(1, retries + 1):
        try:
            conn = pymysql.connect(host=host, user=user,
                                   password=password, database=dbname,
                                   connect_timeout=2)
            conn.close()
            logger.info(f"✅ MariaDB disponível ({host})")
            return
        except pymysql.err.OperationalError:
            logger.info(f"⏳ Tentativa {i}/{retries}: aguardando MariaDB...")
        time.sleep(delay)
    logger.warning("⚠️ MariaDB não respondeu após múltiplas tentativas.")

# Initial service waits
db_host = os.getenv("DB_HOST", "mariadb")
db_user = os.getenv("DB_USER", "router_user")
db_pass = os.getenv("DB_PASS", "router_pass")
db_name = os.getenv("DB_NAME", "routerdb")
redis_host = os.getenv("REDIS_HOST", "redis")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_pass = os.getenv("REDIS_PASSWORD", "")

wait_for_service(db_host, 3306, "MariaDB TCP")
wait_for_service(redis_host, redis_port, "Redis TCP")
wait_for_redis(redis_host, redis_port, redis_pass or "")
wait_for_mariadb(db_host, db_user, db_pass, db_name)

logger.info("✅ Dependências estão OK — iniciando NSGA multimodal.")

# ============================================================
# Setup Banco + Redis
# ============================================================
DB_URL = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:3306/{settings.DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

def get_redis():
    try:
        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_pass or None,
            db=int(os.getenv("REDIS_DB", "0"))
        )
        r.ping()
        return r
    except Exception as exc:
        logger.warning("[nsga-mm] Falha Redis: %s", exc)
        return None

redis_client = get_redis()

# ============================================================
# CONSTANTES MULTIMODAIS
# ============================================================

MODALITIES = ["text", "vision", "multimodal"]

REDIS_KEY_WEIGHTS = {
    m: f"nsga:weights:{m}" for m in MODALITIES
}

REDIS_KEY_CANDIDATES = {
    "text": "nsga:candidate_models:text",
    "vision": "nsga:candidate_models:vision",
    "multimodal": "nsga:candidate_models:multimodal",
}

UPDATE_INTERVAL_S = int(settings.get("NSGA_UPDATE_INTERVAL_S", "300"))
QUERY_LOG_LOOKBACK_MINUTES = int(settings.get("NSGA_LOOKBACK_MINUTES", "180"))
QUERY_LOG_MAX_ROWS = int(settings.get("NSGA_LOOKBACK_MAXROWS", "2000"))

# ============================================================
# Tabela multimodal nsga_weights
# ============================================================
DDL_WEIGHTS_MM = """
CREATE TABLE IF NOT EXISTS nsga_weights (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    modality VARCHAR(32) NOT NULL,
    model VARCHAR(255) NOT NULL,
    weight FLOAT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_mod_model (modality, model)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

with engine.begin() as conn:
    conn.execute(text(DDL_WEIGHTS_MM))

logger.info("Tabela nsga_weights multimodal OK.")

# ============================================================
# 🔍 Carregar modelos candidatos por modalidade (CORRIGIDO)
# ============================================================

def load_candidate_models(modality: str) -> List[str]:
    """
    Retorna a lista de modelos da modalidade.
    Origem: Redis -> Settings -> Fallback.
    """
    # 1️⃣ Tentar Redis
    try:
        if redis_client:
            raw = redis_client.get(REDIS_KEY_CANDIDATES.get(modality, ""))
            if raw:
                data = json.loads(raw)
                if isinstance(data, list) and data:
                    return [str(x) for x in data]
    except Exception:
        pass

    # 2️⃣ Settings (Mapeamento Correto)
    data = []
    if modality == "text":
        data = settings.CANDIDATE_MODELS_LIST
    elif modality == "vision":
        data = settings.CANDIDATE_VISION_MODELS_LIST
    elif modality == "multimodal":
        data = settings.CANDIDATE_MULTIMODAL_MODELS_LIST

    if data:
        return data

    # 3️⃣ Fallback de Segurança (Evita Crash)
    logger.warning(f"[nsga-mm] ⚠️ Nenhum modelo encontrado para '{modality}'. Usando fallback.")
    
    if modality == "text":
        return ["ollama/phi4:latest", "ollama/mistral:7b"]
    elif modality in ("vision", "multimodal"):
        return ["ollama/llava:7b"]
    
    return []


# ============================================================
# 📚 Leitura do histórico EMA por modalidade
# ============================================================

def load_recent_ema(modality: str, limit: int = 5000) -> List[Dict[str, Any]]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT model, ema_latency, ema_cost, ema_quality, ema_alignment
                    FROM ema_history
                    WHERE modality = :m
                    ORDER BY updated_at DESC
                    LIMIT :lim
                """),
                {"m": modality, "lim": limit},
            ).mappings().all()

        return [
            {
                "model": r["model"],
                "latency": float(r["ema_latency"]),
                "cost": float(r["ema_cost"]),
                "quality": float(r["ema_quality"]),
                "alignment": float(r["ema_alignment"]),
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning(f"[nsga-mm] Falha ao ler ema_history modality={modality}: {e}")
        return []


# ============================================================
# 📊 Agregar EMA em uma única média por modelo
# ============================================================

def aggregate_ema_by_model(modality: str, models: List[str]) -> Dict[str, Dict[str, float]]:
    rows = load_recent_ema(modality)
    
    # Se não houver histórico, gera dados sintéticos para cold-start
    # Isso permite que o NSGA rode mesmo sem dados reais
    if not rows:
        logger.info(f"[nsga-mm] Cold-start para {modality}. Gerando dados iniciais.")
        return {
            m: {
                "latency": random.uniform(0.5, 2.0),
                "cost": random.uniform(0.001, 0.01),
                "quality": random.uniform(5.0, 8.0),
                "alignment": 1.0,
            }
            for m in models
        }

    grouped = {}
    for r in rows:
        m = r["model"]
        if m not in models: continue
        grouped.setdefault(m, {"lat": [], "cost": [], "qual": [], "aln": []})
        grouped[m]["lat"].append(r["latency"])
        grouped[m]["cost"].append(r["cost"])
        grouped[m]["qual"].append(r["quality"])
        grouped[m]["aln"].append(r["alignment"])

    out = {}
    for m in models:
        g = grouped.get(m)
        if g:
            out[m] = {
                "latency": sum(g["lat"]) / len(g["lat"]),
                "cost": sum(g["cost"]) / len(g["cost"]),
                "quality": sum(g["qual"]) / len(g["qual"]),
                "alignment": sum(g["aln"]) / len(g["aln"]),
            }
        else:
            # Modelo na lista mas sem histórico ainda
            out[m] = {
                "latency": 2.0, "cost": 0.01, "quality": 5.0, "alignment": 1.0
            }

    return out


# ============================================================
# 📈 Métricas Prometheus
# ============================================================

WEIGHT_BY_MODEL_MM = Gauge("nsga_model_weight_mm", "Peso NSGA-II multimodal", ["modality", "model"])
OBS_LATENCY_MM = Gauge("nsga_obs_latency_mm", "EMA latência", ["modality", "model"])
OBS_COST_MM = Gauge("nsga_obs_cost_mm", "EMA custo", ["modality", "model"])
OBS_QUALITY_MM = Gauge("nsga_obs_quality_mm", "EMA qualidade", ["modality", "model"])
OBS_ALIGNMENT_MM = Gauge("nsga_obs_alignment_mm", "EMA alinhamento", ["modality", "model"])

NSGA_RUNS_MM = Counter("nsga_runs_total_mm", "Execuções NSGA-II", ["modality"])
NSGA_LAST_RUN_TS_MM = Gauge("nsga_last_run_timestamp_mm", "Timestamp última execução", ["modality"])
NSGA_BEST_LAT_MM = Gauge("nsga_best_latency_seconds_mm", "Melhor Latência", ["modality"])
NSGA_BEST_COST_MM = Gauge("nsga_best_cost_per_1k_mm", "Melhor Custo", ["modality"])
NSGA_BEST_QUAL_MM = Gauge("nsga_best_quality_mm", "Melhor Qualidade", ["modality"])
NSGA_BEST_ALIGN_MM = Gauge("nsga_best_alignment_mm", "Melhor Alinhamento", ["modality"])

def export_obs_to_prometheus(modality: str, metrics: Dict[str, Dict[str, float]]) -> None:
    for model, obs in metrics.items():
        OBS_LATENCY_MM.labels(modality=modality, model=model).set(obs["latency"])
        OBS_COST_MM.labels(modality=modality, model=model).set(obs["cost"])
        OBS_QUALITY_MM.labels(modality=modality, model=model).set(obs["quality"])
        OBS_ALIGNMENT_MM.labels(modality=modality, model=model).set(obs["alignment"])


# ============================================================
# 🧮 Núcleo NSGA-II (multimodal)
# ============================================================

def run_nsga_for_modality(
    modality: str,
    models: List[str],
    obs_metrics: Dict[str, Dict[str, float]],
    *,
    N_pop: int,
    N_gen: int,
    cxpb: float,
    mutpb: float,
    eta_c: float,
    eta_m: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    
    n = len(models)
    # Se só tiver 1 modelo, peso é 100%
    if n <= 1:
        m = models[0] if models else "default"
        s = obs_metrics.get(m, {"latency": 0, "cost": 0, "quality": 0, "alignment": 0})
        return {m: 1.0}, s

    # Criadores DEAP (Idempotente)
    if not hasattr(creator, "FitnessMultiMM"):
        creator.create("FitnessMultiMM", base.Fitness, weights=(-1.0, -1.0, +1.0, +1.0))
    if not hasattr(creator, "IndividualMM"):
        creator.create("IndividualMM", list, fitness=creator.FitnessMultiMM)

    toolbox = base.Toolbox()
    toolbox.register("attr_w", lambda: random.random())
    toolbox.register("individual", tools.initRepeat, creator.IndividualMM, toolbox.attr_w, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        w_raw = [max(0.0, min(1.0, x)) for x in ind]
        s = sum(w_raw) or 1.0
        w_norm = [x / s for x in w_raw]

        lat = sum(w_norm[i] * obs_metrics[models[i]]["latency"] for i in range(n))
        cst = sum(w_norm[i] * obs_metrics[models[i]]["cost"] for i in range(n))
        qlt = sum(w_norm[i] * obs_metrics[models[i]]["quality"] for i in range(n))
        aln = sum(w_norm[i] * obs_metrics[models[i]]["alignment"] for i in range(n))
        return lat, cst, qlt, aln

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=eta_c)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=eta_m, indpb=1.0/n)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=N_pop)
    hof = tools.ParetoFront()
    algorithms.eaMuPlusLambda(pop, toolbox, mu=N_pop, lambda_=N_pop, cxpb=cxpb, mutpb=mutpb, ngen=N_gen, halloffame=hof, verbose=False)

    # Escolha do melhor indivíduo (Scalarização simples para produção)
    best_weights = None
    best_score = float("-inf")
    best_tuple = (0,0,0,0)

    for ind in hof:
        w_raw = [max(0.0, min(1.0, x)) for x in ind]
        s = sum(w_raw) or 1.0
        w_norm = [x / s for x in w_raw]
        
        vals = evaluate(ind)
        # Score: +Qualidade -Custo -Latência
        score = (vals[2] * 2.0) + (vals[3] * 1.0) - (vals[0] * 0.5) - (vals[1] * 50.0)
        
        if score > best_score:
            best_score = score
            best_weights = w_norm
            best_tuple = vals

    final_weights = {models[i]: best_weights[i] for i in range(n)}
    summary = {
        "latency": best_tuple[0], "cost": best_tuple[1], 
        "quality": best_tuple[2], "alignment": best_tuple[3]
    }
    
    # Métricas ótimas
    NSGA_BEST_LAT_MM.labels(modality=modality).set(best_tuple[0])
    NSGA_BEST_COST_MM.labels(modality=modality).set(best_tuple[1])
    NSGA_BEST_QUAL_MM.labels(modality=modality).set(best_tuple[2])
    
    return final_weights, summary


def compute_efficiency(summary: Dict[str, float]) -> float:
    lat = max(1e-6, float(summary.get("latency", 0.0)))
    cst = max(1e-9, float(summary.get("cost", 0.0)))
    qlt = max(0.0, float(summary.get("quality", 0.0)))
    return float(qlt / (lat * cst))


# ============================================================
# 💾 Persistência
# ============================================================

def persist_weights(modality: str, weights: Dict[str, float]) -> None:
    if not weights: return
    try:
        with engine.begin() as conn:
            for model, weight in weights.items():
                conn.execute(
                    text("""
                        INSERT INTO nsga_weights (modality, model, weight) VALUES (:mod, :m, :w)
                        ON DUPLICATE KEY UPDATE weight = :w, updated_at = CURRENT_TIMESTAMP
                    """),
                    {"mod": modality, "m": model, "w": float(weight)},
                )
    except Exception as e:
        logger.warning(f"[nsga-mm] Erro DB: {e}")

    try:
        if redis_client:
            key = REDIS_KEY_WEIGHTS[modality]
            redis_client.set(key, json.dumps(weights))
    except Exception as e:
        logger.warning(f"[nsga-mm] Erro Redis: {e}")

    for model, weight in weights.items():
        WEIGHT_BY_MODEL_MM.labels(modality=modality, model=model).set(weight)


# ============================================================
# 🎛 Execução
# ============================================================

def one_iteration_with_params(modality: str, N_pop: int, N_gen: int, cxpb: float, mutpb: float, eta_c: float, eta_m: float) -> Dict[str, Any]:
    models = load_candidate_models(modality)
    if not models:
        logger.error(f"Sem modelos para {modality}, abortando iteração.")
        return {"status": "skipped", "reason": "no_models"}

    obs = aggregate_ema_by_model(modality, models)
    export_obs_to_prometheus(modality, obs)

    weights, summary = run_nsga_for_modality(
        modality, models, obs,
        N_pop=N_pop, N_gen=N_gen, cxpb=cxpb, mutpb=mutpb, eta_c=eta_c, eta_m=eta_m,
    )

    persist_weights(modality, weights)
    NSGA_RUNS_MM.labels(modality=modality).inc()
    NSGA_LAST_RUN_TS_MM.labels(modality=modality).set(time.time())

    return {
        "modality": modality, "weights": weights, "summary": summary,
        "efficiency": compute_efficiency(summary)
    }

def _get_env(key, default, cast=str):
    try: return cast(os.getenv(key, default))
    except: return default

def one_iteration_defaults(modality: str) -> Dict[str, Any]:
    return one_iteration_with_params(
        modality,
        _get_env("NSGA_POP", 40, int), _get_env("NSGA_GEN", 30, int),
        _get_env("NSGA_CXPB", 0.9, float), _get_env("NSGA_MUTPB", 0.1, float),
        _get_env("NSGA_ETA_C", 20.0, float), _get_env("NSGA_ETA_M", 20.0, float)
    )

# ============================================================
# 🔁 Loop automático
# ============================================================

def nsga_loop_multimodal():
    time.sleep(10) # Aguarda sistemas subirem
    while True:
        for mod in MODALITIES:
            try:
                one_iteration_defaults(mod)
                logger.info(f"[nsga-mm] Iteração OK: {mod}")
            except Exception as e:
                logger.error(f"[nsga-mm] Erro loop {mod}: {e}")
        time.sleep(UPDATE_INTERVAL_S)

# ============================================================
# 🌐 API FastAPI
# ============================================================
app = FastAPI(title="NSGA Weights Updater (Multimodal)")

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(REGISTRY).decode("utf-8"))

@app.get("/health")
def health():
    return {"status": "ok", "redis": bool(redis_client)}

@app.post("/run/{modality}")
def run_once(modality: str = Path(...), payload: Dict[str, Any] = Body(default={})):
    if modality not in MODALITIES:
        return JSONResponse({"error": "Invalid modality"}, status_code=400)
    
    try:
        # Merge defaults with payload
        params = {
            "N_pop": int(payload.get("N_pop", _get_env("NSGA_POP", 40, int))),
            "N_gen": int(payload.get("N_gen", _get_env("NSGA_GEN", 30, int))),
            "cxpb": float(payload.get("cxpb", _get_env("NSGA_CXPB", 0.9, float))),
            "mutpb": float(payload.get("mutpb", _get_env("NSGA_MUTPB", 0.1, float))),
            "eta_c": float(payload.get("eta_c", _get_env("NSGA_ETA_C", 20.0, float))),
            "eta_m": float(payload.get("eta_m", _get_env("NSGA_ETA_M", 20.0, float))),
        }
        result = one_iteration_with_params(modality, **params)
        return JSONResponse(result)
    except Exception as e:
        logger.exception(f"Erro em /run/{modality}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    t = threading.Thread(target=nsga_loop_multimodal, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=9999, log_level="info")