# -*- coding: utf-8 -*-
"""
nsga_weights_updater.py — Otimizador Multimodal (NSGA-II + UQ Tuning)
---------------------------------------------------------------------
Serviço de Otimização Contínua.

Responsabilidades:
1. Coletar métricas históricas (EMA) do Banco de Dados.
2. Rodar Algoritmo Genético (NSGA-II) para encontrar pesos ótimos (w1, w2, w3) para cada modelo.
3. Ajustar dinamicamente o Limiar de Incerteza (UNCERTAINTY_THRESHOLD) baseado na estabilidade do sistema.
4. Persistir configurações ótimas no Redis/DB para o Router usar.

Correções:
- Mapeamento correto de modelos por modalidade (Text/Vision/Multimodal).
- Fallback de segurança para Cold Start.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import socket
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional

import pymysql
import redis
import uvicorn
from deap import algorithms, base, creator, tools
from fastapi import FastAPI, Body, Path
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import create_engine, text
from prometheus_client import Counter, Gauge, REGISTRY, generate_latest

from app.settings_dynamic import settings

# ============================================================
# Logging & Config
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("nsga-updater")

# Configurações de Ambiente
UPDATE_INTERVAL_S = int(settings.get("NSGA_UPDATE_INTERVAL_S", "300"))
MODALITIES = ["text", "vision", "multimodal"]

# Chaves Redis
REDIS_KEY_WEIGHTS = {m: f"nsga:weights:{m}" for m in MODALITIES}
REDIS_KEY_CANDIDATES = {
    "text": "nsga:candidate_models:text",
    "vision": "nsga:candidate_models:vision",
    "multimodal": "nsga:candidate_models:multimodal",
}

# ============================================================
# Conexões (DB & Redis)
# ============================================================

DB_URL = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

def get_redis_client():
    try:
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD or None,
            db=settings.REDIS_DB,
            socket_timeout=2
        )
        if r.ping(): return r
    except Exception as e:
        logger.warning(f"[NSGA] Redis indisponível: {e}")
    return None

redis_client = get_redis_client()

# ============================================================
# Inicialização de Tabelas
# ============================================================
def init_db_tables():
    DDL = """
    CREATE TABLE IF NOT EXISTS nsga_weights (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        modality VARCHAR(32) NOT NULL,
        model VARCHAR(255) NOT NULL,
        weight FLOAT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uniq_mod_model (modality, model)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(DDL))
        logger.info("[NSGA] Tabela 'nsga_weights' verificada.")
    except Exception as e:
        logger.error(f"[NSGA] Erro ao criar tabelas: {e}")

init_db_tables()


# ============================================================
# 1. Carregamento de Modelos (CORRIGIDO)
# ============================================================
def load_candidate_models(modality: str) -> List[str]:
    """
    Carrega a lista de modelos candidatos para a modalidade específica.
    Ordem: Redis -> Settings -> Fallback Hardcoded.
    """
    # 1. Redis
    try:
        if redis_client:
            raw = redis_client.get(REDIS_KEY_CANDIDATES.get(modality, ""))
            if raw:
                data = json.loads(raw)
                if isinstance(data, list) and data:
                    return [str(x) for x in data]
    except Exception: pass

    # 2. Settings (CORREÇÃO: Mapeamento Explícito)
    if modality == "text":
        candidates = settings.CANDIDATE_MODELS_LIST
    elif modality == "vision":
        candidates = settings.CANDIDATE_VISION_MODELS_LIST
    elif modality == "multimodal":
        candidates = settings.CANDIDATE_MULTIMODAL_MODELS_LIST
    else:
        candidates = []

    if candidates:
        # Filtra duplicatas e vazios
        return list(set([c for c in candidates if c]))

    # 3. Fallback de Segurança (Evita erro 500)
    logger.warning(f"[NSGA] ⚠️ Nenhum modelo encontrado para '{modality}'. Usando fallback de emergência.")
    
    if modality == "text":
        return ["ollama/phi4:latest", "ollama/mistral:7b"]
    elif modality in ("vision", "multimodal"):
        return ["ollama/llava:7b", "ollama/moondream:latest"]
    
    return []


# ============================================================
# 2. Coleta de Dados Históricos (EMA)
# ============================================================
def aggregate_ema_by_model(modality: str, models: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Lê o histórico de performance (EMA) do banco.
    Retorna dicionário: {model: {latency, cost, quality, alignment}}
    """
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT model, ema_latency, ema_cost, ema_quality, ema_alignment
                    FROM ema_history
                    WHERE modality = :m
                """),
                {"m": modality}
            ).mappings().all()
        
        db_data = {r["model"]: dict(r) for r in rows}

    except Exception as e:
        logger.warning(f"[NSGA] Erro ao ler EMA: {e}")
        db_data = {}

    final_data = {}
    
    # Garante que todos os modelos candidatos tenham dados (Synthetic Cold Start)
    for m in models:
        if m in db_data:
            d = db_data[m]
            final_data[m] = {
                "latency": float(d["ema_latency"]),
                "cost": float(d["ema_cost"]),
                "quality": float(d["ema_quality"]),
                "alignment": float(d["ema_alignment"]),
            }
        else:
            # Dados sintéticos conservadores para modelos novos
            final_data[m] = {
                "latency": 2.0,  # Latência média/alta
                "cost": 0.001,   # Custo baixo
                "quality": 5.0,  # Qualidade neutra
                "alignment": 1.0 # Alinhamento máximo
            }
            
    return final_data


# ============================================================
# 3. Métricas Prometheus
# ============================================================
NSGA_RUNS = Counter("nsga_runs_total", "Execuções do NSGA-II", ["modality"])
NSGA_LAST_TS = Gauge("nsga_last_run_ts", "Timestamp da última execução", ["modality"])
NSGA_UQ_THRESH = Gauge("nsga_uq_threshold", "Limiar de Incerteza otimizado", [])

# ============================================================
# 4. Núcleo NSGA-II (Algoritmo Genético)
# ============================================================
def run_nsga_optimization(
    modality: str,
    models: List[str],
    metrics: Dict[str, Dict[str, float]],
    n_pop=40, n_gen=20
) -> Tuple[Dict[str, float], float]:
    """
    Roda o NSGA-II para encontrar os pesos ideais para cada modelo.
    Retorna: (pesos_finais, pontuação_eficiência)
    """
    n = len(models)
    if n == 0: return {}, 0.0
    if n == 1: return {models[0]: 1.0}, 1.0

    # Setup DEAP
    if not hasattr(creator, "FitnessMulti"):
        # Objetivos: Min Latency, Min Cost, Max Quality, Max Alignment
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -50.0, 2.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        # Normaliza pesos do indivíduo (soma = 1)
        s = sum(individual) or 1.0
        w = [x/s for x in individual]
        
        lat = sum(w[i] * metrics[models[i]]["latency"] for i in range(n))
        cst = sum(w[i] * metrics[models[i]]["cost"] for i in range(n))
        qlt = sum(w[i] * metrics[models[i]]["quality"] for i in range(n))
        aln = sum(w[i] * metrics[models[i]]["alignment"] for i in range(n))
        
        return lat, cst, qlt, aln

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0/n)
    toolbox.register("select", tools.selNSGA2)

    # Execução
    pop = toolbox.population(n=n_pop)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=n_pop, lambda_=n_pop, 
                              cxpb=0.9, mutpb=0.1, ngen=n_gen, verbose=False)

    # Seleção do melhor compromisso (Scalarização)
    best_ind = tools.selBest(pop, 1)[0]
    s = sum(best_ind) or 1.0
    norm_weights = [x/s for x in best_ind]
    
    weights_map = {models[i]: norm_weights[i] for i in range(n)}
    
    # Calcula eficiência sistêmica estimada
    eval_metrics = evaluate(best_ind)
    efficiency_score = (eval_metrics[2] / max(0.01, eval_metrics[0])) # Qualidade / Latência
    
    return weights_map, efficiency_score


# ============================================================
# 5. Ajuste Dinâmico de Incerteza (UQ Tuning)
# ============================================================
def tune_uncertainty_threshold(current_efficiency: float) -> float:
    """
    Ajusta o limiar de incerteza (safety mode) baseado na eficiência do sistema.
    - Baixa eficiência -> Sistema instável -> Reduz limiar (Mais seguro/caro).
    - Alta eficiência -> Sistema estável -> Aumenta limiar (Mais econômico/local).
    """
    current_thresh = float(settings.get("UNCERTAINTY_THRESHOLD", 0.45))
    
    # Heurística Simples de Controle
    if current_efficiency < 2.0:
        # Qualidade baixa para a latência paga: Apertar segurança
        new_thresh = max(0.20, current_thresh - 0.05)
        action = "TIGHTEN"
    elif current_efficiency > 4.0:
        # Sistema voando: Relaxar para economizar
        new_thresh = min(0.80, current_thresh + 0.05)
        action = "RELAX"
    else:
        new_thresh = current_thresh
        action = "KEEP"
        
    if action != "KEEP":
        logger.info(f"[UQ-Tuning] Eficiência={current_efficiency:.2f}. {action} Threshold: {current_thresh:.2f} -> {new_thresh:.2f}")
        settings.set("UNCERTAINTY_THRESHOLD", str(new_thresh), actor="nsga-updater")
        NSGA_UQ_THRESH.set(new_thresh)
        
    return new_thresh


# ============================================================
# 6. Persistência
# ============================================================
def persist_results(modality: str, weights: Dict[str, float]):
    # DB
    try:
        with engine.begin() as conn:
            for m, w in weights.items():
                conn.execute(
                    text("""
                        INSERT INTO nsga_weights (modality, model, weight) VALUES (:mod, :m, :w)
                        ON DUPLICATE KEY UPDATE weight = :w
                    """),
                    {"mod": modality, "m": m, "w": w}
                )
    except Exception as e:
        logger.warning(f"[NSGA] Falha DB: {e}")

    # Redis
    try:
        if redis_client:
            redis_client.set(REDIS_KEY_WEIGHTS[modality], json.dumps(weights))
    except Exception as e:
        logger.warning(f"[NSGA] Falha Redis: {e}")


# ============================================================
# 7. Execução (Uma Iteração)
# ============================================================
def run_optimization_cycle(modality: str):
    # 1. Carrega Modelos
    models = load_candidate_models(modality)
    if not models:
        logger.warning(f"[NSGA] Pulo: Sem modelos para {modality}")
        return

    # 2. Carrega Métricas
    metrics = aggregate_ema_by_model(modality, models)

    # 3. Roda NSGA-II
    weights, efficiency = run_nsga_optimization(modality, models, metrics)
    
    # 4. Persiste Pesos
    persist_results(modality, weights)
    
    # 5. Ajusta Incerteza (Apenas na rodada de texto, para evitar conflito)
    if modality == "text":
        tune_uncertainty_threshold(efficiency)

    # 6. Métricas Ops
    NSGA_RUNS.labels(modality=modality).inc()
    NSGA_LAST_TS.labels(modality=modality).set(time.time())
    
    logger.info(f"[NSGA] Ciclo {modality} OK. Eficiência: {efficiency:.2f}")
    return weights


# ============================================================
# API & Loop
# ============================================================
app = FastAPI(title="NSGA-II Worker")

@app.post("/run/{modality}")
def trigger_run(modality: str = Path(...)):
    if modality not in MODALITIES:
        return JSONResponse({"error": "Invalid modality"}, status_code=400)
    try:
        weights = run_optimization_cycle(modality)
        return {"status": "ok", "weights": weights}
    except Exception as e:
        logger.exception("Erro no endpoint")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(REGISTRY).decode("utf-8"))

@app.get("/health")
def health():
    return {"status": "ok"}

def background_loop():
    time.sleep(15) # Warmup wait
    while True:
        for m in MODALITIES:
            try:
                run_optimization_cycle(m)
            except Exception as e:
                logger.error(f"[Loop] Erro em {m}: {e}")
        time.sleep(UPDATE_INTERVAL_S)

if __name__ == "__main__":
    t = threading.Thread(target=background_loop, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=9999)