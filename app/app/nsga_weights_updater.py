# -*- coding: utf-8 -*-
"""
nsga_weights_updater.py — Otimizador Multimodal (NSGA-II + UQ Tuning + Strategy Tuning)
---------------------------------------------------------------------
Serviço de Otimização Contínua.
Agora ajusta também os pesos globais de decisão (Qualidade, Latência, Custo)
baseado no desempenho sistêmico.
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
REDIS_KEY_EFFICIENCY_HISTORY = {m: f"nsga:efficiency_history:{m}" for m in MODALITIES}

# ============================================================
# Conexões (DB & Redis)
# ============================================================

DB_URL = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

def get_redis_client():
    """Resumo do comportamento desta função.

    Returns:
        Valor retornado pela função.
    """
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
    """Resumo do comportamento desta função.

    Returns:
        Valor retornado pela função.
    """
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
# 1. Carregamento de Modelos
# ============================================================
def load_candidate_models(modality: str) -> List[str]:
    # 1. Redis
    """Resumo do comportamento desta função.

    Args:
        modality: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    try:
        if redis_client:
            raw = redis_client.get(REDIS_KEY_CANDIDATES.get(modality, ""))
            if raw:
                data = json.loads(raw)
                if isinstance(data, list) and data:
                    return [str(x) for x in data]
    except Exception: pass

    # 2. Settings
    if modality == "text":
        candidates = settings.CANDIDATE_MODELS_LIST
    elif modality == "vision":
        candidates = settings.CANDIDATE_VISION_MODELS_LIST
    elif modality == "multimodal":
        candidates = settings.CANDIDATE_MULTIMODAL_MODELS_LIST
    else:
        candidates = []

    if candidates:
        return list(set([c for c in candidates if c]))

    # 3. Fallback de Segurança
    logger.warning(f"[NSGA] ⚠️ Nenhum modelo encontrado para '{modality}'. Usando fallback.")
    if modality == "text":
        return ["ollama/phi4:latest", "ollama/mistral:7b"]
    elif modality in ("vision", "multimodal"):
        return ["ollama/llava:7b", "ollama/moondream:latest"]
    return []


# ============================================================
# 2. Coleta de Dados Históricos (EMA)
# ============================================================
def aggregate_ema_by_model(modality: str, models: List[str]) -> Dict[str, Dict[str, float]]:
    """Resumo do comportamento desta função.

    Args:
        modality: Parâmetro de entrada.
        models: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
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
            # Cold start sintético
            final_data[m] = {
                "latency": 2.0, "cost": 0.001, "quality": 5.0, "alignment": 1.0
            }
            
    return final_data


# ============================================================
# 3. Métricas Prometheus
# ============================================================
NSGA_RUNS = Counter("nsga_runs_total", "Execuções do NSGA-II", ["modality"])
NSGA_LAST_TS = Gauge("nsga_last_run_ts", "Timestamp da última execução", ["modality"])
NSGA_UQ_THRESH = Gauge("nsga_uq_threshold", "Limiar de Incerteza otimizado", [])
NSGA_CONVERGENCE_SCORE = Gauge("nsga_worker_convergence_score", "Convergence score from worker", ["modality"])
NSGA_OPTIMIZATION_HEALTH = Gauge("nsga_worker_optimization_health", "Optimization health from worker", ["modality"])


# ============================================================
# 3.1 Convergence Monitoring
# ============================================================
def store_efficiency_history(modality: str, efficiency: float) -> None:
    """Store efficiency value in Redis history list."""
    if not redis_client:
        return
    try:
        key = REDIS_KEY_EFFICIENCY_HISTORY[modality]
        history_size = int(settings.get("NSGA_CONVERGENCE_HISTORY_SIZE", 20))

        # Push new value and trim to history size
        redis_client.lpush(key, str(efficiency))
        redis_client.ltrim(key, 0, history_size - 1)
    except Exception as e:
        logger.warning(f"[NSGA] Failed to store efficiency history: {e}")


def get_efficiency_history(modality: str) -> List[float]:
    """Retrieve efficiency history from Redis."""
    if not redis_client:
        return []
    try:
        key = REDIS_KEY_EFFICIENCY_HISTORY[modality]
        raw_values = redis_client.lrange(key, 0, -1)
        return [float(v) for v in raw_values]
    except Exception as e:
        logger.warning(f"[NSGA] Failed to get efficiency history: {e}")
        return []


def compute_convergence_metrics(history: List[float]) -> Dict[str, float]:
    """
    Compute convergence metrics from efficiency history.

    Returns:
        Dict with trend, variance, and health score
    """
    if len(history) < 3:
        return {"trend": 0.0, "variance": 0.0, "health": 1.0}

    import numpy as np

    # Compute variance (lower is better for convergence)
    variance = float(np.var(history))

    # Compute trend (positive = improving, negative = degrading)
    # Using simple linear regression slope
    x = np.arange(len(history))
    slope = float(np.polyfit(x, history[::-1], 1)[0])  # Reverse because newest is first

    # Compute health score
    # Health degrades if:
    # 1. High variance (unstable)
    # 2. Negative trend (degrading)
    # 3. Efficiency too low

    avg_efficiency = float(np.mean(history))

    # Health scoring logic
    if variance > 1.0 or slope < -0.5:
        health = -1.0  # Stuck or diverging
    elif variance > 0.5 or slope < -0.1 or avg_efficiency < 1.0:
        health = 0.0  # Degraded
    else:
        health = 1.0  # Healthy

    return {
        "trend": slope,
        "variance": variance,
        "health": health,
        "avg_efficiency": avg_efficiency,
    }


def check_optimization_health(modality: str, current_efficiency: float) -> Dict[str, Any]:
    """
    Check NSGA-II optimization health and log warnings if degraded.

    Returns:
        Dict with health metrics and any warnings
    """
    # Store current efficiency
    store_efficiency_history(modality, current_efficiency)

    # Get history and compute metrics
    history = get_efficiency_history(modality)
    metrics = compute_convergence_metrics(history)

    # Update Prometheus metrics
    NSGA_CONVERGENCE_SCORE.labels(modality=modality).set(current_efficiency)
    NSGA_OPTIMIZATION_HEALTH.labels(modality=modality).set(metrics["health"])

    # Log warnings based on health
    warnings = []
    if metrics["health"] == -1.0:
        warning_msg = f"[NSGA] ⚠️ OPTIMIZATION STUCK/DIVERGING for {modality}: trend={metrics['trend']:.3f}, var={metrics['variance']:.3f}"
        logger.warning(warning_msg)
        warnings.append(warning_msg)
    elif metrics["health"] == 0.0:
        warning_msg = f"[NSGA] ⚡ Optimization degraded for {modality}: trend={metrics['trend']:.3f}, var={metrics['variance']:.3f}"
        logger.warning(warning_msg)
        warnings.append(warning_msg)

    return {
        "modality": modality,
        "current_efficiency": current_efficiency,
        "metrics": metrics,
        "warnings": warnings,
        "history_size": len(history),
    }

# ============================================================
# 4. Núcleo NSGA-II (Algoritmo Genético)
# ============================================================
def run_nsga_optimization(
    modality: str,
    models: List[str],
    metrics: Dict[str, Dict[str, float]],
    n_pop=40, n_gen=20
) -> Tuple[Dict[str, float], float, Tuple[float, float, float]]:
    """
    Roda o NSGA-II.
    Retorna: (pesos_modelos, pontuação_eficiência, (lat_media, cost_medio, qual_media))
    """
    n = len(models)
    if n == 0: return {}, 0.0, (0,0,0)
    if n == 1: return {models[0]: 1.0}, 1.0, (metrics[models[0]]["latency"], metrics[models[0]]["cost"], metrics[models[0]]["quality"])

    # Limpa classes anteriores
    if "FitnessMulti" in creator.__dict__: del creator.FitnessMulti
    if "Individual" in creator.__dict__: del creator.Individual

    # Objetivos fixos para o AG: Min Latency, Min Cost, Max Quality, Max Alignment
    # Usamos pesos fixos AQUI para encontrar o Pareto Front ideal matemático.
    # O ajuste dinâmico será feito nos pesos do ROUTER, baseado no resultado daqui.
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -50.0, 2.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        """Resumo do comportamento desta função.

        Args:
            individual: Parâmetro de entrada.

        Returns:
            Valor retornado pela função.
        """
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

    pop = toolbox.population(n=n_pop)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=n_pop, lambda_=n_pop, 
                              cxpb=0.9, mutpb=0.1, ngen=n_gen, verbose=False)

    best_ind = tools.selBest(pop, 1)[0]
    s = sum(best_ind) or 1.0
    norm_weights = [x/s for x in best_ind]
    
    weights_map = {models[i]: norm_weights[i] for i in range(n)}
    
    # Métricas do sistema ideal encontrado
    sys_lat, sys_cst, sys_qlt, _ = evaluate(best_ind)
    efficiency_score = (sys_qlt / max(0.01, sys_lat))
    
    return weights_map, efficiency_score, (sys_lat, sys_cst, sys_qlt)


# ============================================================
# 5. Ajuste Dinâmico de Incerteza (UQ Tuning)
# ============================================================
def tune_uncertainty_threshold(current_efficiency: float) -> float:
    """Resumo do comportamento desta função.

    Args:
        current_efficiency: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    current_thresh = float(settings.get("UNCERTAINTY_THRESHOLD", 0.45))
    
    if current_efficiency < 2.0:
        new_thresh = max(0.20, current_thresh - 0.05)
        action = "TIGHTEN"
    elif current_efficiency > 4.0:
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
# 6. Ajuste Dinâmico de Pesos Globais (Strategy Tuning) - NOVO
# ============================================================
def tune_global_strategy_weights(sys_metrics: Tuple[float, float, float]):
    """
    Ajusta os pesos globais (NSGA_W_QUALITY, etc.) baseado no desempenho
    do melhor indivíduo encontrado pelo AG.
    
    Se o sistema ideal encontrado ainda é lento, aumentamos a penalidade de latência.
    Se a qualidade está baixa, aumentamos o peso da qualidade.
    """
    sys_lat, sys_cst, sys_qlt = sys_metrics
    
    # Lê valores atuais
    w_qual = settings.NSGA_W_QUALITY
    w_lat = settings.NSGA_W_LATENCY
    w_cost = settings.NSGA_W_COST
    
    changes = []

    # --- Lógica de Controle (P-Controller simples) ---
    
    # 1. Controle de Latência (Target: < 3.0s)
    if sys_lat > 3.0:
        # Sistema lento -> Aumenta importância da latência
        new_w_lat = min(2.0, w_lat + 0.1)
        if new_w_lat != w_lat:
            settings.set("NSGA_W_LATENCY", str(round(new_w_lat, 2)), actor="nsga-updater")
            changes.append(f"Lat {w_lat}->{new_w_lat:.2f}")
    elif sys_lat < 1.0:
        # Sistema muito rápido -> Relaxa latência para ganhar qualidade
        new_w_lat = max(0.1, w_lat - 0.05)
        if new_w_lat != w_lat:
            settings.set("NSGA_W_LATENCY", str(round(new_w_lat, 2)), actor="nsga-updater")
            changes.append(f"Lat {w_lat}->{new_w_lat:.2f}")

    # 2. Controle de Qualidade (Target: > 7.0)
    if sys_qlt < 7.0:
        # Qualidade baixa -> Aumenta importância da qualidade
        new_w_qual = min(5.0, w_qual + 0.2)
        if new_w_qual != w_qual:
            settings.set("NSGA_W_QUALITY", str(round(new_w_qual, 2)), actor="nsga-updater")
            changes.append(f"Qual {w_qual}->{new_w_qual:.2f}")

    # 3. Controle de Custo (Target: < $0.01/req)
    if sys_cst > 0.01:
        new_w_cost = min(100.0, w_cost + 5.0)
        if new_w_cost != w_cost:
            settings.set("NSGA_W_COST", str(round(new_w_cost, 2)), actor="nsga-updater")
            changes.append(f"Cost {w_cost}->{new_w_cost:.2f}")

    if changes:
        logger.info(f"[Strategy-Tuning] Ajustes aplicados: {', '.join(changes)}")


# ============================================================
# 6.1 Adaptive Risk Factors Tuning (Phase 5 - Improvement 1)
# ============================================================
def tune_risk_factors() -> Dict[str, Any]:
    """
    Tune risk factors based on observed quality outcomes by model type and UQ level.

    Analyzes query_log data to see if risk factor adjustments improve quality:
    - If SOTA models in high-UQ scenarios underperform, reduce their boost
    - If local models in high-UQ scenarios outperform expectations, increase their factor

    Uses P-controller feedback to make gradual adjustments.

    Returns:
        Dict with adjustments made and metrics
    """
    if not settings.RISK_FACTOR_ADAPT_ENABLED:
        return {"status": "disabled"}

    result = {"adjustments": [], "metrics": {}}

    try:
        with engine.connect() as conn:
            # Query recent data with UQ scores from raw_payload
            rows = conn.execute(
                text("""
                    SELECT
                        chosen_model,
                        quality,
                        JSON_EXTRACT(raw_payload, '$.uncertainty_score') as uq_score
                    FROM query_log
                    WHERE created_at > NOW() - INTERVAL 24 HOUR
                    AND quality IS NOT NULL
                    AND raw_payload IS NOT NULL
                    AND JSON_EXTRACT(raw_payload, '$.uncertainty_score') IS NOT NULL
                    LIMIT 5000
                """)
            ).fetchall()

        if len(rows) < 100:
            return {"status": "insufficient_data", "count": len(rows)}

        # Categorize results
        sota_high_uq = []
        local_high_uq = []
        local_low_uq = []
        uq_threshold = float(settings.get("UNCERTAINTY_THRESHOLD", 0.45))

        for row in rows:
            model = row[0]
            quality = float(row[1]) if row[1] else 5.0
            uq_score = float(row[2]) if row[2] else 0.5

            is_sota = any(m in model.lower() for m in ["gpt-5", "opus", "sonnet", "gemini-3-pro"])
            is_local = "ollama" in model.lower()
            is_high_uq = uq_score > uq_threshold

            if is_high_uq:
                if is_sota:
                    sota_high_uq.append(quality)
                elif is_local:
                    local_high_uq.append(quality)
            else:
                if is_local:
                    local_low_uq.append(quality)

        # Calculate average qualities
        def safe_mean(lst):
            """Resumo do comportamento desta função.

            Args:
                lst: Parâmetro de entrada.

            Returns:
                Valor retornado pela função.
            """
            return sum(lst) / len(lst) if lst else 5.0

        avg_sota_high = safe_mean(sota_high_uq)
        avg_local_high = safe_mean(local_high_uq)
        avg_local_low = safe_mean(local_low_uq)

        result["metrics"] = {
            "sota_high_uq_count": len(sota_high_uq),
            "sota_high_uq_avg_quality": avg_sota_high,
            "local_high_uq_count": len(local_high_uq),
            "local_high_uq_avg_quality": avg_local_high,
            "local_low_uq_count": len(local_low_uq),
            "local_low_uq_avg_quality": avg_local_low,
        }

        adapt_rate = settings.RISK_FACTOR_ADAPT_RATE

        # Tune SOTA high-UQ factor
        current_sota = settings.RISK_FACTOR_SOTA_HIGH_UQ
        if len(sota_high_uq) >= 20:
            # If SOTA is underperforming in high-UQ (< 7.0), reduce boost
            if avg_sota_high < 7.0:
                new_sota = max(1.0, current_sota - adapt_rate)
                if new_sota != current_sota:
                    settings.set("RISK_FACTOR_SOTA_HIGH_UQ", str(round(new_sota, 2)), actor="risk-tuner")
                    result["adjustments"].append(f"SOTA_HIGH: {current_sota:.2f} -> {new_sota:.2f}")
            # If SOTA is performing well, slight boost
            elif avg_sota_high > 8.0:
                new_sota = min(2.0, current_sota + adapt_rate)
                if new_sota != current_sota:
                    settings.set("RISK_FACTOR_SOTA_HIGH_UQ", str(round(new_sota, 2)), actor="risk-tuner")
                    result["adjustments"].append(f"SOTA_HIGH: {current_sota:.2f} -> {new_sota:.2f}")

        # Tune local high-UQ factor
        current_local_high = settings.RISK_FACTOR_LOCAL_HIGH_UQ
        if len(local_high_uq) >= 20:
            # If local models are underperforming in high-UQ, reduce further
            if avg_local_high < 5.0:
                new_local = max(0.3, current_local_high - adapt_rate)
                if new_local != current_local_high:
                    settings.set("RISK_FACTOR_LOCAL_HIGH_UQ", str(round(new_local, 2)), actor="risk-tuner")
                    result["adjustments"].append(f"LOCAL_HIGH: {current_local_high:.2f} -> {new_local:.2f}")
            # If local models are surprisingly good, increase
            elif avg_local_high > 7.0:
                new_local = min(1.0, current_local_high + adapt_rate)
                if new_local != current_local_high:
                    settings.set("RISK_FACTOR_LOCAL_HIGH_UQ", str(round(new_local, 2)), actor="risk-tuner")
                    result["adjustments"].append(f"LOCAL_HIGH: {current_local_high:.2f} -> {new_local:.2f}")

        # Tune local low-UQ factor
        current_local_low = settings.RISK_FACTOR_LOCAL_LOW_UQ
        if len(local_low_uq) >= 20:
            # If local models do well in known territory, boost
            if avg_local_low > 7.5:
                new_local = min(1.5, current_local_low + adapt_rate)
                if new_local != current_local_low:
                    settings.set("RISK_FACTOR_LOCAL_LOW_UQ", str(round(new_local, 2)), actor="risk-tuner")
                    result["adjustments"].append(f"LOCAL_LOW: {current_local_low:.2f} -> {new_local:.2f}")

        # Update Prometheus metrics
        try:
            from app.observability import RISK_FACTOR_CURRENT
            RISK_FACTOR_CURRENT.labels(factor_type="sota_high_uq").set(settings.RISK_FACTOR_SOTA_HIGH_UQ)
            RISK_FACTOR_CURRENT.labels(factor_type="local_high_uq").set(settings.RISK_FACTOR_LOCAL_HIGH_UQ)
            RISK_FACTOR_CURRENT.labels(factor_type="local_low_uq").set(settings.RISK_FACTOR_LOCAL_LOW_UQ)
        except Exception:
            pass

        if result["adjustments"]:
            logger.info(f"[Risk-Tuning] Adjustments: {', '.join(result['adjustments'])}")

        result["status"] = "ok"
        return result

    except Exception as e:
        logger.warning(f"[Risk-Tuning] Failed: {e}")
        return {"status": "error", "error": str(e)}


# ============================================================
# 6.2 UQ Calibration Against Actual Errors (Phase 5 - Improvement 4)
# ============================================================
def calibrate_uncertainty_threshold() -> Dict[str, Any]:
    """
    Calibrate uncertainty threshold based on actual quality outcomes.

    Analyzes if high-UQ queries actually have lower quality than low-UQ queries.
    If the gap is small, the threshold can be relaxed.
    If the gap is large, the threshold should be tightened.

    Returns:
        Dict with calibration results and metrics
    """
    if not settings.UQ_CALIBRATION_ENABLED:
        return {"status": "disabled"}

    result = {"old_threshold": None, "new_threshold": None, "metrics": {}}

    try:
        with engine.connect() as conn:
            # Query data grouped by UQ level
            rows = conn.execute(
                text("""
                    SELECT
                        quality,
                        JSON_EXTRACT(raw_payload, '$.uncertainty_score') as uq_score
                    FROM query_log
                    WHERE created_at > NOW() - INTERVAL 24 HOUR
                    AND quality IS NOT NULL
                    AND raw_payload IS NOT NULL
                    AND JSON_EXTRACT(raw_payload, '$.uncertainty_score') IS NOT NULL
                    LIMIT 5000
                """)
            ).fetchall()

        if len(rows) < 100:
            return {"status": "insufficient_data", "count": len(rows)}

        current_threshold = float(settings.get("UNCERTAINTY_THRESHOLD", 0.45))
        result["old_threshold"] = current_threshold

        high_uq_qualities = []
        low_uq_qualities = []

        for row in rows:
            quality = float(row[0]) if row[0] else 5.0
            uq_score = float(row[1]) if row[1] else 0.5

            if uq_score > current_threshold:
                high_uq_qualities.append(quality)
            else:
                low_uq_qualities.append(quality)

        if len(high_uq_qualities) < 20 or len(low_uq_qualities) < 20:
            return {"status": "insufficient_split", "high_count": len(high_uq_qualities), "low_count": len(low_uq_qualities)}

        avg_quality_high = sum(high_uq_qualities) / len(high_uq_qualities)
        avg_quality_low = sum(low_uq_qualities) / len(low_uq_qualities)
        quality_gap = avg_quality_low - avg_quality_high

        result["metrics"] = {
            "high_uq_count": len(high_uq_qualities),
            "low_uq_count": len(low_uq_qualities),
            "avg_quality_high_uq": avg_quality_high,
            "avg_quality_low_uq": avg_quality_low,
            "quality_gap": quality_gap,
        }

        # Update Prometheus metrics
        try:
            from app.observability import UQ_HIGH_AVG_QUALITY, UQ_LOW_AVG_QUALITY, UQ_VS_ERROR_CORRELATION
            UQ_HIGH_AVG_QUALITY.set(avg_quality_high)
            UQ_LOW_AVG_QUALITY.set(avg_quality_low)
            # Correlation approximation: quality_gap normalized
            correlation_approx = min(1.0, max(-1.0, quality_gap / 5.0))
            UQ_VS_ERROR_CORRELATION.set(correlation_approx)
        except Exception:
            pass

        # Threshold adjustment logic
        gap_relax = settings.UQ_QUALITY_GAP_RELAX
        gap_tighten = settings.UQ_QUALITY_GAP_TIGHTEN

        if quality_gap < gap_relax:
            # High-UQ queries aren't much worse -> relax threshold
            new_threshold = min(0.80, current_threshold + 0.05)
            action = "RELAX"
        elif quality_gap > gap_tighten:
            # High-UQ queries are much worse -> tighten threshold
            new_threshold = max(0.20, current_threshold - 0.05)
            action = "TIGHTEN"
        else:
            new_threshold = current_threshold
            action = "KEEP"

        result["new_threshold"] = new_threshold
        result["action"] = action

        if action != "KEEP":
            settings.set("UNCERTAINTY_THRESHOLD", str(round(new_threshold, 2)), actor="uq-calibrator")
            NSGA_UQ_THRESH.set(new_threshold)
            logger.info(
                f"[UQ-Calibration] {action}: threshold {current_threshold:.2f} -> {new_threshold:.2f} "
                f"(gap={quality_gap:.2f})"
            )

        result["status"] = "ok"
        return result

    except Exception as e:
        logger.warning(f"[UQ-Calibration] Failed: {e}")
        return {"status": "error", "error": str(e)}


# ============================================================
# 7. Persistência
# ============================================================
def persist_results(modality: str, weights: Dict[str, float]):
    """Resumo do comportamento desta função.

    Args:
        modality: Parâmetro de entrada.
        weights: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
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

    try:
        if redis_client:
            redis_client.set(REDIS_KEY_WEIGHTS[modality], json.dumps(weights))
    except Exception as e:
        logger.warning(f"[NSGA] Falha Redis: {e}")


# ============================================================
# 7.1 Judge Feedback Integration (Phase 3.3)
# ============================================================
def tune_weights_from_judge_feedback() -> None:
    """
    Adjust NSGA weights based on recent judge verdicts.

    If error rate is high (>30%), boost quality weight to prioritize
    better-performing models.
    """
    try:
        with engine.connect() as conn:
            # Query recent judge logs for error rate (last 1 hour)
            result = conn.execute(
                text("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN quality < 5.0 THEN 1 ELSE 0 END) as errors
                    FROM query_log
                    WHERE created_at > NOW() - INTERVAL 1 HOUR
                    AND quality IS NOT NULL
                """)
            ).fetchone()

            if not result or result[0] == 0:
                return

            total = int(result[0])
            errors = int(result[1] or 0)
            error_rate = errors / total

            logger.info(f"[Judge-Feedback] Error rate: {error_rate:.2%} ({errors}/{total})")

            if error_rate > 0.30:
                # High error rate - boost quality weight
                current_w_qual = settings.NSGA_W_QUALITY
                new_w_qual = min(5.0, current_w_qual + 0.3)

                if new_w_qual != current_w_qual:
                    settings.set("NSGA_W_QUALITY", str(round(new_w_qual, 2)), actor="judge-feedback")
                    logger.warning(
                        f"[Judge-Feedback] ⚠️ High error rate ({error_rate:.1%}). "
                        f"Boosting NSGA_W_QUALITY: {current_w_qual:.2f} -> {new_w_qual:.2f}"
                    )

    except Exception as e:
        logger.warning(f"[Judge-Feedback] Failed to query judge logs: {e}")


# ============================================================
# 8. Execução (Uma Iteração)
# ============================================================
def run_optimization_cycle(modality: str):
    """Resumo do comportamento desta função.

    Args:
        modality: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    models = load_candidate_models(modality)
    if not models:
        logger.warning(f"[NSGA] Pulo: Sem modelos para {modality}")
        return

    metrics = aggregate_ema_by_model(modality, models)

    # Roda NSGA-II
    weights, efficiency, sys_metrics = run_nsga_optimization(modality, models, metrics)

    persist_results(modality, weights)

    # Check optimization health and log warnings
    health_status = check_optimization_health(modality, efficiency)

    # Ajustes Globais (Apenas na rodada de texto para evitar conflitos de escrita concorrente)
    if modality == "text":
        tune_uncertainty_threshold(efficiency)
        tune_global_strategy_weights(sys_metrics)

        # Also tune based on judge feedback
        tune_weights_from_judge_feedback()

    NSGA_RUNS.labels(modality=modality).inc()
    NSGA_LAST_TS.labels(modality=modality).set(time.time())

    logger.info(f"[NSGA] Ciclo {modality} OK. Eff: {efficiency:.2f}. Health: {health_status['metrics']['health']}")
    return weights


# ============================================================
# API & Loop
# ============================================================
app = FastAPI(title="NSGA-II Worker")

@app.post("/run/{modality}")
def trigger_run(modality: str = Path(...)):
    """Resumo do comportamento desta função.

    Args:
        modality: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
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
    """Resumo do comportamento desta função.

    Returns:
        Valor retornado pela função.
    """
    return PlainTextResponse(generate_latest(REGISTRY).decode("utf-8"))

@app.get("/health")
def health():
    """Resumo do comportamento desta função.

    Returns:
        Valor retornado pela função.
    """
    return {"status": "ok"}


@app.post("/calibration/run")
def trigger_calibration():
    """Manually trigger a calibration cycle."""
    try:
        run_calibration_cycle()
        return {"status": "ok", "message": "Calibration cycle completed"}
    except Exception as e:
        logger.exception("Erro no endpoint calibration")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/calibration/status")
def calibration_status():
    """Get current calibration status and metrics."""
    result = {
        "risk_factors": {
            "sota_high_uq": settings.RISK_FACTOR_SOTA_HIGH_UQ,
            "local_high_uq": settings.RISK_FACTOR_LOCAL_HIGH_UQ,
            "local_low_uq": settings.RISK_FACTOR_LOCAL_LOW_UQ,
            "adapt_enabled": settings.RISK_FACTOR_ADAPT_ENABLED,
        },
        "uncertainty": {
            "threshold": float(settings.get("UNCERTAINTY_THRESHOLD", 0.45)),
            "calibration_enabled": settings.UQ_CALIBRATION_ENABLED,
        },
        "cache": {
            "threshold": float(settings.get("CACHE_THRESHOLD", 0.92)),
            "adapt_enabled": settings.CACHE_THRESHOLD_ADAPT_ENABLED,
            "min": settings.CACHE_THRESHOLD_MIN,
            "max": settings.CACHE_THRESHOLD_MAX,
            "target_hit_rate": settings.CACHE_HIT_RATE_TARGET,
        },
        "predictor": {
            "validation_enabled": settings.PREDICTOR_VALIDATION_ENABLED,
        },
        "judge": {
            "calibration_enabled": settings.JUDGE_CALIBRATION_ENABLED,
            "cache_agreement_target": settings.JUDGE_CACHE_AGREEMENT_TARGET,
        },
    }

    # Add predictor metrics if available
    try:
        from app.online_predictor import get_all_predictor_metrics
        result["predictor"]["models"] = get_all_predictor_metrics()
    except Exception:
        pass

    # Add cache stats
    try:
        from app.semantic_cache import get_l1_cache_stats, get_cache_hit_rate
        result["cache"]["l1_stats"] = get_l1_cache_stats()
        result["cache"]["hit_rate"] = get_cache_hit_rate()
    except Exception:
        pass

    # Add judge calibration metrics
    try:
        from app.judges import get_judge_calibration_metrics
        result["judge"]["models"] = get_judge_calibration_metrics()
    except Exception:
        pass

    return result

def run_calibration_cycle():
    """
    Run all Phase 5 calibration functions.

    Called after NSGA-II optimization in the background loop.
    """
    logger.info("[Calibration] Starting calibration cycle...")

    # 1. Risk Factor Tuning (Improvement 1)
    try:
        result = tune_risk_factors()
        logger.info(f"[Calibration] Risk factors: {result.get('status', 'unknown')}")
    except Exception as e:
        logger.warning(f"[Calibration] Risk factor tuning failed: {e}")

    # 2. UQ Calibration (Improvement 4)
    try:
        result = calibrate_uncertainty_threshold()
        logger.info(f"[Calibration] UQ threshold: {result.get('status', 'unknown')}")
    except Exception as e:
        logger.warning(f"[Calibration] UQ calibration failed: {e}")

    # 3. Cache Threshold Tuning (Improvement 3)
    try:
        from app.semantic_cache import tune_cache_threshold
        import asyncio
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(tune_cache_threshold())
            if result:
                logger.info(f"[Calibration] Cache threshold adjusted to {result}")
        finally:
            loop.close()
    except Exception as e:
        logger.warning(f"[Calibration] Cache threshold tuning failed: {e}")

    # 4. Predictor Calibration (Improvement 2)
    try:
        from app.online_predictor import calibrate_all_predictors, get_all_predictor_metrics
        calibrate_all_predictors()
        metrics = get_all_predictor_metrics()
        for model, m in metrics.items():
            logger.debug(f"[Calibration] Predictor {model}: brier={m.get('brier_score', 0):.3f}")

        # Update Prometheus metrics
        try:
            from app.observability import PREDICTOR_BRIER_SCORE, PREDICTOR_ACCURACY, PREDICTOR_CALIBRATION_TEMP
            for model, m in metrics.items():
                PREDICTOR_BRIER_SCORE.labels(model=model).set(m.get("brier_score", 0.25))
                PREDICTOR_ACCURACY.labels(model=model).set(m.get("accuracy", 0.5))
                PREDICTOR_CALIBRATION_TEMP.labels(model=model).set(m.get("calibration_temp", 1.0))
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"[Calibration] Predictor calibration failed: {e}")

    # 5. Judge Calibration (Improvement 5)
    try:
        from app.judges import calibrate_judges
        result = calibrate_judges()
        logger.info(f"[Calibration] Judge calibration: {result.get('status', 'unknown')}")
    except Exception as e:
        logger.warning(f"[Calibration] Judge calibration failed: {e}")

    logger.info("[Calibration] Calibration cycle complete.")


def background_loop():
    """Resumo do comportamento desta função.

    Returns:
        Valor retornado pela função.
    """
    time.sleep(15)
    calibration_counter = 0

    while True:
        for m in MODALITIES:
            try:
                run_optimization_cycle(m)
            except Exception as e:
                logger.error(f"[Loop] Erro em {m}: {e}")

        # Run calibration every 3rd cycle (to reduce overhead)
        calibration_counter += 1
        if calibration_counter >= 3:
            try:
                run_calibration_cycle()
            except Exception as e:
                logger.error(f"[Loop] Calibration error: {e}")
            calibration_counter = 0

        time.sleep(UPDATE_INTERVAL_S)

if __name__ == "__main__":
    t = threading.Thread(target=background_loop, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=9999)
