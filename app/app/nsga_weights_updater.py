# -*- coding: utf-8 -*-
# Objective: Application runtime code for nsga weights updater.
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
import random
import threading
import time
from typing import Any, Dict, List, Tuple

import redis
import uvicorn
from deap import algorithms, base, creator, tools
from fastapi import FastAPI, Path
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import REGISTRY, generate_latest
from sqlalchemy import text

from app.db import get_engine
from app.services.nsga_metrics import (
    JUDGE_FEEDBACK_ERROR_RATE,
    JUDGE_FEEDBACK_PROXY_TOTAL,
    JUDGE_FEEDBACK_SAMPLED_TOTAL,
    NSGA_CONVERGENCE_SCORE,
    NSGA_LAST_TS,
    NSGA_OPTIMIZATION_HEALTH,
    NSGA_RUNS,
)
from app.services.nsga_tuning import (
    calibrate_uncertainty_threshold,
    tune_global_strategy_weights,
    tune_risk_factors,
    tune_uncertainty_threshold,
)
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


def _db_engine():
    return get_engine()


def get_redis_client():
    """Return redis client.

    This helper centralizes retrieval logic so callers do not have to duplicate lookup behavior."""
    try:
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD or None,
            db=settings.REDIS_DB,
            socket_timeout=2,
        )
        if r.ping():
            return r
    except Exception as e:
        logger.warning(f"[NSGA] Redis indisponível: {e}")
    return None


redis_client = get_redis_client()


# ============================================================
# Inicialização de Tabelas
# ============================================================
def init_db_tables():
    """Execute the init db tables routine.

    This helper encapsulates one focused step used by the surrounding workflow."""
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
        with _db_engine().begin() as conn:
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
    """Load candidate models.

    The function reads the current representation from its backing store or runtime source."""
    try:
        if redis_client:
            raw = redis_client.get(REDIS_KEY_CANDIDATES.get(modality, ""))
            if raw:
                data = json.loads(raw)
                if isinstance(data, list) and data:
                    return [str(x) for x in data]
    except Exception:
        pass

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
    """Execute the aggregate ema by model routine.

    This helper encapsulates one focused step used by the surrounding workflow."""
    try:
        with _db_engine().connect() as conn:
            rows = (
                conn.execute(
                    text("""
                    SELECT model, ema_latency, ema_cost, ema_quality, ema_alignment
                    FROM ema_history
                    WHERE modality = :m
                """),
                    {"m": modality},
                )
                .mappings()
                .all()
            )

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
            final_data[m] = {"latency": 2.0, "cost": 0.001, "quality": 5.0, "alignment": 1.0}

    return final_data


# Métricas Prometheus em services/nsga_metrics (importadas no topo).


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
    modality: str, models: List[str], metrics: Dict[str, Dict[str, float]], n_pop=40, n_gen=20
) -> Tuple[Dict[str, float], float, Tuple[float, float, float]]:
    """
    Roda o NSGA-II.
    Retorna: (pesos_modelos, pontuação_eficiência, (lat_media, cost_medio, qual_media))
    """
    n = len(models)
    if n == 0:
        return {}, 0.0, (0, 0, 0)
    if n == 1:
        return (
            {models[0]: 1.0},
            1.0,
            (metrics[models[0]]["latency"], metrics[models[0]]["cost"], metrics[models[0]]["quality"]),
        )

    # Limpa classes anteriores
    if "FitnessMulti" in creator.__dict__:
        del creator.FitnessMulti
    if "Individual" in creator.__dict__:
        del creator.Individual

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
        """Execute the evaluate routine.

        This helper encapsulates one focused step used by the surrounding workflow."""
        s = sum(individual) or 1.0
        w = [x / s for x in individual]

        lat = sum(w[i] * metrics[models[i]]["latency"] for i in range(n))
        cst = sum(w[i] * metrics[models[i]]["cost"] for i in range(n))
        qlt = sum(w[i] * metrics[models[i]]["quality"] for i in range(n))
        aln = sum(w[i] * metrics[models[i]]["alignment"] for i in range(n))
        return lat, cst, qlt, aln

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0 / n)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=n_pop)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=n_pop, lambda_=n_pop, cxpb=0.9, mutpb=0.1, ngen=n_gen, verbose=False)

    best_ind = tools.selBest(pop, 1)[0]
    s = sum(best_ind) or 1.0
    norm_weights = [x / s for x in best_ind]

    weights_map = {models[i]: norm_weights[i] for i in range(n)}

    # Métricas do sistema ideal encontrado
    sys_lat, sys_cst, sys_qlt, _ = evaluate(best_ind)
    efficiency_score = sys_qlt / max(0.01, sys_lat)

    return weights_map, efficiency_score, (sys_lat, sys_cst, sys_qlt)


# Tuning dinâmico extraído para services/nsga_tuning (roadmap #19).


# ============================================================
# 7. Persistência
# ============================================================
def persist_results(modality: str, weights: Dict[str, float]):
    """Execute the persist results routine.

    This helper encapsulates one focused step used by the surrounding workflow."""
    try:
        with _db_engine().begin() as conn:
            for m, w in weights.items():
                conn.execute(
                    text("""
                        INSERT INTO nsga_weights (modality, model, weight) VALUES (:mod, :m, :w)
                        ON DUPLICATE KEY UPDATE weight = :w
                    """),
                    {"mod": modality, "m": m, "w": w},
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
        with _db_engine().connect() as conn:
            min_samples = int(settings.get("JUDGE_FEEDBACK_MIN_SAMPLES", 30))
            threshold = float(settings.get("JUDGE_FEEDBACK_ERROR_THRESHOLD", 5.0))

            judged_result = conn.execute(
                text("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN quality < :threshold THEN 1 ELSE 0 END) as errors
                    FROM query_log
                    WHERE created_at > NOW() - INTERVAL 1 HOUR
                    AND quality IS NOT NULL
                    AND quality_source = 'judge'
                """),
                {"threshold": threshold},
            ).fetchone()
            proxy_result = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM query_log
                    WHERE created_at > NOW() - INTERVAL 1 HOUR
                    AND quality IS NOT NULL
                    AND quality_source <> 'judge'
                    """
                )
            ).fetchone()

            proxy_total = int((proxy_result[0] if proxy_result else 0) or 0)
            if proxy_total:
                JUDGE_FEEDBACK_PROXY_TOTAL.inc(proxy_total)

            if not judged_result or judged_result[0] == 0:
                JUDGE_FEEDBACK_ERROR_RATE.set(0.0)
                return

            total = int(judged_result[0])
            errors = int(judged_result[1] or 0)
            JUDGE_FEEDBACK_SAMPLED_TOTAL.inc(total)

            if total < min_samples:
                JUDGE_FEEDBACK_ERROR_RATE.set(0.0)
                logger.info(
                    "[Judge-Feedback] Skipping tune: only %s judged samples available (min=%s).",
                    total,
                    min_samples,
                )
                return

            error_rate = errors / total
            JUDGE_FEEDBACK_ERROR_RATE.set(error_rate)

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
    """Run optimization cycle.

    This function coordinates the main execution path for that step."""
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
    """Execute the trigger run routine.

    This helper encapsulates one focused step used by the surrounding workflow."""
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
    """Execute the metrics routine.

    This helper encapsulates one focused step used by the surrounding workflow."""
    return PlainTextResponse(generate_latest(REGISTRY).decode("utf-8"))


@app.get("/health")
def health():
    """Execute the health routine.

    This helper encapsulates one focused step used by the surrounding workflow."""
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
        from app.semantic_cache import get_cache_hit_rate, get_l1_cache_stats

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
        import asyncio

        from app.semantic_cache import tune_cache_threshold

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
            from app.observability import PREDICTOR_ACCURACY, PREDICTOR_BRIER_SCORE, PREDICTOR_CALIBRATION_TEMP

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
    """Execute the background loop routine.

    This helper encapsulates one focused step used by the surrounding workflow."""
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
