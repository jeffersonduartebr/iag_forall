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
import threading
import time
from typing import Any, Dict, List, Tuple

import redis
import uvicorn
from fastapi import FastAPI, Path
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import REGISTRY, generate_latest
from sqlalchemy import text

from app.db import get_engine
from app.nsga_calibration import calibration_status_payload, run_calibration_cycle  # noqa: F401
from app.nsga_core import compute_convergence_metrics, run_nsga_optimization  # noqa: F401  (reexportados)
from app.services.frozen_policy import is_frozen_policy_active
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
    tune_global_strategy_weights,
    tune_uncertainty_threshold,
)
from app.services.reward import DEFAULT_MIN_SHARE, derive_reward_weights, publish_reward_weights
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
_SETTINGS_CANDIDATES = {
    "text": "CANDIDATE_MODELS_LIST",
    "vision": "CANDIDATE_VISION_MODELS_LIST",
    "multimodal": "CANDIDATE_MULTIMODAL_MODELS_LIST",
}
_FALLBACK_CANDIDATES = {
    "text": ("ollama/phi4:latest", "ollama/mistral:7b"),
    "vision": ("ollama/llava:7b", "ollama/moondream:latest"),
    "multimodal": ("ollama/llava:7b", "ollama/moondream:latest"),
}


def _candidates_from_redis(modality: str) -> List[str]:
    try:
        raw = redis_client.get(REDIS_KEY_CANDIDATES.get(modality, "")) if redis_client else None
        data = json.loads(raw) if raw else None
    except Exception:
        return []
    return [str(x) for x in data] if isinstance(data, list) else []


def load_candidate_models(modality: str) -> List[str]:
    """Candidate models for one modality: Redis override, then settings, then a local fallback.

    Duplicates are dropped keeping the configured order, so the NSGA-II
    genome maps to the same models in every process.
    """
    cached = _candidates_from_redis(modality)
    if cached:
        return cached
    setting = _SETTINGS_CANDIDATES.get(modality)
    candidates = getattr(settings, setting) if setting else []
    if candidates:
        return list(dict.fromkeys(c for c in candidates if c))
    logger.warning(f"[NSGA] ⚠️ Nenhum modelo encontrado para '{modality}'. Usando fallback.")
    return list(_FALLBACK_CANDIDATES.get(modality, ()))


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


def publish_reward_shares(modality: str, sys_metrics: Tuple[float, float, float]) -> None:
    """Publish the reward weights derived from this cycle (see ``services.reward``).

    ``NSGA_W_*`` live on raw scales, so they are converted into the share each
    objective contributes to the routing score at the portfolio chosen by the
    NSGA-II. Skipped under frozen policy so eval runs keep a fixed reward.
    """
    if is_frozen_policy_active():
        logger.info(f"[NSGA] Política congelada: pesos da recompensa ({modality}) mantidos.")
        return
    sys_lat, sys_cst, sys_qlt = sys_metrics
    try:
        min_share = float(settings.get("REWARD_WEIGHT_MIN_SHARE", DEFAULT_MIN_SHARE))
    except (TypeError, ValueError):
        min_share = DEFAULT_MIN_SHARE
    weights = derive_reward_weights(
        settings.NSGA_W_QUALITY,
        settings.NSGA_W_LATENCY,
        settings.NSGA_W_COST,
        sys_qlt,
        sys_lat,
        sys_cst,
        min_share=min_share,
    )
    snapshot = {"quality": sys_qlt, "latency_s": sys_lat, "cost_usd": sys_cst}
    if publish_reward_weights(redis_client, modality, weights, snapshot):
        logger.info(
            f"[NSGA] Pesos da recompensa ({modality}): q={weights[0]:.3f} l={weights[1]:.3f} c={weights[2]:.3f}"
        )


# ============================================================
# 7.1 Judge Feedback Integration (Phase 3.3)
# ============================================================
def tune_weights_from_judge_feedback() -> None:
    """
    Adjust NSGA weights based on recent judge verdicts.

    If error rate is high (>30%), boost quality weight to prioritize
    better-performing models. Skipped under frozen policy (NSGA_W_* are part
    of the frozen snapshot).
    """
    if is_frozen_policy_active():
        return
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

    # Recompensa do bandit: publicada após os ajustes globais, para refletir os NSGA_W_* vigentes.
    publish_reward_shares(modality, sys_metrics)

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
    return calibration_status_payload()


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
