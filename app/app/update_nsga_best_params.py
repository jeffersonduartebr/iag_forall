# -*- coding: utf-8 -*-
# Objective: Application runtime code for update nsga best params.
"""
update_nsga_best_params.py  (VERSÃO MULTIMODAL)
------------------------------------------------------------
Atualiza os melhores parâmetros e pesos NSGA-II para:

    - text
    - vision
    - multimodal

Salva em:
  1) nsga_params (modalidade)
  2) nsga_weights (modalidade + modelo)
  3) Redis (nsga:weights:<modality>)

Chamado pelo nsga_meta_optimizer ou manualmente via:

    docker exec -it metaopt python /app/app/update_nsga_best_params.py
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import redis
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.db_url import get_db_url

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [update_nsga] %(message)s")
logger = logging.getLogger("update_nsga")

# ============================================================
# 🔌 Conexões
# ============================================================

# URL, dialeto e porta vêm de app.db (a porta 3306 fixa impedia qualquer outro banco).
DB_URL = get_db_url()

engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600, pool_size=1, max_overflow=1)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

rds: Optional["redis.Redis[str]"]
try:
    rds = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    rds.ping()
    logger.info("[Redis] Conectado com sucesso.")
except Exception as e:
    logger.warning(f"[Redis] Falha ao conectar: {e}")
    rds = None


# ============================================================
# 📥 Carregar melhor trial por modalidade
# ============================================================

def load_best_trial(modality: str) -> Optional[Dict[str, Any]]:
    """Load best trial.

The function reads the current representation from its backing store or runtime source."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT *
                    FROM nsga_meta_results
                    WHERE modality = :m
                    ORDER BY eff_mean DESC, eff_std ASC
                    LIMIT 1
                    """
                ),
                {"m": modality},
            ).mappings().first()

        if not row:
            logger.warning(f"[update_nsga] Nenhum trial para modality={modality}.")
            return None

        logger.info(
            f"[update_nsga] Melhor trial para {modality}: trial_id={row['trial_id']} "
            f"eff_mean={row['eff_mean']:.6f}"
        )
        return dict(row)

    except SQLAlchemyError as e:
        logger.error(f"[update_nsga] Erro load_best_trial modality={modality}: {e}")
        return None


# ============================================================
# 📤 Atualizar melhores hiperparâmetros (por modalidade)
# ============================================================

def update_best_params(modality: str, row: Dict[str, Any]) -> None:
    """Update best params.

This function applies the module-specific mutation logic for the target resource."""
    try:
        modal_id = {"text": 1, "vision": 2, "multimodal": 3}[modality]

        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO nsga_params
                    (id, modality, n_pop, n_gen, cxpb, mutpb, eta_c, eta_m)
                    VALUES (:id, :mod, :np, :ng, :cx, :mu, :ec, :em)
                    ON CONFLICT (id) DO UPDATE SET
                        modality = EXCLUDED.modality,
                        n_pop = EXCLUDED.n_pop,
                        n_gen = EXCLUDED.n_gen,
                        cxpb = EXCLUDED.cxpb,
                        mutpb = EXCLUDED.mutpb,
                        eta_c = EXCLUDED.eta_c,
                        eta_m = EXCLUDED.eta_m,
                        updated_at = CURRENT_TIMESTAMP
                    """
                ),
                dict(
                    id=modal_id,
                    mod=modality,
                    np=row["n_pop"],
                    ng=row["n_gen"],
                    cx=row["cxpb"],
                    mu=row["mutpb"],
                    ec=row["eta_c"],
                    em=row["eta_m"],
                ),
            )

        logger.info(f"[update_nsga] nsga_params atualizado para {modality}.")

    except Exception as e:
        logger.error(f"[update_nsga] Falha ao atualizar params modality={modality}: {e}")


# ============================================================
# 🧮 Calcular pesos por modalidade
# ============================================================

def _positive_floor(values: List[float]) -> float:
    """Smallest strictly positive value; 1.0 when none is positive (the metric then doesn't discriminate)."""
    positive = [v for v in values if v > 0.0]
    return min(positive) if positive else 1.0


def compute_model_weights(modality: str) -> Dict[str, float]:
    """Compute model weights.

The function derives the value needed by the surrounding workflow from the available inputs."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM ema_history WHERE modality = :m"),
                {"m": modality},
            ).mappings().all()

        if not rows:
            logger.warning(f"[update_nsga] Nenhum EMA disponível para modality={modality}.")
            return {}

        # converter para score. Um modelo grátis (custo 0) era dividido por 1e-6 e
        # levava praticamente todo o peso: o piso é agora o menor valor positivo
        # entre os modelos pesados (o grátis conta como o pago mais barato).
        metrics = [
            (row["model"], float(row["ema_latency"]), float(row["ema_quality"]), float(row["ema_cost"])) for row in rows
        ]
        lat_floor = _positive_floor([m[1] for m in metrics])
        cost_floor = _positive_floor([m[3] for m in metrics])
        scores = {}
        for model, lat, qual, cost in metrics:
            score = (qual / 10.0) / max(lat, lat_floor) / max(cost, cost_floor)
            scores[model] = max(0.0, score)

        total = sum(scores.values()) or 1.0
        weights = {m: v / total for m, v in scores.items()}

        logger.info(f"[update_nsga] Pesos {modality}: {weights}")
        return weights

    except Exception as e:
        logger.error(f"[update_nsga] Erro ao calcular pesos para {modality}: {e}")
        return {}


# ============================================================
# 📤 Persistir pesos (DB + Redis)
# ============================================================

def persist_weights(modality: str, weights: Dict[str, float]) -> None:
    """Execute the persist weights routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    if not weights:
        logger.warning(f"[update_nsga] Nenhum peso para persistir modality={modality}.")
        return

    # DB (uma instrução com todos os modelos, executemany); o UPDATE lê EXCLUDED, a linha proposta.
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO nsga_weights (modality, model, weight)
                    VALUES (:mod, :model, :w)
                    ON CONFLICT (modality, model) DO UPDATE SET
                        weight = EXCLUDED.weight,
                        updated_at = CURRENT_TIMESTAMP
                    """
                ),
                [dict(mod=modality, model=model, w=float(weight)) for model, weight in weights.items()],
            )
        logger.info(f"[update_nsga] Pesos gravados em nsga_weights ({modality}).")
    except SQLAlchemyError as e:
        logger.error(f"[update_nsga] Falha ao gravar pesos DB modality={modality}: {e}")

    # Redis
    try:
        if rds:
            rds.set(f"nsga:weights:{modality}", json.dumps(weights))
            logger.info(f"[update_nsga] Pesos publicados no Redis nsga:weights:{modality}.")
    except Exception as e:
        logger.warning(f"[update_nsga] Redis erro modality={modality}: {e}")


# ============================================================
# 🚀 Execução principal
# ============================================================

if __name__ == "__main__":
    logger.info("[update_nsga] Iniciando atualização MULTIMODAL...")

    for modality in ["text", "vision", "multimodal"]:

        logger.info(f"[update_nsga] --- PROCESSANDO MODALIDADE: {modality} ---")

        best = load_best_trial(modality)
        if not best:
            continue

        update_best_params(modality, best)

        weights = compute_model_weights(modality)
        persist_weights(modality, weights)

    logger.info("[update_nsga] Atualização multimodal concluída com sucesso. ✅")
