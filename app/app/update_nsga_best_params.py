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
import os
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import redis

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [update_nsga] %(message)s")
logger = logging.getLogger("update_nsga")

# ============================================================
# 🔌 Conexões
# ============================================================

DB_HOST = os.getenv("DB_HOST", "mariadb")
DB_USER = os.getenv("DB_USER", "router_user")
DB_PASS = os.getenv("DB_PASS", "router_pass")
DB_NAME = os.getenv("DB_NAME", "routerdb")
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"

engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

try:
    rds = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    rds.ping()
    logger.info("[Redis] Conectado com sucesso.")
except Exception as e:
    logger.warning(f"[Redis] Falha ao conectar: {e}")
    rds = None


# ============================================================
# 🧱 Tabelas (MODIFICADAS PARA SUPORTE MULTIMODAL)
# ============================================================

DDL_PARAMS = """
CREATE TABLE IF NOT EXISTS nsga_params (
    id INT PRIMARY KEY,
    modality VARCHAR(32) NOT NULL,
    N_pop INT NOT NULL,
    N_gen INT NOT NULL,
    cxpb FLOAT NOT NULL,
    mutpb FLOAT NOT NULL,
    eta_c FLOAT NOT NULL,
    eta_m FLOAT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
);
"""

DDL_WEIGHTS = """
CREATE TABLE IF NOT EXISTS nsga_weights (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    modality VARCHAR(32) NOT NULL,
    model VARCHAR(255) NOT NULL,
    weight FLOAT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_mod_model (modality, model)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

DDL_RESULTS = """
CREATE TABLE IF NOT EXISTS nsga_meta_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    trial_id INT NOT NULL,
    modality VARCHAR(32) NOT NULL,
    N_pop INT NOT NULL,
    N_gen INT NOT NULL,
    cxpb FLOAT NOT NULL,
    mutpb FLOAT NOT NULL,
    eta_c FLOAT NOT NULL,
    eta_m FLOAT NOT NULL,
    eff_mean FLOAT NOT NULL,
    eff_std  FLOAT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def init_tables():
    """Execute the init tables routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    try:
        with engine.begin() as conn:
            conn.execute(text(DDL_PARAMS))
            conn.execute(text(DDL_WEIGHTS))
            conn.execute(text(DDL_RESULTS))
        logger.info("[update_nsga] Tabelas multimodais verificadas/criadas.")
    except SQLAlchemyError as e:
        logger.error(f"[update_nsga] Falha ao criar tabelas: {e}")


init_tables()


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
                    LIMIT 1;
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
                    (id, modality, N_pop, N_gen, cxpb, mutpb, eta_c, eta_m)
                    VALUES (:id, :mod, :np, :ng, :cx, :mu, :ec, :em)
                    ON DUPLICATE KEY UPDATE
                        modality = :mod,
                        N_pop = :np,
                        N_gen = :ng,
                        cxpb = :cx,
                        mutpb = :mu,
                        eta_c = :ec,
                        eta_m = :em,
                        updated_at = CURRENT_TIMESTAMP;
                    """
                ),
                dict(
                    id=modal_id,
                    mod=modality,
                    np=row["N_pop"],
                    ng=row["N_gen"],
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

        # converter para score
        scores = {}
        for row in rows:
            model = row["model"]
            lat = float(row["ema_latency"])
            qual = float(row["ema_quality"])
            cost = float(row["ema_cost"])

            score = (qual / 10.0) / (lat + 1e-6) / (cost + 1e-6)
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

    # DB
    try:
        with engine.begin() as conn:
            for model, weight in weights.items():
                conn.execute(
                    text(
                        """
                        INSERT INTO nsga_weights (modality, model, weight)
                        VALUES (:mod, :model, :w)
                        ON DUPLICATE KEY UPDATE
                            weight = :w,
                            updated_at = CURRENT_TIMESTAMP;
                        """
                    ),
                    dict(mod=modality, model=model, w=float(weight)),
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
