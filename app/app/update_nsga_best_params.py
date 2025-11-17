# -*- coding: utf-8 -*-
"""
update_nsga_best_params.py
------------------------------------------------------------
Lê a tabela nsga_meta_results → identifica o melhor trial →
atualiza:

1) Tabela nsga_weights (DB)
2) Redis (chave "nsga:weights")
3) Tabela nsga_params (melhores hiperparâmetros para execução futura)

Esse script é chamado automaticamente ao final do
nsga_meta_optimizer.py, mas também pode ser executado manualmente:

    docker exec -it metaopt python /app/app/update_nsga_best_params.py
"""

from __future__ import annotations
import os
import json
import logging
import numpy as np
from typing import Dict, Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

import redis

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [update_nsga] %(message)s")
logger = logging.getLogger("update_nsga")

# ============================================================
# 🔧 Conexões
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
# 🧱 Tabelas necessárias
# ============================================================

DDL_WEIGHTS = """
CREATE TABLE IF NOT EXISTS nsga_weights (
    model VARCHAR(255) PRIMARY KEY,
    weight FLOAT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

DDL_PARAMS = """
CREATE TABLE IF NOT EXISTS nsga_params (
    id INT PRIMARY KEY,
    N_pop INT NOT NULL,
    N_gen INT NOT NULL,
    cxpb FLOAT NOT NULL,
    mutpb FLOAT NOT NULL,
    eta_c FLOAT NOT NULL,
    eta_m FLOAT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

DDL_RESULTS = """
CREATE TABLE IF NOT EXISTS nsga_meta_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    trial_id INT NOT NULL,
    N_pop INT NOT NULL,
    N_gen INT NOT NULL,
    cxpb FLOAT NOT NULL,
    mutpb FLOAT NOT NULL,
    eta_c FLOAT NOT NULL,
    eta_m FLOAT NOT NULL,
    eff_mean FLOAT NOT NULL,
    eff_std  FLOAT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

def init_tables():
    try:
        with engine.begin() as conn:
            conn.execute(text(DDL_WEIGHTS))
            conn.execute(text(DDL_PARAMS))
            conn.execute(text(DDL_RESULTS))
        logger.info("[update_nsga] Tabelas verificadas/criadas.")
    except SQLAlchemyError as e:
        logger.error(f"[update_nsga] Falha ao criar tabelas: {e}")

init_tables()


# ============================================================
# 📥 Buscar melhores resultados do NSGA-meta
# ============================================================

def load_best_trial() -> Dict[str, Any] | None:
    """
    Retorna o melhor trial baseado em eff_mean.
    """
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT *
                    FROM nsga_meta_results
                    ORDER BY eff_mean DESC, eff_std ASC
                    LIMIT 1;
                """)
            ).mappings().first()

        if not row:
            logger.warning("[update_nsga] Nenhum trial encontrado em nsga_meta_results.")
            return None

        logger.info(f"[update_nsga] Melhor trial encontrado: trial_id={row['trial_id']}, eff_mean={row['eff_mean']:.6f}")
        return dict(row)

    except SQLAlchemyError as e:
        logger.error(f"[update_nsga] Erro ao buscar melhor trial: {e}")
        return None


# ============================================================
# 📤 Atualizar hiperparâmetros (N_pop, N_gen, cxpb, mutpb, eta_c, eta_m)
# ============================================================

def update_best_params(row: Dict[str, Any]) -> None:
    """
    Salva o melhor trial na tabela nsga_params (linha única id=1).
    """
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO nsga_params
                    (id, N_pop, N_gen, cxpb, mutpb, eta_c, eta_m)
                    VALUES (1, :np, :ng, :cx, :mu, :ec, :em)
                    ON DUPLICATE KEY UPDATE
                        N_pop = :np,
                        N_gen = :ng,
                        cxpb = :cx,
                        mutpb = :mu,
                        eta_c = :ec,
                        eta_m = :em,
                        updated_at = CURRENT_TIMESTAMP;
                """),
                dict(
                    np=row["N_pop"],
                    ng=row["N_gen"],
                    cx=row["cxpb"],
                    mu=row["mutpb"],
                    ec=row["eta_c"],
                    em=row["eta_m"],
                )
            )
        logger.info("[update_nsga] Tabela nsga_params atualizada com sucesso.")

    except SQLAlchemyError as e:
        logger.error(f"[update_nsga] Falha ao salvar melhores parâmetros: {e}")


# ============================================================
# 🧮 Gerar pesos para cada modelo
# ============================================================

def compute_model_weights() -> Dict[str, float]:
    """
    Calcula pesos relativos para cada modelo com base nas métricas EMA.
    Apenas meta-opt serve para calibrar hiperparâmetros do NSGA;
    aqui calculamos pesos dos modelos pela tabela ema_history.
    """
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM ema_history")).mappings().all()

        if not rows:
            logger.warning("[update_nsga] Nenhum dado encontrado em ema_history.")
            return {}

        # Converte para score usando a mesma lógica da eficiência global
        scores = {}
        for row in rows:
            model = row["model"]
            lat = float(row["ema_latency"])
            qual = float(row["ema_quality"])
            cost = float(row["ema_cost"])

            # Score simples, consistente com eff usada no NSGA:
            score = (qual / 10.0) / (lat + 1e-6) / (cost + 1e-6)
            scores[model] = max(0.0, score)

        total = sum(scores.values()) or 1.0
        weights = {m: v / total for m, v in scores.items()}
        logger.info(f"[update_nsga] Pesos normalizados gerados: {weights}")

        return weights

    except SQLAlchemyError as e:
        logger.error(f"[update_nsga] Falha ao calcular pesos: {e}")
        return {}


# ============================================================
# 📤 Persistir pesos em DB + Redis
# ============================================================

def persist_weights(weights: Dict[str, float]) -> None:
    """
    Persiste os pesos dos modelos no banco + Redis.
    """
    if not weights:
        logger.warning("[update_nsga] Nenhum peso para persistir.")
        return

    # DB
    try:
        with engine.begin() as conn:
            for model, weight in weights.items():
                conn.execute(
                    text("""
                        INSERT INTO nsga_weights (model, weight)
                        VALUES (:model, :w)
                        ON DUPLICATE KEY UPDATE
                            weight = :w,
                            updated_at = CURRENT_TIMESTAMP;
                    """),
                    dict(model=model, w=float(weight))
                )
        logger.info("[update_nsga] Pesos atualizados em nsga_weights.")
    except SQLAlchemyError as e:
        logger.error(f"[update_nsga] Falha ao salvar nsga_weights no DB: {e}")

    # Redis
    try:
        if rds:
            rds.set("nsga:weights", json.dumps(weights))
            logger.info("[update_nsga] Pesos publicados no Redis (nsga:weights).")
    except Exception as e:
        logger.warning(f"[update_nsga] Falha ao publicar pesos no Redis: {e}")


# ============================================================
# 🚀 Execução principal
# ============================================================

if __name__ == "__main__":
    logger.info("[update_nsga] Iniciando atualização dos melhores parâmetros e pesos...")

    # 1) Carrega melhor trial
    row = load_best_trial()
    if not row:
        logger.warning("[update_nsga] Abortado: nenhum trial disponível.")
        exit(0)

    # 2) Atualiza hiperparâmetros
    update_best_params(row)

    # 3) Recalcula pesos dos modelos
    weights = compute_model_weights()

    # 4) Salva pesos (DB + Redis)
    persist_weights(weights)

    logger.info("[update_nsga] Atualização concluída com sucesso ✅")
