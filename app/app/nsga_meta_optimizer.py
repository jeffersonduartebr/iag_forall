# -*- coding: utf-8 -*-
"""
nsga_meta_optimizer.py (MULTIMODAL)
------------------------------------------------------------
Executa meta-otimização NSGA-II por modalidade:

    - text
    - vision
    - multimodal

Cada modalidade executa seu próprio Optuna Study,
chamando o endpoint real do NSGA-II:

    POST /run/<modality>

Resultados são armazenados em:
    - nsga_meta_results (com coluna modality)
    - chamados posteriores atualizam nsga_params e nsga_weights

Ao final, executa update_nsga_best_params.py (multimodal).
"""

from __future__ import annotations

import os
import logging
import subprocess
from typing import Dict, Any

import optuna
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] metaopt: %(message)s"
)
logger = logging.getLogger("metaopt")

# ============================================================
# 🔌 Conexões
# ============================================================

NSGA_URL = os.getenv("NSGA_URL", "http://nsga_weights_updater:9999")
DB_HOST = os.getenv("DB_HOST", "mariadb")
DB_USER = os.getenv("DB_USER", "router_user")
DB_PASS = os.getenv("DB_PASS", "router_pass")
DB_NAME = os.getenv("DB_NAME", "routerdb")
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"

engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)


# ============================================================
# 🧱 Tabela multimodal nsga_meta_results
# ============================================================

DDL_META = """
CREATE TABLE IF NOT EXISTS nsga_meta_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modality VARCHAR(32) NOT NULL,
    trial_id INT NOT NULL,
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

with engine.begin() as conn:
    conn.execute(text(DDL_META))


# ============================================================
# 🧩 Funções auxiliares
# ============================================================

def save_result(modality: str, trial_id: int, params: Dict[str, Any],
                eff_mean: float, eff_std: float) -> None:
    """Escreve uma linha multimodal em nsga_meta_results."""
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO nsga_meta_results (
                        modality, trial_id,
                        N_pop, N_gen, cxpb, mutpb, eta_c, eta_m,
                        eff_mean, eff_std
                    )
                    VALUES (
                        :mod, :trial,
                        :np, :ng, :cx, :mu, :ec, :em,
                        :mean, :std
                    )
                """),
                dict(
                    mod=modality,
                    trial=trial_id,
                    np=int(params["N_pop"]),
                    ng=int(params["N_gen"]),
                    cx=float(params["cxpb"]),
                    mu=float(params["mutpb"]),
                    ec=float(params["eta_c"]),
                    em=float(params["eta_m"]),
                    mean=float(eff_mean),
                    std=float(eff_std)
                )
            )
        logger.info(f"[metaopt] Salvo trial={trial_id} modality={modality}")
    except SQLAlchemyError as e:
        logger.warning("[metaopt] Falha ao salvar no banco: %s", e)


def evaluate_once(modality: str, N_pop: int, N_gen: int,
                  cxpb: float, mutpb: float,
                  eta_c: float, eta_m: float) -> float:
    """Executa NSGA-II real para a modalidade solicitada."""
    payload = dict(
        modality=modality,
        N_pop=N_pop,
        N_gen=N_gen,
        cxpb=cxpb,
        mutpb=mutpb,
        eta_c=eta_c,
        eta_m=eta_m,
    )

    r = requests.post(
        f"{NSGA_URL}/run/{modality}",
        json=payload,
        timeout=1800,
    )
    r.raise_for_status()
    data = r.json()
    eff = float(data.get("efficiency", 0.0))
    return eff


def build_objective(modality: str):
    """Cria uma função objetivo isolada para cada modalidade."""

    def objective(trial: optuna.trial.Trial) -> float:
        N_pop = trial.suggest_int("N_pop", 8, 64, step=4)
        N_gen = trial.suggest_int("N_gen", 5, 40, step=5)
        cxpb = trial.suggest_float("cxpb", 0.60, 0.95)
        mutpb_max = max(0.05, 1.0 - cxpb)
        mutpb = trial.suggest_float("mutpb", 0.05, min(0.40, mutpb_max))
        eta_c = trial.suggest_float("eta_c", 5.0, 40.0)
        eta_m = trial.suggest_float("eta_m", 5.0, 40.0)

        reps = int(os.getenv("METAOPT_REPS", "5"))
        vals = []
        for _ in range(reps):
            try:
                score = evaluate_once(
                    modality, N_pop, N_gen,
                    cxpb, mutpb, eta_c, eta_m
                )
                vals.append(score)
            except Exception as e:
                logger.warning(
                    f"[metaopt] Erro NSGA modality={modality}: {e}"
                )
                vals.append(0.0)

        mean = float(sum(vals) / max(1, len(vals)))
        var = float(sum((x - mean) ** 2 for x in vals) / max(1, len(vals)))
        std = var ** 0.5

        save_result(
            modality,
            trial.number,
            dict(N_pop=N_pop, N_gen=N_gen,
                 cxpb=cxpb, mutpb=mutpb,
                 eta_c=eta_c, eta_m=eta_m),
            mean, std,
        )
        return mean

    return objective


# ============================================================
# 🚀 Execução principal
# ============================================================

if __name__ == "__main__":

    modal_list = ["text", "vision", "multimodal"]

    N_TRIALS = int(os.getenv("METAOPT_TRIALS", "100"))
    STORAGE = os.getenv("OPTUNA_STORAGE", "sqlite:///metaopt.db")

    logger.info("[metaopt] Iniciando meta-otimização MULTIMODAL...")

    for modality in modal_list:

        STUDY_NAME = f"nsga_metaopt_{modality}"

        logger.info(f"[metaopt] --- OPTIMIZING MODALITY: {modality} ---")

        study = optuna.create_study(
            storage=STORAGE,
            study_name=STUDY_NAME,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(
            build_objective(modality),
            n_trials=N_TRIALS,
            n_jobs=1
        )

        logger.info(
            f"[metaopt] [{modality}] BEST trial={study.best_trial.number}, "
            f"eff={study.best_value:.6f}"
        )
        logger.info(f"[metaopt] [{modality}] BEST params={study.best_trial.params}")

    # Final: rodar script de publicação multimodal
    script_path = os.getenv(
        "NSGA_UPDATE_SCRIPT",
        "/app/app/update_nsga_best_params.py"
    )

    try:
        logger.info(f"[metaopt] Executando script final: {script_path}")
        subprocess.check_call(["python", script_path])
        logger.info("[metaopt] Publicação dos parâmetros MULTIMODAIS concluída.")
    except Exception as e:
        logger.warning(f"[metaopt] Falha ao executar update_nsga_best_params.py: {e}")
