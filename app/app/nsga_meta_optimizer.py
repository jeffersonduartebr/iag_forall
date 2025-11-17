# -*- coding: utf-8 -*-
"""
nsga_meta_optimizer.py
------------------------------------------------------------
Meta-otimização (Optuna/TPE) chamando execução REAL do NSGA-II
via HTTP (/run). Salva resultados em nsga_meta_results e, ao
final, executa update_nsga_best_params.py para publicar os
melhores hiperparâmetros no Redis e DB.

Requer:
- Serviço nsga_weights_updater acessível via NSGA_URL
- Tabela nsga_meta_results (DDL embutido)
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

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] metaopt: %(message)s")
logger = logging.getLogger("metaopt")

NSGA_URL = os.getenv("NSGA_URL", "http://nsga_weights_updater:9999")
DB_HOST = os.getenv("DB_HOST", "mariadb")
DB_USER = os.getenv("DB_USER", "router_user")
DB_PASS = os.getenv("DB_PASS", "router_pass")
DB_NAME = os.getenv("DB_NAME", "routerdb")
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

DDL_META = """
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

with engine.begin() as conn:
    conn.execute(text(DDL_META))

def save_result(trial_id:int, params:Dict[str,Any], eff_mean:float, eff_std:float) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO nsga_meta_results
                    (trial_id, N_pop, N_gen, cxpb, mutpb, eta_c, eta_m, eff_mean, eff_std)
                    VALUES (:trial, :np, :ng, :cx, :mu, :ec, :em, :mean, :std)
                """),
                dict(
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
    except SQLAlchemyError as e:
        logger.warning("[metaopt] Falha ao salvar resultado no banco: %s", e)

def evaluate_once(N_pop:int, N_gen:int, cxpb:float, mutpb:float, eta_c:float, eta_m:float) -> float:
    payload = dict(N_pop=N_pop, N_gen=N_gen, cxpb=cxpb, mutpb=mutpb, eta_c=eta_c, eta_m=eta_m)
    r = requests.post(f"{NSGA_URL}/run", json=payload, timeout=1200)
    r.raise_for_status()
    data = r.json()
    eff = float(data.get("efficiency", 0.0))
    return eff

def objective(trial: optuna.trial.Trial) -> float:
    N_pop = trial.suggest_int("N_pop", 8, 64, step=4)
    N_gen = trial.suggest_int("N_gen", 5, 40, step=5)
    # *** Importante: DEAP exige cxpb + mutpb <= 1.0 ***
    # Escolha mutpb máximo condicionado em (1 - cxpb)
    cxpb  = trial.suggest_float("cxpb", 0.60, 0.95)
    mutpb_max = max(0.05, 1.0 - cxpb)  # garante soma <= 1.0
    mutpb = trial.suggest_float("mutpb", 0.05, min(0.40, mutpb_max))
    eta_c = trial.suggest_float("eta_c", 5.0, 40.0)
    eta_m = trial.suggest_float("eta_m", 5.0, 40.0)

    reps = int(os.getenv("METAOPT_REPS", "5"))
    vals = []
    for _ in range(5):
        try:
            vals.append(evaluate_once(N_pop, N_gen, cxpb, mutpb, eta_c, eta_m))
        except Exception as e:
            logger.warning("[metaopt] Erro durante execução real do NSGA: %s", e)
            vals.append(0.0)

    mean = float(sum(vals) / max(1, len(vals)))
    var  = float(sum((x - mean) ** 2 for x in vals) / max(1, len(vals)))
    std  = var ** 0.5

    save_result(trial.number, dict(N_pop=N_pop, N_gen=N_gen, cxpb=cxpb, mutpb=mutpb, eta_c=eta_c, eta_m=eta_m), mean, std)
    return mean

if __name__ == "__main__":
    N_TRIALS = int(os.getenv("METAOPT_TRIALS", "100"))
    STUDY_NAME = os.getenv("METAOPT_STUDY", "nsga_metaopt_real")
    STORAGE = os.getenv("OPTUNA_STORAGE", "sqlite:///metaopt.db")

    study = optuna.create_study(
        storage=STORAGE,
        study_name=STUDY_NAME,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    logger.info("[metaopt] Iniciando meta-otimização real com %d trials (n_jobs=1)...", N_TRIALS)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    logger.info("[metaopt] Melhor trial: %s", study.best_trial.number)
    logger.info("[metaopt] Melhor valor (eff): %.6f", study.best_value)
    logger.info("[metaopt] Melhores params: %s", study.best_trial.params)
    logger.info("[metaopt] Meta-otimização real concluída.")

    # ▶️ Publica os melhores parâmetros no Redis/DB (chama script auxiliar)
    script_path = os.getenv("NSGA_UPDATE_SCRIPT", "/app/app/update_nsga_best_params.py")
    try:
        logger.info("[metaopt] Executando %s para publicar melhores parâmetros...", script_path)
        # Usa o mesmo interpretador do container
        subprocess.check_call(["python", script_path])
        logger.info("[metaopt] Publicação concluída com sucesso.")
    except Exception as e:
        logger.warning("[metaopt] Falha ao executar update_nsga_best_params.py: %s", e)
