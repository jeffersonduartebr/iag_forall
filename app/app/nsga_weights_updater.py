# nsga_weights_updater.py
# ------------------------------------------------------------
# Serviço NSGA-II para cálculo de pesos de seleção entre modelos.
# Lê modelos de settings_dynamic (Redis -> DB -> .env); usa logs 
# recentes de consultas; publica pesos em DB/Redis.
# ------------------------------------------------------------

import os
import re
import json
import math
import time
import random
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from .settings_dynamic import settings

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import redis

# try:
#     # CORRIGIDO: Importa o settings central
#     from app.settings_dynamic import settings
#     # use o util já existente no projeto, se disponível
#     from app.utils.redis_client import get_redis
# except Exception:
#     # Fallback de import (caso o path esteja diferente)
#     import sys
#     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#     from app.settings_dynamic import settings
    
    # fallback minimalista


def get_redis(max_wait_s: int = 0):
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    pwd = os.getenv("REDIS_PASSWORD") or os.getenv("REDIS_PASS") or "SenhaForte"
    try:
        r = redis.Redis(host=host, port=port, password=pwd, db=int(os.getenv("REDIS_DB", "0")))
        r.ping()
        return r
    except Exception:
        return None

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)

# DEAP - Algoritmo evolutivo (NSGA-II)
from deap import base, creator, tools, algorithms

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("nsga-updater")

# ------------------------------------------------------------
# Config / Conexões (Lidas do settings)
# ------------------------------------------------------------
DB_HOST = settings.DB_HOST
DB_USER = settings.DB_USER
DB_PASS = settings.DB_PASS
DB_NAME = settings.DB_NAME
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

# Chave de escrita (específica deste módulo)
REDIS_KEY_WEIGHTS = "nsga:weights" # JSON dict: {model: weight}

# NSGA ciclo (lido do settings, com fallback)
UPDATE_INTERVAL_S = int(settings.get("NSGA_UPDATE_INTERVAL_S", "300"))  # 5 minutos

# Quantidade de logs considerados (lido do settings, com fallback)
QUERY_LOG_LOOKBACK_MINUTES = int(settings.get("NSGA_LOOKBACK_MINUTES", "180"))  # 3h
QUERY_LOG_MAX_ROWS = int(settings.get("NSGA_LOOKBACK_MAXROWS", "2000"))         # até 2000 linhas

# ------------------------------------------------------------
# Prometheus (single-process registry)
# ------------------------------------------------------------
registry = CollectorRegistry()

NSGA_RUNS = Counter("nsga_runs_total", "Total de execuções NSGA-II realizadas", registry=registry)
NSGA_LAST_RUN_TS = Gauge("nsga_last_run_timestamp", "Epoch da última execução do NSGA", registry=registry)
NSGA_POP_FITNESS_AVG = Gauge("nsga_population_fitness_avg", "Fitness médio da população (proxy)", registry=registry)

NSGA_BEST_LATENCY = Gauge("nsga_best_latency_seconds", "Latência esperada (s) do mix ótimo", registry=registry)
NSGA_BEST_COST = Gauge("nsga_best_cost_per_1k", "Custo esperado ($/1k toks) do mix ótimo", registry=registry)
NSGA_BEST_QUALITY = Gauge("nsga_best_quality", "Qualidade esperada (0-10) do mix ótimo", registry=registry)
NSGA_BEST_ALIGNMENT = Gauge("nsga_best_query_alignment", "Alinhamento esperado (0-1) do mix ótimo", registry=registry)

WEIGHT_BY_MODEL = Gauge("nsga_model_weight", "Peso/Probabilidade do modelo no mix", ["model"], registry=registry)
MODEL_OBS_LAT = Gauge("nsga_model_obs_latency", "Latência média observada (s) do modelo", ["model"], registry=registry)
MODEL_OBS_COST = Gauge("nsga_model_obs_cost_per_1k", "Custo médio observado do modelo", ["model"], registry=registry)
MODEL_OBS_QUAL = Gauge("nsga_model_obs_quality", "Qualidade média observada (0-10) do modelo", ["model"], registry=registry)
MODEL_OBS_SR = Gauge("nsga_model_obs_success_rate", "Taxa de sucesso observada (0-1) do modelo", ["model"], registry=registry)
MODEL_ALIGNMENT = Gauge("nsga_model_alignment", "Score de alinhamento (0-1) do modelo ao perfil de query atual", ["model"], registry=registry)

# ------------------------------------------------------------
# Utilidades: leitura dinâmica (Redis -> DB -> .env)
# ------------------------------------------------------------

# REMOVIDO: _parse_models_list
# REMOVIDO: load_candidate_models
# (O settings.CANDIDATE_MODELS_LIST substitui ambos)

# ------------------------------------------------------------
# Esquemas / tabelas necessárias
# ------------------------------------------------------------
def ensure_tables():
    """Cria tabelas necessárias (nsga_weights, query_log) se não existirem."""
    ddl_weights = """
    CREATE TABLE IF NOT EXISTS nsga_weights (
        model VARCHAR(255) PRIMARY KEY,
        weight FLOAT NOT NULL,
        computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """
    # CORRIGIDO: DDL para 'query_log' (bate com db_manager.py)
    ddl_qlogs = """
    CREATE TABLE IF NOT EXISTS query_log (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        query_text TEXT NOT NULL,
        chosen_model VARCHAR(255) NOT NULL,
        answer TEXT,
        quality FLOAT,
        latency_s FLOAT,
        cost_per_1k FLOAT,
        reward FLOAT,
        context_label VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB;
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(ddl_weights))
            conn.execute(text(ddl_qlogs))
    except SQLAlchemyError as e:
        logger.warning(f"[nsga] Falha ao garantir tabelas: {e}")

# ------------------------------------------------------------
# Coleta de dados dos logs recentes
# ------------------------------------------------------------
def load_recent_query_logs(minutes_back: int, max_rows: int) -> List[Dict[str, Any]]:
    """Carrega logs recentes (janela móvel) da tabela 'query_log'."""
    try:
        with engine.connect() as conn:
            since = datetime.utcnow() - timedelta(minutes=minutes_back)
            # CORRIGIDO: Query ajustada para os nomes de coluna corretos
            rs = conn.execute(
                text(
                    """
                    SELECT 
                        created_at as ts, 
                        query_text as query, 
                        chosen_model as model, 
                        latency_s, 
                        quality, 
                        cost_per_1k, 
                        (reward > 0.3) as success
                    FROM query_log
                    WHERE created_at >= :since
                    ORDER BY id DESC
                    LIMIT :lim
                    """
                ),
                {"since": since, "lim": max_rows},
            )
            rows = rs.fetchall()
        out = []
        for r in rows:
            m = r._mapping
            out.append(
                {
                    "ts": m["ts"],
                    "query": m["query"] or "",
                    "model": m["model"] or "",
                    "latency_s": float(m["latency_s"]) if m["latency_s"] is not None else None,
                    "quality": float(m["quality"]) if m["quality"] is not None else None,
                    "cost_per_1k": float(m["cost_per_1k"]) if m["cost_per_1k"] is not None else None,
                    "success": bool(m["success"]) if m["success"] is not None else True,
                }
            )
        return out
    except SQLAlchemyError as e:
        logger.warning(f"[nsga] Falha ao ler query_log: {e}")
        return []

# ------------------------------------------------------------
# Extração de características das queries
# ------------------------------------------------------------
_CODE_HINTS = [
    "def ", "class ", "import ", "from ", "```", "```python", "SELECT ", "INSERT ", "UPDATE ", "DELETE ",
    "for(", "while(", ";", "#include", "console.log", "function(", "pub fn", "package "
]
_MATH_HINTS = [
    "∑", "∫", "√", "≈", "≤", "≥", "theorem", "lemma", "proof", "derivative", "integral", "matrix", "vector",
    "probability", "variance", "covariance", "gradient", "loss", "optimization", "equation"
]

def _is_code_like(s: str) -> bool:
    s2 = s.strip()
    if "```" in s2:
        return True
    s_low = s2.lower()
    return any(h.lower() in s_low for h in _CODE_HINTS)

def _is_math_like(s: str) -> bool:
    s_low = s.lower()
    return any(h.lower() in s_low for h in _MATH_HINTS) or bool(re.search(r"[0-9]\s*[\+\-\*/\^]\s*[0-9]", s_low))

def _estimate_tokens(s: str) -> int:
    # aproximação simples baseada em caracteres
    return max(1, int(len(s) / 4))

def characterize_queries(logs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Gera estatísticas de perfil de query da janela."""
    if not logs:
        return {
            "avg_tokens": 64.0,
            "code_frac": 0.1,
            "math_frac": 0.1,
            "diversity": 0.5,
        }
    toks = []
    code_count = 0
    math_count = 0

    # diversidade (Simpson) por trigramas grosseiros
    grams = {}

    for row in logs:
        q = (row.get("query") or "").strip()
        toks.append(_estimate_tokens(q))
        if _is_code_like(q):
            code_count += 1
        if _is_math_like(q):
            math_count += 1

        # trigrama bem simplificado (apenas letras/números)
        clean = re.sub(r"[^a-z0-9]+", " ", q.lower())
        parts = clean.split()
        for i in range(len(parts) - 2):
            tri = " ".join(parts[i:i+3])
            grams[tri] = grams.get(tri, 0) + 1

    avg_tokens = float(np.mean(toks)) if toks else 64.0
    n = len(logs)
    code_frac = code_count / n
    math_frac = math_count / n

    # índice de Simpson (quanto maior, mais concentrado; usaremos 1 - simpson para "diversidade")
    total = sum(grams.values()) or 1
    simpson = sum((c / total) ** 2 for c in grams.values())
    diversity = max(0.0, min(1.0, 1.0 - simpson))

    return {
        "avg_tokens": float(avg_tokens),
        "code_frac": float(code_frac),
        "math_frac": float(math_frac),
        "diversity": float(diversity),
    }

# ------------------------------------------------------------
# Métricas observadas por modelo
# ------------------------------------------------------------
def per_model_observations(models: List[str], logs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Calcula métricas por modelo a partir dos logs: média de latência, custo, qualidade, taxa de sucesso.
    Se um modelo não tiver dados, usa defaults conservadores.
    """
    by_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}
    for r in logs:
        m = r.get("model") or ""
        if m in by_model:
            by_model[m].append(r)

    out: Dict[str, Dict[str, float]] = {}
    for m in models:
        rows = by_model[m]
        if not rows:
            out[m] = {
                "latency": 8.0,
                "cost_per_1k": 0.02 if "ollama" not in m else 0.001,
                "quality": 5.0,
                "success_rate": 0.95,
            }
            continue

        lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
        qlt = [r["quality"] for r in rows if r.get("quality") is not None]
        cst = [r["cost_per_1k"] for r in rows if r.get("cost_per_1k") is not None]
        succ = [1.0 if r.get("success", True) else 0.0 for r in rows]

        out[m] = {
            "latency": float(np.mean(lat)) if lat else 8.0,
            "cost_per_1k": float(np.mean(cst)) if cst else (0.02 if "ollama" not in m else 0.001),
            "quality": float(np.mean(qlt)) if qlt else 5.0,
            "success_rate": float(np.mean(succ)) if succ else 0.9,
        }

    return out

# ------------------------------------------------------------
# Alinhamento modelo ↔ perfil de query
# ------------------------------------------------------------
def _parse_model_billion_params(model_name: str) -> float:
    """
    Tenta inferir o número de parâmetros em bilhões (ex: ':1.7b', ':4b', ':70b') a partir do nome.
    Se não encontrado, usa heurística: modelos 'gpt' -> 70b; 'mini' ~ 4b; caso geral 8b.
    """
    m = re.search(r":\s*([0-9]+(?:\.[0-9]+)?)\s*b", model_name.lower())
    if m:
        return float(m.group(1))
    lower = model_name.lower()
    if "mini" in lower:
        return 4.0
    if "gpt" in lower:
        return 70.0
    if "qwen2.5" in lower or "qwen3" in lower or "gemma3" in lower or "granite" in lower:
        return 3.0
    return 8.0

def alignment_score(model: str, profile: Dict[str, float]) -> float:
    """
    Score de alinhamento (0..1) do modelo ao perfil de query:
    - queries com muitos tokens beneficiam modelos maiores;
    - queries 'code-like' favorecem modelos com >4B;
    - queries 'math-like' favorecem modelos maiores ainda;
    - diversidade alta favorece modelos 'robustos' (aumenta baseline).
    """
    size_b = _parse_model_billion_params(model)
    avg_toks = profile["avg_tokens"]
    code_frac = profile["code_frac"]
    math_frac = profile["math_frac"]
    diversity = profile["diversity"]

    # contribuição por tamanho vs tokens
    # ideal_size_b ~ avg_toks/256 com saturação
    ideal_size = min(20.0, max(1.0, avg_toks / 256.0))
    size_factor = 1.0 - min(1.0, abs(size_b - ideal_size) / max(ideal_size, 1.0))

    code_factor = 1.0 if (size_b >= 4.0) else 0.6
    math_factor = 1.0 if (size_b >= 7.0) else 0.6

    # mistura ponderada
    score = (
        0.50 * size_factor +
        0.30 * (code_frac * code_factor + (1 - code_frac) * 0.9) +
        0.20 * (math_frac * math_factor + (1 - math_frac) * 0.9)
    )

    # diversidade aumenta baseline (robustez)
    score = 0.8 * score + 0.2 * diversity
    return max(0.0, min(1.0, float(score)))

# ------------------------------------------------------------
# NSGA-II (variáveis = pesos por modelo que somam 1; 0<=w<=1)
# Objetivos: minimizar latência e custo; maximizar qualidade e alinhamento.
# ------------------------------------------------------------
def normalize_weights(w: List[float]) -> List[float]:
    w = [max(0.0, min(1.0, x)) for x in w]
    s = sum(w)
    if s <= 0:
        # volta para uniforme
        return [1.0 / len(w)] * len(w)
    return [x / s for x in w]

def expected_mix_objectives(
    weights: List[float],
    per_model: Dict[str, Dict[str, float]],
    models: List[str],
    align: Dict[str, float],
) -> Tuple[float, float, float, float]:
    """
    Dado um vetor de pesos, calcula objetivos esperados do "mix":
    latency (min), cost (min), quality (max), alignment (max).
    """
    w = weights
    lat = sum(w[i] * per_model[models[i]]["latency"] for i in range(len(models)))
    cst = sum(w[i] * per_model[models[i]]["cost_per_1k"] for i in range(len(models)))
    qlt = sum(w[i] * per_model[models[i]]["quality"] for i in range(len(models)))
    aln = sum(w[i] * align[models[i]] for i in range(len(models)))
    return lat, cst, qlt, aln

def run_nsga(models: List[str], pm_obs: Dict[str, Dict[str, float]], align: Dict[str, float]) -> Dict[str, float]:
    n = len(models)
    if n == 1:
        return {models[0]: 1.0}

    # DEAP setup
    try:
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, +1.0, +1.0))
    except Exception:
        pass
    try:
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    except Exception:
        pass

    toolbox = base.Toolbox()
    toolbox.register("attr_w", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_w, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        w = normalize_weights(individual)
        lat, cst, qlt, aln = expected_mix_objectives(w, pm_obs, models, align)
        # proxy de "fitness médio" para debug
        NSGA_POP_FITNESS_AVG.set((max(0.0, 10.0 - lat) + max(0.0, 10.0 - cst) + qlt + 10.0 * aln) / 4.0)
        return lat, cst, qlt, aln

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0/n)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=40)
    hof = tools.ParetoFront()
    algorithms.eaMuPlusLambda(
        population=pop,
        toolbox=toolbox,
        mu=40,
        lambda_=40,
        cxpb=0.9,
        mutpb=0.1,
        ngen=30,
        stats=None,
        halloffame=hof,
        verbose=False,
    )

    # Escolha do ponto do Pareto: "melhor comprometido" via escalarização simples
    best = None
    best_score = -1e9
    for ind in hof:
        w = normalize_weights(list(ind))
        lat, cst, qlt, aln = expected_mix_objectives(w, pm_obs, models, align)
        # score escalar (ajuste os pesos conforme sua preferência)
        score = -lat - 0.5 * cst + 0.8 * qlt + 5.0 * aln
        if score > best_score:
            best_score = score
            best = (w, (lat, cst, qlt, aln))

    if best is None:
        # fallback uniforme
        return {m: 1.0 / n for m in models}

    w, (lat, cst, qlt, aln) = best
    NSGA_BEST_LATENCY.set(lat)
    NSGA_BEST_COST.set(cst)
    NSGA_BEST_QUALITY.set(qlt)
    NSGA_BEST_ALIGNMENT.set(aln)

    return {models[i]: float(w[i]) for i in range(n)}

# ------------------------------------------------------------
# Persistência de pesos (DB/Redis)
# ------------------------------------------------------------
def persist_weights(weights: Dict[str, float]):
    # DB
    try:
        with engine.begin() as conn:
            for model, w in weights.items():
                conn.execute(
                    text(
                        """
                        INSERT INTO nsga_weights (model, weight, computed_at)
                        VALUES (:m, :w, NOW())
                        ON DUPLICATE KEY UPDATE
                            weight = :w,
                            computed_at = NOW()
                        """
                    ),
                    {"m": model, "w": w},
                )
    except SQLAlchemyError as e:
        logger.warning(f"[nsga] Falha ao gravar nsga_weights: {e}")

    # Redis
    try:
        r = get_redis()
        if r:
            r.set(REDIS_KEY_WEIGHTS, json.dumps(weights))
    except Exception as e:
        logger.warning(f"[nsga] Falha ao publicar pesos no Redis: {e}")

    # Prometheus
    for m, w in weights.items():
        WEIGHT_BY_MODEL.labels(model=m).set(w)

# ------------------------------------------------------------
# Loop principal
# ------------------------------------------------------------
def one_iteration():
    try:
        ensure_tables()
        
        # CORRIGIDO: Lê diretamente do settings
        models = settings.CANDIDATE_MODELS_LIST
        
        if not models:
            logger.warning("[nsga] Sem modelos; interrompendo iteração.")
            return

        logs = load_recent_query_logs(QUERY_LOG_LOOKBACK_MINUTES, QUERY_LOG_MAX_ROWS)
        profile = characterize_queries(logs)
        pm_obs = per_model_observations(models, logs)
        align = {m: alignment_score(m, profile) for m in models}

        # expõe métricas observadas por modelo
        for m in models:
            MODEL_OBS_LAT.labels(model=m).set(pm_obs[m]["latency"])
            MODEL_OBS_COST.labels(model=m).set(pm_obs[m]["cost_per_1k"])
            MODEL_OBS_QUAL.labels(model=m).set(pm_obs[m]["quality"])
            MODEL_OBS_SR.labels(model=m).set(pm_obs[m]["success_rate"])
            MODEL_ALIGNMENT.labels(model=m).set(align[m])

        weights = run_nsga(models, pm_obs, align)
        persist_weights(weights)

        NSGA_RUNS.inc()
        NSGA_LAST_RUN_TS.set(time.time())
        logger.info(f"[nsga] Pesos atualizados: {weights}")
    except Exception as e:
        logger.exception(f"[nsga] Falha na iteração NSGA: {e}")

def nsga_loop():
    # primeira execução imediata
    one_iteration()
    # atualizações periódicas
    while True:
        time.sleep(UPDATE_INTERVAL_S)
        one_iteration()

# ------------------------------------------------------------
# API (FastAPI) para /metrics e /health
# ------------------------------------------------------------
app = FastAPI(title="NSGA Weights Updater")

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(registry).decode("utf-8"))

@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    r = get_redis()
    redis_ok = r is not None

    return JSONResponse(
        {
            "status": "ok" if (db_ok and redis_ok) else "degraded",
            "db": db_ok,
            "redis": redis_ok,
            "last_run_epoch": NSGA_LAST_RUN_TS._value.get() if hasattr(NSGA_LAST_RUN_TS, "_value") else None,
        }
    )

# ------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------
if __name__ == "__main__":
    # roda o loop de otimização num thread separado do servidor HTTP
    t = threading.Thread(target=nsga_loop, daemon=True)
    t.start()

    # inicia servidor HTTP (porta 9999 dentro do container conforme docker-compose)
    uvicorn.run(app, host="0.0.0.0", port=9999, log_level="info")