# bandits.py
import os
import json
import random
import logging
from datetime import datetime
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)

__all__ = [
    "select_model",
    "update_model",
    "bandit_update",   # alias compatível
    "compute_reward",
]

# ======================================================
# CONFIGURAÇÕES BÁSICAS
# ======================================================

DATA_DIR = "/app/data"
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")

os.makedirs(DATA_DIR, exist_ok=True)

# Parâmetros do bandit
EPSILON = 0.1     # exploração
ALPHA = 0.15      # taxa de aprendizado (exponential moving average)
DEFAULT_THETA = [0.55, 0.25, 0.20]  # [w_q, w_c, w_l]

# Tabelas de custo aproximado (USD por 1k tokens)
MODEL_COSTS: Dict[str, float] = {
    "gemini/gemini-2.5-flash-lite": 0.12,
    "gemini/gemini-2.0-flash": 0.15,
    "ollama/phi4": 0.001,
    "ollama/deepseek-r1:8b": 0.001,
}

# Armazena média de recompensa por modelo
_bandit_state: Dict[str, float] = {}

# ======================================================
# SUPORTE A PESOS DO NSGA-II
# ======================================================

def _load_theta() -> Tuple[float, float, float]:
    """Carrega os pesos atuais (w_q, w_c, w_l) definidos pelo NSGA-II."""
    try:
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, "r") as f:
                data = json.load(f)
            theta = data.get("current_best", DEFAULT_THETA)
            if len(theta) == 3 and all(isinstance(x, (float, int)) for x in theta):
                return float(theta[0]), float(theta[1]), float(theta[2])
        logger.warning("[bandit] weights.json inválido, usando padrão.")
    except Exception as e:
        logger.error(f"[bandit] erro ao ler weights.json: {e}")
    return float(DEFAULT_THETA[0]), float(DEFAULT_THETA[1]), float(DEFAULT_THETA[2])

# ======================================================
# FUNÇÕES AUXILIARES
# ======================================================

def _normalize_quality(q: float) -> float:
    return max(0.0, min(1.0, float(q)))

def _normalize_latency(latency: float) -> float:
    cap = 3.0
    return max(0.0, min(1.0, float(latency) / cap))

def _normalize_cost(model: str) -> float:
    cap = 0.30
    cost = MODEL_COSTS.get(model, 0.0)
    return max(0.0, min(1.0, cost / cap))

def _scalar_reward(q: float, c: float, l: float) -> float:
    wq, wc, wl = _load_theta()
    return (wq * q) - (wc * c) - (wl * l)

# ======================================================
# BANDIT CORE (ε-greedy)
# ======================================================

def select_model(candidates: List[str], query: str, **kwargs) -> str:
    """Seleciona modelo usando política ε-greedy."""
    if not _bandit_state:
        for m in candidates:
            _bandit_state[m] = 0.0

    if random.random() < EPSILON:
        choice = random.choice(candidates)
        logger.info(f"[bandit] Exploração (ε={EPSILON:.2f}) → {choice}")
    else:
        choice = max(candidates, key=lambda m: _bandit_state.get(m, 0.0))
        logger.info(f"[bandit] Greedy (aproveitamento) → {choice}")
    return choice


def update_model(model: str, query: str, reward: float, **kwargs) -> float:
    """Atualiza o valor médio estimado do modelo via EMA."""
    prev = _bandit_state.get(model, 0.0)
    _bandit_state[model] = (1 - ALPHA) * prev + ALPHA * float(reward)

    # Persistência para auditoria
    rec = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "reward": float(reward),
        "ema": _bandit_state[model],
        "query_sample": (query or "")[:80],
    }
    try:
        history = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        history.append(rec)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history[-1000:], f, indent=2)
    except Exception as e:
        logger.error(f"[bandit] erro ao salvar histórico: {e}")

    logger.info(f"[bandit] Modelo {model} atualizado | reward={reward:.3f} | média={_bandit_state[model]:.3f}")
    return _bandit_state[model]


# Alias para manter compatibilidade com importações existentes
def bandit_update(model: str, query: str, reward: float, **kwargs) -> float:
    """Backward-compatible alias. Encaminha para update_model()."""
    return update_model(model, query, reward, **kwargs)

# ======================================================
# CÁLCULO COMPLETO DE RECOMPENSA ACOPLADA AO NSGA-II
# ======================================================

def compute_reward(model: str, quality: float, latency: float) -> float:
    """
    Calcula r_t = w_q·Q - w_c·C - w_l·L
    usando pesos do NSGA-II e métricas normalizadas.
    """
    qn = _normalize_quality(quality)
    cn = _normalize_cost(model)
    ln = _normalize_latency(latency)
    reward = _scalar_reward(qn, cn, ln)
    logger.debug(
        f"[bandit] reward={reward:.3f} | q={qn:.3f} c={cn:.3f} l={ln:.3f} θ={_load_theta()}"
    )
    return reward
