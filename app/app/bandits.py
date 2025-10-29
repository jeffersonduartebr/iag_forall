# app/bandits.py
import random
import logging
from datetime import datetime
from typing import Dict, Tuple, List
from .db_manager import insert_history

logger = logging.getLogger(__name__)

EPSILON = 0.1
ALPHA = 0.15
DEFAULT_THETA = [0.55, 0.25, 0.20]

MODEL_COSTS = {
    "gemini/gemini-2.5-flash-lite": 0.12,
    "gemini/gemini-2.0-flash": 0.15,
    "ollama/phi4": 0.001,
    "ollama/deepseek-r1:8b": 0.001,
}

_bandit_state: Dict[str, float] = {}

def _normalize_quality(q: float) -> float:
    return max(0.0, min(1.0, float(q)))

def _normalize_latency(latency: float) -> float:
    cap = 3.0
    return max(0.0, min(1.0, float(latency) / cap))

def _normalize_cost(model: str) -> float:
    cap = 0.30
    cost = MODEL_COSTS.get(model, 0.0)
    return max(0.0, min(1.0, cost / cap))

def _scalar_reward(q: float, c: float, l: float, theta: List[float]) -> float:
    wq, wc, wl = theta
    return (wq * q) - (wc * c) - (wl * l)

def select_model(candidates: List[str], query: str) -> str:
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
    prev = _bandit_state.get(model, 0.0)
    ema = (1 - ALPHA) * prev + ALPHA * reward
    _bandit_state[model] = ema

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "reward": reward,
        "ema": ema,
        "latency_s": kwargs.get("latency_s", 0),
        "quality": kwargs.get("quality", 0),
        "cost_usd": kwargs.get("cost_usd", 0),
        "query_sample": query[:80],
    }

    try:
        insert_history(entry)
    except Exception as e:
        logger.error(f"[bandit] Falha ao gravar no banco: {e}")

    logger.info(f"[bandit] Modelo {model} atualizado | reward={reward:.3f} | média={ema:.3f}")
    return ema

def bandit_update(model: str, query: str, reward: float, **kwargs) -> float:
    return update_model(model, query, reward, **kwargs)

def compute_reward(model: str, quality: float, latency: float) -> float:
    qn = _normalize_quality(quality)
    cn = _normalize_cost(model)
    ln = _normalize_latency(latency)
    reward = _scalar_reward(qn, cn, ln, DEFAULT_THETA)
    return reward
