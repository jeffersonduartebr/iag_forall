# app/bandits.py
import random
import logging
import time # 👈 Adicionado
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from .db_manager import insert_history, get_current_weights # 👈 Adicionado get_current_weights

logger = logging.getLogger(__name__)

EPSILON = 0.1
ALPHA = 0.15
# ❌ DEFAULT_THETA removido, agora é dinâmico

MODEL_COSTS = {
    "gemini/gemini-2.5-flash-lite": 0.12,
    "gemini/gemini-2.0-flash": 0.15,
    "ollama/phi4": 0.001,
    "ollama/deepseek-r1:8b": 0.001,
}

_bandit_state: Dict[str, float] = {}

# ============================================================
# ✅ LÓGICA DE PESOS DINÂMICOS (com cache)
# ============================================================
_weights_cache: Dict[str, List[float]] = {}
_weights_cache_last_updated: float = 0.0
WEIGHTS_CACHE_TTL_S = 300  # 5 minutos

# (Copiado de nsga_weights_updater.py para evitar dependência circular)
MODEL_FAMILIES = {
    "phi": {"default_weights": [0.65, 0.20, 0.15], "cost_target": 0.002, "latency_target": 1.3, "token_key": "max_tokens"},
    "deepseek": {"default_weights": [0.55, 0.25, 0.20], "cost_target": 0.005, "latency_target": 1.4, "token_key": "max_tokens"},
    "llama": {"default_weights": [0.60, 0.25, 0.15], "cost_target": 0.008, "latency_target": 1.5, "token_key": "max_tokens"},
    "mistral": {"default_weights": [0.58, 0.22, 0.20], "cost_target": 0.006, "latency_target": 1.1, "token_key": "max_tokens"},
    "qwen": {"default_weights": [0.50, 0.30, 0.20], "cost_target": 0.009, "latency_target": 1.2, "token_key": "max_tokens"},
    "gemma": {"default_weights": [0.57, 0.23, 0.20], "cost_target": 0.010, "latency_target": 1.0, "token_key": "max_tokens"},
    "gpt": {"default_weights": [0.45, 0.35, 0.20], "cost_target": 0.12, "latency_target": 0.9, "token_key": "max_completion_tokens"},
    "gemini": {"default_weights": [0.50, 0.30, 0.20], "cost_target": 0.15, "latency_target": 1.0, "token_key": "max_tokens"},
    "default": {"default_weights": [0.55, 0.25, 0.20], "cost_target": 0.01, "latency_target": 1.5, "token_key": "max_tokens"},
}

def detect_model_family(model_name: str) -> str:
    """(Copiado de nsga_weights_updater.py)"""
    if not model_name:
        return "default"
    model_name = model_name.lower()
    for fam in MODEL_FAMILIES.keys():
        if fam in model_name:
            return fam
    return "default"

def _get_weights_for_model(model: str) -> List[float]:
    """Obtém pesos do cache ou do DB; usa default como fallback."""
    global _weights_cache, _weights_cache_last_updated
    now = time.time()
    
    # 1. Limpa cache se estiver velho
    if (now - _weights_cache_last_updated) > WEIGHTS_CACHE_TTL_S:
        _weights_cache = {}
        _weights_cache_last_updated = now
        logger.info("[bandit] Cache de pesos NSGA expirou. Limpando.")

    family = detect_model_family(model)
    
    # 2. Verifica cache
    if family in _weights_cache:
        return _weights_cache[family]
    
    # 3. Tenta buscar no Banco de Dados
    try:
        weights_from_db = get_current_weights(family)
        if weights_from_db:
            logger.info(f"[bandit] Carregou pesos do DB para família '{family}': {weights_from_db}")
            _weights_cache[family] = weights_from_db
            return weights_from_db
    except Exception as e:
        logger.warning(f"[bandit] Não foi possível buscar pesos do DB para '{family}': {e}")
    
    # 4. Fallback para o default da família
    default_weights = MODEL_FAMILIES.get(family, MODEL_FAMILIES["default"])["default_weights"]
    logger.warning(f"[bandit] Usando pesos DEFAULT para família '{family}': {default_weights}")
    _weights_cache[family] = default_weights
    return default_weights

# ============================================================
# Funções de Normalização e Recompensa
# ============================================================
def _normalize_quality(q: float) -> float:
    # A qualidade já vem em 0-10, converte para 0-1
    return max(0.0, min(1.0, float(q) / 10.0))

def _normalize_latency(latency: float) -> float:
    cap = 3.0  # latência máxima esperada (3s)
    return max(0.0, min(1.0, float(latency) / cap))

def _normalize_cost(model: str) -> float:
    cap = 0.30 # Custo máximo esperado (0.30 USD/1k)
    cost = MODEL_COSTS.get(model, 0.0)
    return max(0.0, min(1.0, cost / cap))

def _scalar_reward(q: float, c: float, l: float, theta: List[float]) -> float:
    """Função de recompensa escalar ponderada."""
    wq, wc, wl = theta
    # Recompensa é alta qualidade, baixo custo e baixa latência
    return (wq * q) - (wc * c) - (wl * l)

# ============================================================
# Lógica do Bandit (Seleção e Atualização)
# ============================================================
def select_model(candidates: List[str], query: str) -> str:
    """Seleciona modelo (Epsilon-Greedy)."""
    if not _bandit_state:
        for m in candidates:
            _bandit_state[m] = 0.0  # Inicializa valor médio em 0
            
    if random.random() < EPSILON:
        choice = random.choice(candidates)
        logger.info(f"[bandit] Exploração (ε={EPSILON:.2f}) → {choice}")
    else:
        # Ação Greedy: escolhe o modelo com maior valor (recompensa média)
        choice = max(candidates, key=lambda m: _bandit_state.get(m, 0.0))
        logger.info(f"[bandit] Greedy (aproveitamento) → {choice}")
    return choice

def update_model(model: str, query: str, reward: float, **kwargs) -> float:
    """Atualiza o valor médio estimado (EMA) do modelo e salva no banco."""
    prev = _bandit_state.get(model, 0.0)
    # Atualiza valor médio via Média Móvel Exponencial (EMA)
    _bandit_state[model] = (1 - ALPHA) * prev + ALPHA * float(reward)

    # Persistência no banco
    try:
        insert_history(model, float(reward), float(_bandit_state[model]), (query or "")[:80])
    except Exception as db_err:
        logger.error(f"[bandit] Falha ao gravar no banco: {db_err}")

    logger.info(f"[bandit] Modelo {model} atualizado | reward={reward:.3f} | média={_bandit_state[model]:.3f}")
    return _bandit_state[model]


def bandit_update(model: str, query: str, reward: float, **kwargs) -> float:
    """Função wrapper pública para atualização."""
    return update_model(model, query, reward, **kwargs)

def compute_reward(model: str, quality: float, latency: float) -> float:
    """Calcula a recompensa escalar usando pesos DINÂMICOS."""
    qn = _normalize_quality(quality)
    cn = _normalize_cost(model)
    ln = _normalize_latency(latency)
    
    # ✅ LÓGICA CORRIGIDA:
    theta = _get_weights_for_model(model)
    
    reward = _scalar_reward(qn, cn, ln, theta)
    return reward