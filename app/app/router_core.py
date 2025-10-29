# router_core.py
import time
import random
from typing import Dict, Any, List, Tuple

from .providers import call_model
try:
    from .rag_local import build_augmented_prompt, add_document

except ImportError as e:
    import logging
    logging.error(f"[router_core] Falha ao importar RAG: {e}")

from .semantic_cache import get as cache_get, put as cache_put

# ====== Config do roteador ======
CANDIDATE_MODELS = [
    "ollama/phi4",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.5-flash-lite",
    "ollama/deepseek-r1:8b",
    # adicione "openai/gpt-4o" se for usar via OpenAI, etc.
]

EPSILON = 0.1             # exploração no MAB
ALPHA = 0.15              # LR do MAB
RAG_THRESHOLD = 0.75      # τ para ativar RAG
CACHE_MIN_QUALITY = 0.70  # qualidade mínima para reutilizar cache

# Estado do MAB (média móvel de recompensas por modelo)
_bandit_reward: Dict[str, float] = {m: 0.0 for m in CANDIDATE_MODELS}

# ====== NSGA-II: pesos correntes (exemplo; substitua pela sua implementação/pipe real) ======
# theta = [w_quality, w_cost, w_latency], todos normalizados positivos
def get_current_theta() -> Tuple[float, float, float]:
    # Exemplo estático; no seu sistema, ISSO deve vir da sua população NSGA-II
    return (0.55, 0.25, 0.20)

# ====== Normalizações de métricas ======
def normalize_quality(q: float) -> float:
    return max(0.0, min(1.0, q))  # já no [0,1]

def normalize_cost(cost_per_1k: float) -> float:
    # normalize custos para [0,1] com um "cap" guardião
    # (ajuste para sua realidade; aqui supomos custos até 0.30 = 1.0)
    cap = 0.30
    return max(0.0, min(1.0, cost_per_1k / cap))

def normalize_latency(s: float) -> float:
    # normalize latência para [0,1] com cap em 3s
    cap = 3.0
    return max(0.0, min(1.0, s / cap))

# ====== Custo de referência por modelo (ajuste conforme seu pricing) ======
MODEL_REF_COST = {
    "gemini/gemini-2.5-flash-lite": 0.12,  # USD / 1k tokens (exemplo)
    "gemini/gemini-2.0-flash": 0.15,
    "ollama/phi4": 0.001,
    "ollama/deepseek-r1:8b": 0.001,
}

def est_cost_per_1k(model: str) -> float:
    return MODEL_REF_COST.get(model, 0.0)

# ====== Julgamento de qualidade (placeholder) ======
def judge_quality(answer: str, query: str) -> float:
    """
    Substitua pelo seu ensemble de juízes (heurística + LLM-judges).
    Aqui, usamos um placeholder simples: comprimento/estrutura -> score ∈ [0.6, 0.95].
    """
    base = 0.6
    bonus = min(0.35, len(answer.strip()) / 4000.0)  # 4k chars ~ +0.35
    return max(0.0, min(1.0, base + bonus))

# ====== Política de seleção (ε-greedy) ======
def _select_model() -> str:
    if random.random() < EPSILON:
        return random.choice(CANDIDATE_MODELS)
    # Exploração: pega maior recompensa média até aqui
    return max(CANDIDATE_MODELS, key=lambda m: _bandit_reward.get(m, 0.0))

# ====== Atualização do MAB ======
def _bandit_update(model: str, r_t: float) -> None:
    prev = _bandit_reward.get(model, 0.0)
    _bandit_reward[model] = (1 - ALPHA) * prev + ALPHA * r_t

# ====== Recompensa escalar (Eq. de acoplamento) ======
def scalar_reward(q_norm: float, c_norm: float, l_norm: float) -> float:
    wq, wc, wl = get_current_theta()
    # reward = +wq*Q - wc*C - wl*L
    return wq * q_norm - wc * c_norm - wl * l_norm

# ====== Pipeline principal ======
def route_and_answer(
    query: str,
    system_prompt: str | None = None,
    use_rag: bool = True,
) -> Dict[str, Any]:
    # 1) Cache semântico (reuso imediato)
    cached = cache_get(query, min_quality=CACHE_MIN_QUALITY)
    if cached:
        return {
            "answer": cached,
            "from_cache": True,
            "model": None,
            "latency_s": 0.0,
            "quality": 1.0,  # já avaliado previamente
            "cost_per_1k": 0.0,
            "rag_used": False,
        }

    # 2) Seleciona modelo pelo MAB
    model = _select_model()

    # 3) Constrói prompt (RAG adaptativo opcional)
    user_prompt = build_augmented_prompt(query, threshold=RAG_THRESHOLD) if use_rag else query

    # 4) Chama o provedor
    t0 = time.perf_counter()
    resp = call_model(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=2048,
    )
    latency_s = resp["latency_s"]
    answer = resp["text"]

    # 5) Julga qualidade (substitua pelo seu layer de juízes)
    q = judge_quality(answer, query)

    # 6) Normaliza métricas + recompensa
    qn = normalize_quality(q)
    cn = normalize_cost(est_cost_per_1k(model))
    ln = normalize_latency(latency_s)

    r_t = scalar_reward(qn, cn, ln)
    _bandit_update(model, r_t)

    # 7) Atualiza cache (somente se bom o suficiente)
    if q >= CACHE_MIN_QUALITY:
        cache_put(query, answer, quality=q)

    return {
        "answer": answer,
        "from_cache": False,
        "model": model,
        "latency_s": latency_s,
        "quality": q,
        "cost_per_1k": est_cost_per_1k(model),
        "rag_used": (user_prompt != query),
        "reward": r_t,
        "theta": get_current_theta(),
    }
