import random
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# ------------------------------------------------------
# Estado interno dos bandits (exploração x exploração)
# ------------------------------------------------------
_bandit_stats: Dict[str, Dict[str, float]] = {}

# Parâmetros de controle
EPSILON = 0.2  # probabilidade de explorar
ALPHA = 0.3    # taxa de aprendizado (decadência do histórico)


def select_model(
    candidates: List[str],
    query_text: str,
    *,
    temperature: float = 0.7,
    enable_rag_for_answer: bool = False
) -> str:
    """
    Seleciona o modelo a ser usado com base em um esquema epsilon-greedy.
    Quanto maior o reward médio histórico, maior a chance de ser escolhido.
    """
    try:
        # Inicializa estatísticas se necessário
        for model in candidates:
            _bandit_stats.setdefault(model, {"avg_reward": 0.0, "count": 0})

        explore = random.random() < EPSILON
        if explore:
            chosen = random.choice(candidates)
            logger.info(f"[bandit] Explorando modelo aleatório: {chosen}")
            return chosen

        # Exploitation — escolhe o modelo com maior reward médio
        best_model = max(_bandit_stats, key=lambda m: _bandit_stats[m]["avg_reward"])
        logger.info(f"[bandit] Exploitando modelo com melhor média: {best_model}")
        return best_model

    except Exception as e:
        logger.error(f"[bandit] Erro na seleção: {e}")
        # fallback seguro
        return random.choice(candidates)


def update_model(
    model_name: str,
    query_text: str,
    reward: float,
    *,
    temperature: float = 0.7,
    enable_rag_for_answer: bool = False
):
    """
    Atualiza o valor médio de recompensa (reward) para o modelo selecionado.
    """
    try:
        entry = _bandit_stats.setdefault(model_name, {"avg_reward": 0.0, "count": 0})
        old_avg = entry["avg_reward"]
        count = entry["count"]

        new_avg = (1 - ALPHA) * old_avg + ALPHA * reward
        entry.update({"avg_reward": round(new_avg, 4), "count": count + 1})

        logger.info(f"[bandit] Modelo {model_name} atualizado | reward={reward:.3f} | média={new_avg:.3f}")

    except Exception as e:
        logger.error(f"[bandit] Falha ao atualizar modelo {model_name}: {e}")


def reset_bandits():
    """Limpa o histórico de decisões (útil para testes)."""
    global _bandit_stats
    _bandit_stats = {}
    logger.info("[bandit] Histórico de bandits resetado com sucesso.")


def get_bandit_stats() -> Dict[str, Dict[str, float]]:
    """Retorna um snapshot seguro das estatísticas atuais."""
    return {k: v.copy() for k, v in _bandit_stats.items()}
