import random
import logging
from typing import List, Dict, Tuple
from .settings import settings
from .observability import ROUTER_CHOSEN

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# Mock de métricas internas dos modelos
# (em um ambiente real, seriam coletadas dinamicamente)
# ------------------------------------------------------
_model_metrics: Dict[str, Dict[str, float]] = {
    "phi4:latest": {"quality": 7.1, "latency": 1.3, "cost": 0.002},
    "gemini-2.0-flash": {"quality": 8.7, "latency": 0.9, "cost": 0.12},
    "gemini-2.5-flash": {"quality": 8.3, "latency": 1.1, "cost": 0.18},
}


def choose_top2_models(
    candidates: List[str], min_quality: float, query_text: str
) -> List[str]:
    """
    Escolhe os 2 melhores modelos com base em métricas internas.
    Retorna uma lista de nomes de modelos (sempre 2, se possível).
    """
    results = []
    for c in candidates:
        m = _model_metrics.get(c, {"quality": 5.0, "latency": 1.5, "cost": 0.2})
        results.append((c, m["quality"], m["latency"], m["cost"]))

    # Ordena por maior qualidade, depois menor latência e custo
    ranked = sorted(results, key=lambda x: (-x[1], x[2], x[3]))

    top = [r[0] for r in ranked[:2]]
    logger.info(f"[router_strategy] top2={top} (query='{query_text[:40]}...')")
    return top


def update_metrics(model_name: str, cost: float, latency: float, quality: float):
    """
    Atualiza métricas locais de desempenho dos modelos.
    Em uma versão mais complexa, isso seria persistido em Redis/Prometheus.
    """
    try:
        if model_name not in _model_metrics:
            _model_metrics[model_name] = {"quality": quality, "latency": latency, "cost": cost}
        else:
            # Média ponderada simples (suavização)
            prev = _model_metrics[model_name]
            _model_metrics[model_name] = {
                "quality": round((prev["quality"] * 0.7 + quality * 0.3), 3),
                "latency": round((prev["latency"] * 0.7 + latency * 0.3), 3),
                "cost": round((prev["cost"] * 0.7 + cost * 0.3), 5),
            }
        logger.info(f"[router_strategy] métricas atualizadas para {model_name}: {_model_metrics[model_name]}")
    except Exception as e:
        logger.error(f"[router_strategy] falha ao atualizar métricas de {model_name}: {e}")


def get_metrics_snapshot() -> Dict[str, Dict[str, float]]:
    """Retorna uma cópia imutável das métricas atuais."""
    return {k: v.copy() for k, v in _model_metrics.items()}
