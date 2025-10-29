import random
import logging
from typing import List
from .settings import settings
from .metrics_collector import update_model_metrics, get_snapshot
from .observability import ROUTER_CHOSEN

logger = logging.getLogger(__name__)


def choose_top2_models(
    candidates: List[str], min_quality: float, query_text: str
) -> List[str]:
    """
    Escolhe os 2 melhores modelos com base nas métricas dinâmicas.
    Retorna uma lista de nomes de modelos (sempre 2, se possível).
    """
    snapshot = get_snapshot() or {}
    results = []

    for c in candidates:
        m = snapshot.get(c, {"quality": 5.0, "latency": 1.5, "cost": 0.2})
        results.append((c, m["quality"], m["latency"], m["cost"]))

    # Ordena por maior qualidade, depois menor latência e custo
    ranked = sorted(results, key=lambda x: (-x[1], x[2], x[3]))

    top = [r[0] for r in ranked[:2]]
    logger.info(f"[router_strategy] top2={top} (query='{query_text[:40]}...')")
    return top


def update_metrics(model_name: str, cost: float, latency: float, quality: float):
    """
    Atualiza as métricas do modelo em tempo real (coletadas no router_core/providers).
    """
    update_model_metrics(model_name, latency=latency, quality=quality, cost=cost)
