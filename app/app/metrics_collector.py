# app/metrics_collector.py
import threading
import time
from typing import Dict

_lock = threading.Lock()
_model_metrics: Dict[str, Dict[str, float]] = {}

def update_model_metrics(model_name: str, latency: float, quality: float, cost: float):
    """
    Atualiza as métricas dinâmicas de um modelo.
    Pode ser chamada por qualquer parte do sistema (ex: router_core, providers).
    """
    with _lock:
        if model_name not in _model_metrics:
            _model_metrics[model_name] = {"quality": quality, "latency": latency, "cost": cost}
        else:
            prev = _model_metrics[model_name]
            _model_metrics[model_name] = {
                "quality": round((prev["quality"] * 0.7 + quality * 0.3), 3),
                "latency": round((prev["latency"] * 0.7 + latency * 0.3), 3),
                "cost": round((prev["cost"] * 0.7 + cost * 0.3), 5),
            }

def get_snapshot() -> Dict[str, Dict[str, float]]:
    """Retorna uma cópia das métricas atuais (thread-safe)."""
    with _lock:
        return {k: v.copy() for k, v in _model_metrics.items()}
