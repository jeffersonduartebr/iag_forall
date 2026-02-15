"""Módulo principal: descreve responsabilidades e integrações deste arquivo."""

# prometheus_setup.py
# ----------------------------------------------------------
# Inicialização segura do Prometheus em modo multiprocess.
# - Cria diretório de métricas se não existir.
# - Remove arquivos antigos (para evitar UnicodeDecodeError).
# - Reconfigura o registro multiprocess global.
# ----------------------------------------------------------

import os
import shutil
from prometheus_client import CollectorRegistry, multiprocess, generate_latest, CONTENT_TYPE_LATEST

PROM_DIR = os.getenv("PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus_multiproc_dir")


def setup_prometheus():
    """Limpa e configura o diretório multiprocess."""
    # Cria o diretório se não existir
    os.makedirs(PROM_DIR, exist_ok=True)

    # Remove arquivos antigos (evita UnicodeDecodeError)
    for f in os.listdir(PROM_DIR):
        try:
            path = os.path.join(PROM_DIR, f)
            if os.path.isfile(path):
                os.remove(path)
        except Exception as e:
            print(f"[prometheus_setup] Falha ao limpar {f}: {e}")

    print(f"[prometheus_setup] Diretório de métricas pronto: {PROM_DIR}")


def prometheus_registry():
    """Retorna o registro multiprocess configurado."""
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry


def prometheus_metrics():
    """Gera a saída /metrics para FastAPI."""
    registry = prometheus_registry()
    data = generate_latest(registry)
    return data, CONTENT_TYPE_LATEST
