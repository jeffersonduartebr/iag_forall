# Objective: Application runtime code for prometheus setup.
"""Prepare and expose the Prometheus multiprocess registry.

The application runs with multiple worker processes, so Prometheus metrics need
special handling to aggregate per-process files safely. This module provides the
small bootstrap layer used by the API startup sequence and the ``/metrics``
endpoint:

- ``setup_prometheus()`` prepares the multiprocess directory used by the
  Prometheus client library.
- ``prometheus_registry()`` builds a registry that reads all process shards.
- ``prometheus_metrics()`` returns the serialized payload consumed by FastAPI.

The functions here are intentionally lightweight because they run early during
application startup and must remain safe even when the metrics directory already
contains stale files from a previous process.
"""

# prometheus_setup.py
# ----------------------------------------------------------
# Inicialização segura do Prometheus em modo multiprocess.
# - Cria diretório de métricas se não existir.
# - Remove arquivos antigos (para evitar UnicodeDecodeError).
# - Reconfigura o registro multiprocess global.
# ----------------------------------------------------------

import os

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess

PROM_DIR = os.getenv("PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus_multiproc_dir")


def setup_prometheus():
    """Initialize the Prometheus multiprocess directory for the current runtime.

    The Prometheus Python client stores one file per process in multiprocess
    mode. During local restarts those files may become stale and cause confusing
    export errors, so startup clears leftover regular files while preserving the
    directory itself.
    """
    # Cria o diretório se não existir
    os.makedirs(PROM_DIR, exist_ok=True)

    # Optionally remove stale multiprocess files (disabled by default).
    if os.getenv("PROMETHEUS_WIPE_ON_STARTUP", "0").strip() in ("1", "true", "yes"):
        for f in os.listdir(PROM_DIR):
            try:
                path = os.path.join(PROM_DIR, f)
                if os.path.isfile(path):
                    os.remove(path)
            except Exception as e:
                print(f"[prometheus_setup] Falha ao limpar {f}: {e}")

    print(f"[prometheus_setup] Diretório de métricas pronto: {PROM_DIR}")


def prometheus_registry():
    """Build a registry that aggregates metrics across all running processes.

    Returns:
        CollectorRegistry: A registry configured with the multiprocess collector.
    """
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry


def prometheus_metrics():
    """Serialize the current Prometheus registry for the HTTP metrics endpoint.

    Returns:
        tuple[bytes, str]: The raw payload expected by FastAPI responses and the
        matching Prometheus content type header.
    """
    registry = prometheus_registry()
    data = generate_latest(registry)
    return data, CONTENT_TYPE_LATEST
