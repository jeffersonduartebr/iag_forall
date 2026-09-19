# Objective: Regression tests that run in a fresh interpreter, outside the conftest resets.
"""Runtime paths exercised in a new process.

The autouse ``mock_dependencies`` fixture calls ``reset_provider_runtime_state()``
before every test, which used to *create* ``providers_async._http_client``. In a
real process nothing did, so the first ``get_http_client()`` raised
``AttributeError`` (Ollama health unhealthy, ``/query`` answering 502), and no
in-process test could notice. These tests start a clean interpreter instead.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONPATH": str(ROOT / "app"),
        # Portas fechadas: DB/Redis falham rápido, sem esperar resolução de nomes do compose.
        "REDIS_HOST": "127.0.0.1",
        "REDIS_PORT": "1",
        "DB_HOST": "127.0.0.1",
        "DB_PORT": "1",
    }
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)


def test_shared_http_client_works_in_a_fresh_process():
    code = """
import asyncio
import app.providers_async as pa
from app.providers._infra import close_http_client

async def main():
    client = await pa.get_http_client()
    assert client is await pa.get_http_client(), "o cliente deve ser compartilhado"
    await close_http_client()
    assert pa._http_client is None
    print("ok")

asyncio.run(main())
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")
