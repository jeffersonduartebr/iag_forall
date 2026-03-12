"""Módulo `tests/conftest.py`: descreve responsabilidades e integrações deste arquivo."""

import os
import sys
from unittest.mock import MagicMock

import pytest

# Prioriza o pacote de aplicação real em app/app (importável como "app").
APP_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if APP_PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, APP_PACKAGE_ROOT)

# Importa os módulos para garantir que estão disponíveis antes do mocking
try:
    import app.utils.redis_client
    import app.settings_dynamic
except ImportError:
    pass  # Ignora se não conseguir importar


def pytest_configure(config):
    """Register shared custom markers even when pytest.ini is not mounted."""
    config.addinivalue_line("markers", "integration: tests that require external services or credentials")
    config.addinivalue_line("markers", "slow: long-running tests")

@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    """
    Mocka automaticamente conexões externas para TODOS os testes.
    """
    # Mock Redis
    mock_redis = MagicMock()
    try:
        monkeypatch.setattr("app.utils.redis_client.get_redis", lambda *args: mock_redis)
        monkeypatch.setattr("app.settings_dynamic.get_redis", lambda *args: mock_redis)
    except (ImportError, AttributeError):
        pass  # Ignora se os módulos não existirem

    # Mock SQLAlchemy Engine
    mock_engine = MagicMock()
    monkeypatch.setattr("sqlalchemy.create_engine", lambda *args, **kwargs: mock_engine)

    # Mock ChromaDB
    try:
        monkeypatch.setattr("chromadb.PersistentClient", MagicMock())
    except (ImportError, AttributeError):
        pass

    # Mock Settings Defaults para consistência
    monkeypatch.setenv("NSGA_W_QUALITY", "1.0")
    monkeypatch.setenv("NSGA_W_LATENCY", "0.5")
    monkeypatch.setenv("NSGA_W_COST", "50.0")
