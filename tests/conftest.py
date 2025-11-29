import pytest
import sys
import os
from unittest.mock import MagicMock

# Adiciona o diretório raiz ao path para importar 'app' corretamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    """
    Mocka automaticamente conexões externas para TODOS os testes.
    """
    # Mock Redis
    mock_redis = MagicMock()
    monkeypatch.setattr("app.app.utils.redis_client.get_redis", lambda *args: mock_redis)
    monkeypatch.setattr("app.app.settings_dynamic.get_redis", lambda *args: mock_redis)
    
    # Mock SQLAlchemy Engine
    mock_engine = MagicMock()
    monkeypatch.setattr("sqlalchemy.create_engine", lambda *args, **kwargs: mock_engine)
    
    # Mock ChromaDB
    monkeypatch.setattr("chromadb.PersistentClient", MagicMock())
    
    # Mock Settings Defaults para consistência
    monkeypatch.setenv("NSGA_W_QUALITY", "1.0")
    monkeypatch.setenv("NSGA_W_LATENCY", "0.5")
    monkeypatch.setenv("NSGA_W_COST", "50.0")