import pytest
from app.settings import settings


@pytest.fixture(scope="session")
def default_settings():
    """Retorna uma cópia das configurações padrão."""
    return settings
