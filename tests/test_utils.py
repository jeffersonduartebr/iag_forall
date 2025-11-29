import pytest
from app.app.utils.pricing import get_model_cost
from app.app.settings_dynamic import settings

def test_pricing_logic():
    # GPT-4o: Input $2.50/1M, Output $10.00/1M (Valores do seed)
    # 1000 tokens in = $0.0025
    # 1000 tokens out = $0.0100
    cost = get_model_cost("openai/gpt-4o", 1000, 1000)
    assert cost == pytest.approx(0.0125, rel=1e-3)

def test_pricing_fallback():
    # Modelo desconhecido deve retornar 0 ou default seguro
    cost = get_model_cost("modelo_inexistente", 1000, 1000)
    assert cost >= 0.0

def test_settings_defaults():
    """Garante que configurações críticas têm defaults."""
    assert settings.NSGA_W_QUALITY == 1.0
    assert settings.CACHE_TTL_DAYS == 7
    assert isinstance(settings.CANDIDATE_MODELS_LIST, list)