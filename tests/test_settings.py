from app.settings import Settings


def test_settings_defaults():
    """Testa se os defaults estão corretamente definidos."""
    s = Settings()
    assert s.OLLAMA_MODEL == "phi4:latest"
    assert s.MAX_TOKENS > 0
    assert isinstance(s.COSTS_USD_PER_1K, dict)


def test_settings_load_custom_cost(monkeypatch):
    """Carrega custo customizado de variável de ambiente."""
    monkeypatch.setenv("COST_PHI4_LATEST", "0.002")
    s = Settings()
    assert s.COSTS_USD_PER_1K["phi4:latest"] == 0.002
