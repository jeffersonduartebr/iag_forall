import pytest
from app import bandits


def test_bandit_initial_selection():
    """Verifica se o bandit consegue selecionar um modelo válido."""
    candidates = ["phi4", "gemini-2.0-flash"]
    chosen = bandits.select_model(candidates, query_text="Teste")
    assert chosen in candidates


def test_bandit_update_and_retrieve_stats():
    """Atualiza o modelo e verifica se as estatísticas foram salvas."""
    bandits.reset_bandits()
    bandits.update_model("phi4", "Pergunta X", reward=0.75)
    stats = bandits.get_bandit_stats()
    assert "phi4" in stats
    assert stats["phi4"]["avg_reward"] >= 0.0


def test_bandit_explore_exploit(monkeypatch):
    """Força modo exploit (ε = 0) para testar comportamento determinístico."""
    bandits.reset_bandits()
    candidates = ["phi4", "gemini"]
    monkeypatch.setattr(bandits, "EPSILON", 0.0)
    chosen = bandits.select_model(candidates, "Pergunta Y")
    assert chosen in candidates


def test_bandit_reset_clears_stats():
    """Verifica se o reset realmente limpa as estatísticas."""
    bandits.reset_bandits()
    bandits.update_model("phi4", "Pergunta", 0.5)
    bandits.reset_bandits()
    stats = bandits.get_bandit_stats()
    assert stats == {}
