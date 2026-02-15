"""Módulo `tests/test_runtime_state.py`: descreve responsabilidades e integrações deste arquivo."""

from app import runtime_state as rs


def test_reset_runtime_state_calls_all(monkeypatch):
    """Testa reset runtime state calls all."""
    calls = []
    monkeypatch.setattr(rs, "reset_provider_runtime_state", lambda: calls.append("provider"))
    monkeypatch.setattr(rs, "reset_reliability_runtime_state", lambda: calls.append("reliability"))
    monkeypatch.setattr(rs, "reset_router_runtime_state", lambda: calls.append("router"))
    monkeypatch.setattr(rs, "reset_bandits_runtime_state", lambda: calls.append("bandits"))
    monkeypatch.setattr(rs, "reset_vectorstore_runtime_state", lambda: calls.append("vectorstore"))

    rs.reset_runtime_state()
    assert calls == ["provider", "reliability", "router", "bandits", "vectorstore"]
