import pytest
from app import providers


def test_safe_call_model_success(monkeypatch):
    """Mocka chamada de modelo e valida retorno correto."""
    def fake_completion(*args, **kwargs):
        return {"choices": [{"message": {"content": "Resposta teste"}}]}, {"latency": 0.05}

    monkeypatch.setattr(providers, "completion", fake_completion)

    resp, meta = providers.safe_call_model(
        model="phi4:latest",
        prompt="Explique o NSGA-II",
        max_tokens=16,
        temperature=0.5,
    )

    assert isinstance(resp, str)
    assert "Resposta" in resp
    assert "latency" in meta


def test_safe_call_model_exception(monkeypatch):
    """Simula erro no provedor e espera retorno vazio e log de falha."""
    def fake_completion(*args, **kwargs):
        raise RuntimeError("Falha simulada")

    monkeypatch.setattr(providers, "completion", fake_completion)
    resp, meta = providers.safe_call_model("phi4:latest", "pergunta", max_tokens=4)
    assert resp == ""
    assert isinstance(meta, dict)
