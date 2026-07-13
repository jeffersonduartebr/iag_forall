# Objective: Tests for expert account service helpers.
"""Tests for expert account registration and password hashing."""

import pytest
from app.services.expert_accounts import (
    hash_password,
    normalize_email,
    validate_email,
    verify_password,
)


def test_normalize_and_validate_email():
    assert normalize_email("  Maria@Uni.Edu ") == "maria@uni.edu"
    assert validate_email("maria@uni.edu") is True
    assert validate_email("invalid") is False


def test_password_hash_roundtrip():
    stored = hash_password("senha-segura-123")
    assert verify_password("senha-segura-123", stored) is True
    assert verify_password("wrong", stored) is False


def test_register_expert_account(monkeypatch):
    # Importa o módulo atual (sys.modules) e faz patch/chamada por ele: outros
    # testes fazem sys.modules.pop(...), o que pode trocar o objeto de módulo e
    # deixar `register_expert_account` (importado na coleção) resolvendo nomes
    # contra um módulo diferente do que o patch por string alcançaria.
    from app.services import expert_accounts as ea

    created = {}

    def fake_create(**kwargs):
        created.update(kwargs)
        return 7

    monkeypatch.setattr(ea, "get_expert_account_by_email", lambda email: None)
    monkeypatch.setattr(ea, "_db_create_expert_account", fake_create)
    monkeypatch.setattr(
        ea,
        "get_expert_account_by_id",
        lambda account_id: {
            "id": account_id,
            "email": "maria@uni.edu",
            "display_name": "Maria",
            "phone": "+5511999999999",
            "enabled": 1,
        },
    )
    monkeypatch.setattr(ea, "upsert_expert_profile", lambda *a, **k: None)
    # get_expert_profile é importado tardiamente de roadmap_features (não é atributo
    # de módulo de expert_accounts), então o patch precisa mirar a origem.
    monkeypatch.setattr("app.roadmap_features.get_expert_profile", lambda uid: {"theme_ids": []})

    out = ea.register_expert_account(
        display_name="Maria",
        email="maria@uni.edu",
        phone="+5511999999999",
        password="senha1234",
    )
    assert out["email"] == "maria@uni.edu"
    assert created["email"] == "maria@uni.edu"
    assert ea.verify_password("senha1234", created["password_hash"])


def test_register_duplicate_email(monkeypatch):
    from app.services import expert_accounts as ea

    monkeypatch.setattr(ea, "get_expert_account_by_email", lambda email: {"id": 1, "email": email})
    with pytest.raises(ValueError, match="já cadastrado"):
        ea.register_expert_account(
            display_name="Maria",
            email="maria@uni.edu",
            phone=None,
            password="senha1234",
        )
