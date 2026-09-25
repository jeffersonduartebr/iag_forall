# Objective: Behavioural coverage for expert account registration, login and admin updates.
"""expert_accounts: validation errors, authentication outcomes and admin update paths."""

import pytest


@pytest.fixture
def ea(monkeypatch):
    # Importado no teste: outros testes fazem sys.modules.pop e trocam o objeto de módulo.
    from app.services import expert_accounts

    monkeypatch.setattr(expert_accounts, "_PBKDF2_ITERATIONS", 1000)  # hash rápido nos testes
    monkeypatch.setattr("app.roadmap_features.get_expert_profile", lambda email: {"theme_ids": ["hist"]})
    return expert_accounts


@pytest.fixture
def db(ea, monkeypatch):
    """Tiny in-memory account table behind the module's DB helpers."""
    rows = {}
    profiles = []

    def create(**kw):
        account_id = len(rows) + 1
        rows[account_id] = {"id": account_id, "enabled": 1, **kw}
        return account_id

    def update(account_id, **kw):
        rows[account_id].update({k: v for k, v in kw.items() if v is not None})

    monkeypatch.setattr(ea, "_db_create_expert_account", create)
    monkeypatch.setattr(ea, "_db_update_expert_account", update)
    monkeypatch.setattr(ea, "_db_list_expert_accounts", lambda: list(rows.values()))
    monkeypatch.setattr(ea, "get_expert_account_by_id", lambda i: rows.get(i))
    monkeypatch.setattr(
        ea, "get_expert_account_by_email", lambda e: next((r for r in rows.values() if r["email"] == e), None)
    )
    monkeypatch.setattr(ea, "upsert_expert_profile", lambda email, **kw: profiles.append((email, kw)))
    return rows, profiles


@pytest.mark.parametrize(
    ("email", "password", "message"),
    [("sem-arroba", "senha1234", "E-mail inválido"), ("", "senha1234", "E-mail inválido"), ("a@x.org", "curta", "8")],
)
def test_register_rejects_invalid_input(ea, db, email, password, message):
    with pytest.raises(ValueError, match=message):
        ea.register_expert_account(display_name="Ana", email=email, phone=None, password=password)
    assert db[0] == {}


def test_register_normalizes_and_hides_hash(ea, db):
    rows, profiles = db
    out = ea.register_expert_account(display_name="  Ana  ", email=" A@X.org ", phone="  ", password="senha1234")
    assert rows[1]["email"] == "a@x.org" and rows[1]["display_name"] == "Ana" and rows[1]["phone"] is None
    assert "password_hash" not in out
    assert out["theme_ids"] == ["hist"] and out["enabled"] is True
    assert profiles == [("a@x.org", {"display_name": "Ana", "theme_ids": []})]


def test_register_when_account_cannot_be_reread(ea, db, monkeypatch):
    monkeypatch.setattr(ea, "get_expert_account_by_id", lambda i: None)
    out = ea.register_expert_account(display_name="Ana", email="a@x.org", phone=None, password="senha1234")
    assert out == {"id": 1, "email": "a@x.org"}


def test_authenticate_outcomes(ea, db):
    rows, _ = db
    ea.register_expert_account(display_name="Ana", email="a@x.org", phone=None, password="senha1234")
    assert ea.authenticate_expert_account(" A@x.org", "senha1234")["id"] == 1
    assert ea.authenticate_expert_account("a@x.org", "errada99") is None
    assert ea.authenticate_expert_account("b@x.org", "senha1234") is None
    rows[1]["enabled"] = 0
    assert ea.authenticate_expert_account("a@x.org", "senha1234") is None


@pytest.mark.parametrize("stored", ["", "md5$1$s$abc", "pbkdf2_sha256$nao-numero$s$abc", None])
def test_verify_password_rejects_malformed_hashes(ea, stored):
    assert ea.verify_password("qualquer", stored) is False


def test_list_accounts_public_strips_secrets(ea, db):
    ea.register_expert_account(display_name="Ana", email="a@x.org", phone="1", password="senha1234")
    items = ea.list_expert_accounts_public()
    assert len(items) == 1
    assert "password_hash" not in items[0] and items[0]["phone"] == "1"


def test_public_view_of_nothing_is_empty(ea):
    assert ea._public_account_view(None) == {}


def test_admin_update_missing_account(ea, db):
    assert ea.update_expert_account_admin(42, display_name="X") is None


def test_admin_update_rejects_short_password(ea, db):
    rows, _ = db
    ea.register_expert_account(display_name="Ana", email="a@x.org", phone=None, password="senha1234")
    before = rows[1]["password_hash"]
    with pytest.raises(ValueError, match="8"):
        ea.update_expert_account_admin(1, password="curta")
    assert rows[1]["password_hash"] == before


def test_admin_update_changes_fields_and_profile(ea, db):
    rows, profiles = db
    ea.register_expert_account(display_name="Ana", email="a@x.org", phone=None, password="senha1234")
    out = ea.update_expert_account_admin(
        1, display_name=" Ana Maria ", phone=" 55 ", password="nova-senha", enabled=False
    )
    assert out["display_name"] == "Ana Maria" and out["phone"] == "55" and out["enabled"] is False
    assert ea.verify_password("nova-senha", rows[1]["password_hash"])
    assert profiles[-1] == ("a@x.org", {"display_name": "Ana Maria"})


def test_admin_update_without_name_leaves_profile_alone(ea, db):
    _, profiles = db
    ea.register_expert_account(display_name="Ana", email="a@x.org", phone=None, password="senha1234")
    ea.update_expert_account_admin(1, enabled=True)
    assert len(profiles) == 1  # só o do registo


def test_admin_update_account_vanishes_after_write(ea, db, monkeypatch):
    rows, _ = db
    ea.register_expert_account(display_name="Ana", email="a@x.org", phone=None, password="senha1234")
    lookups = iter([rows[1], None])
    monkeypatch.setattr(ea, "get_expert_account_by_id", lambda i: next(lookups))
    assert ea.update_expert_account_admin(1, enabled=False) is None
