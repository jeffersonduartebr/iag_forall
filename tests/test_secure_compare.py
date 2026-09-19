# Objective: Regression tests for timing-safe secret comparison with non-ASCII input.
"""Non-ASCII tokens/passwords must be rejected (or accepted) instead of raising TypeError -> 500."""

from types import SimpleNamespace

from app.api import admin_auth_routes, auth
from app.utils.secure_compare import secret_equals


def test_secret_equals_handles_any_text():
    assert secret_equals("abc", "abc") is True
    assert secret_equals("senhá", "senhá") is True
    assert secret_equals("ç", "abc") is False
    assert secret_equals("", "abc") is False


def test_api_key_auth_rejects_non_ascii_header(monkeypatch):
    fake = SimpleNamespace(ADMIN_TOKEN="admin-token", ADMIN_TOKEN_PREVIOUS="", get=lambda key, default=None: "k1,k2")
    monkeypatch.setattr(auth, "settings", fake)
    assert auth._auth_from_api_key("vÂ\x9b®»") is None  # antes: TypeError (500)
    assert auth._auth_from_api_key("k2").method == "api_key"
    assert auth._auth_from_api_key("admin-token").roles == ["admin"]


def test_ui_login_accepts_non_ascii_password(monkeypatch):
    values = {"ADMIN_UI_USERNAME": "admin", "ADMIN_UI_PASSWORD": "senhá-forte"}
    monkeypatch.setattr(
        admin_auth_routes, "settings", SimpleNamespace(get=lambda key, default=None: values.get(key, default))
    )
    assert admin_auth_routes._validate_ui_credentials("admin", "senhá-forte") is True
    assert admin_auth_routes._validate_ui_credentials("admín", "x") is False
