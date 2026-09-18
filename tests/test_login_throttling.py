# Objective: Test coverage for login throttling keyed by trusted client IP.
"""Admin login throttling: spoofed X-Forwarded-For does not reset the limit; state stays bounded."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import admin_auth_routes as auth
from app.utils.client_ip import parse_trusted_proxies, resolve_client_ip


@pytest.fixture(autouse=True)
def _clean_attempts():
    auth._LOGIN_ATTEMPTS.clear()
    yield
    auth._LOGIN_ATTEMPTS.clear()


def test_resolve_client_ip_honors_only_trusted_proxies():
    trusted = parse_trusted_proxies(" 10.0.0.1, 10.0.0.2 ,")
    assert trusted == {"10.0.0.1", "10.0.0.2"}
    assert resolve_client_ip("10.0.0.1", "203.0.113.5, 10.0.0.1", trusted) == "203.0.113.5"
    assert resolve_client_ip("198.51.100.7", "203.0.113.5", trusted) == "198.51.100.7"
    assert resolve_client_ip(None, "203.0.113.5", trusted) == "unknown"


def test_rotating_forwarded_for_does_not_reset_limit(monkeypatch):
    monkeypatch.setattr(auth, "_validate_ui_credentials", lambda u, p: False)
    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    statuses = [
        client.post(
            "/admin/auth/login",
            json={"username": "admin", "password": "errada"},
            headers={"X-Forwarded-For": f"203.0.113.{i}"},
        ).status_code
        for i in range(auth._MAX_ATTEMPTS + 1)
    ]
    assert statuses[:-1] == [401] * auth._MAX_ATTEMPTS
    assert statuses[-1] == 429  # a chave é o peer (sem proxy confiável), não o cabeçalho


def test_attempt_state_is_pruned_and_bounded(monkeypatch):
    monkeypatch.setattr(auth, "_MAX_TRACKED_CLIENTS", 3)
    for i in range(5):
        auth._record_failed_login(f"ip{i}")
    assert list(auth._LOGIN_ATTEMPTS) == ["ip2", "ip3", "ip4"]

    auth._LOGIN_ATTEMPTS["old"] = [0.0]  # tentativa fora da janela
    auth._check_rate_limit("old")
    assert "old" not in auth._LOGIN_ATTEMPTS
