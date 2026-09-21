# Objective: A session token must not be able to disable the platform's authentication.
"""``PUT /admin/settings`` accepted any catalog key, including the auth domain.

Anyone holding an admin *session* could set ``REQUIRE_API_AUTH=0``,
``TRUST_HEADER_ROLES=1``, ``ADMIN_UI_CORS_ORIGINS=*`` or rotate ``JWT_SECRET``
— that is, disable the authentication of the whole installation from the
browser. A stolen session was enough.

The split is by *authorization*, not by behaviour: what counts as known, what
needs a restart and how a value is serialised are identical on both sides. Only
the credential required differs, and it mirrors a distinction the codebase
already makes for the RBAC routes.
"""

import pytest
from app.config.settings_catalog import SETTINGS_BY_DOMAIN, is_security_setting, split_by_security
from fastapi.testclient import TestClient

ADMIN = {"X-Admin-Token": "test-admin-token-for-ci"}


@pytest.fixture(scope="module")
def client():
    from app import main

    with TestClient(main.app) as c:
        yield c

    from app.middleware import backpressure
    from app.settings_dynamic import _invalidate_cache

    backpressure.BackpressureSemaphore._instance = None
    backpressure._backpressure = None
    _invalidate_cache()


# ---------------------------------------------------------------------------
# What counts as a security key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "REQUIRE_API_AUTH",
        "TRUST_HEADER_ROLES",
        "JWT_SECRET",
        "ADMIN_UI_CORS_ORIGINS",
        "ENV",
        "TRUSTED_PROXY_IPS",
        "ROADMAP_AUTO_DDL",
    ],
)
def test_the_auth_domain_is_security(key):
    assert is_security_setting(key)


def test_a_credential_outside_the_auth_domain_is_still_security():
    """``REDIS_PASSWORD`` lives in the ``redis`` domain, not in ``auth``.

    This is why the rule is "auth domain OR credential" and not just the
    domain: a hand-written list would have missed it.
    """
    assert "REDIS_PASSWORD" in SETTINGS_BY_DOMAIN["redis"]
    assert is_security_setting("REDIS_PASSWORD")


@pytest.mark.parametrize(
    "key",
    [
        "BANDIT_EPSILON",
        "NSGA_W_QUALITY",
        "CACHE_THRESHOLD",
        "CANDIDATE_MODELS_LIST",
        "JUDGES_ENABLED",
    ],
)
def test_an_operational_key_is_not_security(key):
    assert not is_security_setting(key)


@pytest.mark.parametrize(
    "key",
    [
        "MAX_TOKENS_DEFAULT",
        "RAG_CONTEXT_TOKEN_BUDGET",
        "RAG_FULL_CONTEXT_TOKEN_BUDGET",
        "RAG_LIGHT_CONTEXT_TOKEN_BUDGET",
        "REWARD_LATENCY_TOKENS_PER_S",
        "REWARD_DEFAULT_COMPLETION_TOKENS",
        "ROUTER_SIMPLE_QUERY_MAX_TOKENS",
        "RAG_SIMPLE_QUERY_BYPASS_ENABLED",
    ],
)
def test_a_token_count_is_not_a_credential(key):
    """The predicate matched markers as *substrings*, so these eight were
    hidden from the operator in ``/admin/settings`` as if they were secrets —
    ``RAG_SIMPLE_QUERY_BYPASS_ENABLED`` because "BYPASS" contains "PASS"."""
    from app.config.secrets_redaction import is_secret_key

    assert not is_secret_key(key)
    assert not is_security_setting(key)


@pytest.mark.parametrize(
    "key", ["ADMIN_TOKEN", "JWT_SECRET", "API_KEYS", "REDIS_PASSWORD", "DB_PASS", "OPENROUTER_API_KEY"]
)
def test_a_real_credential_is_still_recognised(key):
    """Narrowing the rule must not un-hide anything that is genuinely secret."""
    from app.config.secrets_redaction import is_secret_key

    assert is_secret_key(key)


def test_the_operator_can_see_their_token_budgets():
    """The regression this closes, end to end over the redaction."""
    from app.config.secrets_redaction import redact_secrets

    out = redact_secrets({"MAX_TOKENS_DEFAULT": 2048, "ADMIN_TOKEN": "segredo"})
    assert out["MAX_TOKENS_DEFAULT"] == 2048
    assert out["ADMIN_TOKEN"] == "***REDACTED***"


def test_one_rule_not_two():
    """Two copies of the predicate is how a key ends up masked in one place and
    stored in the clear in another."""
    from app.config.secrets_redaction import is_secret_key
    from app.config.settings_encryption import is_secret

    assert is_secret is is_secret_key


# ---------------------------------------------------------------------------
# Splitting a batch
# ---------------------------------------------------------------------------


def test_a_mixed_batch_is_split_preserving_order():
    operational, security = split_by_security(
        ["BANDIT_EPSILON", "JWT_SECRET", "CACHE_THRESHOLD", "REQUIRE_API_AUTH"]
    )
    assert operational == ["BANDIT_EPSILON", "CACHE_THRESHOLD"]
    assert security == ["JWT_SECRET", "REQUIRE_API_AUTH"]


def test_an_empty_batch_splits_cleanly():
    assert split_by_security([]) == ([], [])


# ---------------------------------------------------------------------------
# The endpoints
# ---------------------------------------------------------------------------


def test_the_operational_endpoint_refuses_a_security_key(client):
    """The whole point: an admin session cannot disable the authentication."""
    response = client.put(
        "/admin/settings", headers=ADMIN, json={"settings": {"REQUIRE_API_AUTH": "0"}}
    )
    assert response.status_code == 403
    body = response.json()["detail"]
    assert body["error"] == "security_settings_refused"
    assert body["keys"] == ["REQUIRE_API_AUTH"]
    assert "/admin/settings/security" in body["message"]


def test_a_mixed_batch_is_refused_whole(client):
    """Applying half a batch would be the worst of the three outcomes."""
    response = client.put(
        "/admin/settings",
        headers=ADMIN,
        json={"settings": {"BANDIT_EPSILON": "0.2", "JWT_SECRET": "novo"}},
    )
    assert response.status_code == 403
    assert response.json()["detail"]["keys"] == ["JWT_SECRET"]


def test_the_security_endpoint_refuses_an_operational_key(client):
    response = client.put(
        "/admin/settings/security", headers=ADMIN, json={"settings": {"BANDIT_EPSILON": "0.2"}}
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "operational_settings_refused"


def test_the_security_endpoint_rejects_a_session_token(client):
    """It takes the master token, not an admin JWT — the same distinction the
    RBAC routes already make, and for the same reason."""
    response = client.put(
        "/admin/settings/security",
        headers={"Authorization": "Bearer uma-sessao-qualquer"},
        json={"settings": {"REQUIRE_API_AUTH": "1"}},
    )
    assert response.status_code in {401, 403}


def test_both_endpoints_refuse_an_anonymous_caller(client):
    for path in ("/admin/settings", "/admin/settings/security"):
        assert client.put(path, json={"settings": {}}).status_code in {401, 403}
