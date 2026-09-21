# Objective: The CORS allowlist must take effect without restarting the process.
"""``ADMIN_UI_CORS_ORIGINS`` was frozen at import time.

Starlette's ``CORSMiddleware`` reads ``allow_origins`` once, in its
constructor, and the middleware stack is built while ``main`` is imported.
Changing the setting through ``/admin/settings`` persisted it, invalidated the
caches, published the reload — and had no effect at all until someone restarted
the container. The catalog marked it ``requires_restart``, which was honest but
also an admission.
"""

import pytest
from app.middleware.cors import DynamicCORSMiddleware, parse_origins
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

BOOT = ["https://arranque.exemplo"]


@pytest.fixture
def app_with_origins(monkeypatch):
    """A minimal app whose allowlist the test can change mid-flight."""
    configured = {"value": "https://permitida.exemplo"}

    monkeypatch.setattr(
        "app.settings_dynamic.settings.get",
        lambda key, default=None: configured["value"] if key == "ADMIN_UI_CORS_ORIGINS" else default,
    )

    async def _ok(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/ping", _ok)])
    app.add_middleware(
        DynamicCORSMiddleware,
        fallback_origins=BOOT,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return TestClient(app), configured


def allowed_origin(response):
    return response.headers.get("access-control-allow-origin")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_a_comma_separated_list_is_split():
    assert parse_origins("https://a, https://b ,https://c") == ["https://a", "https://b", "https://c"]


@pytest.mark.parametrize("raw", ["", None, "   ", ",,,"])
def test_an_empty_configuration_yields_no_origins(raw):
    assert parse_origins(raw) == []


# ---------------------------------------------------------------------------
# The property that was missing
# ---------------------------------------------------------------------------


def test_a_change_takes_effect_without_a_restart(app_with_origins):
    """This is the whole point of the module."""
    client, configured = app_with_origins

    before = client.get("/ping", headers={"Origin": "https://nova.exemplo"})
    assert allowed_origin(before) is None

    configured["value"] = "https://permitida.exemplo,https://nova.exemplo"

    after = client.get("/ping", headers={"Origin": "https://nova.exemplo"})
    assert allowed_origin(after) == "https://nova.exemplo"


def test_an_origin_removed_at_runtime_stops_being_allowed(app_with_origins):
    """Revoking has to work as fast as granting, or the setting is a trap."""
    client, configured = app_with_origins
    assert allowed_origin(client.get("/ping", headers={"Origin": "https://permitida.exemplo"}))

    configured["value"] = "https://outra.exemplo"
    assert allowed_origin(client.get("/ping", headers={"Origin": "https://permitida.exemplo"})) is None


def test_a_disallowed_origin_gets_no_header(app_with_origins):
    client, _ = app_with_origins
    assert allowed_origin(client.get("/ping", headers={"Origin": "https://intrusa.exemplo"})) is None


def test_a_wildcard_allows_everything(app_with_origins):
    client, configured = app_with_origins
    configured["value"] = "*"
    assert allowed_origin(client.get("/ping", headers={"Origin": "https://qualquer.exemplo"}))


# ---------------------------------------------------------------------------
# The cache-poisoning guard
# ---------------------------------------------------------------------------


def test_the_response_varies_on_origin(app_with_origins):
    """Without ``Vary: Origin`` a shared cache would hand one origin's response
    to another — a dynamic allowlist makes that a live bug, not a theoretical
    one."""
    client, _ = app_with_origins
    response = client.get("/ping", headers={"Origin": "https://permitida.exemplo"})
    assert "origin" in response.headers.get("vary", "").lower()


def test_the_preflight_also_varies_on_origin(app_with_origins):
    client, _ = app_with_origins
    response = client.options(
        "/ping",
        headers={"Origin": "https://permitida.exemplo", "Access-Control-Request-Method": "GET"},
    )
    assert "origin" in response.headers.get("vary", "").lower()
    assert allowed_origin(response) == "https://permitida.exemplo"


def test_a_preflight_from_a_disallowed_origin_is_not_granted(app_with_origins):
    client, _ = app_with_origins
    response = client.options(
        "/ping",
        headers={"Origin": "https://intrusa.exemplo", "Access-Control-Request-Method": "GET"},
    )
    assert allowed_origin(response) is None


# ---------------------------------------------------------------------------
# Degrading
# ---------------------------------------------------------------------------


def test_an_unreadable_setting_falls_back_to_the_boot_value(monkeypatch):
    """Not to an empty list: a Redis outage must not become "no origin is
    allowed", which would break the whole frontend over an infrastructure
    failure."""

    def explode(key, default=None):
        raise RuntimeError("redis em baixo")

    monkeypatch.setattr("app.settings_dynamic.settings.get", explode)

    async def _ok(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/ping", _ok)])
    app.add_middleware(DynamicCORSMiddleware, fallback_origins=BOOT, allow_credentials=True)
    client = TestClient(app)

    assert allowed_origin(client.get("/ping", headers={"Origin": BOOT[0]})) == BOOT[0]
    assert allowed_origin(client.get("/ping", headers={"Origin": "https://intrusa.exemplo"})) is None


def test_an_empty_configuration_falls_back_too(app_with_origins):
    """An operator who blanks the field has almost certainly made a mistake;
    locking themselves out of the admin UI is a harsh way to find out."""
    client, configured = app_with_origins
    configured["value"] = ""
    assert allowed_origin(client.get("/ping", headers={"Origin": BOOT[0]})) == BOOT[0]


# ---------------------------------------------------------------------------
# Where it may be changed from
# ---------------------------------------------------------------------------


def test_the_setting_is_now_runtime_mutable():
    from app.config.settings_catalog import metadata_for

    assert metadata_for("ADMIN_UI_CORS_ORIGINS")["mutability"] == "runtime_safe"


def test_but_it_still_needs_the_master_token():
    """Runtime-mutable and security-classified is the right pair: the browser
    must not be able to widen its own CORS policy."""
    from app.config.settings_catalog import is_security_setting

    assert is_security_setting("ADMIN_UI_CORS_ORIGINS")
