# Objective: Four defects found while deciding what the frontend may call.
"""Analysing the API surface turned up four things worth pinning.

None of them is exotic. Each is an ordering or precedence mistake that reads as
correct until you ask who is allowed to reach the line below.
"""

import pytest

# ---------------------------------------------------------------------------
# The expert portal trusted a client-supplied identity over a signed one
# ---------------------------------------------------------------------------


class TestExpertIdentity:
    """``X-User-Id`` came *before* the JWT in ``_expert_id``.

    An authenticated expert only had to send a colleague's id to read and write
    as them: the profile, the email and phone, their assessments, and above all
    the human labels that calibrate the judge. A header the client chooses must
    not outrank a signed identity.
    """

    @staticmethod
    def expert_id(x_user_id, authorization=None, auth=None):
        from app.api.expert_routes import _expert_id

        return _expert_id(x_user_id, auth or {}, authorization)

    def test_a_signed_identity_wins_over_the_header(self, monkeypatch):
        monkeypatch.setattr(
            "app.api.expert_routes._roles_from_jwt",
            lambda authorization: ("perito-real", ["expert_reviewer"], {}),
        )
        assert self.expert_id("outro-perito", authorization="Bearer x") == "perito-real"

    def test_the_header_is_ignored_unless_the_deployment_trusts_headers(self, monkeypatch):
        monkeypatch.setattr("app.api.expert_routes._roles_from_jwt", lambda a: (None, [], {}))
        monkeypatch.setattr(
            "app.api.expert_routes.settings.get",
            lambda key, default=None: "0" if key == "TRUST_HEADER_ROLES" else default,
        )
        assert self.expert_id("outro-perito", auth={"username": "eu"}) == "eu"

    def test_the_header_is_honoured_where_headers_are_trusted(self, monkeypatch):
        """Internal services and identity federation present themselves this way."""
        monkeypatch.setattr("app.api.expert_routes._roles_from_jwt", lambda a: (None, [], {}))
        monkeypatch.setattr(
            "app.api.expert_routes.settings.get",
            lambda key, default=None: "1" if key == "TRUST_HEADER_ROLES" else default,
        )
        assert self.expert_id("servico-interno", auth={"username": "eu"}) == "servico-interno"

    def test_a_settings_backend_that_raises_does_not_trust_the_header(self, monkeypatch):
        """Failing open here would re-open the IDOR during a Redis outage."""
        monkeypatch.setattr("app.api.expert_routes._roles_from_jwt", lambda a: (None, [], {}))

        def explode(key, default=None):
            raise RuntimeError("redis em baixo")

        monkeypatch.setattr("app.api.expert_routes.settings.get", explode)
        assert self.expert_id("outro-perito", auth={"username": "eu"}) == "eu"


# ---------------------------------------------------------------------------
# /v1/query dropped the resolved auth context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v1_query_forwards_the_auth_context(monkeypatch):
    """It accepted ``auth`` and passed it nowhere.

    Downstream that meant the tenant budget and the owner recorded on a queued
    job were empty for anyone entering through /v1.
    """
    from app import main

    seen = {}

    async def _route_query(req, request=None, auth=None):
        seen["auth"] = auth
        return {"ok": True}

    monkeypatch.setattr(main, "route_query", _route_query)
    sentinel = object()
    await main.v1_route_query(object(), None, sentinel)
    assert seen["auth"] is sentinel


# ---------------------------------------------------------------------------
# Two smaller ones
# ---------------------------------------------------------------------------


def test_the_pricing_error_does_not_echo_the_database_exception():
    """A SQLAlchemy error carries the DSN: host, user and database name."""
    import inspect

    from app.api import admin_models_routes

    source = inspect.getsource(admin_models_routes.models_pricing)
    assert 'detail=f"Pricing unavailable: {exc}"' not in source
    assert "logger.error" in source


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app import main

    with TestClient(main.app) as c:
        yield c

    # O TestClient faz o startup real da app, e isso deixa dois resíduos que um
    # teste posterior herda: a cache LRU de settings preenchida com os valores
    # reais, e o semáforo de backpressure já construído a partir deles. Como o
    # semáforo lê MAX_CONCURRENT_REQUESTS uma só vez, no __init__, um teste que
    # monkeypatch `settings.get` e espere outro limite recebe o antigo.
    from app.middleware import backpressure
    from app.settings_dynamic import _invalidate_cache

    # Há dois singletons, não um: o atributo de classe `_instance` e o
    # `_backpressure` ao nível do módulo, que `get_backpressure()` consulta
    # primeiro. Repor só o da classe deixa o antigo em uso — foi exactamente
    # isso que fez este resíduo levar duas tentativas a diagnosticar.
    backpressure.BackpressureSemaphore._instance = None
    backpressure._backpressure = None
    _invalidate_cache()


#: O conftest fixa-o antes de a app ser importada.
ADMIN = {"X-Admin-Token": "test-admin-token-for-ci"}


@pytest.mark.parametrize("step", ["5s", "30s", "1m", "2h", "500ms"])
def test_a_sane_dashboard_step_is_accepted(client, step):
    """Não 422: a forma passa e o pedido segue para o handler."""
    assert client.get(f"/admin/dashboard/series?step={step}", headers=ADMIN).status_code != 422


@pytest.mark.parametrize("step", ["0s", "1", "abc", "5s; drop", "-1s", "1y", "999999s"])
def test_a_nonsense_dashboard_step_is_rejected(client, step):
    """Ia sem validação para o `query_range` do Prometheus. Não é PromQL — as
    cinco consultas são constantes — mas `step=1s` sobre 24 h pede 86 400
    pontos por série, cinco séries de cada vez, e é o chamador que escolhe."""
    assert client.get(f"/admin/dashboard/series?step={step}", headers=ADMIN).status_code == 422


def test_an_anonymous_caller_cannot_probe_the_validation(client):
    """A prova de que a autenticação passou a preceder a validação.

    Antes, um `step` inválido devolvia 422 a quem não tivesse credenciais —
    ou seja, dava para enumerar as regras de validação de cada rota sem
    autenticação nenhuma. Agora a resposta é a mesma, válido ou não: 401.
    """
    assert client.get("/admin/dashboard/series?step=lixo").status_code in {401, 403}
    assert client.get("/admin/dashboard/series?step=5s").status_code in {401, 403}


# ---------------------------------------------------------------------------
# Eval runs: authenticate before the lookup
# ---------------------------------------------------------------------------


def test_the_eval_handlers_authenticate_before_loading_the_run():
    """Loading first let an anonymous caller tell 404 from 401 — an existence
    oracle over run ids — and forced a database query with no credentials."""
    import inspect

    from app.api import eval_routes

    for name in (
        "get_eval",
        "get_eval_results",
        "get_eval_significance",
        "get_eval_academic_report",
        "get_eval_gate_report",
        "execute_eval",
    ):
        source = inspect.getsource(getattr(eval_routes, name))
        assert "_authorized_run(" in source, name
        assert "get_eval_run(run_id)" not in source, name


def test_the_helper_authorises_in_two_phases():
    """Swapping the two lines is not enough: authorisation is scoped to the
    run's tenant, and the tenant is only known after the lookup."""
    import inspect

    from app.api.eval_routes import _authorized_run

    source = inspect.getsource(_authorized_run)
    first = source.index("require_admin_or_role(**kwargs)")
    lookup = source.index("get_eval_run(run_id)")
    scoped = source.index('tenant_id=run.get("tenant_id")')
    assert first < lookup < scoped
