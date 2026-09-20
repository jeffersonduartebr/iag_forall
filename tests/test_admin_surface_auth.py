# Objective: Every admin route must reject an unauthenticated caller.
"""Admin authentication is a manual first line, not a dependency.

Each handler calls ``_auth(...)`` or ``resolve_admin_session(...)`` itself.
That works, and it is fragile in a specific way: a new route that forgets the
call is **public**, and nothing in the type system, the router or the review
diff points at it. Static analysis cannot settle it either — some handlers
delegate to a helper that authenticates, so grepping for the call produces
false positives.

The only reliable check is behavioural: call every ``/admin/*`` route with no
credentials and require a refusal. This test enumerates the routes from the app
itself, so a route added tomorrow is covered without anyone remembering to.
"""

import pytest
from fastapi.testclient import TestClient

REFUSAL = {401, 403}

#: O FastAPI valida o corpo do pedido **antes** de o handler correr, e a
#: autenticação do admin é a primeira linha do handler. Um POST/PUT com corpo
#: inválido devolve 422 sem nunca chegar à verificação de credenciais — o que
#: também significa que um chamador anónimo consegue sondar quais os campos
#: obrigatórios de cada rota. É um vazamento pequeno, mas é real, e impede que
#: o teste distinga "sem credenciais" de "corpo inválido".
#:
#: A propriedade que se pode afirmar sempre, e que é a que interessa, é mais
#: forte do que o código exacto: **nunca uma resposta de sucesso**.
NEVER_SUCCEEDS = REFUSAL | {404, 405, 422}

#: Rotas de autenticação: são a porta de entrada, têm de aceitar quem ainda não
#: tem credenciais. Devolvem 401/422 por outras razões, não por falta de sessão.
PUBLIC_ADMIN_ROUTES = {
    ("POST", "/admin/auth/login"),
    ("POST", "/admin/auth/expert-login"),
    ("POST", "/admin/auth/logout"),
}


def admin_routes():
    """(method, path) for every admin route the app actually serves."""
    from app import main

    out = []
    for route in main.app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/admin/"):
            continue
        for method in sorted(getattr(route, "methods", set()) - {"HEAD", "OPTIONS"}):
            if (method, path) not in PUBLIC_ADMIN_ROUTES:
                out.append((method, path))
    return sorted(out)


def concrete(path: str) -> str:
    """Substitute a value for every path parameter; auth runs before lookup."""
    out = []
    for segment in path.split("/"):
        out.append("test-id" if segment.startswith("{") else segment)
    return "/".join(out)


@pytest.fixture(scope="module")
def client():
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


def test_the_admin_surface_is_not_empty():
    """A bug in the enumeration would make every assertion below vacuous."""
    assert len(admin_routes()) > 50


@pytest.mark.parametrize("method,path", admin_routes())
def test_an_admin_route_never_succeeds_for_an_anonymous_caller(client, method, path):
    response = client.request(method, concrete(path), json={})
    assert response.status_code in NEVER_SUCCEEDS, (
        f"{method} {path} respondeu {response.status_code} sem credenciais"
    )


@pytest.mark.parametrize("method,path", admin_routes())
def test_an_admin_route_never_succeeds_with_a_wrong_token(client, method, path):
    response = client.request(method, concrete(path), json={}, headers={"X-Admin-Token": "errado"})
    assert response.status_code in NEVER_SUCCEEDS, (
        f"{method} {path} respondeu {response.status_code} com um token inválido"
    )


def bodyless_admin_routes():
    """Routes with no request body, where nothing can precede the auth check.

    Classificado pela assinatura e não pelo método. A validação precede a
    autenticação para **qualquer** entrada validada — corpo ou query — e há um
    DELETE com um parâmetro de query obrigatório para o qual o 422 chega antes
    de as credenciais serem vistas. Só os parâmetros de caminho não contam,
    porque o teste substitui-os por um valor.
    """
    from app import main

    out = []
    for route in main.app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/admin/"):
            continue
        dependant = getattr(route, "dependant", None)
        required = list(getattr(dependant, "body_params", None) or [])
        required += [q for q in (getattr(dependant, "query_params", None) or []) if q.required]
        if required:
            continue
        for method in sorted(getattr(route, "methods", set()) - {"HEAD", "OPTIONS"}):
            if (method, path) not in PUBLIC_ADMIN_ROUTES:
                out.append((method, path))
    return sorted(out)


@pytest.mark.parametrize("method,path", bodyless_admin_routes())
def test_a_bodyless_admin_route_refuses_explicitly(client, method, path):
    """Here 401/403 is required, not merely "not success".

    Sem corpo para validar, nada corre antes da verificação de credenciais, por
    isso um 422 ou um 500 aqui significaria que a rota chegou a executar.
    """
    response = client.request(method, concrete(path))
    assert response.status_code in REFUSAL, (
        f"{method} {path} respondeu {response.status_code} sem credenciais"
    )
