# Objective: A database outage must not make /query unavailable.
"""The governance pre-checks read MariaDB and propagated their exceptions.

There is no exception handler in this project, so a database outage turned
every request into ``500 {"detail":"Internal Server Error"}``. The contrast is
what makes it worth fixing: the LLM call needs no database at all — the request
failed on a governance pre-check, not on the work it asked for.
"""

import pytest
from app.services.governance_degradation import resolve_budget, resolve_policy


@pytest.fixture
def fail_closed(monkeypatch):
    def _set(value):
        monkeypatch.setattr(
            "app.services.governance_degradation.settings.get",
            lambda key, default=None: value if key == "TENANT_BUDGET_FAIL_CLOSED" else default,
        )

    return _set


def test_a_healthy_budget_result_passes_through():
    assert resolve_budget({"exceeded": False}, "acme") == {"exceeded": False}


def test_a_failed_budget_read_degrades_by_default(fail_closed):
    fail_closed("0")
    assert resolve_budget(RuntimeError("mariadb em baixo"), "acme") is None


def test_fail_closed_reverses_that(fail_closed):
    """For deployments where an overrun costs more than an outage."""
    fail_closed("1")
    with pytest.raises(RuntimeError, match="mariadb em baixo"):
        resolve_budget(RuntimeError("mariadb em baixo"), "acme")


def test_a_settings_backend_that_is_also_down_still_degrades(monkeypatch):
    """Redis and MariaDB usually fail together; this must not raise twice."""

    def explode(key, default=None):
        raise RuntimeError("redis em baixo")

    monkeypatch.setattr("app.services.governance_degradation.settings.get", explode)
    assert resolve_budget(RuntimeError("mariadb em baixo"), "acme") is None


def test_a_healthy_policy_passes_through():
    assert resolve_policy({"id": "p1"}) == {"id": "p1"}


def test_a_failed_policy_read_falls_back_to_the_defaults():
    """A missing policy means the defaults apply, which is what a deployment
    with no configured policy already does."""
    assert resolve_policy(RuntimeError("mariadb em baixo")) is None


def test_the_degradation_is_counted():
    from app.services.governance_degradation import GOVERNANCE_DEGRADED_TOTAL

    before = GOVERNANCE_DEGRADED_TOTAL.labels(check="active_policy")._value.get()
    resolve_policy(RuntimeError("boom"))
    assert GOVERNANCE_DEGRADED_TOTAL.labels(check="active_policy")._value.get() == before + 1


def test_degraded_budget_pass_serves_the_request_instead_of_crashing():
    """Production regression (2026-09-24): resolve_budget's fail-open None hit ``None.allowed`` -> HTTP 500."""
    from types import SimpleNamespace

    from app.services.query_runtime import _raise_if_budget_exceeded

    _raise_if_budget_exceeded(SimpleNamespace(modality="text"), None)
    _raise_if_budget_exceeded(SimpleNamespace(modality="text"), SimpleNamespace(allowed=True))


def test_deploy_step_fails_the_deploy_when_tables_cannot_be_created(monkeypatch):
    import pytest

    from app import governance_ddl

    class _Falha:
        def begin(self):
            raise RuntimeError("sem banco")

    monkeypatch.setattr(governance_ddl, "get_engine", lambda: _Falha())
    with pytest.raises(RuntimeError):
        governance_ddl.criar_tabelas_de_governanca()


def test_deploy_step_runs_every_governance_ddl(monkeypatch):
    from contextlib import contextmanager

    from app import governance_ddl

    executados = []

    class _Motor:
        @contextmanager
        def begin(self):
            yield type("C", (), {"execute": lambda self, sql: executados.append(str(sql))})()

    monkeypatch.setattr(governance_ddl, "get_engine", lambda: _Motor())
    governance_ddl.criar_tabelas_de_governanca()
    assert any("tenant_budgets" in sql for sql in executados)
    assert len(executados) == len(governance_ddl.DDL_STATEMENTS) + 1
