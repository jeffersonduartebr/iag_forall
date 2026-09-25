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


def test_migration_0008_creates_every_governance_table_the_runtime_knows():
    """The governance tables have one owner (Alembic 0008): a fresh production database used to lack them."""
    import importlib.util
    import re
    from pathlib import Path

    from app.roadmap_features import DDL_STATEMENTS

    path = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "0008_governance_tables.py"
    spec = importlib.util.spec_from_file_location("m0008", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    runtime = {re.search(r"CREATE TABLE IF NOT EXISTS (\w+)", ddl).group(1) for ddl in DDL_STATEMENTS}
    assert runtime <= set(mod.TABLES) and all("IF NOT EXISTS" in ddl for ddl in mod.DDL)
    assert mod.down_revision == "0007_decision_audit"
