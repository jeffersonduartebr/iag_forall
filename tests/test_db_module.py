# Objective: Test coverage for db module behavior and regressions.
"""Test coverage for db module behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


from types import SimpleNamespace

import pytest

from app import db


def test_get_db_url_with_config():
    """Testa get db url with config."""
    url = db.get_db_url(
        {"host": "h", "port": 3306, "user": "u", "password": "p", "database": "d"}
    )
    assert url == "mysql+pymysql://u:p@h:3306/d"


def test_engine_singleton_close_and_lazy(monkeypatch):
    """Testa engine singleton close and lazy."""
    db._engine = None
    db._engine_initialized = False

    class _Conn:
        """Represent `_Conn` within this module.

The class groups the state and behavior required for Conn."""
        def __enter__(self):
            """Execute the enter routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Execute the exit routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return False

        def execute(self, *_a, **_k):
            """Execute the execute routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return None

    class _Pool:
        """Represent `_Pool` within this module.

The class groups the state and behavior required for Pool."""
        def size(self):
            """Execute the size routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return 5

        def checkedin(self):
            """Execute the checkedin routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return 3

        def checkedout(self):
            """Execute the checkedout routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return 2

        def overflow(self):
            """Execute the overflow routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return 0

        def invalidatedcount(self):
            """Execute the invalidatedcount routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return 0

    class _Engine:
        """Represent `_Engine` within this module.

The class groups the state and behavior required for Engine."""
        def __init__(self):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self.pool = _Pool()
            self.disposed = False

        def connect(self):
            """Execute the connect routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return _Conn()

        def dispose(self):
            """Execute the dispose routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            self.disposed = True

    eng = _Engine()
    monkeypatch.setattr(db, "create_engine", lambda *a, **k: eng)

    e1 = db.get_engine()
    e2 = db.get_engine()
    assert e1 is e2
    assert db.get_pool_stats()["status"] == "healthy"
    assert db.check_db_health()["healthy"] is True

    # lazy accessor
    assert db.engine() is eng
    assert db.engine.pool is eng.pool

    db.close_engine()
    assert eng.disposed is True


def test_engine_failure_and_pool_stats_error(monkeypatch):
    """Testa engine failure and pool stats error."""
    db._engine = None
    db._engine_initialized = False
    monkeypatch.setattr(db, "create_engine", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        db.get_engine()

    # second call after failed init
    with pytest.raises(RuntimeError):
        db.get_engine()

    db._engine = SimpleNamespace(pool=SimpleNamespace(size=lambda: (_ for _ in ()).throw(RuntimeError("x"))))
    stats = db.get_pool_stats()
    assert stats["status"].startswith("error:")

    # health failure path
    monkeypatch.setattr(db, "get_engine", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    h = db.check_db_health()
    assert h["healthy"] is False
    assert "db down" in h["error"]
