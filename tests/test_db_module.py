"""Módulo `tests/test_db_module.py`: descreve responsabilidades e integrações deste arquivo."""

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
        """Classe `_Conn`: concentra responsabilidades de test db module."""
        def __enter__(self):
            """Executa enter."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Executa exit."""
            return False

        def execute(self, *_a, **_k):
            """Executa execute."""
            return None

    class _Pool:
        """Classe `_Pool`: concentra responsabilidades de test db module."""
        def size(self):
            """Executa size."""
            return 5

        def checkedin(self):
            """Executa checkedin."""
            return 3

        def checkedout(self):
            """Executa checkedout."""
            return 2

        def overflow(self):
            """Executa overflow."""
            return 0

        def invalidatedcount(self):
            """Executa invalidatedcount."""
            return 0

    class _Engine:
        """Classe `_Engine`: concentra responsabilidades de test db module."""
        def __init__(self):
            """Inicializa estado interno necessário para uso da classe."""
            self.pool = _Pool()
            self.disposed = False

        def connect(self):
            """Executa connect."""
            return _Conn()

        def dispose(self):
            """Executa dispose."""
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
