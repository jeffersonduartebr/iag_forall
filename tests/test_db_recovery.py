# Objective: A transient database outage must not disable the process for good.
"""``get_engine`` used to poison itself permanently after one failure.

``_engine_initialized`` was set before the ``try`` and never reset except by
``close_engine``, so a RuntimeError was raised on every subsequent call for the
life of the process. Two seconds of MariaDB unavailability at process start —
a restart, a rolling deploy — and the API stayed without a database until the
container was restarted. The trigger is easy to hit: ``settings_dynamic``
executes DDL at *import* time inside a ``try/except`` that only warns.
"""

import pytest

from app import db as db_module

#: O conftest substitui `app.db.get_engine` por um fake em TODOS os testes.
#: Aqui é precisamente a função real que está sob teste, por isso é guardada
#: no import do módulo — antes de qualquer fixture correr — e reposta.
_REAL_GET_ENGINE = db_module.get_engine


@pytest.fixture(autouse=True)
def clean_engine_state(monkeypatch):
    monkeypatch.setattr(db_module, "get_engine", _REAL_GET_ENGINE)
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_engine_initialized", False)
    monkeypatch.setattr(db_module, "_engine_retry_after", 0.0)
    yield


def make_engine_fail(monkeypatch, fail: list):
    """create_engine raises while ``fail[0]`` is true."""

    class _Engine:
        def connect(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            return None

    def _create_engine(*args, **kwargs):
        if fail[0]:
            raise RuntimeError("mariadb indisponível")
        return _Engine()

    monkeypatch.setattr(db_module, "create_engine", _create_engine)
    monkeypatch.setattr(db_module, "get_db_url", lambda: "mysql+pymysql://x/y")


def test_a_transient_outage_is_retried_once_the_backoff_passes(monkeypatch):
    fail = [True]
    make_engine_fail(monkeypatch, fail)
    monkeypatch.setattr(db_module, "ENGINE_RETRY_BACKOFF_S", 5.0)

    fake_now = [1000.0]
    monkeypatch.setattr(db_module.time, "monotonic", lambda: fake_now[0])

    with pytest.raises(RuntimeError, match="mariadb indisponível"):
        _REAL_GET_ENGINE()

    # Dentro do backoff: recusa depressa, sem martelar o servidor.
    with pytest.raises(RuntimeError, match="nova tentativa"):
        _REAL_GET_ENGINE()

    # O MariaDB recupera e o backoff passa.
    fail[0] = False
    fake_now[0] += 6.0
    assert _REAL_GET_ENGINE() is not None, "o engine ficou envenenado para sempre"


def test_the_backoff_refuses_without_touching_the_database(monkeypatch):
    attempts = []

    def _create_engine(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("mariadb indisponível")

    monkeypatch.setattr(db_module, "create_engine", _create_engine)
    monkeypatch.setattr(db_module, "get_db_url", lambda: "mysql+pymysql://x/y")
    monkeypatch.setattr(db_module.time, "monotonic", lambda: 1000.0)

    with pytest.raises(RuntimeError):
        _REAL_GET_ENGINE()
    for _ in range(5):
        with pytest.raises(RuntimeError):
            _REAL_GET_ENGINE()

    assert len(attempts) == 1, "o backoff não impediu novas tentativas"


def test_close_engine_clears_the_backoff(monkeypatch):
    monkeypatch.setattr(db_module, "_engine_retry_after", 99999.0)
    db_module.close_engine()
    assert db_module._engine_retry_after == 0.0


def test_the_connection_carries_socket_timeouts():
    """PyMySQL defaults to read_timeout=None: a server that accepts the
    connection but never answers blocks the thread indefinitely."""
    args = db_module._connect_args()
    assert args["connect_timeout"] > 0
    assert args["read_timeout"] > 0
    assert args["write_timeout"] > 0


def test_the_engine_is_created_with_those_timeouts(monkeypatch):
    captured = {}

    def _create_engine(url, **kwargs):
        captured.update(kwargs)

        class _E:
            def connect(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, *a, **k):
                return None

        return _E()

    monkeypatch.setattr(db_module, "create_engine", _create_engine)
    monkeypatch.setattr(db_module, "get_db_url", lambda: "mysql+pymysql://x/y")
    _REAL_GET_ENGINE()
    assert "read_timeout" in captured["connect_args"]
