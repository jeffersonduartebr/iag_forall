# Objective: Test coverage for settings dynamic runtime behavior and regressions.
"""Focused tests for runtime behaviors in settings_dynamic."""

from __future__ import annotations

from types import SimpleNamespace

from app import settings_dynamic as sd


class _Ctx:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        return False


class _Metric:
    def __init__(self):
        self.values = []

    def set(self, value):
        self.values.append(value)


def test_redis_db_helpers_and_runtime_set(monkeypatch):
    """Redis/DB accessors and DynamicSettings.set should update all layers."""
    redis_calls = {"set": [], "publish": []}

    class _Redis:
        def get(self, key):
            return b"123" if key.endswith("A") else None

        def set(self, key, value):
            redis_calls["set"].append((key, value))

        def publish(self, channel, key):
            redis_calls["publish"].append((channel, key))

    monkeypatch.setattr(sd, "_get_rds", lambda: _Redis())

    assert sd._get_from_redis("A") == "123"

    db_exec = []

    class _Conn:
        def execute(self, stmt, params=None):
            db_exec.append((str(stmt), params))
            return SimpleNamespace(fetchone=lambda: ("db-value",))

    monkeypatch.setattr(sd, "engine", SimpleNamespace(connect=lambda: _Ctx(_Conn()), begin=lambda: _Ctx(_Conn())))
    assert sd._get_from_db("A") == "db-value"

    invalidated = []
    monkeypatch.setattr(sd, "_invalidate_cache", lambda: invalidated.append(True))
    settings = sd.DynamicSettings()
    settings.set("MAX_TOKENS_DEFAULT", "4096", actor="test", source="unit")

    assert any("INSERT INTO settings_dynamic" in sql for sql, _ in db_exec)
    assert redis_calls["set"][-1] == ("settings:MAX_TOKENS_DEFAULT", "4096")
    assert redis_calls["publish"][-1] == (sd.REDIS_RELOAD_CHANNEL, "MAX_TOKENS_DEFAULT")
    assert invalidated


def test_settings_helpers_metadata_and_pool_metrics(monkeypatch):
    """Settings helpers should cover metadata, runtime mutability, and DB pool metrics."""
    settings = sd.DynamicSettings()
    assert settings.keys("runtime")
    assert settings.can_update_runtime("MAX_TOKENS_DEFAULT") is True
    assert settings.can_update_runtime("OLLAMA_HOST") is False

    stats = settings.validate_runtime_updates(
        {"MAX_TOKENS_DEFAULT": 1, "OLLAMA_HOST": "http://x", "UNKNOWN_X": "1"}
    )
    assert stats["runtime_safe"] == ["MAX_TOKENS_DEFAULT"]
    assert stats["requires_restart"] == ["OLLAMA_HOST"]
    assert stats["unknown"] == ["UNKNOWN_X"]

    pool = SimpleNamespace(
        size=lambda: 7,
        checkedin=lambda: 5,
        checkedout=lambda: 2,
        overflow=lambda: 1,
        invalidatedcount=lambda: 0,
    )
    monkeypatch.setattr(sd, "engine", SimpleNamespace(pool=pool))
    assert sd.get_db_pool_stats()["size"] == 7

    observed = {name: _Metric() for name in ("DB_POOL_SIZE", "DB_POOL_CHECKED_IN", "DB_POOL_CHECKED_OUT", "DB_POOL_OVERFLOW")}
    monkeypatch.setitem(
        __import__("sys").modules,
        "app.observability",
        SimpleNamespace(**observed),
    )
    sd.update_db_pool_metrics()
    assert observed["DB_POOL_SIZE"].values[-1] == 7


def test_validate_critical_settings_and_defaults():
    """Critical settings validation should catch invalid timeout and NSGA values."""
    ok = SimpleNamespace(
        MIN_TIMEOUT=10,
        MAX_TIMEOUT=20,
        NSGA_W_QUALITY=1.0,
        NSGA_W_LATENCY=0.5,
        NSGA_W_COST=10.0,
    )
    assert sd.validate_critical_settings(ok) == []

    bad = SimpleNamespace(
        MIN_TIMEOUT=30,
        MAX_TIMEOUT=20,
        NSGA_W_QUALITY=-1,
        NSGA_W_LATENCY=0,
        NSGA_W_COST=0,
    )
    errors = sd.validate_critical_settings(bad)
    assert "MIN_TIMEOUT must be <= MAX_TIMEOUT" in errors
    assert "NSGA weights must be non-negative" in errors


def test_reload_listener_invalidates_cache_and_stop(monkeypatch):
    """Reload listener should subscribe, invalidate cache on message, and stop cleanly."""
    invalidated = []
    monkeypatch.setattr(sd, "_invalidate_cache", lambda: invalidated.append(True))

    messages = [
        {"type": "message", "data": b"MAX_TOKENS_DEFAULT"},
        None,
    ]

    class _PubSub:
        def subscribe(self, channel):
            self.channel = channel

        def get_message(self, timeout=1.0):
            if messages:
                msg = messages.pop(0)
                if msg is None:
                    sd._reload_listener_stop.set()
                return msg
            sd._reload_listener_stop.set()
            return None

        def close(self):
            return None

    class _Redis:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def pubsub(self):
            return _PubSub()

        def close(self):
            return None

    started = []

    class _Thread:
        def __init__(self, target=None, daemon=None, name=None):
            self.target = target
            self._alive = False

        def start(self):
            started.append(True)
            self._alive = True
            self.target()
            self._alive = False

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            self._alive = False

    monkeypatch.setitem(__import__("sys").modules, "redis", SimpleNamespace(Redis=lambda **kwargs: _Redis(**kwargs)))
    monkeypatch.setattr(sd.threading, "Thread", _Thread)
    monkeypatch.setattr(sd, "settings", SimpleNamespace(REDIS_HOST="redis", REDIS_PORT=6379, REDIS_DB=0, REDIS_PASSWORD=""))

    sd.stop_reload_listener()
    sd.start_reload_listener()
    assert started
    assert invalidated
    sd.stop_reload_listener()


def test_load_json_list_and_resilient_failures(monkeypatch):
    """Small helpers should tolerate missing backends and parsing edge cases."""
    assert sd._load_json_list('["a","b"]') == ["a", "b"]
    assert sd._load_json_list("a,b") == ["a", "b"]

    monkeypatch.setattr(sd, "_get_rds", lambda: None)
    assert sd._get_from_redis("A") is None

    class _BrokenConn:
        def execute(self, stmt, params=None):
            raise RuntimeError("db down")

    monkeypatch.setattr(sd, "engine", SimpleNamespace(connect=lambda: _Ctx(_BrokenConn())))
    assert sd._get_from_db("A") is None

    monkeypatch.setattr(sd, "engine", SimpleNamespace(pool=property(lambda self: None)))
    assert sd.get_db_pool_stats()["size"] == 0


def test_bootstrap_engine_proxy_and_snapshot_resilience(monkeypatch):
    """Bootstrap helpers should fall back cleanly and snapshot should tolerate getter failures."""
    class _Engine:
        pool = "pool-obj"

        def begin(self):
            return "begin"

        def connect(self):
            return "connect"

    monkeypatch.setitem(__import__("sys").modules, "app.db", SimpleNamespace(get_engine=lambda: _Engine()))
    assert sd._get_settings_engine().pool == "pool-obj"
    assert sd.engine.begin() == "begin"
    assert sd.engine.connect() == "connect"
    assert sd.engine.pool == "pool-obj"

    monkeypatch.delitem(__import__("sys").modules, "app.db", raising=False)
    captured = {}
    monkeypatch.setitem(
        __import__("sys").modules,
        "sqlalchemy",
        SimpleNamespace(create_engine=lambda url, **kwargs: captured.setdefault("value", (url, kwargs)) or "engine"),
    )
    fallback = sd._get_settings_engine()
    assert fallback == captured["value"]

    settings = sd.DynamicSettings()
    monkeypatch.setattr(settings, "get", lambda key, default=None: (_ for _ in ()).throw(RuntimeError("lookup fail")) if key == "DB_HOST" else f"value:{key}")
    snap = settings.snapshot(only_known=False)
    assert snap["DB_HOST"] is None
    assert snap["DB_USER"] == "value:DB_USER"


def test_set_snapshot_keys_and_listener_guards(monkeypatch):
    """Dynamic settings helpers should tolerate DB/Redis publish failures and listener reentry."""
    warnings = []
    infos = []
    monkeypatch.setattr(sd.logger, "warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(sd.logger, "info", lambda msg: infos.append(msg))

    class _BrokenConn:
        def execute(self, *args, **kwargs):
            raise RuntimeError("db fail")

    class _Ctx:
        def __enter__(self):
            return _BrokenConn()

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Redis:
        def set(self, *args, **kwargs):
            return None

        def publish(self, *args, **kwargs):
            raise RuntimeError("publish fail")

    monkeypatch.setattr(sd, "engine", SimpleNamespace(begin=lambda: _Ctx()))
    monkeypatch.setattr(sd, "_get_rds", lambda: _Redis())
    invalidated = []
    monkeypatch.setattr(sd, "_invalidate_cache", lambda: invalidated.append(True))

    settings = sd.DynamicSettings()
    settings.set("MAX_TOKENS_DEFAULT", "2048", actor="test", source="unit")
    assert invalidated
    assert warnings
    assert infos

    keys = settings.keys("runtime")
    assert "MAX_TOKENS_DEFAULT" in keys
    assert settings.metadata("MAX_TOKENS_DEFAULT")

    sd._reload_listener_thread = SimpleNamespace(is_alive=lambda: True)
    sd.start_reload_listener()
