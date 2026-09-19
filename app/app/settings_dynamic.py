# -*- coding: utf-8 -*-
# Objective: Application runtime code for settings dynamic.
"""Resolve runtime settings through a layered configuration strategy.

This module is the operational settings facade used across the application.
Values may come from multiple sources depending on setting type:

1. Secret/bootstrap keys (env only): ``ADMIN_TOKEN``, ``API_KEYS``, ``JWT_SECRET``, ``DB_PASS``, etc.
2. Emergency overrides: ``FORCE_<KEY>`` environment variables.
3. Runtime tunables (Redis -> MariaDB -> env -> catalog defaults).

The implementation adds three behaviors on top of plain key lookup:

- short-lived in-process caching to reduce Redis and database traffic
- runtime mutability metadata so the admin API can reject restart-only changes
- a Redis pub/sub listener that invalidates local caches when settings change

The public ``settings`` object is intentionally lightweight so the rest of the
runtime can treat settings as a normal attribute-based configuration source.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from app.utils.redis_client import ensure_redis_connected, get_redis_async_safe

from .config.settings_catalog import (
    SETTING_METADATA,
    SETTINGS_DEFAULTS,
    is_known_setting,
    is_runtime_mutable,
    known_setting_keys,
)
from .config.settings_env import DB_HOST_ENV, DB_NAME_ENV, DB_PASS_ENV, DB_PORT_ENV, DB_USER_ENV
from .config.settings_properties import TypedSettingsMixin
from .config.settings_sources import decode_redis_value, resolve_setting_value, resolve_setting_value_async
from .config.settings_types import as_bool, as_float, as_int, as_list
from .config.settings_validation import _read_setting  # noqa: F401  (reexportado)
from .config.settings_validation import validate_critical_settings as _validate_critical_settings

logger = logging.getLogger(__name__)

# ============================================================
# Redis / Banco básicos
# ============================================================

REDIS_PREFIX = "settings:"
REDIS_RELOAD_CHANNEL = "settings:reload"


def _get_rds():
    """Return a Redis client suitable for dynamic settings access.

    The helper first attempts to reuse the shared async-safe client and then
    falls back to a best-effort connection bootstrap. Returning ``None`` is
    acceptable and simply means Redis-backed overrides are unavailable.
    """
    return get_redis_async_safe() or ensure_redis_connected(max_wait_s=0.0, min_retry_interval_s=2.0)


def _get_settings_engine():
    """
    Get database engine for settings.
    Uses lazy import to avoid circular dependencies during startup.
    """
    try:
        from app.db import get_engine

        return get_engine()
    except Exception:
        # Fallback: create a temporary engine for bootstrap
        from sqlalchemy import create_engine

        db_url = f"mysql+pymysql://{DB_USER_ENV}:{DB_PASS_ENV}@{DB_HOST_ENV}:{DB_PORT_ENV}/{DB_NAME_ENV}"
        return create_engine(db_url, pool_pre_ping=True, pool_recycle=300)


# Legacy alias for backward compatibility
engine: Any = property(lambda self: _get_settings_engine())


class _EngineProxy:
    """Backward-compatible proxy around the centralized SQLAlchemy engine.

    Older parts of the codebase still import ``settings_dynamic.engine``. The
    proxy preserves that interface while delegating all real work to the engine
    factory in ``app.db``.
    """

    def begin(self):
        """Open a transactional context using the shared settings database engine."""
        return _get_settings_engine().begin()

    def connect(self):
        """Open a direct database connection using the shared settings engine."""
        return _get_settings_engine().connect()

    @property
    def pool(self):
        """Expose the underlying connection pool for operational metrics."""
        return _get_settings_engine().pool


engine = _EngineProxy()

DDL = """
CREATE TABLE IF NOT EXISTS settings_dynamic (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    setting_key VARCHAR(512) NOT NULL UNIQUE,
    setting_value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;
"""

try:
    with engine.begin() as conn:
        conn.execute(text(DDL))
except Exception as e:
    logger.warning(f"[settings_dynamic] Falha ao criar tabela: {e}")


# ============================================================
# LRU Cache interno com TTL
# ============================================================


class LRUCache:
    """Small thread-safe cache used to reduce repeated settings lookups.

    The cache stores resolved setting values for a short period because many
    runtime paths read the same keys on every request. It is deliberately simple
    and only supports the operations needed by the settings facade.
    """

    def __init__(self, maxsize: int = 512, ttl_s: int = 30):
        """Create a bounded cache with LRU eviction and optional TTL."""
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Return a cached value when present and still fresh."""
        now = time.time()
        with self._lock:
            if key not in self._data:
                return None
            value, ts = self._data[key]
            if self.ttl_s > 0 and (now - ts) > self.ttl_s:
                try:
                    del self._data[key]
                except KeyError:
                    pass
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """Insert or refresh one cached setting value."""
        now = time.time()
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (value, now)
            if len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        """Remove all cached entries immediately."""
        with self._lock:
            self._data.clear()


SETTINGS_CACHE_SIZE = int(os.getenv("SETTINGS_CACHE_SIZE", "2000"))  # Optimized for high-capacity
SETTINGS_CACHE_TTL_S = int(os.getenv("SETTINGS_CACHE_TTL_S", "300"))  # 5 min - reduces Redis/DB lookups by ~80%

_lru = LRUCache(maxsize=SETTINGS_CACHE_SIZE, ttl_s=SETTINGS_CACHE_TTL_S)


def _invalidate_cache():
    """Clear the local settings cache after a runtime configuration change."""
    global _last_prime
    _lru.clear()
    _last_prime = 0.0  # a próxima leitura recarrega tudo em lote
    logger.info("[settings_dynamic] Cache LRU invalidado.")


# ============================================================
# Auxiliares de acesso Redis / DB
# ============================================================


def _get_from_redis(key: str) -> Optional[str]:
    """Read one settings override from Redis, returning ``None`` on failure."""
    rds = _get_rds()
    if not rds:
        return None
    try:
        return decode_redis_value(rds.get(f"{REDIS_PREFIX}{key}"))
    except Exception:
        pass
    return None


async def _get_from_redis_async(key: str) -> Optional[str]:
    """Async Redis read for request handlers that already run on the event loop."""
    try:
        from app.utils.redis_client import get_redis_async

        rds = await get_redis_async()
        if not rds:
            return None
        return decode_redis_value(await rds.get(f"{REDIS_PREFIX}{key}"))
    except Exception:
        return None


def _get_from_db(key: str) -> Optional[str]:
    """Read one persisted setting from MariaDB, returning ``None`` on failure."""
    try:
        with engine.connect() as conn:
            r = conn.execute(
                text("SELECT setting_value FROM settings_dynamic WHERE setting_key=:k LIMIT 1"),
                {"k": key},
            ).fetchone()
        if r:
            return r[0]
    except Exception:
        pass
    return None


# ============================================================
# Recarga em lote (1 SELECT + 1 MGET em vez de GET+SELECT por chave)
# ============================================================
_PRIME_MIN_INTERVAL_S = float(os.getenv("SETTINGS_PRIME_MIN_INTERVAL_S", "5"))
_prime_lock = threading.Lock()
_last_prime = 0.0


def _all_from_db() -> Optional[Dict[str, str]]:
    """Every persisted setting in one query (``None`` when the DB is unavailable)."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT setting_key, setting_value FROM settings_dynamic")).fetchall()
        return {str(row[0]): row[1] for row in rows}
    except Exception:
        return None


def _many_from_redis(keys: List[str]) -> Optional[Dict[str, str]]:
    """Redis overrides for ``keys`` in one MGET (``None`` when Redis is unavailable)."""
    rds = _get_rds()
    if not rds:
        return None
    try:
        values = rds.mget([f"{REDIS_PREFIX}{key}" for key in keys])
        decoded = {key: decode_redis_value(raw) for key, raw in zip(keys, values) if raw is not None}
        return {key: value for key, value in decoded.items() if value is not None}
    except Exception:
        return None


def _prime_cache() -> bool:
    """Resolve every catalogued/persisted setting into the LRU with one DB query and one MGET.

    Same precedence as the per-key path (``resolve_setting_value`` over the
    preloaded dicts). Rate-limited and non-blocking: concurrent callers skip it
    and fall back to the per-key path. Returns whether it ran.
    """
    global _last_prime
    now = time.monotonic()
    if now - _last_prime < _PRIME_MIN_INTERVAL_S or not _prime_lock.acquire(blocking=False):
        return False
    try:
        _last_prime = now
        db_values = _all_from_db()
        keys = sorted(set(SETTINGS_DEFAULTS) | set(db_values or {}))
        redis_values = _many_from_redis(keys)
        if db_values is None and redis_values is None:
            return False
        for key in keys:
            resolve_setting_value(
                key=key,
                fallback=None,
                defaults=SETTINGS_DEFAULTS,
                cache_get=lambda _key: None,
                cache_set=_lru.set,
                redis_get=(redis_values or {}).get,
                db_get=(db_values or {}).get,
                env_get=os.getenv,
            )
        return True
    finally:
        _prime_lock.release()


def _set_to_redis(key: str, val: str):
    """Write one resolved setting value to Redis as a best-effort cache/update."""
    rds = _get_rds()
    if not rds:
        return
    try:
        rds.set(f"{REDIS_PREFIX}{key}", val)
    except Exception as e:
        logger.debug(f"[settings_dynamic] Falha ao gravar Redis para {key}: {e}")


def _load_json_list(raw: Optional[str]) -> List[str]:
    """Parse a JSON-encoded list setting into a normalized string list."""
    return as_list(raw)


# ============================================================
# Classe Principal (Integrada)
# ============================================================


class DynamicSettings(TypedSettingsMixin):
    """Expose layered runtime settings through a compact facade.

    Most of the codebase accesses settings through attributes on this object.
    The facade keeps lookup behavior centralized, consistent, and easy to
    instrument while still supporting hot-reload semantics.
    """

    DEFAULTS: Dict[str, str] = SETTINGS_DEFAULTS
    METADATA: Dict[str, Dict[str, str]] = SETTING_METADATA

    def get(self, key: str, fallback: Any = None) -> Any:
        """Resolve one setting through cache, Redis, DB, environment, and defaults."""
        if _lru.get(key) is None:
            _prime_cache()
        return resolve_setting_value(
            key=key,
            fallback=fallback,
            defaults=self.DEFAULTS,
            cache_get=_lru.get,
            cache_set=_lru.set,
            redis_get=_get_from_redis,
            db_get=_get_from_db,
            env_get=os.getenv,
        )

    async def get_async(self, key: str, fallback: Any = None) -> Any:
        """Resolve one setting without blocking the event loop on Redis I/O."""
        if _lru.get(key) is None:
            await asyncio.to_thread(_prime_cache)
        return await resolve_setting_value_async(
            key=key,
            fallback=fallback,
            defaults=self.DEFAULTS,
            cache_get=_lru.get,
            cache_set=_lru.set,
            redis_get_async=_get_from_redis_async,
            db_get=_get_from_db,
            env_get=os.getenv,
        )

    def _get_str(self, key: str, default: str = "") -> str:
        """Get a setting coerced to string."""
        value = self.get(key, default)
        return default if value is None else str(value)

    def _get_int(self, key: str, default: int) -> int:
        """Get a setting coerced to int."""
        return as_int(self.get(key, default), default)

    def _get_float(self, key: str, default: float) -> float:
        """Get a setting coerced to float."""
        return as_float(self.get(key, default), default)

    def _get_bool(self, key: str, default: bool = False) -> bool:
        """Get a setting coerced to bool."""
        return as_bool(self.get(key, "1" if default else "0"), default)

    def _get_list(self, key: str, default: str = "[]") -> List[str]:
        """Get a setting coerced to string list."""
        return as_list(self.get(key, default))

    def set(self, key: str, value: str, actor: str = "system", source: str = "internal") -> None:
        """Persist a setting update and broadcast cache invalidation.

        The update is written to MariaDB, mirrored to Redis when available, and
        published on the reload channel so all running processes can evict stale
        local cache entries.
        """
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO settings_dynamic (setting_key, setting_value)
                        VALUES (:k, :v)
                        ON DUPLICATE KEY UPDATE setting_value = :v
                    """),
                    {"k": key, "v": value},
                )
        except Exception as e:
            logger.warning(f"Falha ao gravar DB ({key}): {e}")
        _set_to_redis(key, value)
        _invalidate_cache()
        rds = _get_rds()
        if rds:
            try:
                rds.publish(REDIS_RELOAD_CHANNEL, key)
            except Exception:
                pass
        logger.info(f"[settings] '{key}' atualizado por {actor} via {source}.")

    def snapshot(self, only_known: bool = False, redact: bool = True) -> Dict[str, Any]:
        """Return a point-in-time view of known settings and selected env values."""
        from .config.secrets_redaction import redact_secrets

        keys = (
            known_setting_keys()
            if only_known
            else known_setting_keys()
            + [
                "DB_HOST",
                "DB_USER",
                "DB_NAME",
                "DB_PORT",
                "OLLAMA_HOST",
                "OLLAMA_BASE_URL",
            ]
        )
        out = {}
        for k in keys:
            try:
                out[k] = self.get(k)
            except Exception:
                out[k] = None
        return redact_secrets(out) if redact else out

    def keys(self, domain: Optional[str] = None) -> List[str]:
        """Return known setting keys, optionally filtered by catalog domain."""
        if not domain:
            return known_setting_keys()
        return [key for key, meta in self.METADATA.items() if meta.get("domain") == domain]

    def metadata(self, key: str) -> Dict[str, str]:
        """Return catalog metadata for one known setting."""
        return dict(self.METADATA.get(key, {}))

    def can_update_runtime(self, key: str) -> bool:
        """Return whether one setting can be updated dynamically via admin API."""
        return is_runtime_mutable(key)

    def validate_runtime_updates(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an admin update payload against the runtime mutability catalog."""
        unknown = [key for key in payload if not is_known_setting(key)]
        restart_required = [key for key in payload if is_known_setting(key) and not is_runtime_mutable(key)]
        runtime_safe = [key for key in payload if key not in unknown and key not in restart_required]
        return {
            "runtime_safe": runtime_safe,
            "requires_restart": restart_required,
            "unknown": unknown,
        }


settings = DynamicSettings()


# ============================================================
# DB Connection Pool Metrics (Quick Win #10)
# ============================================================


def get_db_pool_stats() -> dict:
    """Get database connection pool statistics."""
    try:
        pool = engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalidatedcount() if hasattr(pool, "invalidatedcount") else 0,
        }
    except Exception:
        return {"size": 0, "checked_in": 0, "checked_out": 0, "overflow": 0, "invalid": 0}


def update_db_pool_metrics():
    """Update Prometheus metrics for DB pool (call periodically)."""
    try:
        from app.observability import (
            DB_POOL_CHECKED_IN,
            DB_POOL_CHECKED_OUT,
            DB_POOL_OVERFLOW,
            DB_POOL_SIZE,
        )

        stats = get_db_pool_stats()
        DB_POOL_SIZE.set(stats["size"])
        DB_POOL_CHECKED_IN.set(stats["checked_in"])
        DB_POOL_CHECKED_OUT.set(stats["checked_out"])
        DB_POOL_OVERFLOW.set(stats["overflow"])
    except Exception:
        pass  # Metrics not available


def validate_critical_settings(settings_obj: Optional[Any] = None) -> List[str]:
    """Validate critical runtime settings (see ``config.settings_validation``)."""
    return _validate_critical_settings(settings_obj or settings)


# ============================================================
# Hot-reload listener via Redis Pub/Sub (COM RECONEXÃO)
# ============================================================

_reload_listener_thread: Optional[threading.Thread] = None
_reload_listener_stop = threading.Event()


def start_reload_listener() -> None:
    """Executa a responsabilidade descrita por este método.

    Returns:
        Valor produzido pela execução.
    """
    global _reload_listener_thread
    if _reload_listener_thread and _reload_listener_thread.is_alive():
        return
    _reload_listener_stop.clear()

    # Usamos uma conexão dedicada para o listener, sem timeout
    def _bg():
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        while not _reload_listener_stop.is_set():
            listener = None
            pubsub = None
            try:
                # Conexão dedicada para o PubSub com timeout infinito
                # Importa redis aqui para evitar dependência circular se fosse no topo
                import redis

                listener = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD or None,
                    socket_timeout=None,  # <--- O SEGREDO
                    socket_keepalive=True,
                )

                pubsub = listener.pubsub()
                pubsub.subscribe(REDIS_RELOAD_CHANNEL)
                logger.info(f"[settings_dynamic] Listener hot-reload conectado: {REDIS_RELOAD_CHANNEL}")

                while not _reload_listener_stop.is_set():
                    msg = pubsub.get_message(timeout=1.0)
                    if msg and msg.get("type") == "message":
                        key = msg.get("data")
                        if isinstance(key, bytes):
                            key = key.decode()
                        logger.info(f"[settings_dynamic] Hot-reload: '{key}'.")
                        _invalidate_cache()

            except Exception as e:
                logger.error(f"[settings_dynamic] Erro no listener: {e}. Reconectando em 5s...")
                _reload_listener_stop.wait(5)
            finally:
                try:
                    if pubsub:
                        pubsub.close()
                except Exception:
                    pass
                try:
                    if listener:
                        listener.close()
                except Exception:
                    pass

    _reload_listener_thread = threading.Thread(target=_bg, daemon=True, name="settings-reload-listener")
    _reload_listener_thread.start()


def stop_reload_listener() -> None:
    """Executa a responsabilidade descrita por este método.

    Returns:
        Valor produzido pela execução.
    """
    global _reload_listener_thread
    _reload_listener_stop.set()
    if _reload_listener_thread and _reload_listener_thread.is_alive():
        _reload_listener_thread.join(timeout=1.0)
    _reload_listener_thread = None
