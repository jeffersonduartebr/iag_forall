# app/utils/redis_client.py
# ------------------------------------------------------------
# Cliente Redis SÍNCRONO, seguro e com cache interno opcional.
# Agora com suporte a senha e tentativas automáticas.
# ------------------------------------------------------------

import os
import redis
import logging
import time

logger = logging.getLogger(__name__)

_redis_client = None  # cache global do cliente


def get_redis(max_wait_s: int = 5):
    """
    Retorna cliente Redis síncrono.
    Agora com:
    ✅ Suporte a senha (REDIS_PASSWORD)
    ✅ Tentativas automáticas até max_wait_s
    ✅ Log mais claro
    """
    global _redis_client

    if _redis_client:
        return _redis_client

    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    password = os.getenv("REDIS_PASSWORD") or None

    deadline = time.time() + max_wait_s

    while time.time() < deadline:
        try:
            _redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,
                socket_timeout=2,
                socket_connect_timeout=2,
            )

            # testa conexão
            _redis_client.ping()
            logger.info(
                f"[redis_client] Conectado a redis://{host}:{port}/{db} "
                f"(auth={'sim' if password else 'não'})"
            )
            return _redis_client

        except Exception as e:
            logger.warning(f"[redis_client] Redis indisponível: {e}")
            time.sleep(1)

    logger.error(
        f"[redis_client] Falha ao conectar ao Redis após {max_wait_s}s "
        f"({host}:{port}, auth={'sim' if password else 'não'})"
    )
    return None
