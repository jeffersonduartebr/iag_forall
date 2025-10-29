# app/utils/redis_client.py
import os, time, logging, redis
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://:SenhaForte@redis:6379/0")

_redis = None

def get_redis(max_wait_s: int = 30):
    global _redis
    if _redis:
        return _redis
    deadline = time.time() + max_wait_s
    last_err = None
    while time.time() < deadline:
        try:
            r = redis.Redis.from_url(REDIS_URL)
            r.ping()
            _redis = r
            return _redis
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    logger.warning(f"[redis] indisponível após {max_wait_s}s: {last_err}")
    return None  # permita o warmup continuar sem Redis
