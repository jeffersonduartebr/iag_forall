# semantic_cache.py
import os
import json
import hashlib
from typing import Optional

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)

def _key(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"semcache:{h}"

def put(query: str, answer: str, quality: float) -> None:
    rds.set(_key(query), json.dumps({"answer": answer, "quality": quality}), ex=60*60*24)  # TTL 24h

def get(query: str, min_quality: float = 0.0) -> Optional[str]:
    v = rds.get(_key(query))
    if not v:
        return None
    try:
        data = json.loads(v)
        if float(data.get("quality", 0.0)) >= min_quality:
            return data.get("answer")
        return None
    except Exception:
        return None
