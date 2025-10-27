# app/semantic_cache.py
import time
from typing import Optional
from .settings import settings
from .embeddings import get_embedding
from .vectorstore import chroma_client

def _now() -> float:
    return time.time()

def sem_cache_get(query: str, model: str, temperature: float, max_tokens: int) -> Optional[str]:
    """Busca resposta similar no cache semântico (Chroma)."""
    if not getattr(settings, "SEM_CACHE_ENABLED", True):
        return None

    emb = get_embedding(query)
    coll = chroma_client.get_or_create_collection("sem_cache")
    res = coll.query(query_embeddings=[emb], n_results=getattr(settings, "CACHE_TOP_K", 3))

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    if not docs:
        return None

    # melhor candidato
    idx = 0
    best_sim = 1.0 - float(dists[idx])  # se distance = 1 - cos_sim
    md = metas[idx] or {}
    sim_thr = float(getattr(settings, "CACHE_SIM_THRESHOLD", 0.86))
    ttl = int(getattr(settings, "CACHE_TTL_SECONDS", 86400))

    if best_sim >= sim_thr \
       and (_now() - float(md.get("ts", 0))) <= ttl \
       and md.get("model") == model \
       and float(md.get("temperature", -1)) == float(temperature) \
       and int(md.get("max_tokens", -1)) == int(max_tokens):
        return docs[idx]

    return None

def sem_cache_put(query: str, answer: str, model: str, temperature: float, max_tokens: int):
    """Armazena resposta no cache semântico."""
    if not getattr(settings, "SEM_CACHE_ENABLED", True):
        return
    emb = get_embedding(query)
    coll = chroma_client.get_or_create_collection("sem_cache")
    coll.add(
        embeddings=[emb],
        documents=[answer],
        metadatas=[{
            "ts": _now(),
            "model": model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens)
        }],
        ids=[f"{hash((query, model, temperature, max_tokens, int(_now())))}"]
    )
