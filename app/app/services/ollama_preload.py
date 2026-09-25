# Objective: Startup warmup for the locally hosted Ollama models.
"""Download and warm the Ollama models the router expects to be available.

This ran as one 18-branch function inside ``main.py``. It is really three
independent decisions — *which* models matter, *which* of those are missing,
and *which* to keep resident — and each one fails differently: a malformed
settings list is a configuration error, a failed pull is a transient network
problem, and a failed warmup only costs latency on the first request. Keeping
them apart is what lets each one swallow its own failure without hiding the
other two.

Every model list here is normalised to the ``ollama/<name>`` form, because that
is the identifier the router and the bandits use; anything else is dropped
rather than guessed at.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Sequence, Set

from app.providers_async import (
    fetch_ollama_tags,
    get_configured_ollama_warm_models,
    get_http_client,
    warm_ollama_model_runtime,
)
from app.settings_dynamic import settings

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"))
VLM_OLLAMA_MODELS = list(getattr(settings, "VLM_OLLAMA_MODELS", []))

#: An Ollama pull streams progress for as long as the download takes; a large
#: model on a slow link legitimately exceeds any ordinary request timeout.
PULL_TIMEOUT_S = 1200.0


def _parse_model_list(raw: str) -> List[str]:
    """Read a model list that may be JSON or a bare comma-separated string.

    Both spellings exist in deployed ``.env`` files, so a JSON decode error is
    an expected input shape here, not a fault.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw.split(",")
    return parsed if isinstance(parsed, list) else []


def collect_targets(configured: Sequence[str]) -> List[str]:
    """Every ``ollama/``-prefixed model worth having resident, de-duplicated.

    Order is preserved so the warmup falls back to the most relevant models
    first when no explicit warm list is configured.
    """
    models: List[str] = list(configured)
    models.extend(_parse_model_list(os.getenv("CANDIDATE_MODELS_LIST", "[]")))
    models.extend(_parse_model_list(os.getenv("JUDGE_MODELS", "[]")))

    if main_model := os.getenv("OLLAMA_MODEL", ""):
        models.append(main_model)

    models.extend(f"ollama/{name}" for name in VLM_OLLAMA_MODELS)

    if embed_model := settings.get("EMBED_TEXT_MODEL", "all-minilm"):
        models.append(embed_model if embed_model.startswith("ollama/") else f"ollama/{embed_model}")

    models.append("ollama/all-minilm")

    prefixed = [m.strip() for m in models if m.strip().startswith("ollama/")]
    return list(dict.fromkeys(prefixed))


async def _available_tags() -> Set[str]:
    """Models Ollama already has. An unreachable daemon means "pull everything"."""
    try:
        return {m["name"] for m in await fetch_ollama_tags(force_refresh=False)}
    except Exception:
        return set()


def _is_present(name: str, available: Set[str]) -> bool:
    return f"{name}:latest" in available or any(name in tag for tag in available)


async def _pull(name: str) -> None:
    """Stream one model download to completion through the shared HTTP client."""
    client = await get_http_client()
    async with client.stream(
        "POST",
        f"{OLLAMA_HOST}/api/pull",
        json={"name": name},
        timeout=PULL_TIMEOUT_S,
    ) as response:
        response.raise_for_status()
        # The body must be drained for the pull to finish server-side.
        async for _ in response.aiter_lines():
            pass


async def pull_missing(models: Sequence[str]) -> None:
    """Download whatever is not resident yet, one failure at a time."""
    available = await _available_tags()
    for model in models:
        name = model.split("/", 1)[1]
        if _is_present(name, available):
            logger.info(f"[ollama-preload] '{name}' já disponível.")
            continue
        logger.info(f"[ollama-preload] Baixando '{name}'...")
        try:
            await _pull(name)
            logger.info(f"[ollama-preload] '{name}' OK.")
        except Exception as e:
            logger.error(f"[ollama-preload] Falha ao baixar '{name}': {e}")


async def warm_runtimes(models: Sequence[str], configured: Sequence[str]) -> None:
    """Force each model into memory so the first real request is not the warmup."""
    if str(settings.get("OLLAMA_WARMUP_GENERATE_ENABLED", "1")).strip() != "1":
        return
    targets = list(configured) or list(models)[:3]
    for model in targets:
        try:
            await warm_ollama_model_runtime(model)
        except Exception as e:
            logger.warning(f"[ollama-preload] Falha ao aquecer runtime '{model}': {e}")


async def preload_ollama_models() -> None:
    """Startup hook: resolve the target models, download them, warm them.

    Warmup is best-effort by design — the router must be able to start and
    serve cloud traffic even with no local daemon at all.
    """
    if not _claim_warmup():
        logger.info("[ollama-preload] Outro worker já está aquecendo; este não repete.")
        return
    try:
        logger.info("[ollama-preload] Iniciando verificação...")
        configured = get_configured_ollama_warm_models()
        models = collect_targets(configured)
        if not models:
            return
        await pull_missing(models)
        await warm_runtimes(models, configured)
        _mark_warm()
        logger.info("[ollama-preload] Concluído.")
    except Exception as e:
        logger.exception(f"[ollama-preload] Erro geral: {e}")
    finally:
        _release_warmup()  # uma trava esquecida impediria o aquecimento do próximo deploy


#: Cada worker uvicorn roda o startup: sem a trava, os dois aqueciam os mesmos modelos ao mesmo tempo e a GPU
#: recebia cargas em dobro (a tempestade de cold start da carga do Caso 1).
WARMUP_LOCK_KEY, WARMUP_DONE_KEY = "warmup:ollama:lock", "warmup:ollama:done"


def _redis():
    try:
        from app.utils.redis_client import get_redis_sync_nonblocking

        return get_redis_sync_nonblocking()
    except Exception:
        return None


def _claim_warmup() -> bool:
    """One warmup per deployment: the first worker to take the lock does it; no Redis means each warms alone."""
    rds = _redis()
    if rds is None:
        return True
    try:
        return bool(rds.set(WARMUP_LOCK_KEY, os.getpid(), nx=True, ex=int(PULL_TIMEOUT_S)))
    except Exception:
        return True


def _release_warmup() -> None:
    _quiet_redis(lambda rds: rds.delete(WARMUP_LOCK_KEY))


def _mark_warm() -> None:
    _quiet_redis(lambda rds: rds.set(WARMUP_DONE_KEY, f"{time.time():.0f}", ex=86400))


def _quiet_redis(op) -> None:
    rds = _redis()
    try:
        if rds is not None:
            op(rds)
    except Exception as exc:
        logger.debug("[ollama-preload] Redis indisponível para a marca de aquecimento: %s", exc)


def ollama_warmed_at() -> float | None:
    """Epoch of the last completed warmup (None when never, or when Redis cannot tell)."""
    rds = _redis()
    try:
        valor = rds.get(WARMUP_DONE_KEY) if rds is not None else None
        return float(valor) if valor else None
    except Exception:
        return None
