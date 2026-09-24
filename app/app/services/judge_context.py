# -*- coding: utf-8 -*-
# Objective: Judge context: model resolution, retrieved context and image descriptions.
"""Which models judge, what context they see (RAG) and how images are described.

Extracted from ``app.judges`` (re-exported there): the configured judge pool
with a local fallback, the meta-judge and the vision model that turns an image
into text for text-only judges.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import List, Optional

from ..embeddings import embed_text
from ..model_registry import filter_configured_model_names, is_model_configured
from ..providers_async import call_model
from ..settings_dynamic import settings
from ..vectorstore import query_embedding
from .judge_vendors import elegiveis

logger = logging.getLogger("app.judges")

# Meta-Juiz preferencial (deve ser um modelo forte)
META_JUDGE_HINT = str(settings.get("META_JUDGE_PREF", "ollama/phi4:latest"))
IMAGE_DESC_MODEL_HINT = str(settings.get("IMAGE_DESC_MODEL", "ollama/qwen3-vl:8b"))
VISION_VLM_CANDIDATES: List[str] = list(getattr(settings, "CANDIDATE_VISION_MODELS_LIST", []))
MULTIMODAL_VLM_CANDIDATES: List[str] = list(getattr(settings, "CANDIDATE_MULTIMODAL_MODELS_LIST", []))


def _configured_local_fallback() -> str:
    """Return a stable local fallback model for judge-related paths."""
    preferred = [
        getattr(settings, "JUDGES_LOCAL_MODEL", None),
        "ollama/phi4:latest",
        "ollama/qwen3:14b",
        "ollama/gemma3:4b",
    ]
    candidates = [model for model in preferred if isinstance(model, str) and model]
    filtered = filter_configured_model_names(candidates)
    return filtered[0] if filtered else "ollama/phi4:latest"


def _resolve_meta_judge_model() -> str:
    """Resolve the meta-judge: the preferred one, else another judge, never of the evaluated model's company."""
    if is_model_configured(META_JUDGE_HINT) and elegiveis([META_JUDGE_HINT]):
        return META_JUDGE_HINT
    alternativas = _resolve_judge_models() + elegiveis([_configured_local_fallback()])
    return alternativas[0] if alternativas else _configured_local_fallback()


def _resolve_image_desc_model() -> str:
    """Resolve the configured vision model with a local fallback."""
    candidates = []
    if IMAGE_DESC_MODEL_HINT:
        candidates.append(IMAGE_DESC_MODEL_HINT)
    candidates.extend(VISION_VLM_CANDIDATES)
    candidates.extend(MULTIMODAL_VLM_CANDIDATES)
    filtered = filter_configured_model_names(
        [model for model in candidates if isinstance(model, str) and model]
    )
    return filtered[0] if filtered else "ollama/qwen3-vl:8b"


def _resolve_judge_models() -> List[str]:
    """Resolve judge models to the subset configured for the current environment."""
    configured = filter_configured_model_names(
        [model for model in (getattr(settings, "JUDGE_MODELS", []) or []) if isinstance(model, str) and model]
    )
    # Independência: nenhum juiz da mesma empresa do modelo avaliado (vazio = não julgar).
    return elegiveis(configured) or elegiveis([_configured_local_fallback()])


def _image_hash_from_b64(image_b64: Optional[str]) -> Optional[str]:
    """Return a stable image hash used for judge logging and deduplication."""
    if not image_b64:
        return None
    try:
        h = hashlib.sha256()
        h.update(image_b64.encode("utf-8", errors="ignore"))
        return h.hexdigest()
    except Exception:
        return None


async def get_rag_context(query: str, n_results: int = 5, max_chars: int = 1500) -> str:
    """Retrieve a compact RAG context block to assist judge prompts."""
    try:
        vec = await asyncio.to_thread(embed_text, query)
        coll = settings.get("RAG_COLLECTION_NAME", "knowledge_base")

        results = await query_embedding(coll, vec, n_results=n_results)
        if not results or "documents" not in results:
            return ""

        docs = results["documents"][0]
        ctx = "\n\n".join(docs).strip()

        return (ctx[:max_chars] + "...") if len(ctx) > max_chars else ctx
    except Exception as exc:
        logger.warning("[Judges] RAG error: %s", exc)
        return ""


async def _describe_image_if_needed(image_b64: Optional[str], modality: str) -> str:
    """Generate a short technical image description for judge prompts when needed."""
    if not image_b64:
        return ""

    candidates = []
    image_desc_model = _resolve_image_desc_model()
    if image_desc_model:
        candidates.append(image_desc_model)
    candidates.extend(VISION_VLM_CANDIDATES)
    candidates.extend(MULTIMODAL_VLM_CANDIDATES)

    seen = set()
    ordered = []
    for m in candidates:
        if m and m not in seen:
            seen.add(m)
            ordered.append(m)

    prompt = (
        "Descreva tecnicamente o conteúdo da imagem fornecida. "
        "Use poucas frases, sem especulação."
    )

    for model_name in ordered:
        try:
            text_out, _ = await call_model(
                model=model_name,
                prompt=prompt,
                image_b64=image_b64,
                temperature=0.1,
                max_tokens=128,
            )
            if isinstance(text_out, str) and text_out.strip():
                return text_out.strip()
        except Exception:
            pass

    return ""
