# Objective: Decide, on the request path and without waiting, whether a request enters the shadow sample.
"""Cheap and synchronous: a few settings reads, a hash, and one ``.delay()``. Never raises into the request.

The draw is deterministic, ``sha256("aristo-sombra:" + correlation_id) < SHADOW_SAMPLE_RATE``, so the sample is
reproducible from the logged ids. Eligible requests: switch on, tenant in ``SHADOW_TENANT_ALLOWLIST``, a routed
choice (not a pinned instrument call, not a cache hit, not a tool or multi-turn turn), at least one other candidate.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Optional

from . import config
from .config import ConfigSombra
from .linhas import agora_iso
from .metricas import SHADOW_SKIPPED

logger = logging.getLogger(__name__)


def sorteada(request_id: str, taxa: float) -> bool:
    """Deterministic Bernoulli(taxa) draw keyed by the request id."""
    digest = hashlib.sha256(f"aristo-sombra:{request_id}".encode()).hexdigest()
    return int(digest[:12], 16) / 16**12 < taxa


def faixa_incerteza(u: Optional[float]) -> str:
    u = float(u or 0.0)
    return "baixa" if u < 1 / 3 else "media" if u < 2 / 3 else "alta"


def _estrato(cfg: ConfigSombra, ctx: Any, incerteza: float) -> Dict[str, Any]:
    disponiveis = {
        "disciplina": (ctx.hints.get("rag_filter") or {}).get("disciplina"),
        "faixa_incerteza": faixa_incerteza(incerteza),
        "modalidade": ctx.modality,
    }
    return {campo: disponiveis.get(campo) for campo in cfg.estratos}


def _regime(choice: Any) -> str:
    explorou = bool(choice.exploration_mode) or bool((choice.decision.get("bandit") or {}).get("explored"))
    return "exploracao" if explorou else "aproveitamento"


def montar_job(ctx: Any, choice: Any, outcome: Any, final_prompt: str, bundle: Dict[str, Any],
               result: Dict[str, Any], request_id: str, cfg: ConfigSombra, incerteza: float) -> Dict[str, Any]:
    """Everything a paired shadow evaluation needs, captured once: nothing is recomputed or re-retrieved later."""
    meta = result.get("metadata") or {}
    tenant = ctx.tenant_id or ""
    return {
        "request_id": request_id, "episode_id": ctx.hints.get("episode_id"), "participante": ctx.hints.get("user_key"),
        "tenant": tenant, "caso": 1 if "caso1" in tenant else 2, "estrato": _estrato(cfg, ctx, incerteza),
        "p_nominal": cfg.taxa, "regime_entrega": _regime(choice), "p_atribuicao": None,
        "frozen_run_id": None, "criado_em": agora_iso(),  # o worker lê a política congelada, fora do caminho da requisição
        "modelo_entregue": outcome.chosen, "resposta_entregue": result.get("answer") or "",
        "custo_entregue": result.get("estimated_cost_usd"), "latencia_entregue": result.get("latency_s"),
        "tokens_entrada": meta.get("prompt_tokens"), "tokens_saida": meta.get("completion_tokens"),
        "candidatas": [c for c in ctx.candidates if c != outcome.chosen],
        "pergunta": ctx.query, "prompt_final": final_prompt, "system_prompt": ctx.system_prompt,
        "contexto": bundle.get("context") or "", "modalidade": ctx.modality, "image_b64": ctx.image_b64,
        "temperatura": ctx.temperature, "max_tokens": ctx.max_tokens, "response_format": ctx.response_format,
        "incerteza": incerteza, "grounded": bool(bundle.get("grounded")), "retrieval_mode": bundle.get("retrieval_mode"),
        "workload_class": ctx.hints.get("workload_class"), "complexidade": ctx.hints.get("detected_complexity"),
    }


def talvez_agendar(ctx: Any, choice: Any, outcome: Any, final_prompt: str, bundle: Dict[str, Any],
                   result: Dict[str, Any], incerteza: float) -> bool:
    """Sample and enqueue; ``True`` when the request was sent to the shadow queue. Swallows every error."""
    try:
        cfg = config.carregar()
        if not cfg.ligada or (ctx.tenant_id or "") not in cfg.tenants:
            return False
        if ctx.hints.get("pinned_model") or ctx.tools or ctx.messages or not ctx.candidates:
            return False
        from app.correlation import get_correlation_id

        request_id = get_correlation_id() or ""
        if not request_id or not sorteada(request_id, cfg.taxa):
            return False
        job = montar_job(ctx, choice, outcome, final_prompt, bundle, result, request_id, cfg, incerteza)
        from app.tasks import task_shadow_evaluate

        task_shadow_evaluate.delay(job)
        return True
    except Exception as exc:
        SHADOW_SKIPPED.labels(motivo="enfileiramento_falhou").inc()
        logger.warning("[sombra] não enfileirada: %s", type(exc).__name__)
        return False
