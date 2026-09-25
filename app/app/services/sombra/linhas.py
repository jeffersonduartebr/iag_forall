# Objective: Build shadow rows (numbers and hashes only) from a job and a model outcome.
from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any, Dict, Optional

from app.services.query_reliability import abstain_reason, confidence_band, confidence_score, verification_status

from .admissibilidade import regiao_efetiva

_COMUNS = ("request_id", "episode_id", "participante", "tenant", "caso", "estrato", "p_nominal", "regime_entrega",
           "p_atribuicao", "frozen_run_id")


def agora_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def base(job: Dict[str, Any], modelo: str, papel: str, *, executada: bool, status: str) -> Dict[str, Any]:
    """Row skeleton for one configuration of a sampled request."""
    linha = {c: job.get(c) for c in _COMUNS}
    linha.update(
        modelo=modelo, provedor=modelo.split("/", 1)[0], regiao=regiao_efetiva(modelo), papel=papel,
        executada=executada, status=status, motivo_corte=None if executada else status,
        criado_em=job.get("criado_em") or agora_iso(),
    )
    return linha


def sha256(texto: str) -> str:
    return hashlib.sha256((texto or "").encode("utf-8")).hexdigest()


def versao(modelo: str, meta: Optional[Dict[str, Any]]) -> str:
    """Exact model version reported by the provider (e.g. OpenRouter's dated id), else the configured id."""
    try:
        bruto = json.loads((meta or {}).get("raw_payload") or "{}")
        return str(bruto.get("model") or bruto.get("model_version") or bruto.get("modelVersion") or modelo)
    except (TypeError, ValueError, AttributeError):
        return modelo


def teria_abstido(job: Dict[str, Any], texto: str) -> bool:
    """Would the delivery path's uncertainty check have replaced this answer? Pure: no metric, no side effect."""
    answered = bool((texto or "").strip())
    grounded = bool(job.get("grounded"))
    retrieval_used = str(job.get("retrieval_mode") or "no_retrieval") != "no_retrieval"
    score = confidence_score(
        float(job.get("incerteza", 0.5) or 0.5), answered=answered, grounded=grounded,
        retrieval_used=retrieval_used, fallback_used=False, flagged=False,
    )
    band = confidence_band(score)
    verificacao = verification_status(answered=answered, grounded=grounded, score=score)
    motivo = abstain_reason(
        answered=answered, band=band, verification=verificacao, workload_class=str(job.get("workload_class") or "reasoning"),
        complexity=str(job.get("complexidade") or ""), retrieval_used=retrieval_used, grounded=grounded, score=score,
    )
    return motivo is not None
