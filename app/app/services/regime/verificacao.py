# Objective: The delivery path's uncertainty check as a pure function (no metrics, no side effects).
"""Would ``query_reliability.enrich_result_reliability`` replace this answer by an abstention?

Shared by the exploration regime (regenerate a failing exploratory answer with the exploitation configuration)
and the shadow execution (``teria_abstido``). Same functions, same inputs, nothing counted.
"""

from __future__ import annotations

from app.services.query_reliability import abstain_reason, confidence_band, confidence_score, verification_status


def reprovaria(
    texto: str,
    *,
    incerteza: float,
    grounded: bool,
    retrieval_mode: str | None,
    workload_class: str | None,
    complexidade: str | None,
    fallback_usado: bool = False,
) -> bool:
    """``True`` when the answer would fail the uncertainty check (and be replaced by an abstention)."""
    answered = bool((texto or "").strip())
    retrieval_used = str(retrieval_mode or "no_retrieval") != "no_retrieval"
    score = confidence_score(
        float(incerteza if incerteza is not None else 0.5),
        answered=answered,
        grounded=grounded,
        retrieval_used=retrieval_used,
        fallback_used=fallback_usado,
        flagged=False,
    )
    band = confidence_band(score)
    verificacao = verification_status(answered=answered, grounded=grounded, score=score)
    motivo = abstain_reason(
        answered=answered,
        band=band,
        verification=verificacao,
        workload_class=str(workload_class or "reasoning"),
        complexity=str(complexidade or ""),
        retrieval_used=retrieval_used,
        grounded=grounded,
        score=score,
    )
    return motivo is not None
