# Objective: Regenerate an exploratory answer that fails the uncertainty check with the exploitation config.
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from app.services.router_provider_stage import ProviderOutcome, build_result, execute_provider
from app.services.router_stages import RouteChoice, RouteContext

from .verificacao import reprovaria

logger = logging.getLogger(__name__)


async def talvez_regenerar(
    ctx: RouteContext, choice: RouteChoice, outcome: ProviderOutcome, final_prompt: str,
    bundle: Dict[str, Any], result: Dict[str, Any], incerteza: float,
) -> Tuple[RouteChoice, ProviderOutcome, Dict[str, Any]]:
    """Protocol: an exploratory answer that fails the uncertainty check is replaced by the exploitation one.

    The explored answer is never served; the decision record keeps the explored model, the reason and the
    served one. A failed regeneration keeps the explored answer (the delivery path's own check still applies).
    """
    regime = (choice.decision or {}).get("regime")
    if not regime or not regime.get("explorou"):
        return choice, outcome, result
    if outcome.chosen != choice.chosen:
        regime["servido_por_fallback"] = outcome.chosen
    falhou = reprovaria(
        str(result.get("answer") or ""), incerteza=incerteza, grounded=bool(bundle.get("grounded")),
        retrieval_mode=bundle.get("retrieval_mode"), workload_class=ctx.hints.get("workload_class"),
        complexidade=ctx.hints.get("detected_complexity"), fallback_usado=bool(outcome.fallback_used),
    )
    if not falhou:
        return choice, outcome, result
    nova = RouteChoice(chosen=regime["aproveitamento"], top2=[regime["aproveitamento"]], decision=choice.decision)
    try:
        novo_outcome = await execute_provider(ctx, nova, final_prompt)
    except Exception as exc:
        regime.update(regeneracao_falhou=type(exc).__name__)
        logger.warning("[regime] regeneração falhou: %s", type(exc).__name__)
        return choice, outcome, result
    regime.update(regenerado=True, modelo_explorado=choice.chosen, motivo_regeneracao="verificacao_incerteza")
    return nova, novo_outcome, build_result(ctx, nova, novo_outcome, incerteza, bundle)
