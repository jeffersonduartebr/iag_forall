# Objective: The regime's route choice: greedy exploitation vs uniform exploration, with exact probabilities.
from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional

from app.services.router_stages import RouteChoice, RouteContext, _decision_record, _scored_top2

from . import config, janela

logger = logging.getLogger(__name__)


def uniforme(request_id: str, etiqueta: str) -> float:
    """Deterministic U[0,1) from the request id: the draw is reproducible from the logged ids."""
    digest = hashlib.sha256(f"aristo-regime:{etiqueta}:{request_id}".encode()).hexdigest()
    return int(digest[:13], 16) / 16**13


def _redis():
    try:
        from app.utils.redis_client import get_redis_sync_nonblocking

        return get_redis_sync_nonblocking()
    except Exception:
        return None


async def _medias(ctx: RouteContext) -> Dict[str, float]:
    """Posterior mean reward per model in the request's bandit context (namespaced like the bandit's own)."""
    try:
        from app.bandits import _auto_context_labels_async, _get_ctx_stats_async

        rotulos = await _auto_context_labels_async(ctx.query, ctx.modality)
        estat = await _get_ctx_stats_async(rotulos[0] if rotulos else "global")
        return {m: float(s.get("mean", 0.0) or 0.0) for m, s in (estat or {}).items()}
    except Exception as exc:
        logger.warning("[regime] estatísticas do bandit indisponíveis: %s", type(exc).__name__)
        return {}


async def _pool_catalogo(ctx: RouteContext, modelos: List[str]) -> List[str]:
    """OpenRouter catalogue models under exploration (only while the explorer is on and within its daily caps)."""
    if ctx.modality != "text":
        return []
    try:
        from app.openrouter_exploration_policy import load_exploration_config
        from app.openrouter_exploration_state import _get_daily_count, _get_daily_usd
        from app.openrouter_explorer import _build_exploration_pool, _get_redis, _openrouter_configured
        from app.settings_dynamic import settings

        cfg = load_exploration_config(settings)
        if not cfg.enabled or not _openrouter_configured():
            return []
        rds = await _get_redis()
        if await _get_daily_count(rds) >= cfg.max_per_day:
            return []
        if cfg.max_usd_per_day > 0 and await _get_daily_usd(rds) >= cfg.max_usd_per_day:
            return []
        return [m for m in await _build_exploration_pool(set(modelos), cfg, ctx.modality) if m not in modelos]
    except Exception as exc:
        logger.warning("[regime] catálogo de exploração indisponível: %s", type(exc).__name__)
        return []


def _congelada(rds: Any) -> bool:
    try:
        from app.services.frozen_policy import REDIS_FROZEN_PREFIX

        return bool(rds.get(f"{REDIS_FROZEN_PREFIX}active"))
    except Exception:
        return False


def _epsilon(cfg: config.ConfigRegime, ctx: RouteContext, bracos: List[str], rds: Any, episodio: str) -> tuple:
    """Effective exploration probability and, when it is zero, why."""
    participante = ctx.hints.get("user_key")
    if not bracos:
        return 0.0, "sem_bracos", None
    if cfg.em_aquecimento():  # antes do campo: sem teto por participante e sem exigir participante
        return (
            (0.0, "politica_congelada", None)
            if rds is not None and _congelada(rds)
            else (cfg.epsilon_aquecimento, None, None)
        )
    if not participante:
        return 0.0, "sem_participante", None
    if rds is None:
        return 0.0, "sem_redis", None
    if _congelada(rds):
        return 0.0, "politica_congelada", None
    n, x = janela.contagem(rds, str(participante), episodio, cfg.janela)
    if not janela.pode_explorar(n, x, cfg.teto):
        return 0.0, "teto_janela", (n, x)
    return cfg.epsilon, None, (n, x)


async def rota_do_regime(ctx: RouteContext, modelos: List[str], incerteza: float) -> Optional[RouteChoice]:
    """The route for a regime tenant (``None`` for every other tenant: normal routing applies)."""
    cfg = config.carregar()
    if not modelos or (ctx.tenant_id or "") not in cfg.tenants:
        return None
    from app.correlation import get_correlation_id

    request_id = get_correlation_id() or uuid.uuid4().hex
    top2, pontuados, pesos = await _scored_top2(ctx, modelos, incerteza)
    ctx.scored_candidates, ctx.strategy_weights = pontuados, pesos
    medias, ordem = await _medias(ctx), {c.model: i for i, c in enumerate(pontuados)}
    guloso = max(modelos, key=lambda m: (medias.get(m, 0.0), -ordem.get(m, len(modelos))))
    aquecimento = cfg.em_aquecimento()
    # No aquecimento a exploração fica nas candidatas configuradas: com o catálogo (~80 modelos) cada uma
    # receberia ~0,6% do tráfego e o bandit não aprenderia a compará-las. O catálogo segue no seu mecanismo.
    catalogo = [] if aquecimento else await _pool_catalogo(ctx, modelos)
    bracos = [m for m in modelos if m != guloso] + catalogo
    rds, episodio = _redis(), janela.episodio_de(ctx.hints.get("episode_id"), request_id)
    eps, motivo, janela_antes = _epsilon(cfg, ctx, bracos, rds, episodio)
    explorou = uniforme(request_id, "explorar") < eps
    escolhido = bracos[int(uniforme(request_id, "braco") * len(bracos))] if explorou else guloso
    if ctx.hints.get("user_key") and rds is not None and not aquecimento:  # o campo começa com janelas limpas
        janela.registrar(rds, str(ctx.hints["user_key"]), episodio, cfg.janela, explorou)
    info = {
        "explorou": explorou,
        "p_atribuicao": (eps / len(bracos)) if explorou else 1.0 - eps,
        "epsilon_nominal": cfg.epsilon,
        "epsilon_efetivo": eps,
        "motivo_sem_exploracao": motivo,
        "aproveitamento": guloso,
        "k_bracos": len(bracos),
        "bracos": bracos,  # o conjunto sorteado: com o catálogo, muda de um dia para o outro
        "k_catalogo": len(catalogo),
        "episodio": episodio,
        "janela_antes": {"n": janela_antes[0], "x": janela_antes[1]} if janela_antes else None,
        "teto": cfg.teto,
        "janela_episodios": cfg.janela,
        "regenerado": False,
        "fase": "aquecimento" if aquecimento else "campo",
    }
    decisao = _decision_record(ctx, escolhido, top2, incerteza) or {}
    decisao["regime"] = info
    do_catalogo = escolhido in catalogo
    exploracao = {"openrouter_exploration": True, "regime": True, "exploration_pool_size": len(catalogo)}
    return RouteChoice(
        chosen=escolhido,
        top2=[escolhido],
        decision=decisao,
        exploration_mode=do_catalogo,
        exploration_info=exploracao if do_catalogo else {},
    )
