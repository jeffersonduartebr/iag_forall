# Objective: Run one sampled request in shadow: every admissible candidate, same inputs, same judges (R2-R9).
"""Runs in the Celery ``shadow_queue`` after the answer was delivered. Every failure is recorded and swallowed.

Order of checks for the request: switch -> Redis -> per-tenant hourly cap -> daily budget. A request cut at any of
these still gets one row per configuration (``executada = 0``), so the analysis can estimate the effective
inclusion probability (R9). Per candidate: admissibility (allowlist, region, modality) -> local GPU busy ->
budget -> global slot -> call (``SHADOW_TIMEOUT_S``) -> judges -> spend.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from app.settings_dynamic import settings

from . import config, cotas, registro
from .admissibilidade import motivo_recusa
from .config import ConfigSombra
from .contexto import modo_sombra
from .juizes import julgar, painel_para
from .linhas import agora_iso, base, sha256, teria_abstido, versao
from .metricas import SHADOW_CALLS, SHADOW_COST, SHADOW_SKIPPED
from .suspensoes import fechar_vencidas, suspender

logger = logging.getLogger(__name__)
_LOCAL_SEMAFOROS: Dict[int, asyncio.Semaphore] = {}


def _redis():
    try:
        from app.utils.redis_client import get_redis_sync_nonblocking

        return get_redis_sync_nonblocking()
    except Exception:
        return None


def _frozen_run_id(rds: Any) -> Optional[str]:
    """Run id of the active frozen policy, recorded on every row (the shadow runs normally while frozen)."""
    try:
        from app.services.frozen_policy import REDIS_FROZEN_PREFIX

        ativo = rds.get(f"{REDIS_FROZEN_PREFIX}active")
        return (ativo.decode() if isinstance(ativo, bytes) else str(ativo)) if ativo else None
    except Exception:
        return None


def _cortada(job: Dict[str, Any], motivo: str) -> List[Dict[str, Any]]:
    SHADOW_SKIPPED.labels(motivo=motivo).inc()
    modelos = [(job["modelo_entregue"], "entregue")] + [(c, "sombra") for c in job.get("candidatas", [])]
    return [base(job, m, papel, executada=False, status=motivo) for m, papel in modelos]


async def _vaga(rds: Any, cfg: ConfigSombra) -> bool:
    limite = time.monotonic() + cfg.timeout_s
    while not cotas.tomar_vaga(rds, cfg):
        if time.monotonic() >= limite:
            return False
        await asyncio.sleep(0.5)
    return True


async def _julgar(job: Dict[str, Any], cfg: ConfigSombra, chamar, modelo: str, texto: str) -> Dict[str, Any]:
    from app.services.quality_semantics import is_formative

    painel = painel_para(modelo, cfg, job.get("modalidade", "text"))
    notas = await julgar(
        chamar, painel, pergunta=job.get("pergunta", ""), resposta=texto, contexto=job.get("contexto", ""),
        pesos_brutos=settings.get("JUDGE_RUBRIC_WEIGHTS", None), formativa=is_formative(),
    )
    return {"painel": painel, **notas}


async def _gerar(job: Dict[str, Any], cfg: ConfigSombra, chamar, modelo: str):
    kwargs = {k: job.get(k) for k in ("image_b64", "response_format", "system_prompt")}
    inicio = time.monotonic()
    texto, meta = await asyncio.wait_for(
        chamar(model=modelo, prompt=job["prompt_final"], modality=job.get("modalidade", "text"),
               temperature=job.get("temperatura"), max_tokens=job.get("max_tokens"), timeout_seconds=cfg.timeout_s, **kwargs),
        timeout=cfg.timeout_s,
    )
    return texto or "", meta or {}, time.monotonic() - inicio


async def _avaliar(job: Dict[str, Any], cfg: ConfigSombra, rds: Any, chamar, modelo: str) -> Dict[str, Any]:
    motivo = motivo_recusa(modelo, cfg, job.get("modalidade", "text"))
    if motivo is None and modelo.startswith("ollama/") and cotas.gpu_ocupada():
        motivo = "gpu_ocupada"
    if motivo is None and cotas.orcamento_esgotado(rds, cfg):
        motivo = "orcamento_esgotado"
    if motivo:
        SHADOW_SKIPPED.labels(motivo=motivo).inc()
        return base(job, modelo, "sombra", executada=True, status=motivo)
    local = _LOCAL_SEMAFOROS.setdefault(max(1, cfg.max_local), asyncio.Semaphore(max(1, cfg.max_local)))
    if not await _vaga(rds, cfg):
        return base(job, modelo, "sombra", executada=True, status="timeout")
    try:
        if modelo.startswith("ollama/"):
            async with local:
                texto, meta, latencia = await _gerar(job, cfg, chamar, modelo)
        else:
            texto, meta, latencia = await _gerar(job, cfg, chamar, modelo)
        notas = await _julgar(job, cfg, chamar, modelo, texto)
    except asyncio.TimeoutError:
        return base(job, modelo, "sombra", executada=True, status="timeout")
    except Exception as exc:  # nunca o texto: só o tipo do erro
        logger.warning("[sombra] %s falhou: %s (%s)", modelo, type(exc).__name__, getattr(exc, "category", "-"))
        return base(job, modelo, "sombra", executada=True, status="erro")
    finally:
        cotas.soltar_vaga(rds)
    custo = float(meta.get("call_cost_usd") or meta.get("cost_per_1k") or 0.0)
    _gastar(rds, cfg, modelo, custo + notas.pop("custo_juizes", 0.0))
    return {**base(job, modelo, "sombra", executada=True, status="ok"), **notas,
            "versao_modelo": versao(modelo, meta), "teria_abstido": teria_abstido(job, texto), "custo_usd": custo,
            "latencia_s": round(latencia, 3), "tokens_entrada": meta.get("prompt_tokens"),
            "tokens_saida": meta.get("completion_tokens"), "sha256_texto": sha256(texto), "concluido_em": agora_iso()}


def _gastar(rds: Any, cfg: ConfigSombra, modelo: str, usd: float) -> None:
    SHADOW_COST.labels(model=modelo).inc(max(0.0, usd))
    hora = cotas.somar_gasto(rds, cfg, usd)
    if hora is not None:
        # Linha fixa: o alerta de log do Cloud Monitoring procura por ela (antes_18h=1).
        logger.warning("[sombra] orcamento diario esgotado hora_local=%.2f antes_18h=%d", hora, int(hora < 18))
        suspender(rds, cfg, "orcamento_esgotado", None)


async def _entregue(job: Dict[str, Any], cfg: ConfigSombra, rds: Any, chamar) -> Dict[str, Any]:
    texto = job.get("resposta_entregue") or ""
    notas = await _julgar(job, cfg, chamar, job["modelo_entregue"], texto)
    _gastar(rds, cfg, job["modelo_entregue"], notas.pop("custo_juizes", 0.0))
    return {**base(job, job["modelo_entregue"], "entregue", executada=True, status="ok"), **notas,
            "versao_modelo": job.get("versao_entregue") or job["modelo_entregue"],
            "teria_abstido": teria_abstido(job, texto), "custo_usd": job.get("custo_entregue"),
            "latencia_s": job.get("latencia_entregue"), "tokens_entrada": job.get("tokens_entrada"),
            "tokens_saida": job.get("tokens_saida"), "sha256_texto": sha256(texto), "concluido_em": agora_iso()}


async def executar(job: Dict[str, Any], *, chamar=None, rds: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Evaluate one sampled request in shadow and persist its rows; returns the rows (tests/audit)."""
    cfg = config.carregar()
    with modo_sombra():
        linhas = await _executar(job, cfg, chamar, rds)
    for linha in linhas:
        SHADOW_CALLS.labels(model=linha["modelo"], status=linha["status"]).inc()
    registro.gravar(linhas)
    return linhas


async def _executar(job: Dict[str, Any], cfg: ConfigSombra, chamar, rds: Optional[Any]) -> List[Dict[str, Any]]:
    if not cfg.ligada:
        return _cortada(job, "desligada")
    rds = rds if rds is not None else _redis()
    if rds is None:
        return _cortada(job, "sem_redis")
    if not cotas.contar_amostra_tenant(rds, cfg, str(job.get("tenant"))):
        suspender(rds, cfg, "teto_tenant", str(job.get("tenant")))
        return _cortada(job, "teto_tenant")
    if cotas.orcamento_esgotado(rds, cfg):
        suspender(rds, cfg, "orcamento_esgotado", None)
        return _cortada(job, "orcamento_esgotado")
    fechar_vencidas(rds, cfg)
    job["frozen_run_id"] = _frozen_run_id(rds)
    if chamar is None:
        from app.providers_async import call_model

        chamar = call_model
    try:
        linhas = [await _entregue(job, cfg, rds, chamar)]
        linhas += await asyncio.gather(*[_avaliar(job, cfg, rds, chamar, c) for c in job.get("candidatas", [])])
    except Exception as exc:
        logger.warning("[sombra] requisição %s falhou: %s", job.get("request_id"), type(exc).__name__)
        return _cortada(job, "erro")
    paineis = {tuple(linha["painel"]) for linha in linhas if linha.get("painel") is not None}
    for linha in linhas:
        linha["painel_uniforme"] = len(paineis) <= 1
    return list(linhas)
