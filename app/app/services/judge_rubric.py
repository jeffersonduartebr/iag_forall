# -*- coding: utf-8 -*-
# Objective: Three-dimension judging rubric (thesis ch. 5, judges layer).
"""Build, parse and aggregate the three-dimension judge rubric.

Each judge scores the same answer on three dimensions, 0 to 10:

- ``clareza``: clarity and cohesion of the text;
- ``acuracia``: conceptual accuracy (agreement with the reference, when given);
- ``alinhamento``: pedagogical alignment with the educational context and level.

The aggregate quality is the weighted mean ``Q = sum(w_i * d_i) / sum(w_i)``
(default weights 0.3 / 0.5 / 0.2). Because every judge rates every dimension,
the spread between judges on one dimension is a genuine inter-rater agreement
measure. An unparseable judge output is a *failed* rating (``None``), never a
zero, so infrastructure errors do not masquerade as wrong answers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import statistics
import unicodedata
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

Rating = Dict[str, float]
RateFn = Callable[[str], Awaitable[Tuple[Optional[Rating], Dict[str, Any]]]]
OnRatingFn = Callable[[str, Rating, Dict[str, Any]], None]

RUBRIC_DIMENSIONS = ("clareza", "acuracia", "alinhamento")
DEFAULT_RUBRIC_WEIGHTS: Dict[str, float] = {"clareza": 0.3, "acuracia": 0.5, "alinhamento": 0.2}

#: The two dimensions that measure the *answer*, as opposed to how it was
#: delivered. Q_tech renormalises them over their own weights (0.3 + 0.5 = 0.8),
#: which is what makes the formative score a product of a technical term and a
#: delivery term rather than a sum that lets one pay for the other.
TECH_DIMENSIONS = ("clareza", "acuracia")

_ALIASES = {
    "clareza": "clareza",
    "clareza_coesao": "clareza",
    "clarity": "clareza",
    "acuracia": "acuracia",
    "acuracia_conceitual": "acuracia",
    "accuracy": "acuracia",
    "alinhamento": "alinhamento",
    "alinhamento_pedagogico": "alinhamento",
    "alignment": "alinhamento",
}

_RUBRIC_TEMPLATE = """
Você é um avaliador educacional imparcial. Avalie a RESPOSTA DO MODELO à PERGUNTA
segundo a rubrica abaixo, atribuindo a cada dimensão uma nota de 0 a 10.

PERGUNTA: {query}
{ref_block}{rag_block}{img_block}
RESPOSTA DO MODELO: {answer}

### RUBRICA
1. clareza (clareza e coesão): a resposta é bem estruturada, fluente e compreensível?
2. acuracia (acurácia conceitual): o conteúdo é tecnicamente correto, sem inconsistências
   nem imprecisões terminológicas? {accuracy_hint}
3. alinhamento (alinhamento pedagógico): a resposta é adequada ao contexto educacional e
   ao nível de complexidade esperado do público-alvo?

### INSTRUÇÕES
- Avalie cada dimensão de forma independente.
- Não deixe o tamanho da resposta influenciar as notas.
- Pense passo a passo dentro da tag <reasoning>.

### FORMATO DE SAÍDA OBRIGATÓRIO
<reasoning>
Justifique brevemente cada nota.
</reasoning>
<scores>
{{"clareza": <0-10>, "acuracia": <0-10>, "alinhamento": <0-10>}}
</scores>
"""


def build_rubric_prompt(
    query: str,
    answer: str,
    reference: Optional[str] = None,
    rag_context: str = "",
    image_description: str = "",
) -> str:
    """Render the rubric prompt; the reference (gabarito) anchors the accuracy score."""
    if reference:
        ref_block = f"\nGABARITO OFICIAL (GROUND TRUTH): {reference}\n"
        accuracy_hint = "Compare com o GABARITO OFICIAL: contradizê-lo implica nota baixa."
    else:
        ref_block = ""
        accuracy_hint = "Avalie a precisão factual e lógica."
    rag_block = f"\nCONTEXTO ADICIONAL (RAG):\n{rag_context}\n" if rag_context else ""
    img_block = f"\nDESCRIÇÃO DA IMAGEM:\n{image_description}\n" if image_description else ""
    return _RUBRIC_TEMPLATE.format(
        query=query,
        answer=answer,
        ref_block=ref_block,
        rag_block=rag_block,
        img_block=img_block,
        accuracy_hint=accuracy_hint,
    )


def _canonical_key(key: Any) -> Optional[str]:
    text = unicodedata.normalize("NFKD", str(key)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z]+", "_", text.lower()).strip("_")
    return _ALIASES.get(text)


def _extract_json_object(text: str, tag: str = "scores") -> Optional[Dict[str, Any]]:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.IGNORECASE | re.DOTALL)
    candidates = [match.group(1)] if match else []
    candidates += re.findall(r"\{[^{}]*\}", text)
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_rubric_scores(text: Optional[str]) -> Optional[Dict[str, float]]:
    """Extract the three dimension scores (clamped to 0-10); ``None`` if incomplete."""
    if not text:
        return None
    obj = _extract_json_object(text)
    if not obj:
        return None
    scores: Dict[str, float] = {}
    for key, value in obj.items():
        dim = _canonical_key(key)
        if dim is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if score == score:  # NaN guard
            scores[dim] = max(0.0, min(10.0, score))
    if any(dim not in scores for dim in RUBRIC_DIMENSIONS):
        return None
    return scores


def parse_rubric_weights(raw: Any) -> Dict[str, float]:
    """Parse ``JUDGE_RUBRIC_WEIGHTS`` (JSON or mapping), falling back to the defaults."""
    obj = raw
    if isinstance(raw, (str, bytes)):
        try:
            obj = json.loads(raw)
        except (TypeError, ValueError):
            return dict(DEFAULT_RUBRIC_WEIGHTS)
    if not isinstance(obj, Mapping):
        return dict(DEFAULT_RUBRIC_WEIGHTS)
    weights: Dict[str, float] = {}
    for key, value in obj.items():
        dim = _canonical_key(key)
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if dim is not None and weight >= 0:
            weights[dim] = weight
    if any(dim not in weights for dim in RUBRIC_DIMENSIONS) or sum(weights.values()) <= 0:
        return dict(DEFAULT_RUBRIC_WEIGHTS)
    return weights


def weighted_quality(
    scores: Mapping[str, float],
    weights: Mapping[str, float],
    dims: Sequence[str] = RUBRIC_DIMENSIONS,
) -> float:
    """``Q = sum(w_i * d_i) / sum(w_i)`` over ``dims`` (0-10 scale).

    ``dims`` defaults to the full rubric, so every existing caller is unchanged.
    Passing :data:`TECH_DIMENSIONS` yields Q_tech: the same weights renormalised
    over clarity and accuracy alone, which is the technical half of the
    formative score.
    """
    total = sum(weights[d] for d in dims)
    if total <= 0:
        weights, total = DEFAULT_RUBRIC_WEIGHTS, sum(DEFAULT_RUBRIC_WEIGHTS[d] for d in dims)
    return sum(weights[d] * scores[d] for d in dims) / total


def combine_ratings(
    ratings: Sequence[Mapping[str, float]],
    weights: Mapping[str, float],
    aggregate: str = "mean",
) -> Optional[Dict[str, Any]]:
    """Aggregate valid judge ratings into Q, per-dimension scores and dispersion.

    ``aggregate="median"`` is used when a meta-judge joined a disagreeing pair,
    so the third rating breaks the tie per dimension instead of averaging it in.
    """
    valid: List[Mapping[str, float]] = [r for r in ratings if r]
    if not valid:
        return None
    reducer = statistics.median if aggregate == "median" else statistics.fmean
    dimensions = {d: float(reducer([r[d] for r in valid])) for d in RUBRIC_DIMENSIONS}
    judge_quality = [weighted_quality(r, weights) for r in valid]
    spread = len(valid) > 1
    return {
        "quality": weighted_quality(dimensions, weights),
        "dimensions": dimensions,
        "judge_quality": judge_quality,
        "dispersion": {
            "quality_std": statistics.pstdev(judge_quality) if spread else 0.0,
            "by_dimension": {
                d: (statistics.pstdev([r[d] for r in valid]) if spread else 0.0) for d in RUBRIC_DIMENSIONS
            },
        },
        "n_judges": len(valid),
        "aggregate": aggregate,
    }


def judge_scoring_mode(settings: Any) -> str:
    """Return ``rubric`` (default) or ``binary`` from ``JUDGE_SCORING_MODE``."""
    try:
        mode = str(settings.get("JUDGE_SCORING_MODE", "rubric") or "rubric").strip().lower()
    except Exception:
        mode = "rubric"
    return mode if mode in ("rubric", "binary") else "rubric"


async def rate_with_rubric(
    call_model: Callable[..., Awaitable[Tuple[str, Any]]],
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[Optional[Rating], Dict[str, Any]]:
    """Ask one judge for rubric scores; ``(None, {})`` on call or parse failure."""
    try:
        text_out, meta = await call_model(model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    except Exception as exc:
        logger.warning("[Judges] Falha juiz %s: %s", model, exc)
        return None, {}
    ratings = parse_rubric_scores(text_out)
    if ratings is None:
        logger.warning("[Judges] Saída de rubrica ilegível do juiz %s", model)
    return ratings, (meta if isinstance(meta, dict) else {})


async def score_with_rubric(
    judge_models: Sequence[str],
    rate: RateFn,
    weights: Mapping[str, float],
    *,
    meta_model: Optional[Callable[[], str]] = None,
    disagreement: float = 3.0,
    on_rating: Optional[OnRatingFn] = None,
) -> Optional[Dict[str, Any]]:
    """Run the judges in parallel and aggregate their valid ratings.

    ``rate(model)`` returns ``(ratings | None, provider_meta)``; failed judges
    are dropped. When exactly two judges disagree on Q by more than
    ``disagreement`` points, the meta-judge (resolved lazily) also rates and
    each dimension takes the median of the three. ``on_rating`` is called for
    every valid rating (persistence, judge fitness). Returns ``None`` when no
    judge produced a valid rating.
    """
    results = await asyncio.gather(*[rate(model) for model in judge_models])
    used: List[str] = []
    ratings: List[Rating] = []
    for model, (rating, info) in zip(judge_models, results):
        if rating:
            used.append(model)
            ratings.append(rating)
            if on_rating:
                on_rating(model, rating, info)
    if not ratings:
        return None

    aggregate = "mean"
    if len(ratings) == 2 and meta_model is not None:
        gap = abs(weighted_quality(ratings[0], weights) - weighted_quality(ratings[1], weights))
        if gap > disagreement:
            referee = meta_model()
            logger.info("[Judges] Divergência de rubrica (ΔQ=%.1f). Chamando Meta-Juiz %s.", gap, referee)
            rating, info = await rate(referee)
            if rating:
                used.append(referee)
                ratings.append(rating)
                aggregate = "median"
                if on_rating:
                    on_rating(referee, rating, info)

    payload = combine_ratings(ratings, weights, aggregate=aggregate)
    if payload is not None:
        payload["judges"] = used
    return payload
