# -*- coding: utf-8 -*-
# Objective: Application runtime code for judges.
"""Evaluate router answers with heuristic and model-based judges.

Two LLM scoring modes are available (``JUDGE_SCORING_MODE``):

- ``rubric`` (default): two judges rate clarity, conceptual accuracy and
  pedagogical alignment on 0-10 (``services.judge_rubric``); Q is the weighted
  mean and a meta-judge breaks large disagreements by per-dimension median.
- ``binary``: CORRECT/INCORRECT verdicts with a binary meta-judge, kept for
  reference-guided benchmarks with ground truth.

Both use a short-lived verdict cache and adaptive judge selection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from .db import get_engine
from .providers_async import call_model
from .services.judge_cache import VERDICT_CACHE_SIZE, VERDICT_CACHE_TTL_S, VerdictCache  # noqa: F401
from .services.judge_calibration import (  # noqa: F401  (API pública reexportada)
    JUDGE_CALIBRATION_DDL,
    _ensure_judge_calibration_table,
    calibrate_judges,
    get_judge_calibration_metrics,
    record_judge_calibration,
    update_calibration_cache_status,
)
from .services.judge_context import (  # noqa: F401  (reexportados; chamadores resolvem por este módulo)
    IMAGE_DESC_MODEL_HINT,
    META_JUDGE_HINT,
    MULTIMODAL_VLM_CANDIDATES,
    VISION_VLM_CANDIDATES,
    _configured_local_fallback,
    _describe_image_if_needed,
    _image_hash_from_b64,
    _resolve_image_desc_model,
    _resolve_judge_models,
    _resolve_meta_judge_model,
    get_rag_context,
)
from .services.judge_rubric import (
    build_rubric_prompt,
    judge_scoring_mode,
    parse_rubric_weights,
    rate_with_rubric,
    score_with_rubric,
    weighted_quality,
)
from .services.judge_selection import (  # noqa: F401  (reexportados)
    EPSILON_RANDOM,
    MIN_FITNESS,
    W_FIT,
    W_QC,
    JudgeStats,
    SelectedJudge,
    _adaptive_threshold,
    _choose_two,
    _score_candidate,
)
from .settings_dynamic import settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] judges: %(message)s",
    )


# ============================================================
# 🚀 JUDGE VERDICT CACHE (Performance Optimization)
# ============================================================




_verdict_cache = VerdictCache()
_rubric_cache = VerdictCache()  # payloads da rubrica (dict), separados dos vereditos binários


def get_verdict_cache_stats() -> Dict[str, Any]:
    """Expose verdict-cache statistics for admin endpoints and observability."""
    return _verdict_cache.stats()

# ============================================================
# ⚙️ Banco de dados
# ============================================================

def _get_judge_engine():
    """Return the shared database engine for judge persistence."""
    return get_engine()


# ============================================================
# 📏 Configurações principais
# ============================================================

def _safe_setting_float(key: str, default: float) -> float:
    """Read one float setting defensively for judge configuration."""
    try:
        return float(settings.get(key, default))
    except Exception:
        return float(default)


def _safe_setting_int(key: str, default: int) -> int:
    """Read one integer setting defensively for judge configuration."""
    try:
        return int(settings.get(key, default))
    except Exception:
        return int(default)


ALPHA_DECAY = _safe_setting_float("JUDGES_FITNESS_DECAY", 0.90)
CONSIST_WINDOW_MIN = _safe_setting_int("JUDGES_WINDOW_MIN", 180)

# Meta-Juiz preferencial (deve ser um modelo forte)

# Aumentado para permitir CoT (Raciocínio)
MAX_TOKENS_JUDGE = 512
TEMP_JUDGE = 0.0 # Temperatura zero para determinismo máximo













# ============================================================
# 📊 Estruturas auxiliares
# ============================================================





# ============================================================
# 🧱 Utilitários
# ============================================================





# ============================================================
# 🔧 Garantir tabelas
# ============================================================

def _ensure_judge_logs_table() -> None:
    """Placeholder for judge log bootstrap kept for backward compatibility.

    Table creation is currently handled elsewhere in the application startup
    path, so this function remains as a stable no-op hook for older callers.
    """
    pass


# ============================================================
# 📈 Carregamento de métricas históricas
# ============================================================

_JUDGE_STATS_TTL_S = 60.0
_judge_stats_cache: Dict[int, Tuple[float, Dict[str, JudgeStats]]] = {}


def _load_judge_stats_cached(window_minutes: int) -> Dict[str, JudgeStats]:
    """``_load_judge_stats`` with a 60 s in-process cache (one aggregation per minute)."""
    now = time.monotonic()
    hit = _judge_stats_cache.get(window_minutes)
    if hit is not None and now - hit[0] < _JUDGE_STATS_TTL_S:
        return hit[1]
    stats = _load_judge_stats(window_minutes)
    _judge_stats_cache[window_minutes] = (now, stats)
    return stats


def _load_judge_stats(window_minutes: int) -> Dict[str, JudgeStats]:
    """Load recent judge-performance aggregates used for adaptive selection."""
    since = datetime.utcnow() - timedelta(minutes=window_minutes)
    stats: Dict[str, JudgeStats] = {}

    try:
        with _get_judge_engine().connect() as conn:
            rs = conn.execute(
                text(
                    """
                    SELECT judge_model,
                           AVG(avg_score) AS a_score,
                           AVG(avg_latency) AS a_lat,
                           AVG(avg_cost) AS a_cost,
                           AVG(consistency) AS a_cons,
                           AVG(fitness) AS a_fit
                    FROM judge_performance_log
                    WHERE window_end >= :since
                    GROUP BY judge_model
                    """
                ),
                {"since": since},
            ).fetchall()

        for r in rs:
            m = r._mapping
            stats[m["judge_model"]] = JudgeStats(
                model=m["judge_model"],
                avg_score=float(m["a_score"] or 0.7),
                avg_latency=float(m["a_lat"] or 2.0),
                avg_cost=float(m["a_cost"] or 0.001),
                consistency=float(m["a_cons"] or 0.8),
                fitness=float(m["a_fit"] or 0.5),
            )

    except Exception as exc:
        logger.info("[Judges] Histórico indisponível: %s", exc)

    return stats


# ============================================================
# 🔢 Seleção adaptativa de juízes
# ============================================================





# ============================================================
# 🔎 RAG para juízes
# ============================================================



# ============================================================
# ⭐ Heurístico simples
# ============================================================

def heuristic_score(answer: str) -> float:
    """Estimate answer quality cheaply when LLM judging is disabled or blended."""
    try:
        s = len(answer.strip())
        if s == 0:
            return 0.0
        score = min(1.0, 0.2 + s / 500.0)
        if any(p in answer for p in [".", "?", "!"]):
            score += 0.2
        return min(score, 1.0)
    except Exception:
        return 0.0


# ============================================================
# 🧮 Persistência (Métricas + Logs)
# ============================================================

def _persist_judge_metrics(judge_model, score, latency, cost, consistency, fitness):
    """Persist one aggregate judge-performance sample to the database."""
    try:
        with _get_judge_engine().begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO judge_performance_log
                    (judge_model, avg_score, avg_latency, avg_cost,
                     consistency, fitness, window_start, window_end)
                    VALUES (:jm, :ascore, :alat, :acost, :cons, :fit,
                            NOW() - INTERVAL 10 MINUTE, NOW())
                    """
                ),
                {
                    "jm": judge_model, "ascore": score, "alat": latency,
                    "acost": cost, "cons": consistency, "fit": fitness,
                },
            )
    except Exception as exc:
        logger.warning("[Judges] persist metrics fail: %s", exc)

def _persist_judge_log(query, answer, judge_model, score, modality, image_hash=None, rubric=None):
    """Persist one raw judge evaluation event for audit and analysis.

    ``rubric`` (per-dimension scores) is stored in ``judge_logs.rubric_json``.
    """
    try:
        q_short = query[:2000] if query else ""
        a_short = answer[:4000] if answer else ""
        params = {
            "q": q_short,
            "a": a_short,
            "jm": judge_model,
            "sc": score,
            "mod": modality,
            "ih": image_hash,
        }
        if rubric is None:
            sql = """
                INSERT INTO judge_logs
                (query, answer, judge_model, score_before, score_after,
                 event_type, modality, image_hash, created_at)
                VALUES (:q, :a, :jm, :sc, :sc, 'evaluation', :mod, :ih, NOW())
            """
        else:
            sql = """
                INSERT INTO judge_logs
                (query, answer, judge_model, score_before, score_after,
                 event_type, modality, image_hash, rubric_json, created_at)
                VALUES (:q, :a, :jm, :sc, :sc, 'evaluation', :mod, :ih, :rubric, NOW())
            """
            params["rubric"] = json.dumps(rubric, ensure_ascii=False)

        with _get_judge_engine().begin() as conn:
            conn.execute(text(sql), params)
    except Exception as exc:
        logger.warning("[Judges] persist log fail: %s", exc)


# ============================================================
# 🖼️ Descrição automática da imagem para juízes
# ============================================================



# ============================================================
# ⚖️ Core LLM scoring (XML + BINARY + META-JUDGE)
# ============================================================

def _extract_binary_verdict(text: str) -> float:
    """
    Extrai o veredito binário da resposta do juiz.
    Retorna 10.0 (CORRECT) ou 0.0 (INCORRECT).
    """
    if not text: return 0.0

    # 1. Tenta extrair de XML (Mais robusto)
    match = re.search(r"<verdict>\s*(.*?)\s*</verdict>", text, re.IGNORECASE | re.DOTALL)
    if match:
        content = match.group(1).strip().upper()
        if "INCORRECT" in content: return 0.0
        if "CORRECT" in content: return 10.0

    # 2. Fallback: Procura no texto inteiro
    text_upper = text.upper()
    if "VERDICT: INCORRECT" in text_upper or "VEREDITO: INCORRETO" in text_upper: return 0.0
    if "VERDICT: CORRECT" in text_upper or "VEREDITO: CORRETO" in text_upper: return 10.0

    return 0.0


async def _meta_evaluate_binary(query, answer, conflicting_verdicts, base_prompt, reference=None):
    """
    Meta-Juiz para desempate binário.
    Acionado quando há conflito direto (0 vs 10).
    """
    meta_model = _resolve_meta_judge_model()

    v1_model, v1_score = conflicting_verdicts[0]
    v2_model, v2_score = conflicting_verdicts[1]

    v1_text = "CORRETO" if v1_score > 5 else "INCORRETO"
    v2_text = "CORRETO" if v2_score > 5 else "INCORRETO"

    ref_block = f"\nGABARITO OFICIAL: {reference}\n" if reference else ""

    arb_prompt = f"""
Você é um Juiz Supremo de IA. Existe um conflito entre dois avaliadores sobre a resposta abaixo.
Sua tarefa é decidir quem está certo.

PERGUNTA: {query}
{ref_block}
RESPOSTA DO MODELO: {answer}

--- CONFLITO ---
Avaliador 1 ({v1_model}): Veredito {v1_text}
Avaliador 2 ({v2_model}): Veredito {v2_text}
----------------

INSTRUÇÕES:
1. Analise a resposta friamente em relação à pergunta (e ao gabarito, se houver).
2. Decida se a resposta é FACTUALMENTE CORRETA ou INCORRETA.
3. Dê o veredito final de desempate.

SAÍDA OBRIGATÓRIA:
<reasoning>
Explique quem está certo e por quê.
</reasoning>
<verdict>
CORRECT ou INCORRECT
</verdict>
"""

    try:
        text_out, _ = await call_model(
            model=meta_model,
            prompt=arb_prompt,
            temperature=0.0,
            max_tokens=512,
        )
        return _extract_binary_verdict(text_out)
    except Exception as e:
        logger.error(f"[MetaJudge] Falha: {e}")
        return 0.0


async def _llm_pair_score(query, answer, use_rag, modality, image_b64, reference=None):
    """Score one answer with two judges and an optional meta-judge tie-breaker.

    The function first checks the verdict cache, then evaluates the answer with
    the selected judge pair. When the two judges disagree on the binary verdict,
    a stronger meta-judge resolves the conflict.
    """
    cached_score = _verdict_cache.get(query, answer)
    if cached_score is not None:
        logger.debug(f"[Judges] Cache HIT: score={cached_score}")
        return cached_score

    ctx = await get_rag_context(query) if use_rag else ""
    img_desc = await _describe_image_if_needed(image_b64, modality)

    rag_block = f"\nCONTEXTO ADICIONAL (RAG):\n{ctx}\n" if ctx else ""
    img_block = f"\nDESCRIÇÃO DA IMAGEM:\n{img_desc}\n" if img_desc else ""

    # --- ESTRATÉGIA 1: REFERENCE-GUIDED ---
    if reference:
        ref_block = f"\nGABARITO OFICIAL (GROUND TRUTH): {reference}\n"
        task_desc = "Compare a RESPOSTA DO MODELO com o GABARITO OFICIAL."
    else:
        ref_block = ""
        task_desc = "Avalie a precisão factual e lógica da resposta."

    # --- ESTRATÉGIA 3: BINARY CLASSIFICATION + CoT ---
    prompt = f"""
Você é um juiz técnico imparcial. Sua tarefa é avaliar se a resposta do modelo está CORRETA ou INCORRETA.

PERGUNTA: {query}
{ref_block}
{rag_block}
{img_block}
RESPOSTA DO MODELO: {answer}

### INSTRUÇÕES DE AVALIAÇÃO:
1. Pense passo a passo dentro da tag <reasoning>.
2. {task_desc}
3. Ignore o estilo, tom ou tamanho do texto. Foque apenas na FATUALIDADE e LÓGICA.
4. Se a resposta final contradizer o gabarito ou contiver erros factuais graves, o veredito é INCORRECT.
5. Se a resposta final estiver correta (mesmo que breve), o veredito é CORRECT.

### FORMATO DE SAÍDA OBRIGATÓRIO:
<reasoning>
Descreva aqui os erros ou acertos encontrados.
</reasoning>
<verdict>
CORRECT ou INCORRECT
</verdict>
"""

    judge_models_all = _resolve_judge_models()
    stats = await asyncio.to_thread(_load_judge_stats_cached, CONSIST_WINDOW_MIN)
    selected = _choose_two(judge_models_all, stats)

    # --- PARALLEL JUDGE EVALUATION (Performance Optimization) ---
    # Evaluate judges in parallel using asyncio.gather for -500ms to -1s latency reduction
    async def _evaluate_single_judge(sj: SelectedJudge) -> Tuple[str, float, float, Dict[str, Any]]:
        """Execute one judge model call and return normalized scoring metadata."""
        try:
            text_out, meta = await call_model(
                model=sj.model,
                prompt=prompt,
                temperature=0.0,  # Determinístico
                max_tokens=MAX_TOKENS_JUDGE,
            )

            # Extração Binária (10.0 ou 0.0)
            score10 = _extract_binary_verdict(text_out)
            score01 = score10 / 10.0

            lat = float(meta.get("latency", 2.0))
            cost = float(meta.get("cost_per_1k", 0.001))

            speed_term = 1.0 - min(lat, 10.0) / 10.0
            fitness = (score01 * 0.7) + (speed_term * 0.3)
            consistency = 1.0

            await asyncio.to_thread(
                _persist_judge_metrics,
                judge_model=sj.model,
                score=score01,
                latency=lat,
                cost=cost,
                consistency=consistency,
                fitness=fitness,
            )
            await asyncio.to_thread(
                _persist_judge_log,
                query=query,
                answer=answer,
                judge_model=sj.model,
                score=score10,
                modality=modality,
                image_hash=None,
            )

            return (sj.model, score10, score01, meta)

        except Exception as exc:
            logger.warning("[Judges] Falha juiz %s: %s", sj.model, exc)
            return (sj.model, 0.0, 0.0, {"latency": 5.0})

    # Run all judge evaluations in parallel
    judge_tasks = [_evaluate_single_judge(sj) for sj in selected]
    results = await asyncio.gather(*judge_tasks, return_exceptions=False)

    if not results:
        _verdict_cache.set(query, answer, 0.0)
        return 0.0

    # --- LÓGICA DE DESEMPATE (META-JUIZ) ---
    if len(results) == 1:
        final_score = results[0][2]  # score01
        _verdict_cache.set(query, answer, final_score)
        return final_score

    score_a = results[0][2]  # score01
    score_b = results[1][2]  # score01

    # Se houver conflito (0 vs 1)
    if score_a != score_b:
        logger.info(f"[Judges] Conflito ({results[0][0]}={score_a} vs {results[1][0]}={score_b}). Chamando Meta-Juiz.")

        conflicting_data = [
            (results[0][0], results[0][1]),
            (results[1][0], results[1][1])
        ]

        final_score_10 = await _meta_evaluate_binary(
            query, answer, conflicting_data, prompt, reference
        )
        final_score = final_score_10 / 10.0
        _verdict_cache.set(query, answer, final_score)
        return final_score

    # Se concordaram (Early Exit - ambos concordam, não precisa meta-juiz)
    _verdict_cache.set(query, answer, score_a)
    return score_a


# ============================================================
# 📐 Rubrica de 3 dimensões (clareza, acurácia, alinhamento)
# ============================================================

async def _llm_rubric_score(query, answer, use_rag, modality, image_b64, reference=None) -> Optional[Dict[str, Any]]:
    """Score one answer on the three-dimension rubric with two judges.

    Orchestration (failed judges dropped, meta-judge median on disagreement) lives
    in ``services.judge_rubric.score_with_rubric``. Returns ``None`` when no judge
    produced a valid rating.
    """
    cached = _rubric_cache.get(query, answer)
    if cached is not None:
        return cached

    ctx = await get_rag_context(query) if use_rag else ""
    img_desc = await _describe_image_if_needed(image_b64, modality)
    prompt = build_rubric_prompt(query, answer, reference=reference, rag_context=ctx, image_description=img_desc)
    try:
        weights = parse_rubric_weights(settings.get("JUDGE_RUBRIC_WEIGHTS", None))
    except Exception:
        weights = parse_rubric_weights(None)

    rated: List[Tuple[str, Dict[str, float], Dict[str, Any]]] = []

    def _persist_rated() -> None:
        for model, ratings, meta in rated:
            q01 = weighted_quality(ratings, weights) / 10.0
            lat = float(meta.get("latency", 2.0) or 2.0)
            fit = q01 * 0.7 + (1.0 - min(lat, 10.0) / 10.0) * 0.3
            cost = float(meta.get("cost_per_1k", 0.0) or 0.0)
            _persist_judge_metrics(judge_model=model, score=q01, latency=lat, cost=cost, consistency=1.0, fitness=fit)
            _persist_judge_log(query, answer, model, q01 * 10.0, modality, rubric=ratings)

    stats = await asyncio.to_thread(_load_judge_stats_cached, CONSIST_WINDOW_MIN)
    selected = _choose_two(_resolve_judge_models(), stats)
    payload = await score_with_rubric(
        [sj.model for sj in selected],
        lambda model: rate_with_rubric(call_model, model, prompt, TEMP_JUDGE, MAX_TOKENS_JUDGE),
        weights,
        meta_model=_resolve_meta_judge_model,
        disagreement=_safe_setting_float("JUDGE_RUBRIC_DISAGREEMENT", 3.0),
        on_rating=lambda model, ratings, meta: rated.append((model, ratings, meta)),
    )
    if rated:
        await asyncio.to_thread(_persist_rated)  # uma ida à thread para todas as gravações
    if payload is not None:
        _rubric_cache.set(query, answer, payload)
    return payload


# ============================================================
# 🌐 API pública
# ============================================================

async def llm_based_score(query, answer, use_rag, modality, image_b64, reference=None):
    """Public wrapper around the pairwise LLM judging path."""
    return await _llm_pair_score(
        query=query,
        answer=answer,
        use_rag=use_rag,
        modality=modality,
        image_b64=image_b64,
        reference=reference
    )


async def judge_answer(
    query: str,
    answer: str,
    use_rag: bool = False,
    modality: str = "text",
    image_b64: Optional[str] = None,
    reference: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Run the configured judge pipeline and return normalized judge results.

    The function can emit heuristic-only, LLM-only, or hybrid outputs depending
    on runtime settings. Every returned item contains a ``judge_id`` and a
    normalized ``score`` in the 0-1 range used by the router. In rubric mode the
    LLM item (``llm_rubric``) also carries ``dimensions`` and ``dispersion``; if
    every LLM judge fails in ``llm`` mode the item is ``heuristic_fallback``.
    """
    if not answer or not isinstance(answer, str):
        return [{"judge_id": "heuristic", "score": 0.0}]

    mode = (settings.JUDGES_MODE or "hybrid").lower().strip()

    results = []

    if mode in ("heuristic", "hybrid"):
        base = heuristic_score(answer)
        results.append({"judge_id": "heuristic", "score": round(base, 3)})

    if mode in ("llm", "hybrid") and judge_scoring_mode(settings) == "rubric":
        rubric = await _llm_rubric_score(
            query=query,
            answer=answer,
            use_rag=use_rag,
            modality=modality,
            image_b64=image_b64,
            reference=reference,
        )
        if rubric is not None:
            results.append({
                "judge_id": "llm_rubric",
                "score": round(rubric["quality"] / 10.0, 3),
                "dimensions": rubric["dimensions"],
                "dispersion": rubric["dispersion"],
                "n_judges": rubric["n_judges"],
                "judges": rubric.get("judges", []),
            })
        elif mode == "llm":
            # Nenhum juiz LLM respondeu: aproximação heurística, sinalizada como tal.
            results.append({"judge_id": "heuristic_fallback", "score": round(heuristic_score(answer), 3)})
    elif mode in ("llm", "hybrid"):
        score_llm = await llm_based_score(
            query=query,
            answer=answer,
            use_rag=use_rag,
            modality=modality,
            image_b64=image_b64,
            reference=reference
        )
        results.append({"judge_id": "llm", "score": round(score_llm, 3)})

    valid = [r for r in results if "score" in r]
    return valid or [{"judge_id": "fallback", "score": 0.0}]
