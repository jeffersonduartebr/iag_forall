# -*- coding: utf-8 -*-
"""
judges.py (CORRIGIDO: Persistência de Logs Detalhados)
----------------------------------------------------
Sistema de avaliação de respostas.

Correções:
✔ Grava em 'judge_logs' (detalhe) além de 'judge_performance_log' (stats).
✔ Usa 'await' corretamente.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence, Tuple, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from .settings_dynamic import settings
from .providers_async import call_model
from .vectorstore import query_embedding
from .embeddings import embed_text

import asyncio

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] judges: %(message)s",
    )

# ============================================================
# ⚙️ Banco de dados
# ============================================================

DB_HOST = settings.DB_HOST
DB_USER = settings.DB_USER
DB_PASS = settings.DB_PASS
DB_NAME = settings.DB_NAME
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"

engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)


# ============================================================
# 📏 Configurações principais
# ============================================================

ALPHA_DECAY = float(settings.get("JUDGES_FITNESS_DECAY", 0.90))
MIN_FITNESS = float(settings.get("JUDGES_MIN_FITNESS", 0.30))
CONSIST_WINDOW_MIN = int(settings.get("JUDGES_WINDOW_MIN", 180))
DISAGREE_PERCENT = float(settings.get("JUDGES_DISAGREE_PERCENT", 0.20))

META_JUDGE_HINT = str(settings.get("META_JUDGE_PREF", "openai/gpt-4o-mini"))
MAX_TOKENS_JUDGE = int(settings.get("JUDGES_MAX_TOKENS", 32))
TEMP_JUDGE = float(settings.get("JUDGES_TEMPERATURE", 0.2))

W_FIT = float(settings.get("JUDGES_WEIGHT_FITNESS", 0.6))
W_QC = float(settings.get("JUDGES_WEIGHT_QC", 0.4))
EPSILON_RANDOM = float(settings.get("JUDGES_EPSILON", 0.10))

IMAGE_DESC_MODEL_HINT = str(settings.get("IMAGE_DESC_MODEL", "openai/gpt-4o-mini"))

VISION_VLM_CANDIDATES: List[str] = list(
    getattr(settings, "CANDIDATE_VISION_MODELS_LIST", [])
)
MULTIMODAL_VLM_CANDIDATES: List[str] = list(
    getattr(settings, "CANDIDATE_MULTIMODAL_MODELS_LIST", [])
)


# ============================================================
# 📊 Estruturas auxiliares
# ============================================================

@dataclass
class JudgeStats:
    model: str
    avg_score: float = 0.7
    avg_latency: float = 2.0
    avg_cost: float = 0.001
    consistency: float = 0.8
    fitness: float = 0.5


@dataclass
class SelectedJudge:
    model: str
    weight: float


# ============================================================
# 🧱 Utilitários
# ============================================================

def _adaptive_threshold(values: Sequence[float], base: float) -> float:
    if not values:
        return base
    median_val = statistics.median(values)
    return max(base, min(0.9, median_val * 0.6))


def _image_hash_from_b64(image_b64: Optional[str]) -> Optional[str]:
    if not image_b64:
        return None
    try:
        h = hashlib.sha256()
        h.update(image_b64.encode("utf-8", errors="ignore"))
        return h.hexdigest()
    except Exception:
        return None


# ============================================================
# 🔧 Garantir tabelas
# ============================================================

def _ensure_judge_logs_table() -> None:
    ddl_logs = """
        CREATE TABLE IF NOT EXISTS judge_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            query TEXT,
            answer TEXT,
            judge_model VARCHAR(255),
            score_before FLOAT,
            fallback_model VARCHAR(255),
            score_after FLOAT,
            event_type VARCHAR(50),
            modality VARCHAR(32) DEFAULT 'text',
            image_hash VARCHAR(128) NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci;
    """

    ddl_perf = """
        CREATE TABLE IF NOT EXISTS judge_performance_log (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            judge_model VARCHAR(255) NOT NULL,
            avg_score FLOAT DEFAULT 0,
            avg_latency FLOAT DEFAULT 0,
            avg_cost FLOAT DEFAULT 0,
            consistency FLOAT DEFAULT 0,
            fitness FLOAT DEFAULT 0,
            window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            window_end TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_judge_model (judge_model),
            INDEX idx_window_end (window_end)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl_logs))
            conn.execute(text(ddl_perf))
    except SQLAlchemyError as exc:
        logger.warning("[Judges] Falha ao criar tabelas: %s", exc)


# ============================================================
# 📈 Carregamento de métricas históricas
# ============================================================

def _load_judge_stats(window_minutes: int) -> Dict[str, JudgeStats]:
    since = datetime.utcnow() - timedelta(minutes=window_minutes)
    stats: Dict[str, JudgeStats] = {}

    try:
        _ensure_judge_logs_table()

        with engine.connect() as conn:
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

def _score_candidate(s: JudgeStats) -> float:
    qc = s.avg_score / max(s.avg_cost, 1e-6)
    qc_norm = min(10.0, 1.0 + qc ** 0.25)
    return max(0.0, W_FIT * s.fitness + W_QC * (qc_norm / 10.0))


def _choose_two(models: List[str], stats: Dict[str, JudgeStats]) -> List[SelectedJudge]:
    fitness_vals = [stats.get(m, JudgeStats(m)).fitness for m in models]
    thr = _adaptive_threshold(fitness_vals, MIN_FITNESS)
    valid = [m for m in models if stats.get(m, JudgeStats(m)).fitness >= thr]

    if len(valid) < 2:
        valid = models[:]

    if random.random() < EPSILON_RANDOM and len(valid) >= 2:
        picks = random.sample(valid, k=2)
        return [SelectedJudge(p, 1.0) for p in picks]

    scored = [(m, _score_candidate(stats.get(m, JudgeStats(m)))) for m in valid]
    total = sum(w for _, w in scored) or 1.0

    weights = [(m, w / total) for m, w in scored]

    def pick(wlist):
        r = random.random()
        acc = 0.0
        for name, w in wlist:
            acc += w
            if r <= acc:
                return name
        return wlist[-1][0]

    first = pick(weights)
    rest = [(m, w) for m, w in weights if m != first]
    second = pick(rest) if rest else first

    return [SelectedJudge(first, 1.0), SelectedJudge(second, 1.0)]


# ============================================================
# 🔎 RAG para juízes
# ============================================================

async def get_rag_context(query: str, n_results: int = 5, max_chars: int = 1500) -> str:
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


# ============================================================
# ⭐ Heurístico simples
# ============================================================

def heuristic_score(answer: str) -> float:
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
    """Salva estatísticas de performance do juiz."""
    try:
        _ensure_judge_logs_table()
        with engine.begin() as conn:
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

def _persist_judge_log(query, answer, judge_model, score, modality, image_hash=None):
    """Salva o LOG detalhado do julgamento individual."""
    try:
        # Trunca textos muito longos para economizar banco
        q_short = query[:2000] if query else ""
        a_short = answer[:4000] if answer else ""
        
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO judge_logs
                    (query, answer, judge_model, score_before, score_after, 
                     event_type, modality, image_hash, created_at)
                    VALUES (:q, :a, :jm, :sc, :sc, 'evaluation', :mod, :ih, NOW())
                    """
                ),
                {
                    "q": q_short,
                    "a": a_short,
                    "jm": judge_model,
                    "sc": score,
                    "mod": modality,
                    "ih": image_hash
                },
            )
    except Exception as exc:
        logger.warning("[Judges] persist log fail: %s", exc)


# ============================================================
# 🖼️ Descrição automática da imagem para juízes
# ============================================================

async def _describe_image_if_needed(image_b64: Optional[str], modality: str) -> str:
    if not image_b64:
        return ""

    candidates = []
    if IMAGE_DESC_MODEL_HINT:
        candidates.append(IMAGE_DESC_MODEL_HINT)
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


# ============================================================
# ⚖️ Core LLM scoring
# ============================================================

_SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)(?:\s*/\s*10)?")

def _extract_score(text: str) -> float:
    if not text: return 0.0
    clean = text.strip().lower()
    m = _SCORE_RE.search(clean)
    if m:
        try:
            return float(m.group(1))
        except:
            pass
            
    if any(w in clean for w in ["excelente", "ótima"]): return 9.0
    if any(w in clean for w in ["boa", "adequada"]): return 7.0
    if any(w in clean for w in ["regular", "parcial"]): return 5.0
    if any(w in clean for w in ["ruim", "fraca"]): return 3.0
    return 0.0


async def _meta_evaluate(query, answer, pair_scores, base_prompt):
    judge_models = getattr(settings, "JUDGE_MODELS", []) or [META_JUDGE_HINT]
    stats = _load_judge_stats(CONSIST_WINDOW_MIN)

    scored_all = [
        (m, _score_candidate(stats.get(m, JudgeStats(m)))) for m in judge_models
    ]
    scored_all.sort(key=lambda x: x[1], reverse=True)

    candidates = [META_JUDGE_HINT] + [m for m, _ in scored_all[:3]]

    a, b = pair_scores
    arb_prompt = (
        base_prompt
        + f"\n\nNotas preliminares:"
          f"\n- {a[0]}: {a[1]:.2f}/10"
          f"\n- {b[0]}: {b[1]:.2f}/10"
          f"\nReavalie e responda somente com um número."
    )

    for cm in candidates[:2]:
        try:
            text_out, _ = await call_model(
                model=cm,
                prompt=arb_prompt,
                temperature=TEMP_JUDGE,
                max_tokens=MAX_TOKENS_JUDGE,
            )
            n10 = _extract_score(text_out)
            return max(0.0, min(n10 / 10.0, 1.0))
        except:
            continue
    return 0.0


async def _llm_pair_score(query, answer, use_rag, modality, image_b64):
    ctx = await get_rag_context(query) if use_rag else ""
    img_desc = await _describe_image_if_needed(image_b64, modality)

    rag_block = f"\nContexto adicional:\n{ctx}\n" if ctx else ""
    img_block = ""
    if image_b64:
        img_block = "\nO usuário enviou uma imagem.\n"
        if img_desc:
            img_block += f"Descrição automática:\n{img_desc}\n"
            
    ih = _image_hash_from_b64(image_b64)

    prompt = (
        "Você é um avaliador técnico de respostas de IA.\n"
        "Avalie de 0 a 10 apenas a CORREÇÃO, CLAREZA e RELEVÂNCIA.\n\n"
        f"Modalidade: {modality}\n\n"
        f"Pergunta: {query}\n\nResposta: {answer}\n\n"
        f"{rag_block}{img_block}"
        "Responda SOMENTE com um número de 0 a 10."
    )

    judge_models_all = getattr(settings, "JUDGE_MODELS", []) or ["openai/gpt-4o-mini"]
    stats = _load_judge_stats(CONSIST_WINDOW_MIN)
    selected = _choose_two(judge_models_all, stats)

    results = []

    for sj in selected:
        try:
            text_out, meta = await call_model(
                model=sj.model,
                prompt=prompt,
                temperature=TEMP_JUDGE,
                max_tokens=MAX_TOKENS_JUDGE,
            )
            score10 = _extract_score(text_out)
            score01 = max(0.0, min(score10 / 10.0, 1.0))

            lat = float(meta.get("latency", 2.0))
            cost = float(meta.get("cost_per_1k", 0.001))

            speed_term = 1.0 - min(lat, 10.0) / 10.0
            fitness = (score01 * 0.7) + (speed_term * 0.3)
            consistency = 1.0 - abs(score01 - 0.5)

            # 1. Grava estatísticas agregadas
            _persist_judge_metrics(
                judge_model=sj.model,
                score=score01,
                latency=lat,
                cost=cost,
                consistency=consistency,
                fitness=fitness,
            )
            
            # 2. Grava o log DETALHADO (AQUI ERA O PONTO FALTANTE)
            _persist_judge_log(
                query=query,
                answer=answer,
                judge_model=sj.model,
                score=score10, # Salva nota 0-10 para legibilidade
                modality=modality,
                image_hash=ih
            )

            results.append((sj.model, score10, score01, meta))

        except Exception as exc:
            logger.warning("[Judges] Falha juiz %s: %s", sj.model, exc)
            results.append((sj.model, 0.0, 0.0, {"latency": 5.0}))

    if len(results) >= 2:
        n1, n2 = results[0][1], results[1][1]
        bigger, smaller = max(n1, n2), min(n1, n2)
        if bigger > 0 and (bigger - smaller) / bigger >= DISAGREE_PERCENT:
            meta = await _meta_evaluate(
                query, answer, [(results[0][0], n1), (results[1][0], n2)], prompt
            )
            base = (results[0][2] + results[1][2]) / 2.0
            return round(0.6 * meta + 0.4 * base, 3)

        return round((results[0][2] + results[1][2]) / 2.0, 3)

    return results[0][2] if results else 0.0


# ============================================================
# 🌐 API pública
# ============================================================

async def llm_based_score(query, answer, use_rag, modality, image_b64):
    return await _llm_pair_score(
        query=query,
        answer=answer,
        use_rag=use_rag,
        modality=modality,
        image_b64=image_b64,
    )


async def judge_answer(
    query: str,
    answer: str,
    use_rag: bool = False,
    modality: str = "text",
    image_b64: Optional[str] = None,
) -> List[Dict[str, Any]]:

    if not answer or not isinstance(answer, str):
        return [{"judge_id": "heuristic", "score": 0.0}]

    mode = (settings.JUDGES_MODE or "hybrid").lower().strip()

    results = []

    if mode in ("heuristic", "hybrid"):
        base = heuristic_score(answer)
        results.append({"judge_id": "heuristic", "score": round(base, 3)})

    if mode in ("llm", "hybrid"):
        score_llm = await llm_based_score(
            query=query,
            answer=answer,
            use_rag=use_rag,
            modality=modality,
            image_b64=image_b64,
        )
        results.append({"judge_id": "llm", "score": round(score_llm, 3)})

    valid = [r for r in results if "score" in r]
    return valid or [{"judge_id": "fallback", "score": 0.0}]