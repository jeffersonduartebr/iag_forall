# -*- coding: utf-8 -*-
"""
judges.py
----------------------------------------------------
Sistema de avaliação de respostas (juízes LLM + heurístico + RAG)
com seleção adaptativa e meta-avaliação automática (desacordo ≥ 20%).

Principais pontos:
- Seleciona 2 juízes por rodada dentre `settings.JUDGE_MODELS`.
- Persistência das métrricas no banco (judge_logs e judge_performance_log).
- Seleção ponderada por fitness × (qualidade/custo) + exploração ε-greedy.
- Meta-avaliação automática com terceiro juiz em caso de divergência.
"""

from __future__ import annotations

import logging
import random
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from .settings_dynamic import settings
from .providers import call_model
from .vectorstore import query_embedding
from .embeddings import embed_text

logger = logging.getLogger(__name__)

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
# 🧱 Funções utilitárias
# ============================================================

def _ema(prev: float, new: float, alpha: float) -> float:
    return alpha * prev + (1.0 - alpha) * new


def _adaptive_threshold(values: Sequence[float], base: float) -> float:
    if not values:
        return base
    median_val = statistics.median(values)
    return max(base, min(0.9, median_val * 0.6))


def _ensure_judge_logs_table() -> None:
    """Cria tabelas se não existirem."""
    ddl = """
        CREATE TABLE IF NOT EXISTS judge_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            query TEXT,
            answer TEXT,
            judge_model VARCHAR(255),
            score_before FLOAT,
            fallback_model VARCHAR(255),
            score_after FLOAT,
            event_type VARCHAR(50)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci;
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
            conn.execute(text(ddl))
    except SQLAlchemyError as exc:
        logger.warning("[Judges] Falha ao garantir tabelas: %s", exc)


# ============================================================
# 📈 Métricas históricas
# ============================================================

def _load_judge_stats(window_minutes: int) -> Dict[str, JudgeStats]:
    since = datetime.utcnow() - timedelta(minutes=window_minutes)
    stats: Dict[str, JudgeStats] = {}
    try:
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
        logger.info("[Judges] Métricas históricas indisponíveis: %s", exc)
    return stats


# ============================================================
# 🔢 Seleção de juízes
# ============================================================

def _score_candidate(s: JudgeStats) -> float:
    qc = s.avg_score / max(1e-6, s.avg_cost)
    qc_norm = min(10.0, 1.0 + (qc ** 0.25))
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

    def weighted_pick(wlist: List[Tuple[str, float]]) -> str:
        r = random.random()
        acc = 0.0
        for name, w in wlist:
            acc += w
            if r <= acc:
                return name
        return wlist[-1][0]

    first = weighted_pick(weights)
    rest = [(m, w) for m, w in weights if m != first]
    second = weighted_pick(rest) if rest else first
    return [SelectedJudge(first, 1.0), SelectedJudge(second, 1.0)]


# ============================================================
# 🔎 RAG opcional
# ============================================================

async def get_rag_context(query: str, n_results: int = 5, max_chars: int = 1500) -> str:
    try:
        query_vec = await embed_text(query)
        collection = settings.get("RAG_COLLECTION_NAME", "knowledge_base")
        results = await query_embedding(collection, query_vec, n_results=n_results)
        if not results or "documents" not in results or not results["documents"]:
            return ""
        docs = results["documents"][0]
        context = "\n\n".join(docs).strip()
        return (context[:max_chars] + "...") if len(context) > max_chars else context
    except Exception as exc:
        logger.warning("[Judges] Falha ao obter contexto RAG: %s", exc)
        return ""


# ============================================================
# 🧠 Heurístico simples
# ============================================================

def heuristic_score(answer: str) -> float:
    try:
        length = len(answer.strip())
        if length == 0:
            return 0.0
        score = min(1.0, 0.2 + (length / 500.0))
        if any(p in answer for p in [".", "?", "!"]):
            score += 0.2
        return min(score, 1.0)
    except Exception:
        return 0.0


# ============================================================
# 💾 Persistência de performance dos juízes
# ============================================================

def _persist_judge_metrics(judge_model: str,
                           score: float,
                           latency: float,
                           cost: float,
                           consistency: float,
                           fitness: float) -> None:
    """Grava uma amostra de métricas do juiz no banco."""
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
                    "jm": judge_model,
                    "ascore": float(score),
                    "alat": float(latency),
                    "acost": float(cost),
                    "cons": float(consistency),
                    "fit": float(fitness),
                },
            )
    except Exception as exc:
        logger.warning("[Judges] Falha ao gravar métricas (%s): %s", judge_model, exc)


# ============================================================
# ⚖️ Julgamento LLM (2 juízes + meta-avaliação)
# ============================================================

async def _llm_pair_score(query: str, answer: str, use_rag: bool) -> float:
    context = await get_rag_context(query) if use_rag else ""
    rag_block = f"\nContexto adicional (via RAG):\n{context}\n" if context else ""

    prompt = (
        "Você é um avaliador de respostas de IA.\n"
        "Avalie a resposta abaixo considerando:\n"
        "1️⃣ Correção técnica e factual\n"
        "2️⃣ Clareza e coerência textual\n"
        "3️⃣ Relevância ao que foi perguntado\n\n"
        f"Pergunta: {query}\n\nResposta do modelo: {answer}\n\n"
        f"{rag_block}Responda SOMENTE com um número entre 0 e 10.\n"
        "FORMATO OBRIGATÓRIO: apenas o número (ex.: 8.7)"
    )

    judge_models_all = getattr(settings, "JUDGE_MODELS", []) or ["openai/gpt-4o-mini"]
    stats = _load_judge_stats(CONSIST_WINDOW_MIN)
    selected = _choose_two(judge_models_all, stats)
    results: List[Tuple[str, float, float, Dict[str, float]]] = []

    for sj in selected:
        try:
            text_out, meta = call_model(
                model=sj.model,
                prompt=prompt,
                temperature=TEMP_JUDGE,
                max_tokens=MAX_TOKENS_JUDGE,
            )
            note_10 = _extract_score(text_out)
            note_01 = max(0.0, min(note_10 / 10.0, 1.0))
            latency = float(meta.get("latency", 2.0))
            cost = float(meta.get("cost_per_1k", 0.001))
            # Fitness simples: 70% nota, 30% rapidez relativa (cap 10s)
            fitness = (note_01 * 0.7) + (1.0 - min(latency, 10.0) / 10.0) * 0.3
            # Consistência: quanto mais perto de 0.5 (meio da faixa), menor;
            # aqui tratamos como "conservador": distância do meio reduz consistência
            consistency = 1.0 - abs(note_01 - 0.5)

            _persist_judge_metrics(
                judge_model=sj.model,
                score=note_01,
                latency=latency,
                cost=cost,
                consistency=consistency,
                fitness=fitness,
            )
            results.append((sj.model, note_10, note_01, meta))
        except Exception as exc:
            logger.warning("[Judges] Falha no juiz %s: %s", sj.model, exc)
            results.append((sj.model, 0.0, 0.0, {"latency": 5.0, "cost_per_1k": 0.0}))

    if len(results) >= 2:
        n1_10, n2_10 = results[0][1], results[1][1]
        maior, menor = max(n1_10, n2_10), min(n1_10, n2_10)
        if maior > 0 and (maior - menor) / maior >= DISAGREE_PERCENT:
            meta_score = await _meta_evaluate(
                query,
                answer,
                [(results[0][0], n1_10), (results[1][0], n2_10)],
                prompt,
            )
            base_pair = (results[0][2] + results[1][2]) / 2.0
            return round(0.6 * meta_score + 0.4 * base_pair, 3)
        return round((results[0][2] + results[1][2]) / 2.0, 3)

    return results[0][2] if results else 0.0


async def _meta_evaluate(
    query: str,
    answer: str,
    pair_scores: List[Tuple[str, float]],
    base_prompt: str,
) -> float:
    judge_models_all = getattr(settings, "JUDGE_MODELS", []) or [META_JUDGE_HINT]
    stats = _load_judge_stats(CONSIST_WINDOW_MIN)

    scored_all = [
        (m, _score_candidate(stats.get(m, JudgeStats(m)))) for m in judge_models_all
    ]
    scored_all.sort(key=lambda x: x[1], reverse=True)
    candidates = [META_JUDGE_HINT] + [m for m, _ in scored_all[:3]]

    a, b = pair_scores
    arb_prompt = (
        base_prompt
        + f"\n\nNotas preliminares:"
          f"\n- {a[0]}: {a[1]:.2f}/10"
          f"\n- {b[0]}: {b[1]:.2f}/10"
          f"\nReavalie e responda SOMENTE com um número entre 0 e 10."
    )

    for cm in candidates[:2]:
        try:
            text_out, _ = call_model(
                model=cm,
                prompt=arb_prompt,
                temperature=TEMP_JUDGE,
                max_tokens=MAX_TOKENS_JUDGE,
            )
            note_10 = _extract_score(text_out)
            return max(0.0, min(note_10 / 10.0, 1.0))
        except Exception as exc:
            logger.warning("[Judges] Falha meta-avaliação %s: %s", cm, exc)

    return 0.0


# ============================================================
# 🔍 Extração robusta
# ============================================================

_SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)(?:\s*/\s*10)?")

def _extract_score(text: str) -> float:
    if not text:
        return 0.0
    clean = text.strip().lower()
    match = _SCORE_RE.search(clean)
    if match:
        try:
            return max(0.0, min(float(match.group(1)), 10.0))
        except ValueError:
            return 0.0
    if any(w in clean for w in ["excelente", "ótima", "perfeita"]):
        return 9.0
    if any(w in clean for w in ["boa", "adequada", "razoável"]):
        return 7.0
    if any(w in clean for w in ["regular", "parcial"]):
        return 5.0
    if any(w in clean for w in ["ruim", "fraca", "errada"]):
        return 3.0
    if any(w in clean for w in ["péssima", "horrível"]):
        return 1.0
    return 0.0


# ============================================================
# 🌐 API pública
# ============================================================

async def get_rag_context_blocking(query: str) -> str:
    """Compat: helper explícito para recuperar contexto RAG."""
    return await get_rag_context(query)


async def llm_based_score(query: str, answer: str, use_rag: bool) -> float:
    """Ponto de entrada usado externamente para obter nota 0–1."""
    return await _llm_pair_score(query, answer, use_rag)


def log_fallback_event(
    query: str,
    answer: str,
    model: str,
    score_before: float | None,
    fallback_model: str,
    score_after: float,
    event_type: str,
) -> None:
    """Registra evento (fallback/substituição) no banco."""
    try:
        _ensure_judge_logs_table()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO judge_logs
                    (query, answer, judge_model, score_before,
                     fallback_model, score_after, event_type)
                    VALUES (:q, :a, :jm, :sb, :fb, :sa, :ev)
                    """
                ),
                {
                    "q": query,
                    "a": answer,
                    "jm": model,
                    "sb": score_before,
                    "fb": fallback_model,
                    "sa": score_after,
                    "ev": event_type,
                },
            )
    except Exception as exc:
        logger.error("[Judges] Erro ao registrar auditoria: %s", exc)


async def judge_answer(
    query: str,
    answer: str,
    use_rag: bool = False,
) -> List[Dict[str, Any]]:
    """
    Retorna lista com ao menos um dicionário {"judge_id": ..., "score": ...}
    onde 'score' está em escala 0–1.
    """
    try:
        if not answer or not isinstance(answer, str):
            logger.warning("[Judges] Resposta vazia — score = 0.")
            return [{"judge_id": "heuristic", "score": 0.0}]

        mode = (settings.JUDGES_MODE or "hybrid").lower().strip()
        results: List[Dict[str, Any]] = []

        # Heurístico sempre no modo "heuristic" ou "hybrid"
        if mode in ("heuristic", "hybrid"):
            base_score = heuristic_score(answer)
            results.append({"judge_id": "heuristic", "score": round(base_score, 3)})

        # LLM-based nos modos "llm" ou "hybrid"
        if mode in ("llm", "hybrid"):
            llm_score = await _llm_pair_score(query, answer, use_rag)
            results.append({"judge_id": "llm", "score": round(llm_score, 3)})

        valid = [r for r in results if "score" in r]
        return valid or [{"judge_id": "fallback", "score": 0.0}]
    except Exception as exc:
        logger.error("[Judges] Erro inesperado no julgamento: %s", exc)
        return [{"judge_id": "error", "score": 0.0}]
