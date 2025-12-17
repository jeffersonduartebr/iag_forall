# -*- coding: utf-8 -*-
"""
judges.py (VERSÃO FINAL: Binary Verdict + Tie-Breaker Meta-Judge)
-----------------------------------------------------------------
Sistema de avaliação de respostas.

Mudanças Arquiteturais:
1. Avaliação Binária: O juiz decide apenas entre CORRECT (10) ou INCORRECT (0).
2. Chain-of-Thought (CoT): Obrigatório para evitar "chutes".
3. Meta-Juiz de Desempate: Acionado apenas em conflito direto (0 vs 10).
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

# Meta-Juiz preferencial (deve ser um modelo forte)
META_JUDGE_HINT = str(settings.get("META_JUDGE_PREF", "openai/gpt-5.1"))

# Aumentado para permitir CoT (Raciocínio)
MAX_TOKENS_JUDGE = 512 
TEMP_JUDGE = 0.0 # Temperatura zero para determinismo máximo

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
    # Assume-se que o db_manager.py ou alembic já criou as tabelas
    pass 


# ============================================================
# 📈 Carregamento de métricas históricas
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
    try:
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
    try:
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
    meta_model = META_JUDGE_HINT
    
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

    judge_models_all = getattr(settings, "JUDGE_MODELS", []) or ["openai/gpt-5.1"]
    stats = _load_judge_stats(CONSIST_WINDOW_MIN)
    selected = _choose_two(judge_models_all, stats)

    results = []

    for sj in selected:
        try:
            text_out, meta = await call_model(
                model=sj.model,
                prompt=prompt,
                temperature=0.0, # Determinístico
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

            _persist_judge_metrics(
                judge_model=sj.model,
                score=score01,
                latency=lat,
                cost=cost,
                consistency=consistency,
                fitness=fitness,
            )
            
            _persist_judge_log(
                query=query,
                answer=answer,
                judge_model=sj.model,
                score=score10,
                modality=modality,
                image_hash=None
            )

            results.append((sj.model, score10, score01, meta))

        except Exception as exc:
            logger.warning("[Judges] Falha juiz %s: %s", sj.model, exc)
            results.append((sj.model, 0.0, 0.0, {"latency": 5.0}))

    if not results:
        return 0.0

    # --- LÓGICA DE DESEMPATE (META-JUIZ) ---
    if len(results) == 1:
        return results[0][1] # score01

    score_a = results[0][2] # score01
    score_b = results[1][2] # score01

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
        return final_score_10 / 10.0

    # Se concordaram
    return score_a


# ============================================================
# 🌐 API pública
# ============================================================

async def llm_based_score(query, answer, use_rag, modality, image_b64, reference=None):
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
            reference=reference
        )
        results.append({"judge_id": "llm", "score": round(score_llm, 3)})

    valid = [r for r in results if "score" in r]
    return valid or [{"judge_id": "fallback", "score": 0.0}]
