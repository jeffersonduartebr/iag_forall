# router_core.py
import logging
import random
import time
import asyncio
from .providers import call_model, _ensure_ollama_model
from app.semantic_cache import check_cache, store_cache
from app.observability import (
    ROUTER_MODEL_COST, ROUTER_QUALITY_AVG, ROUTER_COST_SAVINGS,
    ROUTER_LOCAL_USAGE_RATIO, ROUTER_COST_PER_QUERY
)
from .settings import settings
from app.bandits import select_model
from app.judges import judge_answer
from app.metrics_collector import update_model_metrics  
import numpy as np 

logger = logging.getLogger(__name__)

CANDIDATE_MODELS = settings.CANDIDATE_MODELS_LIST

# Modelos proibidos (embedding, não devem ser usados para texto)
BLOCKED_PREFIXES = ("nomic-embed", "text-embedding", "bge-", "e5-")


# -------------------------------------------------------------------
# Função principal de roteamento + cache semântico
# -------------------------------------------------------------------
async def route_and_answer(
    query: str, 
    system_prompt: str = "", 
    use_rag: bool = False,
    max_tokens: int = 1024,      # 👈 Re-adicionado
    temperature: float = 0.5     # 👈 Re-adicionado
):
    """
    Decide qual modelo usar e gera resposta via call_model().
    Agora com cache semântico híbrido (Redis + ChromaDB).
    """
    start_time = time.time()

    # 0️⃣ Verifica cache semântico antes de processar
    cached = await check_cache(query)
    if cached:
        logger.info(
            f"[router_core] ✅ Cache HIT — similaridade={cached['similarity']:.2f}. "
            f"Retornando resposta do cache sem chamar modelo."
        )
        return {
            "model": "semantic_cache",
            "answer": cached["text"],
            "latency_s": round(time.time() - start_time, 3),
            "cost_per_1k": 0.0,
            "quality": 9.5,  # confiança alta para cache
            "metadata": {"cached": True, "similarity": cached["similarity"]},
        }

    # 1️⃣ Filtra modelos válidos
    valid_models = [
        m for m in CANDIDATE_MODELS
        if isinstance(m, str) and not any(m.startswith(prefix) for prefix in BLOCKED_PREFIXES)
    ]
    if not valid_models:
        logger.error("[router_core] Nenhum modelo válido disponível para geração.")
        raise RuntimeError("Nenhum modelo válido disponível para geração.")

    # 2️⃣ Seleção (placeholder para bandit/NSGA)
    chosen = select_model(valid_models, query)
    logger.info(f"[router_core] Modelo selecionado (via bandit): {chosen}")

    # 3️⃣ Garante que o modelo esteja baixado (Ollama)
    try:
        # Lógica dinâmica: só verifica/baixa se for um modelo "ollama/"
        if chosen.startswith("ollama/"):
            # (Usando to_thread como corrigimos anteriormente)
            await asyncio.to_thread(_ensure_ollama_model, chosen.replace("ollama/", ""))
    except Exception as e:
        logger.warning(f"[router_core] Falha ao verificar modelo '{chosen}': {e}")  

    # 4️⃣ Monta prompt final
    system_prompt = system_prompt or ""
    prompt = f"{system_prompt.strip()}\n\nUsuário: {query.strip()}".strip()

    # 5️⃣ Chama modelo LLM
    try:
        text, meta = call_model(
            model=chosen,
            prompt=prompt,
            temperature=temperature, # 👈 Passa o parâmetro
            max_tokens=max_tokens,   # 👈 Passa o parâmetro
        )
    except Exception as e:
        logger.exception(f"[router_core] Erro ao chamar modelo '{chosen}': {e}")
        text, meta = f"[Erro ao processar com modelo {chosen}: {e}]", {"latency_s": 0.0}

    # 5.5️⃣ Avalia a qualidade da resposta (NOVO PASSO)
    try:
        # Chama a função de julgamento
        judge_scores = await judge_answer(query, text, use_rag)
        
        # Calcula a média das pontuações (ex: [heuristic, llm])
        valid_scores = [s["score"] for s in judge_scores if "score" in s]
        if not valid_scores:
            logger.warning("[router_core] Nenhum score válido retornado pelos juízes.")
            quality_score = 0.0
        else:
            quality_score = float(np.mean(valid_scores))
            
        # Converte de 0-1 (do juiz) para 0-10 (da métrica)
        final_quality = round(quality_score * 10.0, 2)
        
    except Exception as e:
        logger.error(f"[router_core] Falha ao avaliar resposta: {e}")
        final_quality = 0.0 # Penaliza falhas no julgamento

    # 6️⃣ Armazena no cache semântico
    try:
        await store_cache(query, text)
        logger.info("[router_core] 🧠 Resposta armazenada no cache semântico.")
    except Exception as e:
        logger.warning(f"[router_core] Falha ao armazenar no cache: {e}")

    # 7️⃣ Monta retorno final padronizado
    result = {
        "model": chosen,
        "answer": text,
        "latency_s": round(time.time() - start_time, 2),
        "cost_per_1k": 0.001 if "ollama" in chosen else 0.15,  # placeholder de custo
              
        
        "quality": final_quality,
        
        "metadata": meta,
    }

    logger.info(
        f"[router_core] Resposta final [{chosen}] | {result['latency_s']:.2f}s | "
        f"Q={result['quality']:.2f}"
    )
    # --- Atualiza métricas Prometheus ---
    ROUTER_MODEL_COST.labels(model=chosen).inc(result["cost_per_1k"])
    ROUTER_QUALITY_AVG.labels(model=chosen).set(result["quality"])
    ROUTER_COST_PER_QUERY.set(result["cost_per_1k"])

    # Calcula economia simulada (baseline: modelo GPT-5 = 0.12 USD/1k)
    baseline_cost = 0.12
    if "ollama" in chosen:
        saved = baseline_cost - result["cost_per_1k"]
        if saved > 0:
            ROUTER_COST_SAVINGS.inc(saved)

    # Calcula % de uso local
    local_models = sum(1 for m in ["ollama", "local"] if m in chosen)
    total_models = 1
    ROUTER_LOCAL_USAGE_RATIO.set(local_models / total_models)
    
    try:
        update_model_metrics(
            model_name=chosen,
            latency=result["latency_s"],
            quality=result["quality"],
            cost=result["cost_per_1k"]
        )
    except Exception as e:
        logger.warning(f"[router_core] Falha ao atualizar métricas dinâmicas: {e}")
    
    return result