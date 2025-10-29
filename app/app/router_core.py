# router_core.py
import logging
import random
import time
from .providers import call_model, _ensure_ollama_model

logger = logging.getLogger(__name__)

# Modelos candidatos (geração de texto apenas)
CANDIDATE_MODELS = [
    "ollama/deepseek-r1:8b",
    "ollama/phi4",
    "ollama/deepseek-r1:1.5b",
]

# Modelos proibidos (embedding, não devem ser usados para texto)
BLOCKED_PREFIXES = ("nomic-embed", "text-embedding", "bge-", "e5-")

# -------------------------------------------------------------------
# Função de roteamento
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Função de roteamento
# -------------------------------------------------------------------
async def route_and_answer(query: str, system_prompt: str = "", use_rag: bool = False):
    """
    Decide qual modelo usar e gera resposta via call_model().
    Se use_rag=True, pode adicionar contexto de RAG.
    """
    start_time = time.time()

    # 1️⃣ Filtra apenas modelos válidos
    valid_models = [
        m for m in CANDIDATE_MODELS
        if isinstance(m, str) and not any(m.startswith(prefix) for prefix in BLOCKED_PREFIXES)
    ]

    if not valid_models:
        logger.error("[router_core] Nenhum modelo válido disponível para geração.")
        raise RuntimeError("[router_core] Nenhum modelo válido disponível para geração.")

    # 2️⃣ Seleção simples (placeholder para bandit/NSGA)
    chosen = random.choice(valid_models)
    if not isinstance(chosen, str):
        logger.error(f"[router_core] Valor inválido selecionado: {chosen!r}")
        raise TypeError(f"[router_core] Modelo selecionado não é string: {chosen!r}")

    logger.info(f"[router_core] Selecionado modelo: {chosen}")

    # 3️⃣ Garante que o modelo esteja baixado (Ollama)
    try:
        _ensure_ollama_model(chosen.replace("ollama/", ""))
    except Exception as e:
        logger.warning(f"[router_core] Falha ao verificar modelo '{chosen}': {e}")

    # 4️⃣ Monta prompt final
    system_prompt = system_prompt or ""
    prompt = f"{system_prompt.strip()}\n\nUsuário: {query.strip()}".strip()

    # 5️⃣ Chama modelo
    try:
        text, meta = call_model(
            model=chosen,
            prompt=prompt,
            temperature=0.7,
            max_tokens=2048
        )
    except Exception as e:
        logger.exception(f"[router_core] Falha ao chamar modelo '{chosen}': {e}")
        text, meta = f"[Erro ao processar com modelo {chosen}: {e}]", {"latency_s": 0.0}

    # 6️⃣ Estrutura de retorno padronizada
    result = {
        "model": chosen,
        "answer": text,
        "latency_s": round(time.time() - start_time, 2),
        "cost_per_1k": 0.001 if "ollama" in chosen else 0.12,
        "quality": round(random.uniform(7.0, 9.5), 2),  # Placeholder até os juízes entrarem
        "metadata": meta,
    }

    logger.info(
        f"[router_core] Resposta final [{chosen}] | {result['latency_s']:.2f}s | Q={result['quality']:.2f}"
    )
    return result

