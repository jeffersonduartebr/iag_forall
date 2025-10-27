# app/providers.py
import time
import logging
from typing import Optional, Tuple, Dict, Any
from litellm import completion
from .settings import settings
from .semantic_cache import sem_cache_get, sem_cache_put

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _llm_call_kwargs(model: str, prompt: str, *, max_tokens: int, temperature: float, api_base: Optional[str]) -> Dict[str, Any]:
    """
    Monta kwargs para LiteLLM.completion considerando ollama/X e base_url.
    """
    messages = [{"role": "user", "content": prompt}]
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Se for ollama, liteLLM espera `model="ollama/<nome>"`
    if model == settings.OLLAMA_MODEL or model.startswith("ollama/"):
        m = model if model.startswith("ollama/") else f"ollama/{model}"
        kwargs["model"] = m
        if api_base:  # aponta pro servidor do Ollama
            kwargs["api_base"] = api_base

    return kwargs

def safe_call_model(model: str, prompt: str, *, max_tokens: int, temperature: float, api_base: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Chamada robusta:
      - cache semântico antes/depois
      - normalização do meta
      - logs
    """
    # Cache semântico
    cached = sem_cache_get(prompt, model, temperature, max_tokens)
    if cached is not None:
        logger.info(f"[providers] Cache HIT para modelo={model}")
        meta = {"usage": {"prompt_tokens": 0, "completion_tokens": 0}, "latency_s": 0.0, "cached": True}
        return cached, meta

    start = time.time()
    try:
        kwargs = _llm_call_kwargs(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            api_base=api_base,
        )
        resp = completion(**kwargs)

        # LiteLLM pode retornar dict-like; padroniza extração
        text = resp.choices[0].message["content"]
        usage_obj = getattr(resp, "usage", None) or {"prompt_tokens": 0, "completion_tokens": 0}
        pt = getattr(usage_obj, "prompt_tokens", None) if hasattr(usage_obj, "prompt_tokens") else usage_obj.get("prompt_tokens", 0)
        ct = getattr(usage_obj, "completion_tokens", None) if hasattr(usage_obj, "completion_tokens") else usage_obj.get("completion_tokens", 0)

        latency = time.time() - start
        meta = {"usage": {"prompt_tokens": pt or 0, "completion_tokens": ct or 0}, "latency_s": latency}

        logger.info(f"[providers] Modelo={kwargs['model']} | Tokens={meta['usage']} | Latência={latency:.2f}s")

        # Cache put
        sem_cache_put(prompt, text, model, temperature, max_tokens)
        return text, meta

    except Exception as e:
        logger.error(f"[providers] Erro ao chamar modelo {model}: {e}")
        # Resposta controlada
        return "Desculpe, não consegui gerar uma resposta agora.", {"usage": {"prompt_tokens": 0, "completion_tokens": 0}, "latency_s": 0.0}
