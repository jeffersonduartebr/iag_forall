import time
import logging
from typing import Optional, Tuple, Dict, Any
from litellm import completion
from .settings import settings
import litellm

logger = logging.getLogger(__name__)

def safe_completion(**kwargs):
    """Wrapper tolerante a respostas incompletas do Gemini 2.5."""
    try:
        resp = litellm.completion(**kwargs)
        # Corrige caso o campo 'parts' falte
        for candidate in getattr(resp, "choices", []):
            msg = getattr(candidate, "message", {})
            if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], dict):
                if "parts" not in msg["content"]:
                    msg["content"]["parts"] = [{"text": msg["content"].get("text", "")}]
        return resp
    except Exception as e:
        import logging
        logging.error(f"[providers] Fallback executado: {e}")
        return type("FakeResponse", (), {"choices": [{"message": {"content": ""}}]})()
    
def safe_call_model(
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    api_base: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Função segura para chamada de modelos LLM.
    Garante retorno padronizado: (texto, meta_dict)
    """
    start = time.time()

    # Corrige prefixo do provider
    if model == "phi4" or model.startswith("phi4:"):
        model_name = f"ollama/{model}"
    else:
        model_name = model
    
    # Monta parâmetros da requisição
    
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if api_base:
        kwargs["api_base"] = api_base

    try:
        logger.info(f"[providers] Chamando modelo '{model}' com temperature={temperature} e max_tokens={max_tokens}")
        resp = safe_completion(**kwargs)
    except Exception as e:
        logger.error(f"[providers] Erro ao chamar modelo {model}: {e}")
        return "[Erro interno: falha na chamada do modelo]", {
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "latency_s": 0.0,
        }

    # Captura tempo de resposta
    latency = time.time() - start

    # --- Normalização da resposta ---
    try:
        text = (
            getattr(resp, "choices", [])[0]
            .message["content"]
            if hasattr(resp, "choices") and resp.choices
            else resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
    except Exception:
        text = str(resp)[:500]  # fallback: stringify resposta parcial

    # --- Normalização do uso (tokens) ---
    usage_data = {}
    try:
        usage_data = getattr(resp, "usage", None)
        if isinstance(usage_data, dict):
            pass  # já está OK
        elif hasattr(usage_data, "__dict__"):
            usage_data = vars(usage_data)
        else:
            usage_data = {"prompt_tokens": 0, "completion_tokens": 0}
    except Exception:
        usage_data = {"prompt_tokens": 0, "completion_tokens": 0}

    # --- Monta meta seguro ---
    meta = {
        "usage": {
            "prompt_tokens": int(usage_data.get("prompt_tokens", 0)),
            "completion_tokens": int(usage_data.get("completion_tokens", 0)),
        },
        "latency_s": round(latency, 4),
    }

    # Segurança extra — nunca retornar lista
    if isinstance(meta, list):
        logger.warning(f"[providers] Meta retornou lista inesperada, convertendo para dict vazio.")
        meta = {"usage": {"prompt_tokens": 0, "completion_tokens": 0}, "latency_s": latency}

    logger.info(
        f"[providers] Modelo={model} | Tokens={meta['usage']} | Latência={latency:.2f}s"
    )

    return text, meta
