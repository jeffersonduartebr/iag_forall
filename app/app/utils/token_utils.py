# -*- coding: utf-8 -*-
import logging
try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = logging.getLogger("token_utils")

# Cache de encoders para evitar recarga
_ENCODERS = {}

def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """
    Conta tokens com precisão usando tiktoken para modelos OpenAI,
    ou fallback heurístico para outros.
    """
    if not text:
        return 0

    # 1. Heurística rápida para modelos não-OpenAI (Llama, Gemma, etc)
    # Geralmente 1 token ~= 3 a 4 caracteres.
    if not model_name.startswith(("gpt-", "o1-", "text-embedding")):
        return max(1, len(text) // 3)

    # 2. Tiktoken para OpenAI
    if tiktoken:
        try:
            if model_name not in _ENCODERS:
                try:
                    _ENCODERS[model_name] = tiktoken.encoding_for_model(model_name)
                except KeyError:
                    # Fallback para o encoder padrão do gpt-4 se o modelo for muito novo
                    _ENCODERS[model_name] = tiktoken.get_encoding("cl100k_base")
            
            return len(_ENCODERS[model_name].encode(text))
        except Exception as e:
            logger.warning(f"Erro no tiktoken para {model_name}: {e}")
            return len(text) // 4
    
    return len(text) // 4