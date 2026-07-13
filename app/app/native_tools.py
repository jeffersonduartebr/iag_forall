# Objective: Classify and route provider-native (server-side) tools carried in the
# canonical tools[] array, distinct from client-executed function tools.
"""Suporte a tools *nativas* (executadas no lado do provedor).

Além das function tools clássicas (``{"type":"function", ...}``, executadas pelo
cliente), os provedores expõem tools *server-side*: o próprio provedor executa a
ferramenta e devolve o resultado inline na resposta. Este módulo classifica essas
tools dentro do array canônico ``tools`` e decide quais modelos podem recebê-las.

Formatos de tool nativa reconhecidos
------------------------------------
- Anthropic (server tools): ``{"type":"web_search_20250305","name":"web_search"}``,
  ``bash_20250124``, ``text_editor_20250124``, ``computer_20250124`` — o ``type`` é
  versionado, por isso o casamento é por *prefixo* (``web_search`` casa
  ``web_search_20250305``).
- Gemini (grounding/execução): ``{"google_search":{}}``,
  ``{"google_search_retrieval":{}}``, ``{"code_execution":{}}`` — dicts SEM chave
  ``type`` cuja chave é uma das chaves de grounding conhecidas.
- OpenAI (hosted tools): ``{"type":"web_search"}``, ``{"type":"code_interpreter"}``,
  ``{"type":"file_search"}``.

Regra geral: uma entrada de ``tools`` é *nativa* quando é um dict cujo ``type`` é
diferente de ``"function"``, OU um dict de grounding do Gemini (sem ``type``).

Módulo puro: não faz I/O. A resolução de provedor usa ``model_registry`` via import
tardio (lazy), evitando ciclos de import — mesmo padrão de ``model_capabilities``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Chaves de grounding/execução do Gemini: tools nativas que NÃO possuem ``type``.
GEMINI_GROUNDING_KEYS: frozenset[str] = frozenset(
    {"google_search", "google_search_retrieval", "code_execution"}
)

# Prefixos de tipo de tool nativa suportados por cada família de provedor. O casamento
# é por prefixo para tolerar tipos versionados (ex.: ``web_search`` casa
# ``web_search_20250305``). Ollama não suporta nenhuma tool nativa.
_ANTHROPIC_NATIVE: frozenset[str] = frozenset({"web_search", "bash", "text_editor", "computer"})
_GEMINI_NATIVE: frozenset[str] = frozenset({"google_search", "google_search_retrieval", "code_execution"})
_OPENAI_NATIVE: frozenset[str] = frozenset({"web_search", "code_interpreter", "file_search"})
_OPENROUTER_NATIVE: frozenset[str] = frozenset({"web_search"})

# Família de provedor → prefixos de tools nativas suportados. ``claude``/``google`` são
# apelidos de ``anthropic``/``gemini`` para tolerar a família como quer que seja resolvida.
NATIVE_TOOL_SUPPORT: Dict[str, frozenset[str]] = {
    "anthropic": _ANTHROPIC_NATIVE,
    "claude": _ANTHROPIC_NATIVE,
    "gemini": _GEMINI_NATIVE,
    "google": _GEMINI_NATIVE,
    "openai": _OPENAI_NATIVE,
    "openrouter": _OPENROUTER_NATIVE,
    "ollama": frozenset(),
}

# Famílias reconhecidas por prefixo de nome de modelo (``openai/``, ``gemini/`` ...).
_KNOWN_FAMILY_PREFIXES: frozenset[str] = frozenset(
    {"openai", "anthropic", "gemini", "google", "ollama", "openrouter"}
)


def is_native_tool(tool: Any) -> bool:
    """Indica se ``tool`` é uma tool *nativa* (executada no provedor).

    É nativa quando: (a) é um dict com ``type`` != ``"function"``; ou (b) é um dict sem
    ``type`` que contém uma chave de grounding do Gemini. Qualquer outra coisa
    (function tool, não-dict, dict vazio) não é nativa.
    """
    if not isinstance(tool, dict):
        return False
    t = tool.get("type")
    if isinstance(t, str):
        return t.strip().lower() != "function"
    if t is not None:
        return False
    return any(key in GEMINI_GROUNDING_KEYS for key in tool)


def native_tool_type(tool: Any) -> str:
    """Retorna o identificador de tipo de uma tool nativa.

    Para tools com ``type`` (Anthropic/OpenAI) devolve o próprio ``type`` (possivelmente
    versionado, ex.: ``web_search_20250305``). Para tools de grounding do Gemini devolve
    a chave de grounding (ex.: ``google_search``). Devolve ``""`` quando ``tool`` não é
    uma tool nativa reconhecível.
    """
    if not isinstance(tool, dict):
        return ""
    t = tool.get("type")
    if isinstance(t, str) and t.strip().lower() != "function":
        return t
    if t is None:
        for key in tool:
            if key in GEMINI_GROUNDING_KEYS:
                return str(key)
    return ""


def split_tools(tools: Any) -> Tuple[List[Any], List[Any]]:
    """Separa o array canônico ``tools`` em ``(function_tools, native_tools)``.

    A ordem relativa de cada grupo é preservada; entradas nativas vão para o segundo
    elemento e as demais (function tools) para o primeiro.
    """
    function_tools: List[Any] = []
    native_tools: List[Any] = []
    for tool in tools or []:
        if is_native_tool(tool):
            native_tools.append(tool)
        else:
            function_tools.append(tool)
    return function_tools, native_tools


def provider_family(model_name: str) -> str:
    """Resolve a família de provedor a partir do nome do modelo.

    Ordem de resolução: (1) provedor autoritativo do ``model_registry``; (2) prefixo do
    nome (``openai/``, ``anthropic/``, ``gemini/``, ``ollama/``, ``openrouter/``);
    (3) marcadores no nome (``claude``→anthropic, ``google``→gemini, ``gpt``/``o1`` etc.→
    openai). Devolve ``""`` quando não é possível resolver.
    """
    if not isinstance(model_name, str) or not model_name:
        return ""
    try:
        from app.model_registry import get_model_registry  # lazy: evita ciclo de import

        cfg = get_model_registry().get(model_name)
        if cfg is not None:
            value = getattr(getattr(cfg, "provider", None), "value", None)
            if isinstance(value, str) and value:
                return value
    except Exception:
        pass
    lname = model_name.lower()
    prefix = lname.split("/", 1)[0] if "/" in lname else ""
    if prefix in _KNOWN_FAMILY_PREFIXES:
        return prefix
    if "claude" in lname or "anthropic" in lname:
        return "anthropic"
    if "gemini" in lname or "google" in lname:
        return "gemini"
    if "gpt" in lname or "openai" in lname or lname.startswith(("o1", "o3", "o4")):
        return "openai"
    return prefix


def _type_supported(native_type: str, supported_prefixes: frozenset[str]) -> bool:
    """Casa ``native_type`` contra o conjunto de prefixos suportados (por prefixo)."""
    if not native_type:
        return False
    return any(native_type.startswith(prefix) for prefix in supported_prefixes)


def provider_supports_native_tools(model_name: str, native_tools: Any) -> bool:
    """Indica se o provedor de ``model_name`` suporta TODAS as tools nativas dadas.

    Sem tools nativas, retorna ``True`` (nada a exigir). Caso contrário, cada tool nativa
    precisa ter seu tipo suportado pela família do provedor (casamento por prefixo, ver
    ``NATIVE_TOOL_SUPPORT``). Entradas não-nativas na lista são ignoradas.
    """
    natives = [t for t in (native_tools or []) if is_native_tool(t)]
    if not natives:
        return True
    supported = NATIVE_TOOL_SUPPORT.get(provider_family(model_name), frozenset())
    if not supported:
        return False
    return all(_type_supported(native_tool_type(t), supported) for t in natives)


def filter_native_tool_capable_model_names(model_names: Any, native_tools: Any) -> List[str]:
    """Mantém apenas os modelos cujo provedor suporta as tools nativas informadas.

    Sem tools nativas, devolve os nomes (string) inalterados. Complementa o filtro de
    capacidade de function calling: use em interseção com ``filter_tool_capable_model_names``.
    """
    names = [m for m in (model_names or []) if isinstance(m, str)]
    natives = [t for t in (native_tools or []) if is_native_tool(t)]
    if not natives:
        return names
    return [m for m in names if provider_supports_native_tools(m, natives)]
