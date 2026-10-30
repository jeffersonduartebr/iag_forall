# -*- coding: utf-8 -*-
# Objective: Effective router configuration for the RBIE contract surface.
"""Leitura da configuracao efetiva do contrato e dos fatos do repositorio.

A configuracao efetiva e gerada a partir do ``config.yaml`` do experimento por
``scripts/rbie/gerar_config_efetiva.py`` e gravada em
``state/rbie/config_efetiva.json``. Assim ``GET /configuracao`` coincide com o
``config.yaml`` por construcao, inclusive depois de uma decisao da pessoa autora
sobre classes de risco.

``GET /saude`` expoe o commit do ARISTO em execucao e se a arvore esta limpa; o
executor do experimento compara esse commit com ``roteador.commit`` do
``config.yaml`` e aborta se divergir.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

RAIZ_ARISTO = Path(__file__).resolve().parents[3]

#: Caminho da configuracao efetiva; o script de subida grava este arquivo.
CAMINHO_CONFIG = Path(
    os.environ.get(
        "ARISTO_CONFIG_EFETIVA",
        str(RAIZ_ARISTO / "state" / "rbie" / "config_efetiva.json"),
    )
)

_CACHE: Dict[str, Any] = {"mtime": None, "dados": None}


def _ler_json() -> Dict[str, Any]:
    """Le a configuracao efetiva, recarregando quando o arquivo muda."""
    try:
        mtime = CAMINHO_CONFIG.stat().st_mtime
    except OSError:
        return {}
    if _CACHE["mtime"] == mtime and _CACHE["dados"] is not None:
        return _CACHE["dados"]
    try:
        dados = json.loads(CAMINHO_CONFIG.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return {}
    _CACHE.update({"mtime": mtime, "dados": dados})
    return dados


def configuracao_bruta() -> Dict[str, Any]:
    """Documento completo gerado (configuracao + especificacoes internas)."""
    return _ler_json()


def configuracao_efetiva() -> Dict[str, Any]:
    """Configuracao servida em ``GET /configuracao``, com o commit e a URL em uso."""
    cfg = dict(_ler_json().get("configuracao") or {})
    if not cfg:
        cfg = {
            "candidatos": [],
            "juizes_internos": [],
            "parametros": {},
            "fatores_antes_da_votacao": None,
            "c_base_usd_por_mil": None,
            "geracao": {},
            "allow_fallbacks": False,
            "cache_semantico": False,
            "recuperacao_interna": False,
            "classes_risco": None,
        }
    cfg["commit"] = commit_avaliado()
    base = os.environ.get("OPENROUTER_BASE_URL")
    if base:
        cfg["openrouter_base_url"] = base
    else:
        cfg["openrouter_base_url"] = (configuracao_bruta().get("openrouter") or {}).get(
            "url_chat", "https://openrouter.ai/api/v1/chat/completions"
        )
    return cfg


def hash_configuracao() -> str:
    """SHA-256 do JSON canonico da configuracao efetiva (identifica o commit avaliado)."""
    canonico = json.dumps(configuracao_efetiva(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str)
    return hashlib.sha256(canonico.encode("utf8")).hexdigest()


# --------------------------------------------------------------------------
# Estado do repositorio
# --------------------------------------------------------------------------

def commit_avaliado() -> str:
    """Commit do repositorio do ARISTO em execucao."""
    try:
        saida = subprocess.run(
            ["git", "-C", str(RAIZ_ARISTO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return saida.stdout.strip()
    except Exception:
        return os.environ.get("ARISTO_COMMIT", "")


def arvore_limpa() -> bool:
    """Verdadeiro se ``git status --porcelain`` do ARISTO estiver vazio."""
    try:
        saida = subprocess.run(
            ["git", "-C", str(RAIZ_ARISTO), "status", "--porcelain"],
            capture_output=True, text=True, timeout=20, check=True,
        )
        return not saida.stdout.strip()
    except Exception:
        return False


# --------------------------------------------------------------------------
# Especificacoes usadas pelo motor de decisao
# --------------------------------------------------------------------------

def modelo(id_modelo: str) -> Dict[str, Any]:
    """Especificacao de um modelo (candidato, juiz do painel ou de referencia)."""
    modelos = configuracao_bruta().get("modelos") or {}
    if id_modelo in modelos:
        return dict(modelos[id_modelo])
    curto = id_modelo.split("/", 1)[-1]
    for chave, valor in modelos.items():
        if chave.split("/", 1)[-1] == curto:
            return dict(valor)
    return {}


def via_modelo(id_modelo: str) -> str:
    """Via de acesso do modelo (``openrouter``, ``vertex`` ou ``anthropic``)."""
    return str((modelo(id_modelo) or {}).get("via") or "openrouter")


def provedor_modelo(id_modelo: str) -> str | None:
    """Nome do provedor como o harness o registra (``provedor`` do config.yaml)."""
    return (modelo(id_modelo) or {}).get("provedor")


def vertex_parametros() -> Dict[str, Any]:
    """Bloco ``vertex`` da configuracao (projeto, localizacao e nomes das variaveis)."""
    return dict(configuracao_bruta().get("vertex") or {})


def vias() -> Dict[str, str]:
    """Mapa modelo -> via, exposto em ``/configuracao`` para o adaptador."""
    return {id_modelo: str(dados.get("via") or "openrouter")
            for id_modelo, dados in (configuracao_bruta().get("modelos") or {}).items()}


def ids_candidatos() -> list[str]:
    return [c["id"] for c in (configuracao_efetiva().get("candidatos") or []) if c.get("id")]


def classes_risco() -> Dict[str, Optional[str]]:
    """Mapa modelo -> classe de risco, como a configuracao efetiva o declara."""
    return dict(configuracao_efetiva().get("classes_risco") or {})


def fatores_antes_da_votacao() -> bool:
    """Etapa dos fatores de risco; ``None`` na configuracao vale como a decisao do artigo (antes)."""
    valor = configuracao_efetiva().get("fatores_antes_da_votacao")
    return True if valor is None else bool(valor)


def parametros() -> Dict[str, Any]:
    return dict(configuracao_efetiva().get("parametros") or {})


def c_base_usd_por_mil() -> float:
    valor = configuracao_efetiva().get("c_base_usd_por_mil")
    return float(valor) if valor is not None else 0.00096


def preco_tabela(id_modelo: str) -> list[float]:
    """Preco de tabela (entrada, saida) em US$ por milhao de tokens."""
    for c in configuracao_efetiva().get("candidatos") or []:
        if c.get("id") == id_modelo:
            return [float(x) for x in c.get("preco_tabela") or [0.0, 0.0]]
    detalhes = (configuracao_bruta().get("juizes") or {}).get("detalhes") or {}
    if id_modelo in detalhes:
        return [float(x) for x in detalhes[id_modelo].get("preco_tabela") or [0.0, 0.0]]
    m = modelo(id_modelo)
    return [float(x) for x in (m.get("preco") or [0.0, 0.0])]


def custo_tabela_usd(id_modelo: str, tokens_entrada: int, tokens_saida: int) -> float:
    """Custo de tabela de uma chamada (US$): tokens x preco por milhao."""
    entrada, saida = preco_tabela(id_modelo)
    return (int(tokens_entrada) * entrada + int(tokens_saida) * saida) / 1_000_000.0


def raciocinio_ativo(raciocinio: Any) -> bool:
    """Verdadeiro quando o raciocinio consome orcamento oculto.

    Mesma regra do harness (``_limite`` de ``src/corpo_requisicao.py``): o
    orcamento de 512 tokens so e somado quando o raciocinio esta ativo. Ele esta
    desligado quando a configuracao do modelo e ``{"enabled": false}`` (via
    OpenRouter) ou tem ``thinking_budget: 0`` (via Vertex); nos demais casos com
    ``raciocinio`` nao nulo, soma-se o orcamento.
    """
    if not raciocinio:
        return False
    if isinstance(raciocinio, dict):
        if raciocinio.get("enabled") is False:
            return False
        if raciocinio.get("thinking_budget") == 0:
            return False
    return True


def limite_geracao(id_modelo: str) -> int:
    """Limite enviado ao provedor: saida visivel mais o orcamento oculto, se houver raciocinio ativo."""
    parametros = geracao(id_modelo)
    if parametros.get("max_tokens"):
        return int(parametros["max_tokens"])
    experimento = configuracao_bruta().get("experimento") or {}
    visivel = int(experimento.get("max_tokens_saida", 1024))
    if raciocinio_ativo(modelo(id_modelo).get("raciocinio")):
        visivel += int(experimento.get("orcamento_raciocinio_tokens", 512))
    return visivel


def geracao(id_modelo: str) -> Dict[str, Any]:
    """Parametros de geracao do modelo, como o harness os envia."""
    return dict((configuracao_efetiva().get("geracao") or {}).get(id_modelo) or {})


def juizes_internos() -> list[str]:
    return [str(j) for j in (configuracao_efetiva().get("juizes_internos") or [])]


def parametros_juiz() -> Dict[str, Any]:
    juizes = configuracao_bruta().get("juizes") or {}
    return {
        "temperatura": float(juizes.get("temperatura", 0.0) or 0.0),
        "max_tokens": int(juizes.get("max_tokens", 300) or 300),
    }


def openrouter_base_url() -> str:
    base = os.environ.get("OPENROUTER_BASE_URL")
    if base:
        return base.rstrip("/")
    return ((configuracao_bruta().get("openrouter") or {}).get("url_chat")
            or "https://openrouter.ai/api/v1/chat/completions")


def modo_seco() -> bool:
    """Verdadeiro quando o OpenRouter esta sendo substituido pelo servidor falso."""
    base = os.environ.get("OPENROUTER_BASE_URL") or ""
    return bool(base) and "openrouter.ai" not in base


def ciclo_evolutivo_s() -> float:
    """Periodicidade do ciclo NSGA-II; o atalho de teste so vale em modo seco."""
    if modo_seco():
        teste = os.environ.get("ARISTO_CICLO_S_TESTE")
        if teste:
            try:
                return max(0.5, float(teste))
            except ValueError:
                pass
    return float(parametros().get("ciclo_evolutivo_s", 300) or 300)
