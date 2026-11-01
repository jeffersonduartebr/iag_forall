# -*- coding: utf-8 -*-
# Objective: LinUCB contextual bandit for the RBIE router contract.
"""Politica LinUCB (bandit contextual continuo) para a escolha do modelo.

Cada braco e um modelo candidato e mantem uma matriz ``A`` (d x d, iniciada como
identidade) e um vetor ``b`` (d, iniciado em zeros). A escolha e o braco que
maximiza ``x^T theta + alpha * sqrt(x^T A^-1 x)``, com ``theta = A^-1 b``; a
atualizacao e ``A += x x^T`` e ``b += r x``, com a recompensa ``r`` na mesma escala
do roteador (0 a 1).

As caracteristicas vem do texto da redacao: embedding do mesmo modelo do
roteador, normalizado, menos a media do treino, projetado nos 64 componentes
principais de ``state/rbie/pca_linucb.json``, normalizado de novo e com um termo
de bias igual a 1. Se o texto da redacao nao vier, usa-se o prompt, para manter
compatibilidade com estados antigos.

``A``, ``b`` e o contador de atualizacoes vivem no dicionario do estado
(``estado["linucb"]``), em listas JSON, para que copia, instantaneo e restauracao
continuem funcionando sem conversao especial.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

RAIZ_ARISTO = Path(__file__).resolve().parents[3]
CAMINHO_PCA = Path(os.environ.get(
    "ARISTO_PCA_LINUCB", str(RAIZ_ARISTO / "state" / "rbie" / "pca_linucb.json")))

ALPHA_PADRAO = 0.25
#: Forca do prior do NSGA-II, em pseudo-observacoes; pequena para que as
#: observacoes reais a superem com o tempo.
N0_PADRAO = 20.0
#: Suavizacao do prior a cada ciclo: o novo valor e uma media com o anterior,
#: sem substituir o que o bandit ja aprendeu.
SUAVIZACAO_PRIOR = 0.5
N_COMPONENTES = 64
DIMENSAO_CARACTERISTICAS = N_COMPONENTES + 1  # componentes + bias

_PCA_CACHE: Dict[str, Any] = {"caminho": None, "media": None, "componentes": None}


def carregar_pca(caminho: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Carrega (media, componentes) do arquivo de PCA, com cache por caminho."""
    destino = Path(caminho) if caminho is not None else CAMINHO_PCA
    if _PCA_CACHE["caminho"] == str(destino) and _PCA_CACHE["media"] is not None:
        return _PCA_CACHE["media"], _PCA_CACHE["componentes"]
    dados = json.loads(destino.read_text(encoding="utf8"))
    media = np.asarray(dados["media"], dtype=np.float64)
    componentes = np.asarray(dados["componentes"], dtype=np.float64)
    if componentes.shape != (N_COMPONENTES, media.size):
        raise ValueError(f"PCA com formato inesperado: {componentes.shape} x media {media.size}")
    _PCA_CACHE.update({"caminho": str(destino), "media": media, "componentes": componentes})
    return media, componentes


def caracteristicas(texto_redacao: str, prompt: str = "",
                    caminho_pca: Optional[Path] = None) -> np.ndarray:
    """Vetor de caracteristicas do contexto: 64 componentes normalizados + bias 1."""
    from app.embeddings import embed_text

    texto = (texto_redacao or "").strip() or (prompt or "").strip()
    vetor = np.asarray(embed_text(texto), dtype=np.float64)
    if vetor.ndim != 1 or vetor.size == 0:
        vetor = np.zeros(768, dtype=np.float64)
    norma = float(np.linalg.norm(vetor))
    if norma > 0:
        vetor = vetor / norma
    media, componentes = carregar_pca(caminho_pca)
    if vetor.size != media.size:
        # embedding degradado: devolve apenas o bias, sem quebrar a requisicao
        return np.concatenate([np.zeros(N_COMPONENTES, dtype=np.float64), [1.0]])
    projetado = componentes @ (vetor - media)
    norma_proj = float(np.linalg.norm(projetado))
    if norma_proj > 0:
        projetado = projetado / norma_proj
    return np.concatenate([projetado, [1.0]])


# --------------------------------------------------------------------------
# Estado do bandit
# --------------------------------------------------------------------------

def alpha_de(estado: Dict[str, Any]) -> float:
    """Alpha do estado (configuravel na criacao; 0,25 por padrao)."""
    return float((estado.get("linucb") or {}).get("alpha", ALPHA_PADRAO))


def _estrutura(estado: Dict[str, Any], bracos: Sequence[str],
               alpha: Optional[float] = None,
               dimensao: Optional[int] = None,
               n0: Optional[float] = None) -> Dict[str, Any]:
    """Garante a estrutura do LinUCB no estado e devolve o dicionario.

    ``dimensao`` vem do vetor de caracteristicas (65 no uso real: 64 componentes
    mais o bias). Um braco cuja matriz tenha dimensao diferente e reiniciado, o
    que so acontece em teste ou depois de mudar o numero de componentes.
    """
    linucb = estado.setdefault("linucb", {})
    linucb.setdefault("alpha", float(alpha if alpha is not None else ALPHA_PADRAO))
    if n0 is not None:
        linucb["n0"] = float(n0)
    linucb.setdefault("n0", N0_PADRAO)
    linucb.setdefault("n_atualizacoes", 0)
    armas = linucb.setdefault("bracos", {})
    d = int(dimensao or DIMENSAO_CARACTERISTICAS)
    for braco in bracos:
        dados = armas.get(braco)
        if dados is None or len(dados.get("A") or []) != d:
            armas[braco] = {
                "A": np.eye(d, dtype=np.float64).tolist(),
                "b": [0.0] * d,
                "b_prior": 0.0,
                "n": 0,
            }
        else:
            dados.setdefault("b_prior", 0.0)
    return linucb


def inicializar(estado: Dict[str, Any], bracos: Sequence[str],
                alpha: Optional[float] = None,
                n0: Optional[float] = None) -> Dict[str, Any]:
    """Cria (ou completa) a estrutura do LinUCB no estado."""
    return _estrutura(estado, bracos, alpha, n0=n0)


def n0_de(estado: Dict[str, Any]) -> float:
    """Forca do prior do NSGA-II usada no estado."""
    return float((estado.get("linucb") or {}).get("n0", N0_PADRAO))


def aplicar_prior(estado: Dict[str, Any], pesos: Dict[str, float],
                  n0: Optional[float] = None,
                  suavizacao: float = SUAVIZACAO_PRIOR) -> None:
    """Converte os pesos da mistura do NSGA-II em prior do LinUCB, com suavizacao.

    Para cada braco, ``b_prior`` recebe uma media (fator ``suavizacao``) entre o
    valor anterior e ``w_a * n0``. O prior entra na escolha como deslocamento na
    coordenada de bias; a matriz A permanece identidade mais as observacoes
    reais, e o vetor b observado nunca e substituido.
    """
    linucb = _estrutura(estado, list(pesos) or configuracao_ids(estado), n0=n0)
    forca = float(linucb.get("n0", N0_PADRAO) if n0 is None else n0)
    beta = max(0.0, min(1.0, float(suavizacao)))
    primeiro = not bool(linucb.get("prior_aplicado"))
    for braco, peso in pesos.items():
        dados = linucb["bracos"].setdefault(
            braco, {"A": np.eye(DIMENSAO_CARACTERISTICAS).tolist(),
                    "b": [0.0] * DIMENSAO_CARACTERISTICAS, "b_prior": 0.0, "n": 0})
        alvo = float(peso) * forca
        if primeiro:
            # primeiro ciclo inicializa o prior com a forca n0
            dados["b_prior"] = alvo
        else:
            anterior = float(dados.get("b_prior", 0.0))
            dados["b_prior"] = (1.0 - beta) * anterior + beta * alvo
    linucb["prior_aplicado"] = True


def configuracao_ids(estado: Dict[str, Any]) -> Sequence[str]:
    """Bracos conhecidos no estado (usado quando ``pesos`` vem vazio)."""
    return list((estado.get("linucb") or {}).get("bracos") or {})


def escolher(estado: Dict[str, Any], caracteristica: np.ndarray,
             bracos: Sequence[str]) -> Tuple[str, Dict[str, Any]]:
    """Escolhe o braco de maior UCB e devolve ``(braco, detalhes)``."""
    x = np.asarray(caracteristica, dtype=np.float64)
    linucb = _estrutura(estado, bracos, dimensao=x.size)
    alpha = float(linucb["alpha"])
    pontuacoes: Dict[str, float] = {}
    bias = np.zeros(DIMENSAO_CARACTERISTICAS if x.size == DIMENSAO_CARACTERISTICAS else x.size,
                    dtype=np.float64)
    bias[-1] = 1.0
    for braco in bracos:
        dados = linucb["bracos"][braco]
        A = np.asarray(dados["A"], dtype=np.float64)
        b = np.asarray(dados["b"], dtype=np.float64) + float(dados.get("b_prior", 0.0)) * bias
        try:
            theta = np.linalg.solve(A, b)
            bonus = float(np.sqrt(max(0.0, float(x @ np.linalg.solve(A, x)))))
        except np.linalg.LinAlgError:
            theta = np.zeros_like(b)
            bonus = 0.0
        pontuacoes[braco] = float(x @ theta) + alpha * bonus
    escolhido = max(bracos, key=lambda braco: pontuacoes[braco])
    detalhes = {
        "politica": "linucb",
        "alpha": alpha,
        "braco": escolhido,
        "ucb": {braco: round(valor, 8) for braco, valor in pontuacoes.items()},
    }
    return escolhido, detalhes


def atualizar(estado: Dict[str, Any], caracteristica: np.ndarray, braco: str,
              recompensa: float) -> None:
    """Atualiza A e b do braco com a recompensa observada (r em [0, 1])."""
    x = np.asarray(caracteristica, dtype=np.float64)
    linucb = _estrutura(estado, [braco], dimensao=x.size)
    r = max(0.0, min(1.0, float(recompensa)))
    dados = linucb["bracos"][braco]
    A = np.asarray(dados["A"], dtype=np.float64) + np.outer(x, x)
    b = np.asarray(dados["b"], dtype=np.float64) + r * x
    dados["A"] = A.tolist()
    dados["b"] = b.tolist()
    dados["n"] = int(dados.get("n", 0)) + 1
    linucb["n_atualizacoes"] = int(linucb.get("n_atualizacoes", 0)) + 1


def resumo(estado: Dict[str, Any]) -> Dict[str, Any]:
    """Resumo do bandit para ``/estados/<id>/resumo``."""
    linucb = estado.get("linucb") or {}
    armas = linucb.get("bracos") or {}
    return {
        "alpha": float(linucb.get("alpha", ALPHA_PADRAO)),
        "n_atualizacoes": int(linucb.get("n_atualizacoes", 0)),
        "n0": float(linucb.get("n0", N0_PADRAO)),
        "n_por_braco": {braco: int((dados or {}).get("n", 0)) for braco, dados in armas.items()},
        "b_prior": {braco: round(float((dados or {}).get("b_prior", 0.0)), 6)
                    for braco, dados in armas.items()},
        "desvio_a": {braco: round(float(np.linalg.norm(
            np.asarray((dados or {}).get("A"), dtype=np.float64) - np.eye(
                len((dados or {}).get("A") or [])))), 6)
            for braco, dados in armas.items()},
        "norma_b": {braco: round(float(np.linalg.norm(
            np.asarray((dados or {}).get("b"), dtype=np.float64))), 6)
            for braco, dados in armas.items()},
    }
