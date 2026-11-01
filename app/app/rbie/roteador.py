# -*- coding: utf-8 -*-
# Objective: Contract route decision, internal judge, reward and NSGA-II cycle.
"""Motor de ``POST /rotear`` e do ciclo evolutivo por estado.

Decisao rapida (por requisicao): contexto semantico por centroides, fatores de
risco aplicados antes da votacao, votos das tres politicas pelo meta-bandit do
ARISTO, chamada ao candidato, juiz interno com Q_int, recompensa da Equacao da
ESPECIFICACAO e atualizacao das estatisticas do estado.

Escala lenta (ciclo por estado): NSGA-II sobre os objetivos latencia, custo,
qualidade e alinhamento calculados do EMA do proprio estado; o controlador
proporcional converte o ponto de operacao do melhor individuo nos pesos
(w_q, w_l, w_c) da recompensa. O relogio e pausavel por estado.
"""

from __future__ import annotations

import asyncio
import copy
import math
import os
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.services import bandit_policy
from app.services.bandit_stats_store import welford_beta
from app.services.judge_rubric import DEFAULT_RUBRIC_WEIGHTS, build_rubric_prompt, \
    parse_rubric_scores, weighted_quality
from app.services.reward import cost_score, derive_reward_weights, latency_score

from . import cliente, configuracao, estado as estado_modulo, interpretacao

QUANTIDADE_MEDIA = 0.15  # EMA_ALPHA de services.ema_store
_COLETOR_ATIVO = False
_COLETOR_THREAD: Optional[threading.Thread] = None


# --------------------------------------------------------------------------
# Centroides e decisao rapida
# --------------------------------------------------------------------------

def _cosseno(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _embedding(prompt: str) -> Optional[np.ndarray]:
    from app.embeddings import embed_text

    try:
        vetor = embed_text(prompt)
    except Exception:
        return None
    if vetor is None:
        return None
    v = np.asarray(vetor, dtype=np.float32)
    if v.ndim != 1 or v.size == 0:
        return None
    return v


def _contexto(estado: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    """Atualiza centroides do estado e devolve o contexto da requisicao."""
    vetor = _embedding(prompt)
    parametros = configuracao.parametros()
    maximo = int(parametros.get("max_centroides", 20) or 20)
    limiar_novo = float(parametros.get("limiar_novo_centroide", 0.35) or 0.35)
    taxa = float(os.environ.get("ARISTO_CENTROID_LEARN_RATE", "0.15") or 0.15)
    centroides: List[Dict[str, Any]] = estado.setdefault("centroides", [])

    if vetor is None:
        return {"id_centroide": centroides[0]["id"] if centroides else 0,
                "novo_centroide": False, "u_q": 1.0, "similaridade": 0.0}

    if not centroides:
        centroides.append({"id": 0, "vec": vetor.tolist(), "count": 1})
        return {"id_centroide": 0, "novo_centroide": True, "u_q": 1.0, "similaridade": 0.0}

    melhor_idx, melhor_sim = 0, -1.0
    for i, c in enumerate(centroides):
        sim = _cosseno(vetor, np.asarray(c["vec"], dtype=np.float32))
        if sim > melhor_sim:
            melhor_idx, melhor_sim = i, sim

    if melhor_sim < limiar_novo and len(centroides) < maximo:
        novo_id = max(int(c["id"]) for c in centroides) + 1
        centroides.append({"id": novo_id, "vec": vetor.tolist(), "count": 1})
        return {"id_centroide": novo_id, "novo_centroide": True,
                "u_q": max(0.0, 1.0 - melhor_sim), "similaridade": melhor_sim}

    c = centroides[melhor_idx]
    novo_vec = (1.0 - taxa) * np.asarray(c["vec"], dtype=np.float32) + taxa * vetor
    norma = float(np.linalg.norm(novo_vec))
    if norma > 0:
        novo_vec = novo_vec / norma
    c["vec"] = novo_vec.astype(np.float32).tolist()
    c["count"] = int(c.get("count", 0)) + 1
    return {"id_centroide": int(c["id"]), "novo_centroide": False,
            "u_q": max(0.0, 1.0 - melhor_sim), "similaridade": melhor_sim}


def _fatores_risco(estado: Dict[str, Any], contexto: Dict[str, Any]) -> Dict[str, float]:
    """Fatores de risco por candidato, aplicados antes da votacao (PV-03.3)."""
    parametros = configuracao.parametros()
    limiar = float(parametros.get("limiar_novidade", 0.45) or 0.45)
    fatores = parametros.get("fatores_risco") or {}
    fronteira = float(fatores.get("fronteira", 1.3) or 1.3)
    compacta = float(fatores.get("compacta", 0.6) or 0.6)
    compacta_baixo = float(fatores.get("compacta_abaixo_limiar", 1.1) or 1.1)
    classes = configuracao.classes_risco()
    acima = float(contexto.get("u_q", 1.0)) > limiar
    aplicados: Dict[str, float] = {}
    for modelo in configuracao.ids_candidatos():
        classe = classes.get(modelo)
        fator = 1.0
        if classe == "fronteira":
            fator = fronteira if acima else 1.0
        elif classe == "compacta":
            fator = compacta if acima else compacta_baixo
        aplicados[modelo] = fator
    return aplicados


def _estatisticas_contexto(estado: Dict[str, Any], id_centroide: int) -> Dict[str, Dict[str, float]]:
    """Estatisticas do contexto; cai no contexto global quando o centroide e novo."""
    estatisticas = estado.setdefault("estatisticas", {})
    cluster = estatisticas.get(f"cluster:{id_centroide}") or {}
    if cluster:
        return cluster
    return estatisticas.get("global") or {}


def _decidir_centroide(estado: Dict[str, Any], prompt: str) -> Tuple[str, Dict[str, Any]]:
    """Decide o modelo: centroides, fatores de risco e votacao das tres politicas."""
    contexto = _contexto(estado, prompt)
    parametros = configuracao.parametros()
    base = _estatisticas_contexto(estado, int(contexto["id_centroide"]))
    fatores = _fatores_risco(estado, contexto)
    ajustadas: Dict[str, Dict[str, float]] = {}
    for modelo in configuracao.ids_candidatos():
        entrada = dict(base.get(modelo) or {})
        fator = float(fatores.get(modelo, 1.0))
        if "mean" in entrada:
            entrada["mean"] = max(0.0, min(1.0, float(entrada.get("mean", 0.0)) * fator))
        ajustadas[modelo] = entrada
    votos: Dict[str, str] = {}
    if configuracao.fatores_antes_da_votacao():
        escolhido, votos = bandit_policy.meta_combine_choices(
            configuracao.ids_candidatos(), ajustadas,
            float(parametros.get("epsilon_base", 0.12)), "ucb1")
    else:
        escolhido, votos = bandit_policy.meta_combine_choices(
            configuracao.ids_candidatos(), base,
            float(parametros.get("epsilon_base", 0.12)), "ucb1")
    contagem: Dict[str, int] = {}
    for vencedor in votos.values():
        contagem[vencedor] = contagem.get(vencedor, 0) + 1
    vencedores = [m for m, n in contagem.items() if n >= 2]
    if vencedores:
        politica = next(p for p, m in votos.items() if m == vencedores[0])
        desempate = False
    else:
        politica = "ucb1"
        desempate = True
    decisao = {
        "votos": votos,
        "desempate_meta_bandit": desempate,
        "politica_vencedora": politica,
        "id_centroide": int(contexto["id_centroide"]),
        "novo_centroide": bool(contexto["novo_centroide"]),
        "u_q": round(float(contexto["u_q"]), 6),
        "acima_limiar_novidade": float(contexto["u_q"]) >
                                 float(parametros.get("limiar_novidade", 0.45) or 0.45),
        "fatores_aplicados": {m: round(f, 6) for m, f in fatores.items()},
    }
    return escolhido, decisao


def _decidir_linucb(estado: Dict[str, Any], texto_redacao: Optional[str],
                    prompt: str) -> Tuple[str, Dict[str, Any], Any]:
    """Decide o modelo pelo LinUCB sobre as caracteristicas da redacao.

    Nao usa centroides, ``_contexto`` nem fatores de risco. A decisao por
    requisicao e sempre do LinUCB nesta politica; o ciclo NSGA-II continua
    atualizando apenas os pesos da recompensa.
    """
    from . import linucb

    caracteristica = linucb.caracteristicas(texto_redacao or "", prompt)
    escolhido, detalhes = linucb.escolher(
        estado, caracteristica, configuracao.ids_candidatos())
    return escolhido, detalhes, caracteristica


# --------------------------------------------------------------------------
# Juiz interno, recompensa e atualizacao
# --------------------------------------------------------------------------

def _modelo_de_empresa(id_modelo: str) -> Optional[str]:
    return (configuracao.modelo(id_modelo) or {}).get("empresa")


def _escolher_juiz(modelo_gerador: str) -> str:
    """Sorteia um juiz do painel entre os elegiveis pela regra de exclusao por empresa."""
    painel = configuracao.juizes_internos()
    empresa = _modelo_de_empresa(modelo_gerador)
    elegiveis = [j for j in painel if _modelo_de_empresa(j) != empresa] or list(painel)
    return random.choice(elegiveis)


async def _avaliar(prompt: str, texto: str, modelo_gerador: str):
    """Chama juiz interno e devolve ``(q_int 0-10, alinhamento 0-1, chamadas)``.

    A instrucao de rubrica do ARISTO pede raciocinio antes das notas. Com o
    limite de 300 tokens do painel, um juiz mais verboso (por exemplo o
    glm-5.3-flash) trunca antes do bloco ``<scores>`` e a nota fica ilegivel. O
    primeiro reenvio do MESMO juiz usa um orcamento maior; se ainda assim nao
    houver nota, tenta o proximo juiz elegivel do painel. Sem nota valida,
    ``q_int`` fica nulo (nao avaliado), como no painel do harness.
    """
    instrucao = build_rubric_prompt(query=prompt, answer=texto)
    juiz = _escolher_juiz(modelo_gerador)
    empresa = _modelo_de_empresa(modelo_gerador)
    elegiveis = [j for j in configuracao.juizes_internos()
                 if _modelo_de_empresa(j) != empresa] or configuracao.juizes_internos()
    candidatos = [juiz] + [j for j in elegiveis if j != juiz][:1]
    limite_padrao = int(configuracao.parametros_juiz()["max_tokens"])
    limite_ampliado = max(limite_padrao, int(os.environ.get("ARISTO_JUIZ_MAX_TOKENS", "1024") or 1024))
    chamadas: List[Dict[str, Any]] = []
    for candidato in candidatos:
        for limite in (limite_padrao, limite_ampliado):
            corpo = cliente.corpo_juiz(candidato, instrucao, max_tokens=limite)
            chamada, saida = await cliente.chamar(candidato, corpo, papel="juiz_interno")
            chamadas.append(chamada)
            if chamada.get("erro") is not None:
                break
            notas = parse_rubric_scores(saida)
            if notas is not None:
                q = weighted_quality(notas, DEFAULT_RUBRIC_WEIGHTS)
                alinhamento = float(notas.get("alinhamento", 0.0)) / 10.0
                return float(q), alinhamento, chamadas
            if chamada.get("finish_reason") != "length":
                break
    return None, 0.0, chamadas


def _recompensa(estado: Dict[str, Any], concordancia: Optional[float], latencia_s: float,
                custo_gerador: float, tokens_entrada: int, tokens_saida: int):
    """Recompensa alinhada a nota humana; devolve ``(valor, disponivel)``.

    A qualidade e a concordancia com as notas humanas; o escore dos juizes nao
    entra. Sem concordancia, usa apenas latencia e custo e marca indisponivel.
    """
    from . import recompensa as recompensa_modulo

    parametros = configuracao.parametros()
    return recompensa_modulo.combinar(
        estado.get("pesos") or {}, concordancia, latencia_s, custo_gerador,
        tokens_entrada, tokens_saida,
        latencia_k=float(parametros.get("latencia_k", 0.12) or 0.12),
        latencia_x0=float(parametros.get("latencia_x0", 20.0) or 20.0),
        c_base_usd_por_mil=configuracao.c_base_usd_por_mil(),
    )


def _atualizar_estatisticas(estado: Dict[str, Any], id_centroide: Optional[int], modelo: str,
                            recompensa: float, latencia_s: float,
                            qualidade: Optional[float],
                            custo_usd: float, alinhamento: float) -> None:
    if id_centroide is not None:
        estatisticas = estado.setdefault("estatisticas", {})
        for chave in (f"cluster:{id_centroide}", "global"):
            contexto = estatisticas.setdefault(chave, {})
            contexto[modelo] = welford_beta(contexto.get(modelo, {}), recompensa)
    from app.services.ema_store import next_ema

    anterior = (estado.get("ema") or {}).get(modelo) or {}
    qualidade_ema = (float(qualidade) if qualidade is not None
                     else float(anterior.get("ema_quality", 5.0)))
    estado.setdefault("ema", {})[modelo] = next_ema(
        anterior, latencia_s, qualidade_ema, custo_usd)
    ema = estado["ema"][modelo]
    ema["ema_alignment"] = alinhamento
    historico = estado.setdefault("historico", [])
    historico.append({"modelo": modelo, "latencia_s": round(latencia_s, 6),
                      "custo_usd": round(custo_usd, 8),
                      "qualidade": (None if qualidade is None else round(qualidade, 4)),
                      "alinhamento": round(alinhamento, 4),
                      "recompensa": round(recompensa, 6), "carimbo_utc": estado_modulo.carimbo_utc()})
    if len(historico) > 200:
        del historico[: len(historico) - 200]


# --------------------------------------------------------------------------
# Ciclo evolutivo (escala lenta)
# --------------------------------------------------------------------------

def rodar_ciclo(estado: Dict[str, Any]) -> None:
    """Um ciclo NSGA-II sobre o EMA do proprio estado, com o controlador proporcional."""
    from app.nsga_core import run_nsga_optimization
    from app.settings_dynamic import settings

    modelos = configuracao.ids_candidatos()
    ema = estado.get("ema") or {}
    metricas: Dict[str, Dict[str, float]] = {}
    for modelo in modelos:
        entrada = ema.get(modelo) or {}
        metricas[modelo] = {
            "latency": float(entrada.get("ema_latency", 2.0)),
            "cost": float(entrada.get("ema_cost", 0.001)),
            "quality": float(entrada.get("ema_quality", 5.0)),
            "alignment": float(entrada.get("ema_alignment", 1.0)),
        }
    pesos_modelos, eficiencia, sys_metricas = run_nsga_optimization(
        "text", modelos, metricas,
        n_pop=int(os.environ.get("ARISTO_NSGA_POP", "40") or 40),
        n_gen=int(os.environ.get("ARISTO_NSGA_GEN", "20") or 20),
    )
    w_qualidade = float(getattr(settings, "NSGA_W_QUALITY", 1.0) or 1.0)
    w_latencia = float(getattr(settings, "NSGA_W_LATENCY", 0.5) or 0.5)
    w_custo = float(getattr(settings, "NSGA_W_COST", 100.0) or 100.0)
    sys_lat, sys_cst, sys_qlt = (float(x) for x in sys_metricas)
    pesos = derive_reward_weights(w_qualidade, w_latencia, w_custo, sys_qlt, sys_lat, sys_cst,
                                  min_share=0.05)
    estado["pesos"] = {"w_q": round(pesos[0], 6), "w_l": round(pesos[1], 6), "w_c": round(pesos[2], 6)}
    estado["origem_pesos"] = "ciclo_evolutivo"
    if estado.get("politica") == "linucb":
        from . import linucb

        linucb.aplicar_prior(estado, pesos_modelos)
    estado["populacao"] = [{"pesos_modelos": {m: round(float(w), 6) for m, w in pesos_modelos.items()},
                            "eficiencia": round(float(eficiencia), 6),
                            "latencia_s": round(sys_lat, 6), "custo_usd": round(sys_cst, 8),
                            "qualidade": round(sys_qlt, 4), "alinhamento": round(float(sys_metricas[3]) if len(sys_metricas) > 3 else 1.0, 4)}]
    estado["tamanho_populacao"] = int(os.environ.get("ARISTO_NSGA_POP", "40") or 40)
    controlador = estado.setdefault("controlador", {"erro": 0.0, "integral": 0.0})
    alvo = 0.55
    controlador["erro"] = round(float(pesos[0]) - alvo, 6)
    controlador["integral"] = round(float(controlador.get("integral", 0.0)) + controlador["erro"], 6)
    estado["ciclos_evolutivos"] = int(estado.get("ciclos_evolutivos", 0)) + 1
    estado["ultimo_ciclo_utc"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _rodar_ciclo_travado(id_estado: str) -> None:
    estado_modulo.TRAVA.acquire()
    try:
        estado = estado_modulo.obter(id_estado)
        estado_modulo.entrar(estado)
        try:
            rodar_ciclo(estado)
        finally:
            estado_modulo.sair(estado)
        estado_modulo.salvar(estado)
    except Exception:
        pass
    finally:
        estado_modulo.TRAVA.release()


def _laco_ciclos() -> None:
    """Relogio do ciclo evolutivo: um laco por processo, respeitando a pausa de cada estado."""
    proximo = time.monotonic() + 1.0
    while True:
        time.sleep(max(0.05, proximo - time.monotonic()))
        proximo = time.monotonic() + 1.0
        try:
            intervalo = configuracao.ciclo_evolutivo_s()
            agora = time.time()
            for id_estado in list(estado_modulo.listar_ids()):
                try:
                    estado = estado_modulo.obter(id_estado)
                    if estado.get("pausado", True):
                        continue
                    ultimo = estado.get("_ultimo_ciclo_epoch") or 0.0
                    if agora - float(ultimo) >= intervalo:
                        _rodar_ciclo_travado(id_estado)
                        estado_modulo.obter(id_estado)["_ultimo_ciclo_epoch"] = agora
                        estado_modulo.salvar(estado_modulo.obter(id_estado))
                except Exception:
                    continue
        except Exception:
            continue


def iniciar_coletor() -> None:
    """Sobe o laco do ciclo evolutivo uma unica vez por processo."""
    global _COLETOR_ATIVO, _COLETOR_THREAD
    if _COLETOR_ATIVO:
        return
    _COLETOR_ATIVO = True
    _COLETOR_THREAD = threading.Thread(target=_laco_ciclos, name="rbie-ciclo-nsga", daemon=True)
    _COLETOR_THREAD.start()


def parar_coletor() -> None:
    global _COLETOR_ATIVO
    _COLETOR_ATIVO = False


# --------------------------------------------------------------------------
# POST /rotear
# --------------------------------------------------------------------------

async def rotear(id_estado: str, id_requisicao: str, prompt: str, semente: Optional[int],
                 metadados: Optional[Dict[str, Any]] = None,
                 texto_redacao: Optional[str] = None,
                 notas_humanas: Optional[List[int]] = None,
                 aprender: Optional[bool] = None) -> Dict[str, Any]:
    """Executa uma requisicao do contrato; devolve o documento de resposta.

    A trava do estado e mantida apenas nas duas secoes criticas: a decisao, que
    le e atualiza centroides e contadores, e a atualizacao final. As chamadas de
    rede ficam fora da trava, o que permite varias requisicoes simultaneas no
    mesmo estado sem serializar a espera das APIs.
    """
    # --- secao critica 1: decisao -------------------------------------------------
    await asyncio.to_thread(estado_modulo.TRAVA.acquire)
    try:
        estado = estado_modulo.obter(id_estado)
        guardada = estado_modulo.resposta_guardada(estado, id_requisicao)
        if guardada is not None:
            repetida = copy.deepcopy(guardada)
            repetida["repetida"] = True
            return repetida

        inicio = time.perf_counter()
        politica_estado = str(estado.get("politica") or "centroide")
        estado_modulo.entrar(estado)
        t_decisao = time.perf_counter()
        caracteristica = None
        if politica_estado == "linucb":
            modelo, decisao, caracteristica = _decidir_linucb(estado, texto_redacao, prompt)
        else:
            modelo, decisao = _decidir_centroide(estado, prompt)
        latencia_decisao = time.perf_counter() - t_decisao
        estado_modulo.salvar(estado)
    finally:
        estado_modulo.TRAVA.release()

    # --- fora da trava: chamadas de rede -----------------------------------------
    chamadas: List[Dict[str, Any]] = []
    corpo = cliente.corpo_gerador(modelo, prompt, semente)
    chamada_gerador, texto = await cliente.chamar(modelo, corpo, papel="gerador")
    chamadas.append(chamada_gerador)
    erro = chamada_gerador.get("erro")
    if erro is not None:
        texto = ""

    q_int: Optional[float] = None
    alinhamento = 0.0
    if erro is None and texto:
        q_int, alinhamento, chamadas_juiz = await _avaliar(prompt, texto, modelo)
        chamadas.extend(chamadas_juiz)
    qualidade = float(q_int or 0.0)

    custo_gerador = cliente.custo_tabela_chamada(chamada_gerador)
    custo_juizes = sum(cliente.custo_tabela_chamada(c) for c in chamadas
                       if c.get("papel") == "juiz_interno")
    latencia_gerador = float(chamada_gerador.get("latencia_s") or 0.0)

    # --- secao critica 2: atualizacao do estado e resposta -----------------------
    await asyncio.to_thread(estado_modulo.TRAVA.acquire)
    try:
        estado = estado_modulo.obter(id_estado)
        aprender_efetivo = (bool(estado.get("aprender", True))
                            if aprender is None else bool(aprender))
        status_parser, niveis_atribuidos = (interpretacao.interpretar_saida(texto)
                                            if texto else ("formato_invalido", None))
        concordancia = interpretacao.qualidade_concordancia(niveis_atribuidos, notas_humanas)
        recompensa, disponivel = _recompensa(
            estado, (concordancia if aprender_efetivo else None), latencia_gerador,
            custo_gerador, int(chamada_gerador.get("tokens_entrada") or 0),
            int(chamada_gerador.get("tokens_saida") or 0))
        if not aprender_efetivo:
            recompensa = None
            disponivel = False

        if aprender_efetivo:
            estado["n_requisicoes"] = int(estado.get("n_requisicoes", 0)) + 1
            if caracteristica is not None:
                from . import linucb

                linucb.atualizar(estado, caracteristica, modelo, float(recompensa or 0.0))
            _atualizar_estatisticas(
                estado, (None if caracteristica is not None else int(decisao["id_centroide"])),
                modelo, float(recompensa or 0.0), latencia_gerador,
                (concordancia * 10.0 if concordancia is not None else None),
                custo_gerador, alinhamento)
            estado_modulo.sair(estado)
        latencia_total = time.perf_counter() - inicio
        resposta = {
            "id_estado": id_estado,
            "modelo": modelo,
            "texto": texto,
            "erro": erro,
            "latencia_decisao_s": round(latencia_decisao, 6),
            "latencia_gerador_s": round(latencia_gerador, 6),
            "latencia_total_s": round(latencia_total, 6),
            "chamadas": chamadas,
            "custo_tabela_gerador_usd": round(custo_gerador, 9),
            "custo_tabela_juizes_internos_usd": round(custo_juizes, 9),
            "q_int": q_int,
            "recompensa": (None if recompensa is None else round(recompensa, 6)),
            "recompensa_disponivel": bool(disponivel),
            "qualidade_concordancia": (None if not aprender_efetivo or concordancia is None
                                       else round(concordancia, 6)),
            "niveis_atribuidos": niveis_atribuidos,
            "status_parser": status_parser,
            "aprender": aprender_efetivo,
            "origem_pesos": estado.get("origem_pesos", "padrao"),
            "pesos": dict(estado.get("pesos") or {}),
            "decisao": decisao,
            "politica": politica_estado,
            "alpha": (float((estado.get("linucb") or {}).get("alpha"))
                      if politica_estado == "linucb" else None),
            "n0": (float((estado.get("linucb") or {}).get("n0"))
                   if politica_estado == "linucb" else None),
            "n_centroides": len(estado.get("centroides") or []),
            "ciclos_evolutivos": int(estado.get("ciclos_evolutivos", 0)),
            "hash_estado_depois": estado_modulo.hash_estado(estado),
            "repetida": False,
        }
        estado_modulo.guardar_resposta(estado, id_requisicao, resposta)
        estado_modulo.salvar(estado)
        return resposta
    finally:
        estado_modulo.TRAVA.release()
