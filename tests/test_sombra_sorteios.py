# Objective: The lean shadow's draws — 25% of the candidates with a known probability, 3 of 4 judges per answer.
from __future__ import annotations

import collections

import pytest
from app.services.sombra import sorteios
from app.services.sombra.juizes import painel_para
from sombra_fakes import cfg

PAINEL = (
    "gemini/gemini-3.1-pro-preview",
    "openrouter/moonshotai/kimi-k3",
    "openrouter/x-ai/grok-4.7",
    "openrouter/openai/gpt-5.6-sol",
)
CANDIDATAS = [f"openrouter/v{i}/m{i}" for i in range(19)]


def test_a_quarter_of_the_candidates_run_with_a_known_probability():
    sorteadas, fora, p = sorteios.candidatas("req-1", CANDIDATAS, 0.25)
    assert (len(sorteadas), len(fora), p) == (5, 14, pytest.approx(5 / 19))
    assert set(sorteadas) | set(fora) == set(CANDIDATAS) and not set(sorteadas) & set(fora)
    assert sorteios.candidatas("req-1", CANDIDATAS, 0.25) == (sorteadas, fora, p)  # reprodutível pelo id


def test_the_candidate_draw_is_uniform():
    contagem = collections.Counter(c for i in range(3800) for c in sorteios.candidatas(f"r{i}", CANDIDATAS, 0.25)[0])
    assert all(abs(n / 3800 - 5 / 19) < 0.03 for n in contagem.values()) and len(contagem) == 19


@pytest.mark.parametrize("fracao, todas", [(1.0, CANDIDATAS), (0.25, [])])
def test_full_fraction_or_no_candidates_runs_all(fracao, todas):
    assert sorteios.candidatas("r", todas, fracao) == (list(todas), [], 1.0)


def test_one_candidate_is_always_drawn():
    assert sorteios.candidatas("r", ["a", "b"], 0.01)[0] != []


@pytest.mark.parametrize(
    "avaliado",
    ["openrouter/deepseek/deepseek-v4-pro", "gemini/gemini-3.8-flash", "openrouter/moonshotai/kimi-k2.6",
     "openrouter/x-ai/grok-4.20", "openrouter/openai/gpt-5.6-luna"],
)
def test_every_answer_gets_three_judges_never_from_its_company(avaliado):
    from app.services.judge_vendors import empresa

    c = cfg(juizes=PAINEL, juizes_por_resposta=3)
    fora = collections.Counter()
    for i in range(400):
        painel = painel_para(avaliado, c, "text", f"req-{i}")
        assert len(painel) == 3 and empresa(avaliado) not in {empresa(j) for j in painel}
        fora.update(set(PAINEL) - set(painel))
    elegiveis = [j for j in PAINEL if empresa(j) != empresa(avaliado)]
    if len(elegiveis) == 4:  # cada juiz fica de fora ~1/4 das vezes
        assert all(abs(fora[j] / 400 - 0.25) < 0.08 for j in PAINEL)


def test_the_judge_order_is_fixed_within_a_request():
    c = cfg(juizes=PAINEL, juizes_por_resposta=3)
    a = painel_para("openrouter/deepseek/deepseek-v4-pro", c, "text", "req-9")
    assert a == painel_para("openrouter/cohere/command-a-plus", c, "text", "req-9")


def test_zero_judges_per_answer_means_the_whole_eligible_panel():
    assert len(painel_para("openrouter/deepseek/x", cfg(juizes=PAINEL, juizes_por_resposta=0), "text", "r")) == 4
