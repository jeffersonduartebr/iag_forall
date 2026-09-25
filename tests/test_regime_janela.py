# Objective: Per-participant exploration window and cap of the Caso 1 protocol (<=15% over 20 episodes).
from __future__ import annotations

from app.services.regime import janela


def test_the_cap_counts_this_request_and_the_whole_window():
    assert not janela.pode_explorar(0, 0, 0.15)  # a 1ª requisição explorada seria 100%
    assert not janela.pode_explorar(5, 0, 0.15)  # 1/6 = 16,7% > 15%
    assert janela.pode_explorar(6, 0, 0.15)  # 1/7 = 14,3%
    assert not janela.pode_explorar(12, 1, 0.15)  # 2/13 = 15,4%
    assert janela.pode_explorar(13, 1, 0.15)  # 2/14 = 14,3%


def test_window_is_the_last_n_distinct_episodes_including_the_current_one(fake_redis):
    for i in range(25):
        janela.registrar(fake_redis, "P1", f"ep{i}", 20, explorou=(i == 0))
    janela.registrar(fake_redis, "P1", "ep24", 20, explorou=True)  # 2ª requisição do episódio atual
    n, x = janela.contagem(fake_redis, "P1", "ep24", 20)
    assert (n, x) == (21, 1)  # ep5..ep24 = 20 episódios, 21 requisições; a exploração de ep0 saiu da janela
    assert janela.contagem(fake_redis, "P1", "novo", 20) == (20, 1)  # novo + ep24..ep6 (ep24 tem 2 requisições)
    assert janela.contagem(fake_redis, "P2", "ep1", 20) == (0, 0)  # cada participante tem sua janela


def test_a_request_without_episode_is_its_own_episode():
    assert janela.episodio_de(None, "abc") == "req:abc" and janela.episodio_de("E7", "abc") == "E7"
