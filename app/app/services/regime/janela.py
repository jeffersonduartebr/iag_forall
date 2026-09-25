# Objective: Per-participant exploration accounting over a moving window of episodes (Redis).
"""Window = the participant's last ``janela`` distinct episodes (the current one included).

- ``regime:episodios:<participante>``: list of episode ids, most recent first (trimmed to ``janela``).
- ``regime:contagem:<participante>:<episodio>``: hash ``n`` (requests) and ``x`` (exploratory assignments).
A request without an episode id is its own episode (``req:<request id>``). Keys expire after 180 days.
Without Redis the window cannot be verified, so the caller forces exploitation.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

_PREFIXO, _TTL_S = "regime:", 180 * 86400


def _txt(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def episodio_de(episode_id: Optional[str], request_id: str) -> str:
    return str(episode_id) if episode_id else f"req:{request_id}"


def contagem(rds: Any, participante: str, episodio: str, janela: int) -> Tuple[int, int]:
    """``(requests, exploratory)`` in the window that this request's episode belongs to."""
    recentes = [_txt(e) for e in rds.lrange(f"{_PREFIXO}episodios:{participante}", 0, janela - 1)]
    episodios = [episodio] + [e for e in recentes if e != episodio][: janela - 1]
    n = x = 0
    for ep in episodios:
        dados = rds.hgetall(f"{_PREFIXO}contagem:{participante}:{ep}") or {}
        n += int(dados.get(b"n", dados.get("n", 0)) or 0)
        x += int(dados.get(b"x", dados.get("x", 0)) or 0)
    return n, x


def registrar(rds: Any, participante: str, episodio: str, janela: int, explorou: bool) -> None:
    lista = f"{_PREFIXO}episodios:{participante}"
    chave = f"{_PREFIXO}contagem:{participante}:{episodio}"
    pipe = rds.pipeline()
    pipe.lrem(lista, 0, episodio)
    pipe.lpush(lista, episodio)
    pipe.ltrim(lista, 0, max(janela, 1) * 2 - 1)  # folga: o episódio atual pode reentrar na frente
    pipe.expire(lista, _TTL_S)
    pipe.hincrby(chave, "n", 1)
    if explorou:
        pipe.hincrby(chave, "x", 1)
    pipe.expire(chave, _TTL_S)
    pipe.execute()


def pode_explorar(n: int, x: int, teto: float) -> bool:
    """Exploring now keeps the window's exploratory share (this request included) at or below ``teto``."""
    return (x + 1) <= teto * (n + 1) + 1e-9
