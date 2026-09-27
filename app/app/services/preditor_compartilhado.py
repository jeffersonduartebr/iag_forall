# Objective: One consistent error predictor across processes: learn under a file lock on fresh state, save at once.
"""The predictor pickles in ``STATE_DIR`` are shared by the feedback worker's processes and the API. Each process
used to load them once, learn on its own copy and save now and then: the last writer won, and the others' learning
was continuously discarded. Now every update takes an exclusive ``flock`` on the model's lock file, reloads the
state if another process saved since this one loaded it, learns, and saves before releasing.
"""

from __future__ import annotations

import fcntl
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


def _mtime(caminho: str) -> Optional[tuple]:
    """Identity of the saved file: the mtime alone is coarse (a few ms), and two saves in the same tick looked like
    no change; every save is an atomic replace, so the inode changes too."""
    try:
        st = os.stat(caminho)
        return st.st_ino, st.st_mtime_ns, st.st_size
    except OSError:
        return None


@contextmanager
def aprendizado(preditor: Any) -> Iterator[None]:
    """Serialized, fresh update of one model's predictor (a no-op wrapper without persistence)."""
    if not getattr(preditor, "persistence_enabled", False) or not getattr(preditor, "save_path", None):
        yield
        return
    os.makedirs(os.path.dirname(preditor.save_path) or ".", exist_ok=True)
    with open(f"{preditor.save_path}.lock", "a") as trava:
        fcntl.flock(trava, fcntl.LOCK_EX)
        try:
            atual = _mtime(preditor.save_path)
            if atual is not None and atual != getattr(preditor, "_mtime_carregado", None):
                preditor._load()
                preditor._load_validation()
            yield
            preditor.save()
            preditor._mtime_carregado = _mtime(preditor.save_path)
        finally:
            fcntl.flock(trava, fcntl.LOCK_UN)
