# -*- coding: utf-8 -*-
# Objective: Atomic per-identity pending-job slots for the asynchronous query queue.
"""Reserve and release one pending slot in a Redis counter without a read-then-write race.

``enqueue_query_job`` used to ``GET`` the counter, compare it with the limit and only
later ``INCR`` it: N concurrent requests all read the same value below the limit and
all got in, so a tenant could overshoot its pending ceiling by the request fan-in.
Here the ``INCR`` comes first — Redis serialises it — and a request that lands over
the limit gives its slot back with ``DECR``. The transient overshoot only ever makes a
concurrent request reject conservatively; it never lets one through.
"""

from __future__ import annotations

from typing import Any, Tuple


def reserve_pending_slot(client: Any, key: str, limit: int, ttl_seconds: int) -> Tuple[bool, int]:
    """Take one slot; return ``(accepted, pending_before)``. Redis errors propagate."""
    pipe = client.pipeline()
    pipe.incr(key)
    pipe.expire(key, ttl_seconds)
    after = int(pipe.execute()[0])
    if after > limit:
        client.decr(key)
        return False, after - 1
    return True, after - 1


def release_pending_slot(client: Any, key: str) -> None:
    """Give back a slot taken by :func:`reserve_pending_slot` (best effort).

    Always a plain ``DECR``: deleting the key when "we were the only one" (as the
    old rollback did) would erase slots other requests reserved in the meantime.
    """
    try:
        client.decr(key)
    except Exception:
        pass
