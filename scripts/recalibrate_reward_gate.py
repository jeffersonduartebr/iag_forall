#!/usr/bin/env python3
# Objective: Report how the dynamic latency threshold shifts the reward distribution, and the new gate value.
"""Recalibrate the promotion gate by quantile, not by value.

``OPENROUTER_EXPLORATION_PROMOTE_MIN_REWARD`` is an absolute threshold (0.72 by
default) on a quantity whose distribution the dynamic latency threshold moves.
Leaving the number alone while the distribution shifts does not keep the policy
constant — it inverts it: local models, which answer quickly and briefly, get a
much tighter deadline and stop clearing the bar, while slow verbose cloud models
get a wider one and start clearing it.

What the gate actually controls is the *promotion rate*. So the recalibrated
value is the reward at the **same quantile** the old threshold sat at, not the
same number.

This script only reads. It prints the two distributions, the quantile 0.72 used
to occupy and the value that occupies it now. Nothing is written to Redis or to
settings; applying the new value is a separate, deliberate act.

Usage::

    python3 scripts/recalibrate_reward_gate.py --days 30
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

#: Rough characters-per-token for answers whose token count was not recorded.
CHARS_PER_TOKEN = 4.0


def load_rows(days: int, limit: int) -> List[Dict[str, Any]]:
    """Recent query_log rows with what is needed to recompute the reward."""
    from app.db import get_engine
    from sqlalchemy import text

    sql = text(
        """
        SELECT chosen_model, modality, quality, latency_s, estimated_cost_usd,
               reward, raw_payload, answer
        FROM query_log
        WHERE created_at >= NOW() - INTERVAL :days DAY
          AND latency_s IS NOT NULL
          AND quality IS NOT NULL
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )
    with get_engine().connect() as conn:
        return [dict(row._mapping) for row in conn.execute(sql, {"days": days, "limit": limit})]


def completion_tokens_of(row: Dict[str, Any]) -> Optional[int]:
    """Token count from the payload, or estimated from the answer length.

    The estimate is coarse and is reported as such: it is good enough to see
    where the distribution moves, not to publish as a measurement.
    """
    payload = row.get("raw_payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            payload = None
    if isinstance(payload, dict):
        for key in ("completion_tokens", "output_tokens"):
            value = payload.get(key)
            if value:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
    answer = row.get("answer") or ""
    return int(len(answer) / CHARS_PER_TOKEN) if answer else None


def quantile_of(values: List[float], target: float) -> float:
    """Fraction of the sample at or below ``target``."""
    if not values:
        return float("nan")
    return sum(1 for v in values if v <= target) / len(values)


def value_at_quantile(values: List[float], quantile: float) -> float:
    """The value sitting at ``quantile`` of the sample."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(quantile * (len(ordered) - 1)))))
    return ordered[index]


def describe(values: List[float]) -> Dict[str, float]:
    """Summary of one reward distribution."""
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "p25": round(value_at_quantile(ordered, 0.25), 4),
        "median": round(statistics.median(ordered), 4),
        "p75": round(value_at_quantile(ordered, 0.75), 4),
        "p95": round(value_at_quantile(ordered, 0.95), 4),
        "max": round(ordered[-1], 4),
        "mean": round(statistics.fmean(ordered), 4),
    }


def recompute(rows: List[Dict[str, Any]], gate: float) -> Dict[str, Any]:
    """Reward under the static and the dynamic threshold, and the gate that preserves the rate."""
    from app.services.reward import (
        DEFAULT_COST_BASELINE_PER_1K,
        LATENCY_X0_S,
        cost_per_1k_from_total,
        cost_score,
        latency_score,
        latency_threshold_s,
        load_reward_weights,
    )

    static: List[float] = []
    dynamic: List[float] = []
    by_class: Dict[str, Dict[str, List[float]]] = {}
    estimated_tokens = 0

    from app.services.formative_observability import classify_model

    for row in rows:
        (w_q, w_l, w_c), _ = load_reward_weights(row.get("modality") or "text")
        quality = max(0.0, min(10.0, float(row.get("quality") or 0.0))) / 10.0
        latency = float(row.get("latency_s") or 0.0)
        tokens = completion_tokens_of(row)
        if tokens is not None and not (row.get("raw_payload") or {}):
            estimated_tokens += 1
        rate = cost_per_1k_from_total(row.get("estimated_cost_usd"), 0, tokens or 0)
        cost_term = w_c * cost_score(rate, DEFAULT_COST_BASELINE_PER_1K)

        before = w_q * quality + w_l * latency_score(latency, x0=LATENCY_X0_S) + cost_term
        after = w_q * quality + w_l * latency_score(latency, x0=latency_threshold_s(tokens)) + cost_term
        before, after = max(0.0, min(1.0, before)), max(0.0, min(1.0, after))

        static.append(before)
        dynamic.append(after)
        bucket = by_class.setdefault(classify_model(row.get("chosen_model") or ""), {"before": [], "after": []})
        bucket["before"].append(before)
        bucket["after"].append(after)

    occupied = quantile_of(static, gate)
    return {
        "rows": len(rows),
        "rows_with_estimated_tokens": estimated_tokens,
        "static_threshold": describe(static),
        "dynamic_threshold": describe(dynamic),
        "gate": {
            "current_value": gate,
            "quantile_it_occupied": round(occupied, 4),
            "recalibrated_value": round(value_at_quantile(dynamic, occupied), 4),
            "promotion_rate_before": round(1 - occupied, 4),
            "promotion_rate_if_value_kept": round(1 - quantile_of(dynamic, gate), 4),
        },
        "by_model_class": {
            name: {"before": describe(v["before"]), "after": describe(v["after"])}
            for name, v in sorted(by_class.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalibra o gate de promoção por quantil.")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument(
        "--gate",
        type=float,
        default=float(os.getenv("OPENROUTER_EXPLORATION_PROMOTE_MIN_REWARD", "0.72")),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = load_rows(args.days, args.limit)
    if not rows:
        raise SystemExit("nenhuma linha de query_log no periodo; nao ha como recalibrar")

    report = recompute(rows, args.gate)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")

    gate = report["gate"]
    print(
        f"\nManter {gate['current_value']} passaria a promover "
        f"{gate['promotion_rate_if_value_kept']:.1%} em vez de {gate['promotion_rate_before']:.1%}."
        f"\nPara preservar a taxa, o gate passa a {gate['recalibrated_value']}."
    )


if __name__ == "__main__":
    main()
