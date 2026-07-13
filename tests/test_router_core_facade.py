# Objective: Tests for extracted router strategy weights helper.
"""Tests for router_core extractions."""

from __future__ import annotations

from types import SimpleNamespace


def test_weights_from_getter_reads_attributes():
    from app.services.router_strategy_weights import _weights_from_getter

    fake_settings = SimpleNamespace(
        NSGA_W_QUALITY=1.2,
        NSGA_W_LATENCY=0.4,
        NSGA_W_COST=90.0,
    )

    weights = _weights_from_getter(lambda name, default=None: default, settings_obj=fake_settings)
    assert weights["w_quality"] == 1.2
    assert weights["w_latency"] == 0.4
    assert weights["w_cost"] == 90.0
