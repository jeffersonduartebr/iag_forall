# Objective: Shared Prometheus metrics for the NSGA-II worker and its tuning helpers.
"""Prometheus metric singletons for nsga_weights_updater and nsga_tuning.

Kept in one module so both importers share the same registered instances — a metric
name may be registered only once in the default registry."""

from __future__ import annotations

from prometheus_client import Counter, Gauge

NSGA_RUNS = Counter("nsga_runs_total", "Execuções do NSGA-II", ["modality"])
NSGA_LAST_TS = Gauge("nsga_last_run_ts", "Timestamp da última execução", ["modality"])
NSGA_UQ_THRESH = Gauge("nsga_uq_threshold", "Limiar de Incerteza otimizado", [])
NSGA_CONVERGENCE_SCORE = Gauge("nsga_worker_convergence_score", "Convergence score from worker", ["modality"])
NSGA_OPTIMIZATION_HEALTH = Gauge("nsga_worker_optimization_health", "Optimization health from worker", ["modality"])
JUDGE_FEEDBACK_ERROR_RATE = Gauge(
    "judge_feedback_error_rate_judged_only",
    "Fraction of judged query_log records with quality below the configured threshold",
)
JUDGE_FEEDBACK_SAMPLED_TOTAL = Counter(
    "judge_feedback_sampled_total",
    "Number of judged query_log rows considered by the judge-feedback tuning logic",
)
JUDGE_FEEDBACK_PROXY_TOTAL = Counter(
    "judge_feedback_proxy_total",
    "Number of proxy-quality query_log rows ignored by the judge-feedback tuning logic",
)
