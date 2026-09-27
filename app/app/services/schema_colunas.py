# Objective: The columns of every table this build writes, as created by the alembic_pg baseline.
"""Expected schema for ``schema_check``: one entry per table the code writes, with every baseline column.

Kept apart from ``schema_check`` so the check stays small. ``tests/test_schema_consistency.py`` compares this
with ``alembic_pg/versions`` (and with the columns each INSERT writes), so a column added to a migration and
forgotten here, or the reverse, fails in CI.
"""

from __future__ import annotations

from typing import Dict, Tuple

REQUIRED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "audit_log": ("id", "actor", "action", "resource", "tenant_id", "metadata", "created_at",),
    "bandit_context_stats": ("id", "context_label", "model", "last_update", "avg_reward", "count", "var", "m2",),
    "correlation_history": (
        "id", "model", "corr_latency_quality", "corr_cost_quality", "corr_fitness_weights", "r2_mean", "generation",
        "timestamp",
    ),
    "ema_history": (
        "id", "model", "modality", "updated_at", "semantics", "ema_latency", "ema_cost", "ema_quality",
        "ema_alignment", "policy_namespace", "updates",
    ),
    "ema_history_log": (
        "id", "created_at", "modality", "model", "ema_latency", "ema_cost", "ema_quality", "ema_alignment",
        "update_num", "semantics",
    ),
    "eval_run_results": (
        "id", "run_id", "prompt_text", "model", "quality", "latency_s", "cost_usd", "metadata_json", "created_at",
    ),
    "eval_runs": (
        "id", "status", "policy_version", "tenant_id", "notes", "prompts_json", "summary_json", "metadata_json",
        "created_at", "updated_at",
    ),
    "expert_accounts": (
        "id", "email", "password_hash", "display_name", "phone", "enabled", "created_at", "updated_at",
    ),
    "expert_assessments": (
        "id", "expert_id", "benchmark_id", "theme", "query_text", "answer", "reference", "eval_run_id",
        "judge_quality", "quality_score", "rubric_json", "notes", "status", "created_at", "updated_at",
    ),
    "expert_profiles": ("user_id", "display_name", "theme_ids", "credentials_note", "created_at", "updated_at",),
    "judge_calibration": (
        "id", "judge_model", "query_hash", "predicted_score", "was_cached", "cache_hit_count", "calibration_score",
        "created_at",
    ),
    "judge_logs": (
        "id", "created_at", "query", "answer", "judge_model", "score_before", "fallback_model", "score_after",
        "event_type", "modality", "image_hash", "rubric_json", "delivery_level", "correlation_id",
    ),
    "judge_performance_log": (
        "id", "created_at", "judge_model", "avg_score", "avg_latency", "avg_cost", "consistency", "fitness",
        "window_start", "window_end",
    ),
    "learned_state": ("chave", "atualizado_em", "valor",),
    "model_metrics": (
        "id", "timestamp", "model", "modality", "latency_ms", "cost_usd", "cost_per_1k", "quality_score",
        "tokens_in", "tokens_out", "embedding_dim", "vision_usage", "fitness", "generation",
    ),
    "model_pricing": ("id", "model", "cost_input_1k", "cost_output_1k", "updated_at",),
    "nsga_meta_results": (
        "id", "timestamp", "trial_id", "modality", "n_pop", "n_gen", "cxpb", "mutpb", "eta_c", "eta_m", "eff_mean",
        "eff_std",
    ),
    "nsga_params": ("id", "updated_at", "modality", "n_pop", "n_gen", "cxpb", "mutpb", "eta_c", "eta_m",),
    "nsga_weights": ("id", "model", "updated_at", "modality", "weight",),
    "openrouter_exploration_stats": (
        "model", "count", "failure_count", "mean_reward", "mean_latency_s", "mean_cost_usd",
        "mean_observed_usd_per_1k", "catalog_prompt_usd_per_1k", "catalog_completion_usd_per_1k",
        "auto_promoted_at", "blocklisted", "stats_json", "updated_at",
    ),
    "policy_versions": ("id", "version", "description", "config_json", "is_active", "created_at", "updated_at",),
    "query_log": (
        "id", "created_at", "query_text", "chosen_model", "modality", "image_provided", "answer",
        "image_output_b64", "query_embedding", "answer_embedding", "quality", "latency_s", "cost_per_1k", "reward",
        "context_label", "raw_payload", "quality_source", "judge_sampled", "predicted_error_prob",
        "quality_semantics", "q_tech", "q_calibrado", "p_entrega", "detected_complexity", "confidence_score",
        "confidence_band", "abstained", "abstain_reason", "grounded", "verification_status", "knowledge_version",
        "review_status", "estimated_cost_usd", "tenant_id", "decision_json", "correlation_id", "participant",
        "episode_id", "prompt_tokens", "completion_tokens", "reasoning_tokens", "finish_reason", "trace_json",
    ),
    "rbac_user_roles": ("id", "user_id", "role_name", "tenant_id", "created_at",),
    "request_failures": (
        "id", "correlation_id", "tenant_id", "participant", "episode_id", "route_path", "status_code", "category",
        "model", "detail_json", "query_text", "modality", "latency_s", "created_at",
    ),
    "response_reviews": (
        "id", "correlation_id", "tenant_id", "query_text", "answer", "chosen_model", "confidence_score",
        "confidence_band", "grounded", "verification_status", "review_status", "review_reason", "reviewer_id",
        "reviewer_notes", "corrected_answer", "metadata_json", "created_at", "updated_at",
    ),
    "settings_dynamic": ("id", "setting_key", "setting_value", "updated_at",),
    "shadow_evaluations": (
        "id", "request_id", "episode_id", "participante", "tenant", "caso", "estrato", "p_nominal", "executada",
        "motivo_corte", "modelo", "provedor", "regiao", "versao_modelo", "papel", "regime_entrega", "p_atribuicao",
        "escores_juizes", "escore_agregado", "painel", "painel_uniforme", "teria_abstido", "custo_usd",
        "latencia_s", "tokens_entrada", "tokens_saida", "sha256_texto", "status", "frozen_run_id", "criado_em",
        "concluido_em", "p_candidata",
    ),
    "shadow_suspensions": ("chave", "motivo", "tenant", "inicio", "fim",),
    "tenant_budgets": ("tenant_id", "daily_usd_limit", "monthly_usd_limit", "enabled", "updated_at",),
    "tenant_usage": (
        "id", "tenant_id", "day_key", "month_key", "requests", "tokens_in", "tokens_out", "cost_usd", "updated_at",
    ),
    "user_feedback": (
        "id", "query_id", "query_text", "model", "modality", "feedback_type", "rating", "user_quality",
        "original_quality", "blended_quality", "reward", "comment", "created_at",
    ),
}
