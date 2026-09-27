# Catálogo de SQL do ARISTO (PostgreSQL)

Uma linha por statement SQL do código: `<id> | <arquivo>::<função ou constante> | <teste em tests/pg>`.
`tests/test_catalogo_sql.py` confere, só lendo arquivos, que todo literal SQL de `app/app/**`, `scripts/**` e
`app/*.py` tem aqui a sua linha (uma por statement) e que o teste citado existe; o job `tests_postgres` roda os
testes contra um PostgreSQL 16 com o esquema do `alembic_pg`. SQL novo sem linha e sem teste não passa no CI.

Fora do catálogo: `scripts/benchmark_v2/` (gabarito SQL do benchmark, roda num SQLite em memória) e os
`SELECT 1` de verificação de conexão (`app.db`, `correlation_metrics.wait_for_db`), sem tabela.

## Estado aprendido
B1 | app/app/services/bandit_stats_store.py::_UPSERT_SQL | test_bandit_upsert_grows_and_never_regresses
B2 | app/app/services/bandit_stats_store.py::_SELECT_SQL | test_bandit_upsert_grows_and_never_regresses
E1 | app/app/services/ema_persistencia.py::_UPSERT | test_ema_upsert_on_the_four_column_key_and_log_every_tenth
E2 | app/app/services/ema_persistencia.py::_LOG | test_ema_upsert_on_the_four_column_key_and_log_every_tenth
E3 | app/app/services/ema_persistencia.py::_LER | test_ema_read_is_scoped_by_semantics_and_period
L1 | app/app/services/estado_duravel.py::_GRAVAR | test_learned_state_upsert_twice_and_read
L2 | app/app/services/estado_duravel.py::_LER | test_learned_state_upsert_twice_and_read
OX1 | app/app/openrouter_explorer.py::_persist_stats_to_db | test_exploration_stats_upsert_is_monotonic_and_keeps_the_promotion
OS1 | app/app/openrouter_exploration_state.py::_stats_do_banco | test_exploration_stats_read_with_expanded_in
C1 | app/app/services/centroides_duraveis.py::primeiro_id | test_centroid_first_id_comes_after_the_largest_cluster_label

## Registro de requisições, juízes e métricas
Q1 | app/app/query_service.py::insert_query_log | test_insert_query_log_writes_every_column_strips_nul_and_keeps_embedding_bytes
F1 | app/app/services/falhas_consulta.py::_INSERT | test_request_failure_is_recorded_with_nul_in_the_question
J1 | app/app/judges.py::_load_judge_stats | test_judge_performance_window_is_ten_minutes_and_is_read_back
J2 | app/app/judges.py::_persist_judge_metrics | test_judge_performance_window_is_ten_minutes_and_is_read_back
J3 | app/app/judges.py::_persist_judge_log | test_judge_log_keeps_the_correlation_id_with_and_without_rubric
J4 | app/app/judges.py::_persist_judge_log | test_judge_log_keeps_the_correlation_id_with_and_without_rubric
JC1 | app/app/services/judge_calibration.py::record_judge_calibration | test_judge_calibration_bool_and_one_hour_cache_window
JC2 | app/app/services/judge_calibration.py::update_calibration_cache_status | test_judge_calibration_bool_and_one_hour_cache_window
JC3 | app/app/services/judge_calibration.py::get_judge_calibration_metrics | test_judge_calibration_metrics_use_the_24_hour_window
MC1 | app/app/metrics_collector.py::_persist_sample | test_metrics_collector_sample_row
UF1 | app/app/user_feedback.py::_persist_feedback | test_user_feedback_insert_and_windowed_stats
UF2 | app/app/user_feedback.py::get_feedback_stats | test_user_feedback_insert_and_windowed_stats
UF3 | app/app/user_feedback.py::get_feedback_stats | test_user_feedback_insert_and_windowed_stats

## Governança (roadmap_features)
G1 | app/app/roadmap_features.py::set_tenant_budget | test_tenant_budget_upsert_read_and_list
G2 | app/app/roadmap_features.py::get_tenant_budget | test_tenant_budget_upsert_read_and_list
G3 | app/app/roadmap_features.py::list_tenant_budgets | test_tenant_budget_upsert_read_and_list
G4 | app/app/roadmap_features.py::_usage_snapshot | test_tenant_usage_sums_on_conflict_and_feeds_the_budget_check
G5 | app/app/roadmap_features.py::_usage_snapshot | test_tenant_usage_sums_on_conflict_and_feeds_the_budget_check
G6 | app/app/roadmap_features.py::record_tenant_usage | test_tenant_usage_sums_on_conflict_and_feeds_the_budget_check
G7 | app/app/roadmap_features.py::get_usage_summary | test_usage_summary_for_one_tenant_and_for_all
G8 | app/app/roadmap_features.py::get_usage_summary | test_usage_summary_for_one_tenant_and_for_all
G9 | app/app/roadmap_features.py::log_audit_event | test_audit_event_round_trip
G10 | app/app/roadmap_features.py::list_audit_events | test_audit_event_round_trip
G11 | app/app/roadmap_features.py::create_policy_version | test_policy_versions_upsert_activate_and_deactivate
G12 | app/app/roadmap_features.py::activate_policy_version | test_policy_versions_upsert_activate_and_deactivate
G13 | app/app/roadmap_features.py::activate_policy_version | test_policy_versions_upsert_activate_and_deactivate
G14 | app/app/roadmap_features.py::activate_policy_version | test_policy_versions_upsert_activate_and_deactivate
G15 | app/app/roadmap_features.py::get_active_policy | test_policy_versions_upsert_activate_and_deactivate
G16 | app/app/roadmap_features.py::list_policy_versions | test_policy_versions_upsert_activate_and_deactivate
G17 | app/app/roadmap_features.py::create_eval_run | test_eval_run_header_results_status_and_aggregate
G18 | app/app/roadmap_features.py::update_eval_run_status | test_eval_run_header_results_status_and_aggregate
G19 | app/app/roadmap_features.py::add_eval_result | test_eval_run_header_results_status_and_aggregate
G20 | app/app/roadmap_features.py::list_eval_run_results | test_eval_run_header_results_status_and_aggregate
G21 | app/app/roadmap_features.py::get_eval_run | test_eval_run_header_results_status_and_aggregate
G22 | app/app/roadmap_features.py::get_eval_run | test_eval_run_header_results_status_and_aggregate
G23 | app/app/roadmap_features.py::list_eval_runs | test_eval_run_header_results_status_and_aggregate
G24 | app/app/roadmap_features.py::create_response_review | test_response_review_returns_its_id_and_is_updated
G25 | app/app/roadmap_features.py::list_response_reviews | test_response_review_returns_its_id_and_is_updated
G26 | app/app/roadmap_features.py::update_response_review | test_response_review_returns_its_id_and_is_updated
G27 | app/app/roadmap_features.py::grant_role | test_rbac_grant_is_idempotent_and_revoke_matches_null_tenant
G28 | app/app/roadmap_features.py::revoke_role | test_rbac_grant_is_idempotent_and_revoke_matches_null_tenant
G29 | app/app/roadmap_features.py::list_roles | test_rbac_grant_is_idempotent_and_revoke_matches_null_tenant
G30 | app/app/roadmap_features.py::list_roles | test_rbac_grant_is_idempotent_and_revoke_matches_null_tenant
G31 | app/app/roadmap_features.py::get_roles_for_user | test_rbac_grant_is_idempotent_and_revoke_matches_null_tenant

## Especialistas (roadmap_experts)
X1 | app/app/roadmap_experts.py::upsert_expert_profile | test_expert_profile_upsert_keeps_fields_not_sent
X2 | app/app/roadmap_experts.py::get_expert_profile | test_expert_profile_upsert_keeps_fields_not_sent
X3 | app/app/roadmap_experts.py::create_expert_account | test_expert_account_returns_its_id_and_logs_in_case_insensitively
X4 | app/app/roadmap_experts.py::get_expert_account_by_email | test_expert_account_returns_its_id_and_logs_in_case_insensitively
X5 | app/app/roadmap_experts.py::get_expert_account_by_id | test_expert_account_update_list_and_get_by_id
X6 | app/app/roadmap_experts.py::list_expert_accounts | test_expert_account_update_list_and_get_by_id
X7 | app/app/roadmap_experts.py::update_expert_account | test_expert_account_update_list_and_get_by_id
X8 | app/app/roadmap_experts.py::create_expert_assessment | test_expert_assessment_upsert_returns_the_same_id_and_keeps_the_judge_quality
X9 | app/app/roadmap_experts.py::list_expert_assessments | test_expert_assessment_list_carries_p_entrega_from_query_log
X10 | app/app/roadmap_experts.py::list_assessed_benchmark_ids | test_expert_assessment_with_null_run_id_never_conflicts
X11 | app/app/roadmap_experts.py::get_expert_assessment_stats | test_expert_assessment_list_carries_p_entrega_from_query_log

## NSGA-II, ajuste e tempos
N1 | app/app/nsga_weights_updater.py::aggregate_ema_by_model | test_nsga_updater_reads_the_ema_and_upserts_weights_on_the_new_unique_key
N2 | app/app/nsga_weights_updater.py::persist_results | test_nsga_updater_reads_the_ema_and_upserts_weights_on_the_new_unique_key
N3 | app/app/nsga_weights_updater.py::tune_weights_from_judge_feedback | test_judge_feedback_uses_only_the_last_hour
N4 | app/app/nsga_weights_updater.py::tune_weights_from_judge_feedback | test_judge_feedback_uses_only_the_last_hour
U1 | app/app/update_nsga_best_params.py::load_best_trial | test_best_trial_uses_lowercase_columns_and_upserts_params
U2 | app/app/update_nsga_best_params.py::update_best_params | test_best_trial_uses_lowercase_columns_and_upserts_params
U3 | app/app/update_nsga_best_params.py::compute_model_weights | test_model_weights_are_computed_from_the_ema_and_upserted_in_one_executemany
U4 | app/app/update_nsga_best_params.py::persist_weights | test_model_weights_are_computed_from_the_ema_and_upserted_in_one_executemany
MO1 | app/app/nsga_meta_optimizer.py::save_result | test_meta_optimizer_saves_a_trial_with_lowercase_columns
T1 | app/app/services/nsga_tuning.py::_recent_quality_rows | test_uncertainty_score_is_read_only_from_valid_json_objects
AT1 | app/app/adaptive_timeout.py::get_ema_latency | test_adaptive_timeout_reads_the_ema_latency
M1 | app/app/services/router_maintenance.py::retention_loop | test_retention_deletes_only_rows_older_than_n_days

## Configuração, esquema, preços e relatórios
SD1 | app/app/settings_dynamic.py::_get_from_db | test_settings_upsert_on_the_exact_key_and_reads
SD2 | app/app/settings_dynamic.py::_all_from_db | test_settings_upsert_on_the_exact_key_and_reads
SD3 | app/app/settings_dynamic.py::DynamicSettings.set | test_settings_upsert_on_the_exact_key_and_reads
DM1 | app/app/db_manager.py::SEED_SQL | test_pricing_seed_is_idempotent_and_is_read_by_both_readers
PR1 | app/app/utils/pricing.py::_refresh_pricing_from_db | test_pricing_seed_is_idempotent_and_is_read_by_both_readers
PR2 | app/app/api/admin_models_routes.py::models_pricing | test_pricing_seed_is_idempotent_and_is_read_by_both_readers
S1 | app/app/services/schema_check.py::_present_columns | test_schema_check_passes_on_the_baseline_and_detects_a_missing_column
R1 | app/app/services/roi_analytics.py::_load_query_rows | test_roi_rows_filter_by_tenant_and_window
CM1 | app/app/correlation_metrics.py::fetch_recent_metrics | test_correlation_window_and_history_append
CM2 | app/app/correlation_metrics.py::persist_correlations | test_correlation_window_and_history_append
CS1 | app/calculate_savings.py::QUERY | test_calculate_savings_reads_thirty_days_without_integer_division
RG1 | scripts/recalibrate_reward_gate.py::load_rows | test_recalibrate_reward_gate_rows_use_a_day_window

## Sombra, pesquisa e regime
SH1 | app/app/services/sombra/registro.py::gravar | test_shadow_rows_write_python_bools_and_legacy_ints_into_boolean_columns
SH2 | app/app/services/sombra/registro.py::abrir_suspensao | test_shadow_suspension_open_is_idempotent_and_close_sets_the_end_once
SH3 | app/app/services/sombra/registro.py::fechar_suspensao | test_shadow_suspension_open_is_idempotent_and_close_sets_the_end_once
P1 | app/app/services/pesquisa/leitura.py::_SOMBRA | test_research_readout_filters_tenant_and_period
P2 | app/app/services/pesquisa/leitura.py::_QUERY_LOG | test_research_readout_filters_tenant_and_period
RC1 | app/app/services/regime/reconstrucao.py::_SQL | test_regime_window_is_rebuilt_from_decision_json
ES1 | scripts/exportar_sombra.py::ler_banco | test_exportar_sombra_reads_the_database_with_optional_filters
