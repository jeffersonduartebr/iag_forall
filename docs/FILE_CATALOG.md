# Catálogo de Arquivos

Documento gerado automaticamente por `scripts/generate_docs_catalog.py`.
Escopo: código Python do projeto (`app/app`, `app`, `alembic`, `tests`).

| Arquivo | Módulo | Classes | Funções |
|---|---|---:|---:|
| `app/app/00providers.py` | providers.py — versão multimodal + UM-RAG compatível (VALIDAÇÃO DE PARÂMETROS) | 0 | 5 |
| `app/app/00rag.py` | Módulo principal: descreve responsabilidades e integrações deste arquivo. | 0 | 2 |
| `app/app/__init__.py` | Módulo principal: descreve responsabilidades e integrações deste arquivo. | 0 | 0 |
| `app/app/ab_testing.py` | ab_testing.py — A/B Testing Infrastructure | 5 | 1 |
| `app/app/adaptive_timeout.py` | adaptive_timeout.py — Adaptive Timeout Calculation | 0 | 8 |
| `app/app/bandits.py` | bandits.py — Meta-Bandit Multimodal Completo + UQ Support | 1 | 34 |
| `app/app/celery_app.py` | Módulo principal: descreve responsabilidades e integrações deste arquivo. | 0 | 0 |
| `app/app/config/__init__.py` | Configuration modules for the LLM Router application. | 0 | 0 |
| `app/app/config/constants.py` | constants.py — Application Constants | 0 | 0 |
| `app/app/correlation.py` | correlation.py — Request Correlation ID Infrastructure | 1 | 4 |
| `app/app/correlation_metrics.py` | correlation_metrics.py — Cálculo, exposição e armazenamento histórico das correlações multiobjetivo (NSGA-II) | 0 | 10 |
| `app/app/dash_control_panel.py` | Módulo principal: descreve responsabilidades e integrações deste arquivo. | 0 | 10 |
| `app/app/db.py` | db.py — Centralized Database Connection Management | 1 | 6 |
| `app/app/db_manager.py` | db_manager.py (FULL SCHEMA + PRICING SEED) | 0 | 5 |
| `app/app/drift_detector.py` | drift_detector.py — Query Distribution Drift Detection | 1 | 2 |
| `app/app/embeddings.py` | embeddings.py — Híbrido Otimizado (CPU Local + Cloud Fallback) | 1 | 10 |
| `app/app/error_handling.py` | error_handling.py — Structured Error Handling and Logging | 3 | 6 |
| `app/app/guardrails.py` | Basic content guardrails (MVP). | 1 | 2 |
| `app/app/health.py` | health.py — Deep Health Checks for All Dependencies | 3 | 9 |
| `app/app/judges/heuristic.py` | Módulo principal: descreve responsabilidades e integrações deste arquivo. | 0 | 4 |
| `app/app/judges/judge.py` | Módulo principal: descreve responsabilidades e integrações deste arquivo. | 0 | 2 |
| `app/app/judges/llm.py` | Módulo principal: descreve responsabilidades e integrações deste arquivo. | 0 | 3 |
| `app/app/judges.py` | judges.py (VERSÃO FINAL: Binary Verdict + Tie-Breaker Meta-Judge) | 3 | 24 |
| `app/app/main.py` | main.py (Production-Ready with Rate Limiting, Compression, Health Checks) | 1 | 51 |
| `app/app/metrics_collector.py` | Coletor de métricas multimodais para o Router LLM. | 0 | 4 |
| `app/app/middleware/__init__.py` | Middleware modules for the LLM Router application. | 0 | 0 |
| `app/app/middleware/backpressure.py` | backpressure.py — Global Concurrency Limit Middleware | 2 | 1 |
| `app/app/middleware/rate_limit.py` | rate_limit.py — Rate Limiting Middleware with Redis Support | 2 | 2 |
| `app/app/model_registry.py` | model_registry.py — Centralized Model Configuration Registry | 4 | 2 |
| `app/app/nsga_meta_optimizer.py` | nsga_meta_optimizer.py (MULTIMODAL) | 0 | 6 |
| `app/app/nsga_weights_updater.py` | nsga_weights_updater.py — Otimizador Multimodal (NSGA-II + UQ Tuning + Strategy Tuning) | 0 | 23 |
| `app/app/observability.py` | observability.py | 1 | 6 |
| `app/app/online_predictor.py` | online_predictor.py — Real-Time Error Prediction Module (Logistic Regression SGD) | 1 | 3 |
| `app/app/prometheus_setup.py` | Módulo principal: descreve responsabilidades e integrações deste arquivo. | 0 | 3 |
| `app/app/providers_async.py` | providers_async.py (FINAL: Suporte a Reasoning/Thinking Explícito) | 9 | 13 |
| `app/app/query_service.py` | query_service.py — versão MULTIMODAL COMPLETA | 1 | 5 |
| `app/app/rag_context_provider.py` | rag_context_provider.py | 0 | 1 |
| `app/app/rag_healthcheck.py` | rag_healthcheck.py | 0 | 3 |
| `app/app/rag_local.py` | rag_local.py — RAG Multimodal Unificado (Com suporte Imagem -> Texto) | 0 | 8 |
| `app/app/reliability.py` | reliability.py — Reliability Patterns for LLM Providers | 5 | 8 |
| `app/app/reranker.py` | reranker.py — Módulo de Re-Ranking (Cross-Encoder) | 0 | 2 |
| `app/app/reset_chroma_collections.py` | Módulo principal: descreve responsabilidades e integrações deste arquivo. | 0 | 1 |
| `app/app/risk_tuner.py` | risk_tuner.py — Adaptive Risk Factor Management | 2 | 1 |
| `app/app/roadmap_features.py` | Roadmap hardening features. | 2 | 26 |
| `app/app/router_core.py` | router_core.py — Multimodal + UM-RAG + Meta-bandit + UQ + Online Learning | 2 | 23 |
| `app/app/router_strategy.py` | router_strategy.py (Versão Final: Filtros Bilaterais de Segurança) | 0 | 4 |
| `app/app/routers/rag_router.py` | rag_router.py (CORRIGIDO: Importação e Async) | 1 | 5 |
| `app/app/runtime_state.py` | Global runtime state reset helpers for tests/dev. | 0 | 1 |
| `app/app/schemas.py` | schemas.py (VERSÃO COMPLETA DE PRODUÇÃO) | 6 | 0 |
| `app/app/semantic_cache.py` | semantic_cache.py — Cache Semântico via ChromaDB (Rápido) | 1 | 10 |
| `app/app/services/__init__.py` | Módulo principal: descreve responsabilidades e integrações deste arquivo. | 0 | 0 |
| `app/app/services/bandit_centroids.py` | Centroid helpers for bandit clustering. | 0 | 2 |
| `app/app/services/bandit_policy.py` | Policy helpers for meta-bandit decisions. | 0 | 5 |
| `app/app/services/router_maintenance.py` | Maintenance helpers for router background services. | 0 | 1 |
| `app/app/services/router_services.py` | Shared helpers for router_core routing/feedback/maintenance. | 0 | 5 |
| `app/app/settings_dynamic.py` | settings_dynamic.py (VERSÃO FINAL: Com Configuração de Amostragem) | 3 | 12 |
| `app/app/sparse_index.py` | sparse_index.py — Gerenciador de Índice BM25 (Busca por Palavras-Chave) | 1 | 0 |
| `app/app/tasks.py` | Celery tasks for background processing. | 0 | 6 |
| `app/app/umrag.py` | umrag.py — Unified Multimodal RAG | 0 | 6 |
| `app/app/update_nsga_best_params.py` | update_nsga_best_params.py  (VERSÃO MULTIMODAL) | 0 | 5 |
| `app/app/user_feedback.py` | user_feedback.py — User Feedback Processing | 3 | 6 |
| `app/app/utils/bkp.redis_client.py` | Módulo principal: descreve responsabilidades e integrações deste arquivo. | 0 | 1 |
| `app/app/utils/pricing.py` | pricing.py - Model Cost Calculator with Redis Caching (Quick Win #4) | 0 | 5 |
| `app/app/utils/redis_client.py` | redis_client.py — Redis Client with Connection Pooling | 0 | 9 |
| `app/app/utils/text_splitter.py` | Módulo principal: descreve responsabilidades e integrações deste arquivo. | 0 | 3 |
| `app/app/utils/token_utils.py` | token_utils.py — Token Counting Utilities | 0 | 4 |
| `app/app/utils/uncertainty.py` | app/app/utils/uncertainty.py | 0 | 2 |
| `app/app/vectorstore.py` | vectorstore.py — RAG Multimodal com Versionamento e Auto-Healing | 0 | 18 |
| `app/__init__.py` | Módulo `app/__init__.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 0 |
| `app/advanced_analytics.py` | advanced_analytics.py — Advanced Scientific Validation for Thesis | 0 | 4 |
| `app/audit_anomalies.py` | audit_anomalies.py — Auditoria de Discrepâncias (Sincronizado) | 0 | 3 |
| `app/benchmark_thesis.py` | benchmark_thesis.py — Phase 1: Generation & Performance (NO JUDGE) | 1 | 15 |
| `app/calculate_savings.py` | calculate_savings.py | 0 | 1 |
| `app/evaluate_results.py` | evaluate_results.py — Phase 2: Batch Evaluation (Parallelized) | 0 | 2 |
| `app/experiment_oml_standard.py` | experiment_oml_standard.py — OML Comparison with Statistical Validation | 0 | 5 |
| `app/populate_vectorstore.py` | populate_vectorstore.py (Versão Final: OCR + Deduplicação + Metadados) | 0 | 8 |
| `app/prestart_vectorstore.py` | prestart_vectorstore.py | 0 | 1 |
| `app/sensitivity_runner.py` | sensitivity_runner.py — Sensitivity Analysis for Thesis (ROBUST FIXED) | 0 | 2 |
| `app/statistical_validation.py` | statistical_validation.py — Validação Estatística (Sincronizado com Benchmark Final) | 0 | 6 |
| `alembic/env.py` | Módulo `alembic/env.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 2 |
| `alembic/versions/0002_add_performance_indices.py` | Add performance indices for frequently queried tables | 0 | 2 |
| `tests/conftest.py` | Módulo `tests/conftest.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 1 |
| `tests/locustfile.py` | Módulo `tests/locustfile.py`: descreve responsabilidades e integrações deste arquivo. | 7 | 0 |
| `tests/test_ab_testing.py` | Tests for A/B testing infrastructure. | 4 | 0 |
| `tests/test_autonomous.py` | test_autonomous.py — Tests for Phase 5 Autonomous Behavior Improvements | 7 | 0 |
| `tests/test_bandits.py` | Módulo `tests/test_bandits.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 2 |
| `tests/test_bandits_more.py` | Módulo `tests/test_bandits_more.py`: descreve responsabilidades e integrações deste arquivo. | 2 | 8 |
| `tests/test_cascade_detector.py` | Tests for cascade failure detection. | 2 | 0 |
| `tests/test_chaos.py` | test_chaos.py — Chaos Testing for Failure Scenarios | 9 | 0 |
| `tests/test_correlation.py` | test_correlation.py — Tests for correlation ID infrastructure | 5 | 0 |
| `tests/test_coverage_boosters.py` | Módulo `tests/test_coverage_boosters.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 5 |
| `tests/test_db_module.py` | Módulo `tests/test_db_module.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 3 |
| `tests/test_drift_detector.py` | Tests for query distribution drift detection. | 3 | 0 |
| `tests/test_embeddings.py` | test_embeddings.py — Tests for Embeddings Module | 8 | 0 |
| `tests/test_health_components.py` | Módulo `tests/test_health_components.py`: descreve responsabilidades e integrações deste arquivo. | 1 | 3 |
| `tests/test_health_readiness_mode.py` | Módulo `tests/test_health_readiness_mode.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 3 |
| `tests/test_integration_pipeline.py` | test_integration_pipeline.py — Integration tests for the full request pipeline | 5 | 1 |
| `tests/test_judges.py` | Módulo `tests/test_judges.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 5 |
| `tests/test_judges_extra.py` | Módulo `tests/test_judges_extra.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 5 |
| `tests/test_main_admin.py` | Módulo `tests/test_main_admin.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 10 |
| `tests/test_main_lifecycle.py` | Módulo `tests/test_main_lifecycle.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 3 |
| `tests/test_main_provider_errors.py` | Módulo `tests/test_main_provider_errors.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 6 |
| `tests/test_metrics_collector.py` | Módulo `tests/test_metrics_collector.py`: descreve responsabilidades e integrações deste arquivo. | 2 | 3 |
| `tests/test_model_registry.py` | Módulo `tests/test_model_registry.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 5 |
| `tests/test_performance.py` | test_performance.py - Testes de Performance para Quick Wins | 8 | 0 |
| `tests/test_pricing_extra.py` | Módulo `tests/test_pricing_extra.py`: descreve responsabilidades e integrações deste arquivo. | 2 | 3 |
| `tests/test_prometheus_setup.py` | Módulo `tests/test_prometheus_setup.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 2 |
| `tests/test_providers.py` | Módulo `tests/test_providers.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 4 |
| `tests/test_providers_async_core_reliability.py` | Módulo `tests/test_providers_async_core_reliability.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 7 |
| `tests/test_providers_async_extra.py` | Módulo `tests/test_providers_async_extra.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 3 |
| `tests/test_providers_ensure_ollama.py` | test_providers_ensure_ollama.py — Extensive tests for _ensure_ollama_model | 6 | 0 |
| `tests/test_providers_reliability.py` | test_providers_reliability.py — Tests for provider reliability patterns | 7 | 0 |
| `tests/test_rag_local_extra.py` | Módulo `tests/test_rag_local_extra.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 3 |
| `tests/test_rag_logic.py` | Módulo `tests/test_rag_logic.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 2 |
| `tests/test_rag_router.py` | Módulo `tests/test_rag_router.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 6 |
| `tests/test_redis_client.py` | test_redis_client.py — Tests for Redis Client Module | 6 | 0 |
| `tests/test_reliability_core_extra.py` | Módulo `tests/test_reliability_core_extra.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 5 |
| `tests/test_reranker_module.py` | Módulo `tests/test_reranker_module.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 2 |
| `tests/test_risk_tuner.py` | Tests for adaptive risk factor management. | 3 | 0 |
| `tests/test_router_core.py` | test_router_core.py — Unit tests for router_core.py | 3 | 0 |
| `tests/test_router_core_extra.py` | Módulo `tests/test_router_core_extra.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 4 |
| `tests/test_router_core_internal.py` | Módulo `tests/test_router_core_internal.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 6 |
| `tests/test_router_maintenance_service.py` | Módulo `tests/test_router_maintenance_service.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 1 |
| `tests/test_router_strategy.py` | Módulo `tests/test_router_strategy.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 5 |
| `tests/test_runtime_state.py` | Módulo `tests/test_runtime_state.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 1 |
| `tests/test_schemas.py` | test_schemas.py — Tests for Pydantic Schema Validation | 8 | 0 |
| `tests/test_semantic_cache.py` | Módulo `tests/test_semantic_cache.py`: descreve responsabilidades e integrações deste arquivo. | 4 | 0 |
| `tests/test_semantic_cache_extra.py` | Módulo `tests/test_semantic_cache_extra.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 3 |
| `tests/test_settings.py` | Módulo `tests/test_settings.py`: descreve responsabilidades e integrações deste arquivo. | 2 | 0 |
| `tests/test_smoke.py` | test_smoke.py — Smoke Tests for LLM Router | 0 | 6 |
| `tests/test_token_utils.py` | test_token_utils.py — Tests for Token Utilities | 4 | 0 |
| `tests/test_user_feedback.py` | Tests for user feedback processing. | 5 | 0 |
| `tests/test_utils.py` | Módulo `tests/test_utils.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 3 |
| `tests/test_vectorstore_extra.py` | Módulo `tests/test_vectorstore_extra.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 4 |
| `tests/test_vision.py` | Módulo `tests/test_vision.py`: descreve responsabilidades e integrações deste arquivo. | 0 | 3 |

