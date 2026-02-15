# Catálogo de Métodos e Funções

Documento gerado automaticamente por `scripts/generate_docs_catalog.py`.
Escopo: código Python do projeto (`app/app`, `app`, `alembic`, `tests`).

## `app/app/00providers.py`

Resumo do arquivo: providers.py — versão multimodal + UM-RAG compatível (VALIDAÇÃO DE PARÂMETROS)

### Funções de módulo

- `heuristic_quality_estimate(text)` (`app/app/00providers.py:130`): Estimativa simples de "qualidade" (0–10) baseada em:
- `_estimate_tokens(text)` (`app/app/00providers.py:149`): Aproximação bem grosseira: ~4 chars por token.
- `_encode_image_vision(image_b64)` (`app/app/00providers.py:159`): Helper genérico para payloads multimodais que usam URL base64.
- `call_model(model, prompt, modality, image_b64, temperature, max_tokens)` (`app/app/00providers.py:175`): Chamada unificada para todos os provedores.
- `_ensure_ollama_model(model_name)` (`app/app/00providers.py:514`): Garante que o modelo Ollama esteja presente:

## `app/app/00rag.py`

Resumo do arquivo: Módulo principal: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `retrieve_context(query, top_k)` (`app/app/00rag.py:11`): (Função original simples) - mantém por compatibilidade.
- `retrieve_context_adaptive(query)` (`app/app/00rag.py:26`): Ativa RAG apenas se a similaridade do top-1 exceder o threshold.

## `app/app/__init__.py`

Resumo do arquivo: Módulo principal: descreve responsabilidades e integrações deste arquivo.

## `app/app/ab_testing.py`

Resumo do arquivo: ab_testing.py — A/B Testing Infrastructure

### Funções de módulo

- `get_ab_test_manager()` (`app/app/ab_testing.py:428`): Get the global A/B test manager instance.

### Classes e métodos

- Classe `ExperimentStatus` (`app/app/ab_testing.py:39`): Experiment status.
- Classe `Variant` (`app/app/ab_testing.py:48`): A/B test variant.
- Classe `Experiment` (`app/app/ab_testing.py:56`): A/B test experiment.
  - `Experiment.to_dict(self)` (`app/app/ab_testing.py:68`): Executa to dict.
  - `Experiment.from_dict(data)` (`app/app/ab_testing.py:83`): Executa from dict.
- Classe `ExperimentCreateRequest` (`app/app/ab_testing.py:102`): Request to create a new experiment.
- Classe `ABTestManager` (`app/app/ab_testing.py:110`): Manages A/B testing experiments.
  - `ABTestManager.__new__(cls)` (`app/app/ab_testing.py:120`): Executa new.
  - `ABTestManager.__init__(self)` (`app/app/ab_testing.py:129`): Inicializa estado interno necessário para uso da classe.
  - `ABTestManager._get_redis(self)` (`app/app/ab_testing.py:138`): Executa get redis.
  - `ABTestManager._load_experiments(self)` (`app/app/ab_testing.py:142`): Load experiments from Redis.
  - `ABTestManager._save_experiments(self)` (`app/app/ab_testing.py:159`): Save experiments to Redis.
  - `ABTestManager._hash_to_bucket(self, key, num_buckets)` (`app/app/ab_testing.py:171`): Hash a key to a bucket number for consistent assignment.
  - `ABTestManager._assign_variant(self, experiment, user_key)` (`app/app/ab_testing.py:177`): Assign a user to a variant using consistent hashing.
  - `ABTestManager.create_experiment(self, request)` (`app/app/ab_testing.py:202`): Create a new experiment.
  - `ABTestManager.start_experiment(self, experiment_id)` (`app/app/ab_testing.py:236`): Start an experiment.
  - `ABTestManager.pause_experiment(self, experiment_id)` (`app/app/ab_testing.py:249`): Pause an experiment.
  - `ABTestManager.complete_experiment(self, experiment_id)` (`app/app/ab_testing.py:261`): Complete an experiment.
  - `ABTestManager.get_assignment(self, experiment_id, user_key)` (`app/app/ab_testing.py:274`): Get variant assignment for a user.
  - `ABTestManager.record_result(self, experiment_id, variant_name, metric_name, value)` (`app/app/ab_testing.py:309`): Record a result for an experiment variant.
  - `ABTestManager.get_experiment_results(self, experiment_id)` (`app/app/ab_testing.py:341`): Get aggregated results for an experiment.
  - `ABTestManager.list_experiments(self, status)` (`app/app/ab_testing.py:392`): List all experiments, optionally filtered by status.
  - `ABTestManager.get_experiment(self, experiment_id)` (`app/app/ab_testing.py:401`): Get a specific experiment.
  - `ABTestManager.delete_experiment(self, experiment_id)` (`app/app/ab_testing.py:405`): Delete an experiment.

## `app/app/adaptive_timeout.py`

Resumo do arquivo: adaptive_timeout.py — Adaptive Timeout Calculation

### Funções de módulo

- `is_reasoning_model(model)` (`app/app/adaptive_timeout.py:43`): Check if a model is a reasoning/thinking model.
- `is_fast_local_model(model)` (`app/app/adaptive_timeout.py:49`): Check if a model is a fast local model.
- `_get_ema_from_redis(model, modality)` (`app/app/adaptive_timeout.py:57`): Try to get cached EMA latency from Redis.
- `_set_ema_to_redis(model, modality, value)` (`app/app/adaptive_timeout.py:73`): Cache EMA latency value in Redis with TTL.
- `get_ema_latency(model, modality)` (`app/app/adaptive_timeout.py:86`): Get EMA latency for a model, with Redis caching.
- `calculate_adaptive_timeout(model, modality, ema_latency)` (`app/app/adaptive_timeout.py:134`): Calculate an adaptive timeout for a model request.
- `get_timeout_for_request(model, modality, user_timeout)` (`app/app/adaptive_timeout.py:197`): Get the timeout to use for a request.
- `get_timeout_status(model, modality)` (`app/app/adaptive_timeout.py:230`): Get timeout calculation status for debugging.

## `app/app/bandits.py`

Resumo do arquivo: bandits.py — Meta-Bandit Multimodal Completo + UQ Support

### Funções de módulo

- `_get_db_engine()` (`app/app/bandits.py:67`): Get database engine from centralized module.
- `_get_rds()` (`app/app/bandits.py:74`): Executa get rds.
- `_safe_setting_float(key, default)` (`app/app/bandits.py:90`): Executa safe setting float.
- `_safe_setting_int(key, default)` (`app/app/bandits.py:98`): Executa safe setting int.
- `_unit(v)` (`app/app/bandits.py:119`): Executa unit.
- `_cosine(a, b)` (`app/app/bandits.py:125`): Executa cosine.
- `_ensure_dim(v)` (`app/app/bandits.py:133`): Executa ensure dim.
- `_sanitize_model_stats(raw)` (`app/app/bandits.py:138`): Centralized sanitization for per-model stats used by selection/update.
- `_acquire_lock(key, ttl)` (`app/app/bandits.py:223`): Executa acquire lock.
- `_release_lock(key)` (`app/app/bandits.py:234`): Executa release lock.
- `_load_centroids(update_matrix_cache)` (`app/app/bandits.py:245`): Carrega centróides de Redis:
- `_save_centroids(cents)` (`app/app/bandits.py:282`): Persiste centróides com reinicialização automática de degenerados.
- `_new_centroid_id(cents)` (`app/app/bandits.py:335`): Executa new centroid id.
- `_nearest_centroid_vec(v, cents, use_cache)` (`app/app/bandits.py:344`): Versão vetorizada: empilha centróides e faz produto escalar.
- `centroids_online_update(query_text)` (`app/app/bandits.py:366`): Atualização ONLINE:
- `_nearest_centroid_label(query_text)` (`app/app/bandits.py:419`): Somente leitura: retorna 'semctx:<id>' do centróide mais próximo.
- `_auto_context_labels(query, modality)` (`app/app/bandits.py:447`): Gera contextos automáticos:
- `_ctx_key(ctx)` (`app/app/bandits.py:485`): Executa ctx key.
- `_get_ctx_stats(ctx)` (`app/app/bandits.py:490`): Lê stats de um contexto:
- `_set_ctx_stats(ctx, stats)` (`app/app/bandits.py:553`): Executa set ctx stats.
- `_upsert_ctx_db(ctx, model, s)` (`app/app/bandits.py:578`): Executa upsert ctx db.
- `_dynamic_epsilon(ctx_stats)` (`app/app/bandits.py:616`): Executa dynamic epsilon.
- `_choose_epsilon_greedy(models, ctx_stats)` (`app/app/bandits.py:621`): Executa choose epsilon greedy.
- `_choose_ucb1(models, ctx_stats)` (`app/app/bandits.py:628`): Executa choose ucb1.
- `_choose_thompson(models, ctx_stats)` (`app/app/bandits.py:633`): Executa choose thompson.
- `_meta_choose_strategy()` (`app/app/bandits.py:642`): Estratégia meta:
- `_meta_combine_choices(models, ctx_stats)` (`app/app/bandits.py:666`): Executa as três estratégias e combina por:
- `select_model(valid_models, query, modality)` (`app/app/bandits.py:687`): Seleção de modelo via Meta-Bandit híbrido.
- `bandit_update(model, query, reward, modality)` (`app/app/bandits.py:735`): Atualiza estatísticas Welford (mean/var) + Beta(α,β) por contexto e modelo.
- `_load_nsga_weights()` (`app/app/bandits.py:830`): Lê pesos NSGA-II de Redis (nsga:weights) ou usa defaults.
- `compute_reward(model, quality, latency_s, cost_per_1k)` (`app/app/bandits.py:860`): Converte métricas em recompensa [0..1] com pesos NSGA-II dinâmicos.
- `get_snapshot()` (`app/app/bandits.py:904`): Helper para router_strategy.py.
- `sample_metrics_from_snapshot(snapshot)` (`app/app/bandits.py:914`): Helper para router_strategy.py.
- `reset_bandits_runtime_state()` (`app/app/bandits.py:930`): Reset in-memory runtime caches/singletons (test/dev utility).

### Classes e métodos

- Classe `CentroidMatrixCache` (`app/app/bandits.py:172`): Cache para matriz de centróides pré-computada.
  - `CentroidMatrixCache.__init__(self)` (`app/app/bandits.py:175`): Inicializa estado interno necessário para uso da classe.
  - `CentroidMatrixCache.update(self, cents)` (`app/app/bandits.py:183`): Atualiza a matriz cache com os centróides atuais.
  - `CentroidMatrixCache.nearest(self, v)` (`app/app/bandits.py:198`): Busca o centróide mais próximo usando a matriz pré-computada.
  - `CentroidMatrixCache.is_stale(self, max_age_s)` (`app/app/bandits.py:212`): Verifica se o cache está desatualizado.

## `app/app/celery_app.py`

Resumo do arquivo: Módulo principal: descreve responsabilidades e integrações deste arquivo.

## `app/app/config/__init__.py`

Resumo do arquivo: Configuration modules for the LLM Router application.

## `app/app/config/constants.py`

Resumo do arquivo: constants.py — Application Constants

## `app/app/correlation.py`

Resumo do arquivo: correlation.py — Request Correlation ID Infrastructure

### Funções de módulo

- `generate_correlation_id()` (`app/app/correlation.py:22`): Generate a new unique correlation ID (UUID4).
- `set_correlation_id(correlation_id)` (`app/app/correlation.py:27`): Set the correlation ID for the current request context.
- `get_correlation_id()` (`app/app/correlation.py:43`): Get the correlation ID for the current request context.
- `clear_correlation_id()` (`app/app/correlation.py:53`): Clear the correlation ID from the current context.

### Classes e métodos

- Classe `CorrelationIdContext` (`app/app/correlation.py:58`): Context manager for correlation ID scope.
  - `CorrelationIdContext.__init__(self, correlation_id)` (`app/app/correlation.py:68`): Inicializa estado interno necessário para uso da classe.
  - `CorrelationIdContext.__enter__(self)` (`app/app/correlation.py:73`): Executa enter.
  - `CorrelationIdContext.__exit__(self, exc_type, exc_val, exc_tb)` (`app/app/correlation.py:78`): Executa exit.

## `app/app/correlation_metrics.py`

Resumo do arquivo: correlation_metrics.py — Cálculo, exposição e armazenamento histórico das correlações multiobjetivo (NSGA-II)

### Funções de módulo

- `_make_db_engine()` (`app/app/correlation_metrics.py:77`): Cria engine SQLAlchemy com parâmetros seguros.
- `_connect_redis()` (`app/app/correlation_metrics.py:87`): Executa connect redis.
- `wait_for_db(max_wait_seconds)` (`app/app/correlation_metrics.py:111`): Espera o banco ficar disponível (com backoff exponencial).
- `ensure_history_table()` (`app/app/correlation_metrics.py:135`): Cria a tabela de histórico de correlação, se não existir.
- `fetch_recent_metrics(window_sql)` (`app/app/correlation_metrics.py:157`): Busca métricas recentes de 'model_metrics'.
- `_safe_corr(a, b)` (`app/app/correlation_metrics.py:179`): Correlação de Pearson segura (retorna 0.0 se variância for zero ou input inválido).
- `compute_correlations(df)` (`app/app/correlation_metrics.py:193`): Calcula correlações por modelo:
- `publish_metrics(corr_data)` (`app/app/correlation_metrics.py:276`): Publica métricas Prometheus por modelo e o R² médio global.
- `persist_correlations(corr_data)` (`app/app/correlation_metrics.py:306`): Salva correlações no banco de dados (correlation_history).
- `main()` (`app/app/correlation_metrics.py:338`): Executa main.

## `app/app/dash_control_panel.py`

Resumo do arquivo: Módulo principal: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `get_system_status()` (`app/app/dash_control_panel.py:46`): Obtém system status.
- `get_dynamic_settings()` (`app/app/dash_control_panel.py:63`): CORRIGIDO: Lê as configurações do módulo centralizado (Redis > DB > .env).
- `save_dynamic_settings(data)` (`app/app/dash_control_panel.py:74`): CORRIGIDO: Salva usando settings.set() para persistir em Redis + DB.
- `fetch_query_history(limit)` (`app/app/dash_control_panel.py:90`): CORRIGIDO: Lê da tabela 'query_log' e colunas corretas.
- `get_nsga_weights()` (`app/app/dash_control_panel.py:108`): NOTA: Esta função lê uma tabela 'nsga_weights' com colunas 'objective' e 'weight'.
- `update_nsga_weight(objective, new_value)` (`app/app/dash_control_panel.py:130`): CORRIGIDO: Adicionado conn.commit()
- `render_content(tab)` (`app/app/dash_control_panel.py:174`): Executa render content.
- `save_variables(n_clicks, temp, tokens, top_p, bandit)` (`app/app/dash_control_panel.py:261`): Executa save variables.
- `refresh_history(n_clicks)` (`app/app/dash_control_panel.py:280`): Executa refresh history.
- `update_nsga_weights_callback(n, w_acc, w_lat, w_cost)` (`app/app/dash_control_panel.py:293`): Executa update nsga weights callback.

## `app/app/db.py`

Resumo do arquivo: db.py — Centralized Database Connection Management

### Funções de módulo

- `_get_db_config()` (`app/app/db.py:30`): Get database configuration from environment variables.
- `get_db_url(config)` (`app/app/db.py:41`): Build the database URL from configuration.
- `get_engine()` (`app/app/db.py:68`): Get the singleton database engine instance.
- `close_engine()` (`app/app/db.py:123`): Close the database engine and dispose of all connections.
- `get_pool_stats()` (`app/app/db.py:147`): Get database connection pool statistics.
- `check_db_health()` (`app/app/db.py:190`): Check database health and return detailed status.

### Classes e métodos

- Classe `_LazyEngine` (`app/app/db.py:229`): Lazy engine accessor for backward compatibility.
  - `_LazyEngine.__getattr__(self, name)` (`app/app/db.py:232`): Executa getattr.
  - `_LazyEngine.__call__(self, *args, **kwargs)` (`app/app/db.py:236`): Executa call.

## `app/app/db_manager.py`

Resumo do arquivo: db_manager.py (FULL SCHEMA + PRICING SEED)

### Funções de módulo

- `seed_pricing_data(conn)` (`app/app/db_manager.py:346`): Popula a tabela model_pricing com valores de mercado atualizados.
- `ensure_table(table_name, ddl, conn)` (`app/app/db_manager.py:403`): Cria a tabela se não existir.
- `ensure_columns(table_name, columns, conn)` (`app/app/db_manager.py:410`): Verifica se as colunas existem. Se não, cria.
- `ensure_indexes(conn)` (`app/app/db_manager.py:424`): Create performance indexes (Quick Win #3).
- `initialize_system()` (`app/app/db_manager.py:448`): Executa o setup completo do banco.

## `app/app/drift_detector.py`

Resumo do arquivo: drift_detector.py — Query Distribution Drift Detection

### Funções de módulo

- `cosine_distance(a, b)` (`app/app/drift_detector.py:38`): Compute cosine distance between two vectors.
- `get_drift_detector()` (`app/app/drift_detector.py:325`): Get the global drift detector instance.

### Classes e métodos

- Classe `QueryDriftDetector` (`app/app/drift_detector.py:57`): Detects drift in query distribution using embedding centroids.
  - `QueryDriftDetector.__new__(cls)` (`app/app/drift_detector.py:68`): Executa new.
  - `QueryDriftDetector.__init__(self)` (`app/app/drift_detector.py:77`): Inicializa estado interno necessário para uso da classe.
  - `QueryDriftDetector._get_redis(self)` (`app/app/drift_detector.py:101`): Executa get redis.
  - `QueryDriftDetector._load_state(self)` (`app/app/drift_detector.py:105`): Load persisted baseline centroid from Redis.
  - `QueryDriftDetector._save_baseline(self)` (`app/app/drift_detector.py:132`): Persist baseline centroid to Redis.
  - `QueryDriftDetector._save_stats(self)` (`app/app/drift_detector.py:148`): Persist drift statistics to Redis.
  - `QueryDriftDetector._compute_centroid(self, embeddings)` (`app/app/drift_detector.py:165`): Compute centroid of a list of embeddings.
  - `QueryDriftDetector.record_query(self, embedding)` (`app/app/drift_detector.py:171`): Record a query embedding and check for drift.
  - `QueryDriftDetector._update_baseline(self, new_embedding)` (`app/app/drift_detector.py:217`): Update baseline centroid with a new embedding using running mean.
  - `QueryDriftDetector._check_drift(self)` (`app/app/drift_detector.py:232`): Check if current query distribution has drifted from baseline.
  - `QueryDriftDetector.get_status(self)` (`app/app/drift_detector.py:272`): Get current drift detector status.
  - `QueryDriftDetector.reset_baseline(self)` (`app/app/drift_detector.py:290`): Reset baseline centroid to rebuild from scratch.
  - `QueryDriftDetector.force_baseline_update(self)` (`app/app/drift_detector.py:305`): Force update baseline from current recent embeddings.

## `app/app/embeddings.py`

Resumo do arquivo: embeddings.py — Híbrido Otimizado (CPU Local + Cloud Fallback)

### Funções de módulo

- `get_local_model()` (`app/app/embeddings.py:124`): Obtém local model.
- `_hash_text(text, model)` (`app/app/embeddings.py:139`): Executa hash text.
- `_norm(vec)` (`app/app/embeddings.py:144`): Executa norm.
- `_save_cache(key, vec)` (`app/app/embeddings.py:149`): Executa save cache.
- `_load_cache(key)` (`app/app/embeddings.py:157`): Executa load cache.
- `_local_cpu_embed(text)` (`app/app/embeddings.py:172`): Gera embedding localmente na CPU, sem chamar Ollama.
- `embed_text(text)` (`app/app/embeddings.py:190`): Executa embed text.
- `get_embedding_cache_stats()` (`app/app/embeddings.py:235`): Retorna estatísticas do cache L1 de embeddings.
- `embed_image(image_b64)` (`app/app/embeddings.py:241`): Executa embed image.
- `embed_multimodal(text, image_b64)` (`app/app/embeddings.py:245`): Executa embed multimodal.

### Classes e métodos

- Classe `EmbeddingL1Cache` (`app/app/embeddings.py:56`): Cache LRU in-memory para embeddings com TTL.
  - `EmbeddingL1Cache.__init__(self, maxsize, ttl_s)` (`app/app/embeddings.py:59`): Inicializa estado interno necessário para uso da classe.
  - `EmbeddingL1Cache.get(self, key)` (`app/app/embeddings.py:68`): Executa get.
  - `EmbeddingL1Cache.set(self, key, vec)` (`app/app/embeddings.py:85`): Executa set.
  - `EmbeddingL1Cache.stats(self)` (`app/app/embeddings.py:95`): Executa stats.

## `app/app/error_handling.py`

Resumo do arquivo: error_handling.py — Structured Error Handling and Logging

### Funções de módulo

- `classify_exception(exc)` (`app/app/error_handling.py:155`): Classify an exception into category, severity, and retry recommendation.
- `log_error(error, category, model, provider, include_traceback, **context)` (`app/app/error_handling.py:188`): Log an error with structured information.
- `log_provider_error(error, model, operation, **context)` (`app/app/error_handling.py:253`): Log a provider-specific error.
- `log_cache_error(error, operation, cache_type, **context)` (`app/app/error_handling.py:282`): Log a cache-related error.
- `log_infrastructure_error(error, component, **context)` (`app/app/error_handling.py:300`): Log an infrastructure-related error.
- `create_error_response(error_info)` (`app/app/error_handling.py:325`): Create a user-friendly error response from ErrorInfo.

### Classes e métodos

- Classe `ErrorCategory` (`app/app/error_handling.py:31`): Categories for error classification and alerting.
- Classe `ErrorSeverity` (`app/app/error_handling.py:65`): Error severity levels for alerting.
- Classe `ErrorInfo` (`app/app/error_handling.py:79`): Structured error information.
  - `ErrorInfo.to_dict(self)` (`app/app/error_handling.py:94`): Convert to dictionary for logging.

## `app/app/guardrails.py`

Resumo do arquivo: Basic content guardrails (MVP).

### Funções de módulo

- `check_input_guardrails(prompt)` (`app/app/guardrails.py:40`): Evaluate input prompt against basic guardrail patterns.
- `sanitize_output_guardrails(text)` (`app/app/guardrails.py:58`): Mask obvious PII in model output.

### Classes e métodos

- Classe `GuardrailDecision` (`app/app/guardrails.py:34`): Classe `GuardrailDecision`: concentra responsabilidades de guardrails.

## `app/app/health.py`

Resumo do arquivo: health.py — Deep Health Checks for All Dependencies

### Funções de módulo

- `check_redis_health()` (`app/app/health.py:96`): Check Redis connectivity and latency.
- `check_database_health()` (`app/app/health.py:112`): Check MariaDB connectivity.
- `check_vectorstore_health()` (`app/app/health.py:129`): Check ChromaDB connectivity.
- `check_ollama_health()` (`app/app/health.py:150`): Check Ollama server availability.
- `check_circuit_breakers_health()` (`app/app/health.py:173`): Check circuit breaker status.
- `get_full_health_check(force_refresh)` (`app/app/health.py:196`): Run all health checks and return comprehensive status.
- `invalidate_health_cache()` (`app/app/health.py:268`): Invalidate the health check cache (call when components change).
- `get_liveness_check()` (`app/app/health.py:273`): Simple liveness check (is the app running?).
- `get_readiness_check()` (`app/app/health.py:278`): Readiness check (is the app ready to serve traffic?).

### Classes e métodos

- Classe `HealthCache` (`app/app/health.py:32`): Thread-safe cache for health check results.
  - `HealthCache.__init__(self, ttl_s)` (`app/app/health.py:35`): Inicializa estado interno necessário para uso da classe.
  - `HealthCache.get(self)` (`app/app/health.py:42`): Get cached health result if still valid.
  - `HealthCache.set(self, result)` (`app/app/health.py:52`): Cache a health check result.
  - `HealthCache.invalidate(self)` (`app/app/health.py:58`): Clear the cache.
- Classe `HealthStatus` (`app/app/health.py:68`): Classe `HealthStatus`: organiza responsabilidades de health.
- Classe `ComponentHealth` (`app/app/health.py:76`): Classe `ComponentHealth`: organiza responsabilidades de health.
  - `ComponentHealth.to_dict(self)` (`app/app/health.py:84`): Executa to dict.

## `app/app/judges/heuristic.py`

Resumo do arquivo: Módulo principal: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `_len_norm(t)` (`app/app/judges/heuristic.py:3`): Executa len norm.
- `score_coherence(q, a)` (`app/app/judges/heuristic.py:11`): Executa score coherence.
- `score_task_fit(q, a)` (`app/app/judges/heuristic.py:18`): Executa score task fit.
- `score_helpfulness(q, a)` (`app/app/judges/heuristic.py:25`): Executa score helpfulness.

## `app/app/judges/judge.py`

Resumo do arquivo: Módulo principal: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `judge_answer(query, answer, use_rag)` (`app/app/judges/judge.py:9`): Executa judge answer.
- `_heuristic_task(judge_id, fn, q, a)` (`app/app/judges/judge.py:32`): Executa heuristic task.

## `app/app/judges/llm.py`

Resumo do arquivo: Módulo principal: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `_build_prompt(user_q, assistant_a, use_rag)` (`app/app/judges/llm.py:15`): Executa build prompt.
- `_score_sync(user_q, assistant_a, judge_id, use_rag)` (`app/app/judges/llm.py:31`): Executa score sync.
- `score(user_q, assistant_a, idx)` (`app/app/judges/llm.py:48`): Executa score.

## `app/app/judges.py`

Resumo do arquivo: judges.py (VERSÃO FINAL: Binary Verdict + Tie-Breaker Meta-Judge)

### Funções de módulo

- `get_verdict_cache_stats()` (`app/app/judges.py:112`): Retorna estatísticas do cache de verdicts.
- `_safe_setting_float(key, default)` (`app/app/judges.py:133`): Executa safe setting float.
- `_safe_setting_int(key, default)` (`app/app/judges.py:141`): Executa safe setting int.
- `_adaptive_threshold(values, base)` (`app/app/judges.py:200`): Executa adaptive threshold.
- `_image_hash_from_b64(image_b64)` (`app/app/judges.py:208`): Executa image hash from b64.
- `_ensure_judge_logs_table()` (`app/app/judges.py:224`): Executa ensure judge logs table.
- `_load_judge_stats(window_minutes)` (`app/app/judges.py:234`): Executa load judge stats.
- `_score_candidate(s)` (`app/app/judges.py:279`): Executa score candidate.
- `_choose_two(models, stats)` (`app/app/judges.py:286`): Executa choose two.
- `get_rag_context(query, n_results, max_chars)` (`app/app/judges.py:325`): Obtém rag context.
- `heuristic_score(answer)` (`app/app/judges.py:348`): Executa heuristic score.
- `_ensure_judge_calibration_table()` (`app/app/judges.py:384`): Ensure judge_calibration table exists.
- `_persist_judge_metrics(judge_model, score, latency, cost, consistency, fitness)` (`app/app/judges.py:393`): Executa persist judge metrics.
- `_persist_judge_log(query, answer, judge_model, score, modality, image_hash)` (`app/app/judges.py:415`): Executa persist judge log.
- `_describe_image_if_needed(image_b64, modality)` (`app/app/judges.py:448`): Executa describe image if needed.
- `_extract_binary_verdict(text)` (`app/app/judges.py:492`): Extrai o veredito binário da resposta do juiz.
- `_meta_evaluate_binary(query, answer, conflicting_verdicts, base_prompt, reference)` (`app/app/judges.py:514`): Meta-Juiz para desempate binário.
- `_llm_pair_score(query, answer, use_rag, modality, image_b64, reference)` (`app/app/judges.py:569`): Executa llm pair score.
- `llm_based_score(query, answer, use_rag, modality, image_b64, reference)` (`app/app/judges.py:710`): Executa llm based score.
- `judge_answer(query, answer, use_rag, modality, image_b64, reference)` (`app/app/judges.py:722`): Executa judge answer.
- `record_judge_calibration(judge_model, query, predicted_score, was_cached)` (`app/app/judges.py:762`): Record a judge's prediction for calibration analysis.
- `update_calibration_cache_status(query)` (`app/app/judges.py:802`): Update calibration records when a response is cached.
- `get_judge_calibration_metrics()` (`app/app/judges.py:829`): Get calibration metrics for all judges.
- `calibrate_judges()` (`app/app/judges.py:892`): Analyze judge calibration and log insights.

### Classes e métodos

- Classe `VerdictCache` (`app/app/judges.py:53`): Cache LRU com TTL para verdicts de juízes.
  - `VerdictCache.__init__(self, maxsize, ttl_s)` (`app/app/judges.py:56`): Inicializa estado interno necessário para uso da classe.
  - `VerdictCache._make_key(self, query, answer)` (`app/app/judges.py:65`): Gera chave de cache baseada em hash(query + answer[:500]).
  - `VerdictCache.get(self, query, answer)` (`app/app/judges.py:70`): Executa get.
  - `VerdictCache.set(self, query, answer, score)` (`app/app/judges.py:87`): Executa set.
  - `VerdictCache.stats(self)` (`app/app/judges.py:98`): Executa stats.
- Classe `JudgeStats` (`app/app/judges.py:179`): Classe `JudgeStats`: organiza responsabilidades de judges.
- Classe `SelectedJudge` (`app/app/judges.py:190`): Classe `SelectedJudge`: organiza responsabilidades de judges.

## `app/app/main.py`

Resumo do arquivo: main.py (Production-Ready with Rate Limiting, Compression, Health Checks)

### Funções de módulo

- `lifespan(_app)` (`app/app/main.py:138`): Executa lifespan.
- `safe_parse_json(payload)` (`app/app/main.py:195`): Executa safe parse json.
- `preload_ollama_models()` (`app/app/main.py:212`): Preload Ollama models asynchronously using httpx.
- `metrics()` (`app/app/main.py:296`): Executa metrics.
- `startup_event()` (`app/app/main.py:302`): Executa startup event.
- `shutdown_event()` (`app/app/main.py:388`): Graceful shutdown handler.
- `_process_query_request(req)` (`app/app/main.py:421`): Process one query request with governance, guardrails and experimentation hooks.
- `route_query(req)` (`app/app/main.py:546`): Executa route query.
- `route_query_stream(req)` (`app/app/main.py:643`): SSE streaming wrapper for query processing.
- `_require_admin(token)` (`app/app/main.py:664`): Executa require admin.
- `_parse_header_roles(x_user_roles)` (`app/app/main.py:679`): Parse comma-separated roles from header.
- `_require_admin_or_role(*, admin_token, user_id, user_roles_header, required_roles, tenant_id)` (`app/app/main.py:686`): Authorize request by admin token or RBAC role.
- `get_settings(x_admin_token)` (`app/app/main.py:715`): Obtém settings.
- `update_settings(payload, x_admin_token)` (`app/app/main.py:722`): Executa update settings.
- `get_circuit_breakers(x_admin_token)` (`app/app/main.py:732`): Get status of all circuit breakers.
- `reset_circuit_breaker(model_name, x_admin_token)` (`app/app/main.py:743`): Reset a specific circuit breaker.
- `get_cascade_status(x_admin_token)` (`app/app/main.py:754`): Get cascade failure detection status.
- `reset_runtime(x_admin_token)` (`app/app/main.py:762`): Reset internal runtime/singleton state for operational recovery.
- `submit_feedback(request)` (`app/app/main.py:776`): Submit user feedback for a model response.
- `feedback_stats(model, hours)` (`app/app/main.py:802`): Get feedback statistics.
- `list_experiments(status, x_admin_token)` (`app/app/main.py:812`): List all A/B experiments.
- `create_experiment(request, x_admin_token)` (`app/app/main.py:833`): Create a new A/B experiment.
- `get_experiment(experiment_id, x_admin_token)` (`app/app/main.py:853`): Get a specific experiment.
- `start_experiment(experiment_id, x_admin_token)` (`app/app/main.py:870`): Start an experiment.
- `pause_experiment(experiment_id, x_admin_token)` (`app/app/main.py:886`): Pause an experiment.
- `complete_experiment(experiment_id, x_admin_token)` (`app/app/main.py:902`): Complete an experiment.
- `get_experiment_results(experiment_id, x_admin_token)` (`app/app/main.py:918`): Get aggregated results for an experiment.
- `delete_experiment(experiment_id, x_admin_token)` (`app/app/main.py:930`): Delete an experiment.
- `upsert_tenant_budget(tenant_id, payload, x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:951`): Create or update tenant budget limits.
- `get_budget(tenant_id, x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:981`): Get tenant budget configuration.
- `get_quota_usage(tenant_id, x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:999`): Get usage summary for one or all tenants.
- `get_audit_events(limit, x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:1017`): Get latest audit events.
- `create_policy(payload, x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:1034`): Create or update a policy version.
- `activate_policy(version, x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:1065`): Activate one policy version.
- `list_policies(x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:1090`): List policy versions.
- `create_eval(payload, x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:1106`): Create an eval run (MVP academic harness).
- `execute_eval(run_id, payload, x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:1146`): Enqueue asynchronous eval execution in Celery.
- `get_eval(run_id, x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:1194`): Get eval run details.
- `list_evals(x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:1215`): List eval runs.
- `get_eval_results(run_id, limit, x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:1231`): Get individual result rows for one eval run.
- `get_eval_significance(run_id, x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:1253`): Get significance report for model comparisons in one eval run.
- `get_eval_task_status(task_id, x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:1274`): Get Celery task status/result for eval execution.
- `cancel_eval_task(task_id, terminate, x_admin_token, x_user_id, x_user_roles)` (`app/app/main.py:1303`): Cancel/revoke a queued eval Celery task.
- `create_role_grant(payload, x_admin_token)` (`app/app/main.py:1328`): Grant a role to a user. Bootstrap is admin-token only.
- `delete_role_grant(payload, x_admin_token)` (`app/app/main.py:1345`): Revoke a role from a user. Bootstrap is admin-token only.
- `get_rbac_roles(user_id, x_admin_token)` (`app/app/main.py:1362`): List RBAC role bindings.
- `v1_route_query(req)` (`app/app/main.py:1376`): V1 Query endpoint - same as /query.
- `v1_health()` (`app/app/main.py:1382`): V1 Health endpoint.
- `health()` (`app/app/main.py:1395`): Deep health check with all component statuses.
- `liveness()` (`app/app/main.py:1411`): Kubernetes liveness probe - is the app running?
- `readiness()` (`app/app/main.py:1417`): Kubernetes readiness probe - is the app ready to serve traffic?

### Classes e métodos

- Classe `CorrelationIdMiddleware` (`app/app/main.py:164`): Middleware that handles correlation ID propagation.
  - `CorrelationIdMiddleware.dispatch(self, request, call_next)` (`app/app/main.py:172`): Executa dispatch.

## `app/app/metrics_collector.py`

Resumo do arquivo: Coletor de métricas multimodais para o Router LLM.

### Funções de módulo

- `_ensure_model_metrics_table()` (`app/app/metrics_collector.py:42`): Versão nova (multimodal).
- `_persist_sample(model_name, modality, latency_s, quality_0_10, cost_usd, cost_per_1k, tokens_in, tokens_out, embedding_dim, vision_usage, generation)` (`app/app/metrics_collector.py:80`): Persiste uma linha de métricas multimodais no MariaDB.
- `update_model_metrics(model_name, latency, quality, cost, *, modality, cost_per_1k, tokens_in, tokens_out, embedding_dim, vision_usage, generation)` (`app/app/metrics_collector.py:146`): Atualiza snapshot e persiste uma amostra multimodal.
- `get_snapshot()` (`app/app/metrics_collector.py:201`): Retorna snapshot (modelo/modality → métricas) thread-safe:

## `app/app/middleware/__init__.py`

Resumo do arquivo: Middleware modules for the LLM Router application.

## `app/app/middleware/backpressure.py`

Resumo do arquivo: backpressure.py — Global Concurrency Limit Middleware

### Funções de módulo

- `get_backpressure()` (`app/app/middleware/backpressure.py:120`): Get or create the global backpressure semaphore.

### Classes e métodos

- Classe `BackpressureSemaphore` (`app/app/middleware/backpressure.py:31`): Global semaphore for request concurrency control.
  - `BackpressureSemaphore.__new__(cls)` (`app/app/middleware/backpressure.py:41`): Executa new.
  - `BackpressureSemaphore.__init__(self)` (`app/app/middleware/backpressure.py:48`): Inicializa estado interno necessário para uso da classe.
  - `BackpressureSemaphore.acquire(self)` (`app/app/middleware/backpressure.py:60`): Try to acquire a slot for processing.
  - `BackpressureSemaphore.release(self)` (`app/app/middleware/backpressure.py:86`): Release a processing slot.
  - `BackpressureSemaphore.current_load(self)` (`app/app/middleware/backpressure.py:98`): Get current number of in-flight requests.
  - `BackpressureSemaphore.max_concurrent(self)` (`app/app/middleware/backpressure.py:103`): Get maximum concurrent requests allowed.
  - `BackpressureSemaphore.get_stats(self)` (`app/app/middleware/backpressure.py:107`): Get backpressure statistics.
- Classe `BackpressureMiddleware` (`app/app/middleware/backpressure.py:128`): Middleware that enforces global concurrency limits.
  - `BackpressureMiddleware.dispatch(self, request, call_next)` (`app/app/middleware/backpressure.py:138`): Executa dispatch.

## `app/app/middleware/rate_limit.py`

Resumo do arquivo: rate_limit.py — Rate Limiting Middleware with Redis Support

### Funções de módulo

- `_get_redis()` (`app/app/middleware/rate_limit.py:40`): Lazy Redis client getter.
- `periodic_cleanup()` (`app/app/middleware/rate_limit.py:282`): Periodically clean up rate limit store to prevent memory bloat.

### Classes e métodos

- Classe `RateLimitStore` (`app/app/middleware/rate_limit.py:51`): Rate limiting store with Redis support and in-memory fallback.
  - `RateLimitStore.__init__(self)` (`app/app/middleware/rate_limit.py:63`): Inicializa estado interno necessário para uso da classe.
  - `RateLimitStore._should_use_redis(self)` (`app/app/middleware/rate_limit.py:71`): Check if Redis is available and should be used.
  - `RateLimitStore.is_rate_limited(self, client_ip, max_requests, window_seconds)` (`app/app/middleware/rate_limit.py:86`): Check if client is rate limited using sliding window algorithm.
  - `RateLimitStore._is_rate_limited_redis(self, client_ip, max_requests, window_seconds)` (`app/app/middleware/rate_limit.py:107`): Redis-based rate limiting using sorted sets.
  - `RateLimitStore._is_rate_limited_memory(self, client_ip, max_requests, window_seconds)` (`app/app/middleware/rate_limit.py:148`): In-memory rate limiting with sliding window.
  - `RateLimitStore.cleanup(self)` (`app/app/middleware/rate_limit.py:172`): Remove old entries to prevent memory bloat.
  - `RateLimitStore.get_stats(self)` (`app/app/middleware/rate_limit.py:194`): Get rate limit store statistics.
- Classe `RateLimitMiddleware` (`app/app/middleware/rate_limit.py:209`): Rate limiting middleware using sliding window algorithm.
  - `RateLimitMiddleware.dispatch(self, request, call_next)` (`app/app/middleware/rate_limit.py:227`): Executa dispatch.

## `app/app/model_registry.py`

Resumo do arquivo: model_registry.py — Centralized Model Configuration Registry

### Funções de módulo

- `get_model_registry()` (`app/app/model_registry.py:511`): Get the global model registry instance.
- `get_model_config(model_name)` (`app/app/model_registry.py:517`): Get model configuration by name.

### Classes e métodos

- Classe `Provider` (`app/app/model_registry.py:28`): Supported LLM providers.
- Classe `Capability` (`app/app/model_registry.py:36`): Model capabilities.
- Classe `ModelConfig` (`app/app/model_registry.py:50`): Configuration for a single model.
  - `ModelConfig.__post_init__(self)` (`app/app/model_registry.py:84`): Executa post init.
  - `ModelConfig.full_name(self)` (`app/app/model_registry.py:90`): Get the full model name with provider prefix.
  - `ModelConfig.supports_vision(self)` (`app/app/model_registry.py:95`): Executa supports vision.
  - `ModelConfig.supports_streaming(self)` (`app/app/model_registry.py:100`): Executa supports streaming.
  - `ModelConfig.supports_reasoning(self)` (`app/app/model_registry.py:105`): Executa supports reasoning.
  - `ModelConfig.calculate_cost(self, input_tokens, output_tokens)` (`app/app/model_registry.py:109`): Calculate the cost for a request.
- Classe `ModelRegistry` (`app/app/model_registry.py:120`): Centralized registry for all model configurations.
  - `ModelRegistry.__new__(cls)` (`app/app/model_registry.py:133`): Singleton pattern.
  - `ModelRegistry.__init__(self)` (`app/app/model_registry.py:140`): Inicializa estado interno necessário para uso da classe.
  - `ModelRegistry._initialize_default_models(self)` (`app/app/model_registry.py:149`): Initialize with default model configurations.
  - `ModelRegistry.register(self, config)` (`app/app/model_registry.py:375`): Register a model configuration.
  - `ModelRegistry.get(self, model_name)` (`app/app/model_registry.py:382`): Get model configuration by name.
  - `ModelRegistry.get_or_default(self, model_name)` (`app/app/model_registry.py:394`): Get model config or return a default configuration.
  - `ModelRegistry.list_models(self, provider, capability)` (`app/app/model_registry.py:415`): List all registered models, optionally filtered.
  - `ModelRegistry.get_fallback_chain(self, model_name, max_depth)` (`app/app/model_registry.py:444`): Get the fallback chain for a model.
  - `ModelRegistry.get_vision_models(self)` (`app/app/model_registry.py:480`): Get all models that support vision.
  - `ModelRegistry.get_reasoning_models(self)` (`app/app/model_registry.py:484`): Get all models that support reasoning/chain-of-thought.
  - `ModelRegistry.get_local_models(self)` (`app/app/model_registry.py:488`): Get all local (Ollama) models.
  - `ModelRegistry.get_cheapest_model(self, capability)` (`app/app/model_registry.py:492`): Get the cheapest model (by output cost), optionally with a specific capability.

## `app/app/nsga_meta_optimizer.py`

Resumo do arquivo: nsga_meta_optimizer.py (MULTIMODAL)

### Funções de módulo

- `save_result(modality, trial_id, params, eff_mean, eff_std)` (`app/app/nsga_meta_optimizer.py:84`): Escreve uma linha multimodal em nsga_meta_results.
- `evaluate_once(modality, N_pop, N_gen, cxpb, mutpb, eta_c, eta_m)` (`app/app/nsga_meta_optimizer.py:120`): Executa NSGA-II real para a modalidade solicitada.
- `build_objective(modality)` (`app/app/nsga_meta_optimizer.py:145`): Cria uma função objetivo isolada para cada modalidade.
- `run_scheduled_optimization(n_trials)` (`app/app/nsga_meta_optimizer.py:199`): Run meta-optimization with optional reduced trials.
- `_scheduled_optimizer_loop()` (`app/app/nsga_meta_optimizer.py:272`): Background loop that runs meta-optimization at scheduled hour.
- `start_scheduled_optimizer()` (`app/app/nsga_meta_optimizer.py:302`): Start the scheduled optimizer background thread.

## `app/app/nsga_weights_updater.py`

Resumo do arquivo: nsga_weights_updater.py — Otimizador Multimodal (NSGA-II + UQ Tuning + Strategy Tuning)

### Funções de módulo

- `get_redis_client()` (`app/app/nsga_weights_updater.py:63`): Obtém redis client.
- `init_db_tables()` (`app/app/nsga_weights_updater.py:83`): Executa init db tables.
- `load_candidate_models(modality)` (`app/app/nsga_weights_updater.py:108`): Executa load candidate models.
- `aggregate_ema_by_model(modality, models)` (`app/app/nsga_weights_updater.py:145`): Executa aggregate ema by model.
- `store_efficiency_history(modality, efficiency)` (`app/app/nsga_weights_updater.py:196`): Store efficiency value in Redis history list.
- `get_efficiency_history(modality)` (`app/app/nsga_weights_updater.py:211`): Retrieve efficiency history from Redis.
- `compute_convergence_metrics(history)` (`app/app/nsga_weights_updater.py:224`): Compute convergence metrics from efficiency history.
- `check_optimization_health(modality, current_efficiency)` (`app/app/nsga_weights_updater.py:268`): Check NSGA-II optimization health and log warnings if degraded.
- `run_nsga_optimization(modality, models, metrics, n_pop, n_gen)` (`app/app/nsga_weights_updater.py:308`): Roda o NSGA-II.
- `tune_uncertainty_threshold(current_efficiency)` (`app/app/nsga_weights_updater.py:373`): Executa tune uncertainty threshold.
- `tune_global_strategy_weights(sys_metrics)` (`app/app/nsga_weights_updater.py:398`): Ajusta os pesos globais (NSGA_W_QUALITY, etc.) baseado no desempenho
- `tune_risk_factors()` (`app/app/nsga_weights_updater.py:453`): Tune risk factors based on observed quality outcomes by model type and UQ level.
- `calibrate_uncertainty_threshold()` (`app/app/nsga_weights_updater.py:608`): Calibrate uncertainty threshold based on actual quality outcomes.
- `persist_results(modality, weights)` (`app/app/nsga_weights_updater.py:723`): Executa persist results.
- `tune_weights_from_judge_feedback()` (`app/app/nsga_weights_updater.py:748`): Adjust NSGA weights based on recent judge verdicts.
- `run_optimization_cycle(modality)` (`app/app/nsga_weights_updater.py:797`): Executa run optimization cycle.
- `trigger_run(modality)` (`app/app/nsga_weights_updater.py:835`): Executa trigger run.
- `metrics()` (`app/app/nsga_weights_updater.py:847`): Executa metrics.
- `health()` (`app/app/nsga_weights_updater.py:852`): Executa health.
- `trigger_calibration()` (`app/app/nsga_weights_updater.py:858`): Manually trigger a calibration cycle.
- `calibration_status()` (`app/app/nsga_weights_updater.py:869`): Get current calibration status and metrics.
- `run_calibration_cycle()` (`app/app/nsga_weights_updater.py:922`): Run all Phase 5 calibration functions.
- `background_loop()` (`app/app/nsga_weights_updater.py:992`): Executa background loop.

## `app/app/observability.py`

Resumo do arquivo: observability.py

### Funções de módulo

- `_ensure_prometheus_dir()` (`app/app/observability.py:35`): Garante que o diretório PROMETHEUS_MULTIPROC_DIR exista.
- `_build_registry()` (`app/app/observability.py:60`): Cria o CollectorRegistry global.
- `render_metrics_response()` (`app/app/observability.py:82`): Retorna (body_bytes, content_type_str) para o endpoint /metrics.
- `_add_correlation_id(logger, method_name, event_dict)` (`app/app/observability.py:691`): Add correlation ID to log events if available.
- `setup_logging(level)` (`app/app/observability.py:703`): Configura o Structlog para JSON com timestamp.
- `json_log(level, event, **fields)` (`app/app/observability.py:742`): Helper manual para logs estruturados em UTF-8.

### Classes e métodos

- Classe `JsonUTF8Renderer` (`app/app/observability.py:680`): Renderizador JSON que mantém acentuação legível.
  - `JsonUTF8Renderer.__call__(self, logger, name, event_dict)` (`app/app/observability.py:683`): Executa call.

## `app/app/online_predictor.py`

Resumo do arquivo: online_predictor.py — Real-Time Error Prediction Module (Logistic Regression SGD)

### Funções de módulo

- `get_predictor(model_name)` (`app/app/online_predictor.py:348`): Obtém predictor.
- `get_all_predictor_metrics()` (`app/app/online_predictor.py:355`): Get calibration metrics for all active predictors.
- `calibrate_all_predictors()` (`app/app/online_predictor.py:367`): Run auto-calibration on all active predictors.

### Classes e métodos

- Classe `OnlineErrorPredictor` (`app/app/online_predictor.py:49`): Classe `OnlineErrorPredictor`: organiza responsabilidades de online predictor.
  - `OnlineErrorPredictor.__init__(self, model_name)` (`app/app/online_predictor.py:51`): Inicializa o preditor de erro online usando Regressão Logística.
  - `OnlineErrorPredictor._init_model(self)` (`app/app/online_predictor.py:87`): Define o pipeline: StandardScaler -> Logistic Regression (SGD).
  - `OnlineErrorPredictor._embed_to_dict(self, embedding)` (`app/app/online_predictor.py:99`): Converte lista de floats para dicionário (formato exigido pelo River).
  - `OnlineErrorPredictor.predict_error_probability(self, embedding)` (`app/app/online_predictor.py:103`): Estima a probabilidade de o modelo ERRAR a resposta para este embedding.
  - `OnlineErrorPredictor.learn(self, embedding, is_correct)` (`app/app/online_predictor.py:139`): Atualiza os pesos do modelo com um novo exemplo rotulado (Feedback Loop).
  - `OnlineErrorPredictor.record_outcome(self, predicted_prob, actual_error)` (`app/app/online_predictor.py:163`): Record a prediction vs actual outcome for calibration analysis.
  - `OnlineErrorPredictor.compute_brier_score(self)` (`app/app/online_predictor.py:190`): Compute Brier score for calibration assessment.
  - `OnlineErrorPredictor.compute_accuracy(self)` (`app/app/online_predictor.py:212`): Compute binary classification accuracy.
  - `OnlineErrorPredictor.get_calibration_metrics(self)` (`app/app/online_predictor.py:224`): Get all calibration metrics for monitoring.
  - `OnlineErrorPredictor.auto_calibrate_temperature(self)` (`app/app/online_predictor.py:239`): Auto-calibrate temperature using Platt scaling on validation log.
  - `OnlineErrorPredictor._compute_brier_with_temp(self, temp)` (`app/app/online_predictor.py:269`): Compute Brier score with a specific temperature value.
  - `OnlineErrorPredictor.save(self)` (`app/app/online_predictor.py:288`): Persiste o modelo treinado em disco.
  - `OnlineErrorPredictor._save_validation(self)` (`app/app/online_predictor.py:302`): Save validation log and calibration parameters.
  - `OnlineErrorPredictor._load(self)` (`app/app/online_predictor.py:316`): Carrega o modelo do disco se existir.
  - `OnlineErrorPredictor._load_validation(self)` (`app/app/online_predictor.py:326`): Load validation log and calibration parameters.

## `app/app/prometheus_setup.py`

Resumo do arquivo: Módulo principal: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `setup_prometheus()` (`app/app/prometheus_setup.py:18`): Limpa e configura o diretório multiprocess.
- `prometheus_registry()` (`app/app/prometheus_setup.py:35`): Retorna o registro multiprocess configurado.
- `prometheus_metrics()` (`app/app/prometheus_setup.py:42`): Gera a saída /metrics para FastAPI.

## `app/app/providers_async.py`

Resumo do arquivo: providers_async.py (FINAL: Suporte a Reasoning/Thinking Explícito)

### Funções de módulo

- `_classify_provider_exception(e)` (`app/app/providers_async.py:117`): Normalize provider exceptions into stable categories.
- `_build_response_meta(result)` (`app/app/providers_async.py:131`): Build a consistent metadata payload from LLMResponse.
- `get_http_client()` (`app/app/providers_async.py:212`): Get or create a global HTTP client with connection pooling.
- `close_http_client()` (`app/app/providers_async.py:241`): Close the global HTTP client. Call on shutdown.
- `_runtime_provider_settings()` (`app/app/providers_async.py:282`): Read provider runtime settings with safe fallback (supports hot reload).
- `_get_adaptive_timeout(model)` (`app/app/providers_async.py:305`): Calculate adaptive timeout based on model type (Quick Win #6).
- `heuristic_quality_estimate(text)` (`app/app/providers_async.py:336`): Executa heuristic quality estimate.
- `_estimate_tokens(text)` (`app/app/providers_async.py:346`): Executa estimate tokens.
- `render_metrics_response()` (`app/app/providers_async.py:351`): Executa render metrics response.
- `call_model(model, prompt, modality, image_b64, temperature, max_tokens)` (`app/app/providers_async.py:779`): Substituição direta (drop-in) para a antiga função call_model.
- `_ensure_ollama_model_async(model_name)` (`app/app/providers_async.py:834`): Async version using pooled httpx client (Quick Win #8).
- `_ensure_ollama_model(model_name)` (`app/app/providers_async.py:858`): Synchronous function to ensure an Ollama model is available.
- `reset_provider_runtime_state()` (`app/app/providers_async.py:926`): Reset global/singleton provider runtime state (test/dev utility).

### Classes e métodos

- Classe `ProviderCallError` (`app/app/providers_async.py:86`): Structured provider call failure.
  - `ProviderCallError.__init__(self, model, message, *, category, retryable)` (`app/app/providers_async.py:89`): Inicializa estado interno necessário para uso da classe.
- Classe `ProviderCircuitOpenError` (`app/app/providers_async.py:104`): Raised when a provider circuit breaker is open.
  - `ProviderCircuitOpenError.__init__(self, model, message)` (`app/app/providers_async.py:107`): Inicializa estado interno necessário para uso da classe.
- Classe `LLMResponse` (`app/app/providers_async.py:359`): Classe `LLMResponse`: organiza responsabilidades de providers async.
- Classe `BaseProvider` (`app/app/providers_async.py:375`): Classe `BaseProvider`: organiza responsabilidades de providers async.
  - `BaseProvider.__init__(self, name, concurrency_limit)` (`app/app/providers_async.py:377`): Inicializa estado interno necessário para uso da classe.
  - `BaseProvider.generate(self, prompt, image_b64, **kwargs)` (`app/app/providers_async.py:384`): Executa generate.
  - `BaseProvider._record_metrics(self, model, latency, cost, success)` (`app/app/providers_async.py:388`): Executa record metrics.
- Classe `OpenAIProvider` (`app/app/providers_async.py:402`): Classe `OpenAIProvider`: organiza responsabilidades de providers async.
  - `OpenAIProvider.__init__(self)` (`app/app/providers_async.py:404`): Inicializa estado interno necessário para uso da classe.
  - `OpenAIProvider.generate(self, prompt, image_b64, **kwargs)` (`app/app/providers_async.py:412`): Executa generate.
- Classe `AnthropicProvider` (`app/app/providers_async.py:470`): Classe `AnthropicProvider`: organiza responsabilidades de providers async.
  - `AnthropicProvider.__init__(self)` (`app/app/providers_async.py:472`): Inicializa estado interno necessário para uso da classe.
  - `AnthropicProvider.generate(self, prompt, image_b64, **kwargs)` (`app/app/providers_async.py:480`): Executa generate.
- Classe `GeminiProvider` (`app/app/providers_async.py:524`): Classe `GeminiProvider`: organiza responsabilidades de providers async.
  - `GeminiProvider.__init__(self)` (`app/app/providers_async.py:577`): Inicializa estado interno necessário para uso da classe.
  - `GeminiProvider.generate(self, prompt, image_b64, **kwargs)` (`app/app/providers_async.py:585`): Executa generate.
- Classe `OllamaProvider` (`app/app/providers_async.py:630`): Classe `OllamaProvider`: organiza responsabilidades de providers async.
  - `OllamaProvider.__init__(self)` (`app/app/providers_async.py:632`): Inicializa estado interno necessário para uso da classe.
  - `OllamaProvider._refresh_concurrency_limit(self)` (`app/app/providers_async.py:638`): Executa refresh concurrency limit.
  - `OllamaProvider.generate(self, prompt, image_b64, **kwargs)` (`app/app/providers_async.py:649`): Executa generate.
- Classe `ProviderFactory` (`app/app/providers_async.py:757`): Classe `ProviderFactory`: organiza responsabilidades de providers async.
  - `ProviderFactory.get_provider(cls, model_name)` (`app/app/providers_async.py:762`): Obtém provider.

## `app/app/query_service.py`

Resumo do arquivo: query_service.py — versão MULTIMODAL COMPLETA

### Funções de módulo

- `_get_engine()` (`app/app/query_service.py:51`): Get database engine from centralized module.
- `_to_blob(vec)` (`app/app/query_service.py:83`): Converte listas/np arrays para bytes binários (LONGBLOB).
- `_safe_json(obj)` (`app/app/query_service.py:93`): Serializa payload multimodal em JSON seguro.
- `ensure_query_log()` (`app/app/query_service.py:129`): Cria a tabela query_log multimodal + embeddings + payload.
- `insert_query_log(*, query_text, model, modality, image_provided, answer, image_output_b64, latency_s, cost_per_1k, quality, reward, context_label, raw_payload, query_embedding, answer_embedding)` (`app/app/query_service.py:187`): Executa insert query log.

### Classes e métodos

- Classe `_EngineProxy` (`app/app/query_service.py:60`): Proxy to centralized engine for backward compatibility.
  - `_EngineProxy.begin(self)` (`app/app/query_service.py:63`): Executa begin.
  - `_EngineProxy.connect(self)` (`app/app/query_service.py:67`): Executa connect.
  - `_EngineProxy.execute(self, *args, **kwargs)` (`app/app/query_service.py:71`): Executa execute.

## `app/app/rag_context_provider.py`

Resumo do arquivo: rag_context_provider.py

### Funções de módulo

- `get_rag_context(query, k, modality, image_b64)` (`app/app/rag_context_provider.py:40`): Recupera contexto para enriquecer um prompt — inclusive para juízes multimodais.

## `app/app/rag_healthcheck.py`

Resumo do arquivo: rag_healthcheck.py

### Funções de módulo

- `rag_healthcheck()` (`app/app/rag_healthcheck.py:43`): Executa rag healthcheck.
- `rag_healthcheck_sync(timeout_s)` (`app/app/rag_healthcheck.py:165`): Wrapper síncrono — útil para scripts, CLIs e contexts não-async.
- `_finalize(report, t0)` (`app/app/rag_healthcheck.py:176`): Executa finalize.

## `app/app/rag_local.py`

Resumo do arquivo: rag_local.py — RAG Multimodal Unificado (Com suporte Imagem -> Texto)

### Funções de módulo

- `_hash_image(image_b64)` (`app/app/rag_local.py:62`): Generate a hash for image caching.
- `_auto_modality(requested, image_b64)` (`app/app/rag_local.py:70`): Executa auto modality.
- `_generate_visual_search_query(image_b64)` (`app/app/rag_local.py:89`): Usa um VLM para descrever a imagem e criar uma string de busca textual.
- `_compute_embedding(query, modality, image_b64)` (`app/app/rag_local.py:156`): Gera o embedding adequado.
- `reciprocal_rank_fusion(vector_results, bm25_results, k)` (`app/app/rag_local.py:196`): Combina duas listas de resultados usando RRF.
- `build_augmented_prompt(query, modality, image_b64, k)` (`app/app/rag_local.py:226`): Executa build augmented prompt.
- `add_document_local(doc_id, text, metadata, modality, image_b64)` (`app/app/rag_local.py:319`): Executa add document local.
- `health()` (`app/app/rag_local.py:345`): Executa health.

## `app/app/reliability.py`

Resumo do arquivo: reliability.py — Reliability Patterns for LLM Providers

### Funções de módulo

- `_runtime_cb_defaults()` (`app/app/reliability.py:34`): Read circuit-breaker defaults at runtime to support hot reload.
- `get_circuit_breaker_manager()` (`app/app/reliability.py:154`): Get the global circuit breaker manager.
- `get_request_deduplicator()` (`app/app/reliability.py:305`): Get the global request deduplicator.
- `execute_with_fallback(primary_model, execute_fn, max_fallbacks)` (`app/app/reliability.py:324`): Execute a request with automatic fallback to alternative models.
- `check_model_health(model_name)` (`app/app/reliability.py:417`): Check the health of a specific model.
- `check_all_models_health()` (`app/app/reliability.py:442`): Check health of all registered models.
- `get_cascade_detector()` (`app/app/reliability.py:654`): Get the global cascade detector instance.
- `reset_reliability_runtime_state()` (`app/app/reliability.py:659`): Reset singleton runtime state (test/dev utility).

### Classes e métodos

- Classe `ModelCircuitBreakerManager` (`app/app/reliability.py:51`): Manages per-model circuit breakers.
  - `ModelCircuitBreakerManager.__new__(cls)` (`app/app/reliability.py:62`): Executa new.
  - `ModelCircuitBreakerManager.__init__(self)` (`app/app/reliability.py:71`): Inicializa estado interno necessário para uso da classe.
  - `ModelCircuitBreakerManager.get_breaker(self, model_name)` (`app/app/reliability.py:80`): Get or create a circuit breaker for a model.
  - `ModelCircuitBreakerManager.get_status(self, model_name)` (`app/app/reliability.py:115`): Get the current status of a model's circuit breaker.
  - `ModelCircuitBreakerManager.get_all_statuses(self)` (`app/app/reliability.py:129`): Get status of all circuit breakers.
  - `ModelCircuitBreakerManager.reset_breaker(self, model_name)` (`app/app/reliability.py:133`): Manually reset a circuit breaker.
  - `ModelCircuitBreakerManager.is_available(self, model_name)` (`app/app/reliability.py:148`): Check if a model is available (circuit not open).
- Classe `InFlightRequest` (`app/app/reliability.py:164`): Tracks an in-flight request.
- Classe `RequestDeduplicator` (`app/app/reliability.py:172`): Deduplicates identical in-flight requests.
  - `RequestDeduplicator.__new__(cls)` (`app/app/reliability.py:183`): Executa new.
  - `RequestDeduplicator.__init__(self)` (`app/app/reliability.py:192`): Inicializa estado interno necessário para uso da classe.
  - `RequestDeduplicator._compute_key(self, query, model, **kwargs)` (`app/app/reliability.py:202`): Compute a unique key for the request.
  - `RequestDeduplicator.deduplicate(self, query, model, execute_fn, **kwargs)` (`app/app/reliability.py:213`): Execute a request with deduplication.
  - `RequestDeduplicator.cleanup_stale(self)` (`app/app/reliability.py:283`): Remove stale in-flight requests.
  - `RequestDeduplicator.get_stats(self)` (`app/app/reliability.py:297`): Get deduplication statistics.
- Classe `FallbackResult` (`app/app/reliability.py:315`): Result of a fallback chain execution.
- Classe `CascadeDetector` (`app/app/reliability.py:467`): Detects cascade failures across circuit breakers.
  - `CascadeDetector.__new__(cls)` (`app/app/reliability.py:491`): Executa new.
  - `CascadeDetector.__init__(self)` (`app/app/reliability.py:500`): Inicializa estado interno necessário para uso da classe.
  - `CascadeDetector.get_failed_model_ratio(self)` (`app/app/reliability.py:523`): Calculate the ratio of models with open circuit breakers.
  - `CascadeDetector.get_severity(self)` (`app/app/reliability.py:542`): Get current cascade failure severity level.
  - `CascadeDetector.get_severity_name(self)` (`app/app/reliability.py:560`): Get severity level as string.
  - `CascadeDetector.is_emergency_mode(self)` (`app/app/reliability.py:572`): Check if system is in emergency mode.
  - `CascadeDetector.is_degraded(self)` (`app/app/reliability.py:577`): Check if system is in any degraded state.
  - `CascadeDetector.get_emergency_fallback(self)` (`app/app/reliability.py:581`): Get a fallback model for emergency routing.
  - `CascadeDetector.get_status(self)` (`app/app/reliability.py:602`): Get comprehensive cascade detection status.
  - `CascadeDetector.check_and_log_warnings(self)` (`app/app/reliability.py:619`): Check cascade status and log warnings if needed.

## `app/app/reranker.py`

Resumo do arquivo: reranker.py — Módulo de Re-Ranking (Cross-Encoder)

### Funções de módulo

- `get_reranker_model()` (`app/app/reranker.py:30`): Obtém reranker model.
- `rerank_documents(query, documents, top_k)` (`app/app/reranker.py:45`): Reordena uma lista de documentos baseada na relevância para a query.

## `app/app/reset_chroma_collections.py`

Resumo do arquivo: Módulo principal: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `reset_incompatible_collections(chroma_path, expected_dim)` (`app/app/reset_chroma_collections.py:5`): Remove coleções do ChromaDB cuja dimensão de embeddings é diferente da esperada.

## `app/app/risk_tuner.py`

Resumo do arquivo: risk_tuner.py — Adaptive Risk Factor Management

### Funções de módulo

- `get_risk_tuner()` (`app/app/risk_tuner.py:318`): Get the global risk tuner instance.

### Classes e métodos

- Classe `PerformanceRecord` (`app/app/risk_tuner.py:40`): Records model performance for a specific uncertainty regime.
  - `PerformanceRecord.avg_quality(self)` (`app/app/risk_tuner.py:47`): Executa avg quality.
  - `PerformanceRecord.success_rate(self)` (`app/app/risk_tuner.py:54`): Executa success rate.
  - `PerformanceRecord.to_dict(self)` (`app/app/risk_tuner.py:60`): Executa to dict.
  - `PerformanceRecord.from_dict(data)` (`app/app/risk_tuner.py:69`): Executa from dict.
- Classe `AdaptiveRiskTuner` (`app/app/risk_tuner.py:78`): Manages adaptive risk factor adjustments based on model performance.
  - `AdaptiveRiskTuner.__new__(cls)` (`app/app/risk_tuner.py:89`): Executa new.
  - `AdaptiveRiskTuner.__init__(self)` (`app/app/risk_tuner.py:98`): Inicializa estado interno necessário para uso da classe.
  - `AdaptiveRiskTuner._get_redis(self)` (`app/app/risk_tuner.py:121`): Executa get redis.
  - `AdaptiveRiskTuner._load_state(self)` (`app/app/risk_tuner.py:125`): Load persisted performance history from Redis.
  - `AdaptiveRiskTuner._save_state(self)` (`app/app/risk_tuner.py:146`): Persist performance history to Redis.
  - `AdaptiveRiskTuner._get_model_type(self, model)` (`app/app/risk_tuner.py:159`): Determine if model is SOTA or local.
  - `AdaptiveRiskTuner.record_outcome(self, model, quality, is_high_uncertainty)` (`app/app/risk_tuner.py:166`): Record the outcome of a model call for risk tuning.
  - `AdaptiveRiskTuner._calculate_adjustment(self, record, current_factor)` (`app/app/risk_tuner.py:200`): Calculate the adjustment for a risk factor based on performance.
  - `AdaptiveRiskTuner.tune_factors(self)` (`app/app/risk_tuner.py:233`): Tune risk factors based on accumulated performance data.
  - `AdaptiveRiskTuner.get_status(self)` (`app/app/risk_tuner.py:290`): Get current status of the risk tuner.
  - `AdaptiveRiskTuner.reset(self)` (`app/app/risk_tuner.py:309`): Reset all performance tracking data.

## `app/app/roadmap_features.py`

Resumo do arquivo: Roadmap hardening features.

### Funções de módulo

- `ensure_roadmap_tables()` (`app/app/roadmap_features.py:147`): Ensure all governance/policy/eval/rbac tables exist.
- `set_tenant_budget(tenant_id, daily_usd_limit, monthly_usd_limit, enabled)` (`app/app/roadmap_features.py:157`): Create/update budget limits for a tenant.
- `get_tenant_budget(tenant_id)` (`app/app/roadmap_features.py:175`): Get budget configuration for tenant.
- `_usage_snapshot(tenant_id)` (`app/app/roadmap_features.py:187`): Executa usage snapshot.
- `check_tenant_budget(tenant_id, projected_cost_usd)` (`app/app/roadmap_features.py:203`): Check whether tenant can proceed considering budget limits.
- `record_tenant_usage(tenant_id, *, cost_usd, tokens_in, tokens_out, requests)` (`app/app/roadmap_features.py:226`): Accumulate daily usage for tenant.
- `get_usage_summary(tenant_id)` (`app/app/roadmap_features.py:265`): Get usage summary for one tenant or all tenants.
- `log_audit_event(actor, action, resource, tenant_id, metadata)` (`app/app/roadmap_features.py:295`): Persist an audit event.
- `list_audit_events(limit)` (`app/app/roadmap_features.py:315`): List latest audit events.
- `create_policy_version(version, config, description)` (`app/app/roadmap_features.py:333`): Create/update policy version.
- `activate_policy_version(version)` (`app/app/roadmap_features.py:348`): Mark one policy as active.
- `get_active_policy()` (`app/app/roadmap_features.py:359`): Fetch active policy version.
- `list_policy_versions(limit)` (`app/app/roadmap_features.py:375`): List policy versions.
- `create_eval_run(run_id, prompts, policy_version, tenant_id, notes)` (`app/app/roadmap_features.py:393`): Create eval run header.
- `update_eval_run_status(run_id, status, summary)` (`app/app/roadmap_features.py:413`): Update eval run status and summary.
- `add_eval_result(run_id, prompt_text, model, quality, latency_s, cost_usd, metadata)` (`app/app/roadmap_features.py:422`): Persist one eval sample result.
- `list_eval_run_results(run_id, limit)` (`app/app/roadmap_features.py:444`): List individual result rows for an eval run.
- `get_eval_run(run_id)` (`app/app/roadmap_features.py:470`): Fetch eval run with aggregated stats.
- `list_eval_runs(limit)` (`app/app/roadmap_features.py:505`): List latest eval runs.
- `grant_role(user_id, role_name, tenant_id)` (`app/app/roadmap_features.py:515`): Grant role to a user.
- `revoke_role(user_id, role_name, tenant_id)` (`app/app/roadmap_features.py:530`): Revoke role from user.
- `list_roles(user_id)` (`app/app/roadmap_features.py:546`): List all role bindings or those from one user.
- `get_roles_for_user(user_id, tenant_id)` (`app/app/roadmap_features.py:561`): Resolve effective roles for user.
- `check_access(*, user_id, required_roles, tenant_id, header_roles)` (`app/app/roadmap_features.py:576`): Check access by required roles.
- `_welch_ttest(a, b)` (`app/app/roadmap_features.py:593`): Compute Welch's t-test with scipy fallback.
- `eval_significance_report(run_id)` (`app/app/roadmap_features.py:608`): Build per-model leaderboard and significance against the top model by quality.

### Classes e métodos

- Classe `BudgetCheck` (`app/app/roadmap_features.py:129`): Classe `BudgetCheck`: concentra responsabilidades de roadmap features.
- Classe `AccessDecision` (`app/app/roadmap_features.py:140`): Classe `AccessDecision`: concentra responsabilidades de roadmap features.

## `app/app/router_core.py`

Resumo do arquivo: router_core.py — Multimodal + UM-RAG + Meta-bandit + UQ + Online Learning

### Funções de módulo

- `_get_db_engine()` (`app/app/router_core.py:83`): Get database engine from centralized module.
- `_safe_setting_int(key, default)` (`app/app/router_core.py:87`): Executa safe setting int.
- `_safe_setting_float(key, default)` (`app/app/router_core.py:95`): Executa safe setting float.
- `_safe_setting_bool(key, default)` (`app/app/router_core.py:103`): Executa safe setting bool.
- `_get_rds()` (`app/app/router_core.py:118`): Executa get rds.
- `_record_dependency_breaker_metrics()` (`app/app/router_core.py:127`): Executa record dependency breaker metrics.
- `_error_budget_window()` (`app/app/router_core.py:137`): Executa error budget window.
- `_record_request_outcome(success)` (`app/app/router_core.py:145`): Executa record request outcome.
- `_is_error_budget_exceeded()` (`app/app/router_core.py:165`): Executa is error budget exceeded.
- `_ema_batch_flusher()` (`app/app/router_core.py:385`): Background thread that periodically flushes EMA batch queue.
- `_cleanup_ema_history_log()` (`app/app/router_core.py:404`): Cleanup old ema_history_log entries (runs daily).
- `_update_db_pool_metrics()` (`app/app/router_core.py:422`): Background thread to update DB pool metrics (Quick Win #10).
- `get_dynamic_strategy_weights(modality)` (`app/app/router_core.py:436`): Recupera os pesos da estratégia (Objetivos) diretamente do Settings Dinâmico.
- `_load_ema_from_db()` (`app/app/router_core.py:458`): Carrega histórico EMA do banco para memória.
- `_persist_ema(modality, model, record)` (`app/app/router_core.py:480`): Queue EMA update for batch persistence (Quick Win #1).
- `_cleanup_old_query_logs()` (`app/app/router_core.py:484`): Limpeza periódica de logs antigos.
- `_cleanup_ema_history()` (`app/app/router_core.py:499`): Limpeza periódica de entradas EMA expiradas.
- `start_background_services()` (`app/app/router_core.py:513`): Start router maintenance background services exactly once.
- `stop_background_services()` (`app/app/router_core.py:535`): Stop router maintenance background services and flush pending EMA batch.
- `_route_and_answer_internal(query, system_prompt, use_rag, max_tokens, temperature, modality, image_b64, rag_modality, use_cache)` (`app/app/router_core.py:556`): Internal implementation of route_and_answer.
- `route_and_answer(query, system_prompt, use_rag, max_tokens, temperature, modality, image_b64, rag_modality, use_cache, timeout_seconds, deduplicate)` (`app/app/router_core.py:821`): Executa o fluxo crítico de resposta com timeout global e deduplicação.
- `process_background_feedback(query, answer, chosen_model, modality, latency_s, cost_val, image_b64, raw_payload, prompt_tokens, completion_tokens)` (`app/app/router_core.py:921`): Executado após a resposta ser enviada ao usuário.
- `reset_router_runtime_state()` (`app/app/router_core.py:1088`): Reset global runtime state for tests/dev.

### Classes e métodos

- Classe `EMAHistoryCache` (`app/app/router_core.py:196`): Cache LRU com TTL para histórico EMA.
  - `EMAHistoryCache.__init__(self, maxsize, ttl_s)` (`app/app/router_core.py:199`): Inicializa estado interno necessário para uso da classe.
  - `EMAHistoryCache.get(self, key)` (`app/app/router_core.py:207`): Executa get.
  - `EMAHistoryCache.set(self, key, value)` (`app/app/router_core.py:226`): Executa set.
  - `EMAHistoryCache.__contains__(self, key)` (`app/app/router_core.py:241`): Executa contains.
  - `EMAHistoryCache.items(self)` (`app/app/router_core.py:245`): Executa items.
  - `EMAHistoryCache.cleanup_expired(self)` (`app/app/router_core.py:250`): Remove entradas expiradas. Retorna número de itens removidos.
  - `EMAHistoryCache.size(self)` (`app/app/router_core.py:267`): Executa size.
- Classe `EMABatchQueue` (`app/app/router_core.py:279`): Queue for batching EMA updates to reduce DB writes.
  - `EMABatchQueue.__init__(self, max_size, flush_interval)` (`app/app/router_core.py:282`): Inicializa estado interno necessário para uso da classe.
  - `EMABatchQueue.add(self, modality, model, record)` (`app/app/router_core.py:290`): Add or update an EMA record in the queue.
  - `EMABatchQueue._flush_locked(self)` (`app/app/router_core.py:301`): Flush all queued updates to DB. Must hold lock.
  - `EMABatchQueue._persist_batch(self, items)` (`app/app/router_core.py:314`): Persist batch of EMA updates to database.
  - `EMABatchQueue.flush(self)` (`app/app/router_core.py:364`): Public flush method.
  - `EMABatchQueue.should_flush(self)` (`app/app/router_core.py:369`): Check if it's time to flush based on interval.
  - `EMABatchQueue.size(self)` (`app/app/router_core.py:373`): Executa size.

## `app/app/router_strategy.py`

Resumo do arquivo: router_strategy.py (Versão Final: Filtros Bilaterais de Segurança)

### Funções de módulo

- `_is_sota(model_name)` (`app/app/router_strategy.py:29`): Executa is sota.
- `_is_local(model_name)` (`app/app/router_strategy.py:33`): Executa is local.
- `_get_circuit_breaker_penalty(model)` (`app/app/router_strategy.py:37`): Calculate a penalty factor based on circuit breaker state.
- `choose_top2_models(candidates, weights, query_text, modality, uncertainty_score, min_quality)` (`app/app/router_strategy.py:70`): Executa choose top2 models.

## `app/app/routers/rag_router.py`

Resumo do arquivo: rag_router.py (CORRIGIDO: Importação e Async)

### Funções de módulo

- `chunk_text(text, chunk_size, overlap)` (`app/app/routers/rag_router.py:40`): Divide o texto em blocos de tamanho fixo com sobreposição.
- `extract_text_from_pdf(file_bytes)` (`app/app/routers/rag_router.py:54`): Extrai texto bruto de um PDF recebido via upload.
- `summarize_text(text)` (`app/app/routers/rag_router.py:63`): Gera título e resumo com LLM.
- `add_doc(file)` (`app/app/routers/rag_router.py:103`): Recebe um arquivo (PDF, MD, TXT), gera embeddings e insere no ChromaDB.
- `ingest_text(req)` (`app/app/routers/rag_router.py:186`): Endpoint para ingestão direta de texto (usado por serviços externos como o Auditor).

### Classes e métodos

- Classe `IngestRequest` (`app/app/routers/rag_router.py:178`): Classe `IngestRequest`: organiza responsabilidades de rag router.

## `app/app/runtime_state.py`

Resumo do arquivo: Global runtime state reset helpers for tests/dev.

### Funções de módulo

- `reset_runtime_state()` (`app/app/runtime_state.py:13`): Reset all known global runtime/singleton states.

## `app/app/schemas.py`

Resumo do arquivo: schemas.py (VERSÃO COMPLETA DE PRODUÇÃO)

### Classes e métodos

- Classe `Modality` (`app/app/schemas.py:26`): Classe `Modality`: organiza responsabilidades de schemas.
- Classe `QueryRequest` (`app/app/schemas.py:37`): Payload de entrada para o endpoint /query.
  - `QueryRequest.query_not_empty(cls, v)` (`app/app/schemas.py:141`): Ensure query is not just whitespace.
  - `QueryRequest.validate_modality(cls, v)` (`app/app/schemas.py:150`): Validate modality is one of the allowed values.
- Classe `JudgeScore` (`app/app/schemas.py:163`): Registro de avaliação de um juiz específico.
- Classe `CandidateResult` (`app/app/schemas.py:173`): Representa o resultado de um modelo candidato (antes da escolha final ou para comparação).
- Classe `RouteDecision` (`app/app/schemas.py:205`): Explica o porquê de um modelo ter sido escolhido.
- Classe `QueryResponse` (`app/app/schemas.py:229`): Resposta final enviada ao cliente.

## `app/app/semantic_cache.py`

Resumo do arquivo: semantic_cache.py — Cache Semântico via ChromaDB (Rápido)

### Funções de módulo

- `get_cache_threshold()` (`app/app/semantic_cache.py:31`): Get current cache threshold from settings (dynamic reload).
- `_compute_sha256(text)` (`app/app/semantic_cache.py:41`): Executa compute sha256.
- `_normalize_modality(mod)` (`app/app/semantic_cache.py:46`): Executa normalize modality.
- `_make_embedding(query, modality, image_b64)` (`app/app/semantic_cache.py:138`): Executa make embedding.
- `check_cache(query, modality, image_b64)` (`app/app/semantic_cache.py:153`): Verifica se existe uma resposta similar no ChromaDB.
- `store_cache(query, answer, modality, image_b64, model_used)` (`app/app/semantic_cache.py:219`): Armazena uma resposta de alta qualidade no ChromaDB e L1 cache.
- `get_l1_cache_stats()` (`app/app/semantic_cache.py:271`): Get L1 cache statistics for monitoring.
- `get_cache_hit_rate()` (`app/app/semantic_cache.py:281`): Calculate current cache hit rate.
- `tune_cache_threshold()` (`app/app/semantic_cache.py:295`): Automatically tune cache threshold based on hit rate.
- `reset_cache_stats()` (`app/app/semantic_cache.py:367`): Reset L1 cache statistics.

### Classes e métodos

- Classe `L1Cache` (`app/app/semantic_cache.py:59`): Thread-safe in-memory cache with TTL and LRU eviction.
  - `L1Cache.__init__(self, maxsize, ttl_seconds)` (`app/app/semantic_cache.py:69`): Inicializa estado interno necessário para uso da classe.
  - `L1Cache.get(self, key)` (`app/app/semantic_cache.py:78`): Retrieve a value from cache.
  - `L1Cache.store(self, key, value)` (`app/app/semantic_cache.py:102`): Store a value in cache.
  - `L1Cache.clear(self)` (`app/app/semantic_cache.py:117`): Clear all cache entries.
  - `L1Cache.stats(self)` (`app/app/semantic_cache.py:122`): Return cache statistics.

## `app/app/services/__init__.py`

Resumo do arquivo: Módulo principal: descreve responsabilidades e integrações deste arquivo.

## `app/app/services/bandit_centroids.py`

Resumo do arquivo: Centroid helpers for bandit clustering.

### Funções de módulo

- `normalize_centroid_vec(vec, dim)` (`app/app/services/bandit_centroids.py:11`): Executa normalize centroid vec.
- `nearest_centroid_from_array(v, cents)` (`app/app/services/bandit_centroids.py:23`): Executa nearest centroid from array.

## `app/app/services/bandit_policy.py`

Resumo do arquivo: Policy helpers for meta-bandit decisions.

### Funções de módulo

- `dynamic_epsilon(ctx_stats, default_epsilon)` (`app/app/services/bandit_policy.py:13`): Executa dynamic epsilon.
- `choose_epsilon_greedy(models, ctx_stats, default_epsilon)` (`app/app/services/bandit_policy.py:30`): Executa choose epsilon greedy.
- `choose_ucb1(models, ctx_stats)` (`app/app/services/bandit_policy.py:57`): Executa choose ucb1.
- `choose_thompson(models, ctx_stats)` (`app/app/services/bandit_policy.py:79`): Executa choose thompson.
- `meta_combine_choices(models, ctx_stats, default_epsilon, preferred_strategy)` (`app/app/services/bandit_policy.py:95`): Executa meta combine choices.

## `app/app/services/router_maintenance.py`

Resumo do arquivo: Maintenance helpers for router background services.

### Funções de módulo

- `create_background_threads(cleanup_old_query_logs, cleanup_ema_history, ema_batch_flusher, cleanup_ema_history_log, update_db_pool_metrics)` (`app/app/services/router_maintenance.py:10`): Executa create background threads.

## `app/app/services/router_services.py`

Resumo do arquivo: Shared helpers for router_core routing/feedback/maintenance.

### Funções de módulo

- `normalize_modality(modality, image_b64)` (`app/app/services/router_services.py:10`): Executa normalize modality.
- `build_final_prompt(query, system_prompt, use_rag, rag_text)` (`app/app/services/router_services.py:18`): Executa build final prompt.
- `parse_meta_cost(meta, chosen_model, cost_lookup)` (`app/app/services/router_services.py:37`): Executa parse meta cost.
- `should_enable_dedup(settings_get, requested)` (`app/app/services/router_services.py:65`): Executa should enable dedup.
- `compute_judge_probability(n_samples, predicted_error_prob, chosen_model, min_sample_rate)` (`app/app/services/router_services.py:70`): Executa compute judge probability.

## `app/app/settings_dynamic.py`

Resumo do arquivo: settings_dynamic.py (VERSÃO FINAL: Com Configuração de Amostragem)

### Funções de módulo

- `_get_rds()` (`app/app/settings_dynamic.py:31`): Executa a responsabilidade descrita por este método.
- `_get_settings_engine()` (`app/app/settings_dynamic.py:47`): Get database engine for settings.
- `_invalidate_cache()` (`app/app/settings_dynamic.py:193`): Executa a responsabilidade descrita por este método.
- `_get_from_redis(key)` (`app/app/settings_dynamic.py:207`): Executa a responsabilidade descrita por este método.
- `_get_from_db(key)` (`app/app/settings_dynamic.py:234`): Executa a responsabilidade descrita por este método.
- `_set_to_redis(key, val)` (`app/app/settings_dynamic.py:259`): Executa a responsabilidade descrita por este método.
- `_load_json_list(raw)` (`app/app/settings_dynamic.py:278`): Executa a responsabilidade descrita por este método.
- `get_db_pool_stats()` (`app/app/settings_dynamic.py:988`): Get database connection pool statistics.
- `update_db_pool_metrics()` (`app/app/settings_dynamic.py:1003`): Update Prometheus metrics for DB pool (call periodically).
- `validate_critical_settings(settings_obj)` (`app/app/settings_dynamic.py:1021`): Validate critical runtime settings.
- `start_reload_listener()` (`app/app/settings_dynamic.py:1086`): Executa a responsabilidade descrita por este método.
- `stop_reload_listener()` (`app/app/settings_dynamic.py:1153`): Executa a responsabilidade descrita por este método.

### Classes e métodos

- Classe `_EngineProxy` (`app/app/settings_dynamic.py:66`): Proxy to centralized engine for backward compatibility.
  - `_EngineProxy.begin(self)` (`app/app/settings_dynamic.py:69`): Executa a responsabilidade descrita por este método.
  - `_EngineProxy.connect(self)` (`app/app/settings_dynamic.py:77`): Executa a responsabilidade descrita por este método.
  - `_EngineProxy.pool(self)` (`app/app/settings_dynamic.py:86`): Executa a responsabilidade descrita por este método.
- Classe `LRUCache` (`app/app/settings_dynamic.py:119`): Define responsabilidades de estado e comportamento.
  - `LRUCache.__init__(self, maxsize, ttl_s)` (`app/app/settings_dynamic.py:121`): Executa a responsabilidade descrita por este método.
  - `LRUCache.get(self, key)` (`app/app/settings_dynamic.py:136`): Executa a responsabilidade descrita por este método.
  - `LRUCache.set(self, key, value)` (`app/app/settings_dynamic.py:159`): Executa a responsabilidade descrita por este método.
  - `LRUCache.clear(self)` (`app/app/settings_dynamic.py:177`): Executa a responsabilidade descrita por este método.
- Classe `DynamicSettings` (`app/app/settings_dynamic.py:305`): Define responsabilidades de estado e comportamento.
  - `DynamicSettings.get(self, key, fallback)` (`app/app/settings_dynamic.py:456`): Executa a responsabilidade descrita por este método.
  - `DynamicSettings.set(self, key, value, actor, source)` (`app/app/settings_dynamic.py:480`): Executa a responsabilidade descrita por este método.
  - `DynamicSettings.snapshot(self, only_known)` (`app/app/settings_dynamic.py:514`): Executa a responsabilidade descrita por este método.
  - `DynamicSettings.CANDIDATE_MODELS_LIST(self)` (`app/app/settings_dynamic.py:539`): Executa a responsabilidade descrita por este método.
  - `DynamicSettings.CANDIDATE_VISION_MODELS_LIST(self)` (`app/app/settings_dynamic.py:548`): Executa a responsabilidade descrita por este método.
  - `DynamicSettings.CANDIDATE_MULTIMODAL_MODELS_LIST(self)` (`app/app/settings_dynamic.py:557`): Executa a responsabilidade descrita por este método.
  - `DynamicSettings.VLM_OLLAMA_MODELS(self)` (`app/app/settings_dynamic.py:566`): Executa a responsabilidade descrita por este método.
  - `DynamicSettings.JUDGE_MODELS(self)` (`app/app/settings_dynamic.py:575`): Executa a responsabilidade descrita por este método.
  - `DynamicSettings.EMBED_TEXT_MODEL(self)` (`app/app/settings_dynamic.py:585`): Executa a responsabilidade descrita por este método.
  - `DynamicSettings.TEXT_EMBEDDING_MODEL(self)` (`app/app/settings_dynamic.py:594`): Obtém o valor da configuração `TEXT_EMBEDDING_MODEL`.
  - `DynamicSettings.IMAGE_EMBEDDING_MODEL(self)` (`app/app/settings_dynamic.py:598`): Obtém o valor da configuração `IMAGE_EMBEDDING_MODEL`.
  - `DynamicSettings.MULTIMODAL_EMBEDDING_MODEL(self)` (`app/app/settings_dynamic.py:602`): Obtém o valor da configuração `MULTIMODAL_EMBEDDING_MODEL`.
  - `DynamicSettings.EMBED_MODEL(self)` (`app/app/settings_dynamic.py:606`): Obtém o valor da configuração `EMBED_MODEL`.
  - `DynamicSettings.EMBED_PROVIDER(self)` (`app/app/settings_dynamic.py:610`): Obtém o valor da configuração `EMBED_PROVIDER`.
  - `DynamicSettings.EMBED_DEVICE(self)` (`app/app/settings_dynamic.py:614`): Obtém o valor da configuração `EMBED_DEVICE`.
  - `DynamicSettings.MAX_TOKENS_DEFAULT(self)` (`app/app/settings_dynamic.py:620`): Obtém o valor da configuração `MAX_TOKENS_DEFAULT`.
  - `DynamicSettings.TEMPERATURE_DEFAULT(self)` (`app/app/settings_dynamic.py:624`): Obtém o valor da configuração `TEMPERATURE_DEFAULT`.
  - `DynamicSettings.BANDIT_EPSILON(self)` (`app/app/settings_dynamic.py:628`): Obtém o valor da configuração `BANDIT_EPSILON`.
  - `DynamicSettings.QUERY_LOG_RETENTION_DAYS(self)` (`app/app/settings_dynamic.py:632`): Obtém o valor da configuração `QUERY_LOG_RETENTION_DAYS`.
  - `DynamicSettings.REDIS_HOST(self)` (`app/app/settings_dynamic.py:638`): Obtém o valor da configuração `REDIS_HOST`.
  - `DynamicSettings.REDIS_PORT(self)` (`app/app/settings_dynamic.py:642`): Obtém o valor da configuração `REDIS_PORT`.
  - `DynamicSettings.REDIS_DB(self)` (`app/app/settings_dynamic.py:646`): Obtém o valor da configuração `REDIS_DB`.
  - `DynamicSettings.REDIS_PASSWORD(self)` (`app/app/settings_dynamic.py:650`): Obtém o valor da configuração `REDIS_PASSWORD`.
  - `DynamicSettings.DB_HOST(self)` (`app/app/settings_dynamic.py:655`): Obtém o valor da configuração `DB_HOST`.
  - `DynamicSettings.DB_PORT(self)` (`app/app/settings_dynamic.py:659`): Obtém o valor da configuração `DB_PORT`.
  - `DynamicSettings.DB_USER(self)` (`app/app/settings_dynamic.py:663`): Obtém o valor da configuração `DB_USER`.
  - `DynamicSettings.DB_PASS(self)` (`app/app/settings_dynamic.py:667`): Obtém o valor da configuração `DB_PASS`.
  - `DynamicSettings.DB_NAME(self)` (`app/app/settings_dynamic.py:671`): Obtém o valor da configuração `DB_NAME`.
  - `DynamicSettings.ADMIN_TOKEN(self)` (`app/app/settings_dynamic.py:676`): Obtém o valor da configuração `ADMIN_TOKEN`.
  - `DynamicSettings.ADMIN_TOKEN_PREVIOUS(self)` (`app/app/settings_dynamic.py:680`): Obtém o valor da configuração `ADMIN_TOKEN_PREVIOUS`.
  - `DynamicSettings.JUDGES_ENABLED(self)` (`app/app/settings_dynamic.py:686`): Obtém o valor da configuração `JUDGES_ENABLED`.
  - `DynamicSettings.JUDGES_MODE(self)` (`app/app/settings_dynamic.py:690`): Obtém o valor da configuração `JUDGES_MODE`.
  - `DynamicSettings.JUDGES_LOCAL_MODEL(self)` (`app/app/settings_dynamic.py:694`): Obtém o valor da configuração `JUDGES_LOCAL_MODEL`.
  - `DynamicSettings.JUDGES_REMOTE_MODEL(self)` (`app/app/settings_dynamic.py:698`): Obtém o valor da configuração `JUDGES_REMOTE_MODEL`.
  - `DynamicSettings.JUDGES_TIMEOUT_S(self)` (`app/app/settings_dynamic.py:702`): Obtém o valor da configuração `JUDGES_TIMEOUT_S`.
  - `DynamicSettings.JUDGE_MIN_SAMPLE_RATE(self)` (`app/app/settings_dynamic.py:706`): Obtém o valor da configuração `JUDGE_MIN_SAMPLE_RATE`.
  - `DynamicSettings.OLLAMA_HOST(self)` (`app/app/settings_dynamic.py:712`): Obtém o valor da configuração `OLLAMA_HOST`.
  - `DynamicSettings.OLLAMA_BASE_URL(self)` (`app/app/settings_dynamic.py:716`): Obtém o valor da configuração `OLLAMA_BASE_URL`.
  - `DynamicSettings.CENTROIDS_DIM(self)` (`app/app/settings_dynamic.py:722`): Obtém o valor da configuração `CENTROIDS_DIM`.
  - `DynamicSettings.CENTROIDS_K(self)` (`app/app/settings_dynamic.py:726`): Obtém o valor da configuração `CENTROIDS_K`.
  - `DynamicSettings.CENTROIDS_MIN_SIM_CREATE(self)` (`app/app/settings_dynamic.py:730`): Obtém o valor da configuração `CENTROIDS_MIN_SIM_CREATE`.
  - `DynamicSettings.CENTROIDS_ENABLE_ONLINE(self)` (`app/app/settings_dynamic.py:734`): Obtém o valor da configuração `CENTROIDS_ENABLE_ONLINE`.
  - `DynamicSettings.CENTROIDS_UPDATE_INTERVAL_S(self)` (`app/app/settings_dynamic.py:738`): Obtém o valor da configuração `CENTROIDS_UPDATE_INTERVAL_S`.
  - `DynamicSettings.CENTROIDS_MIN_RECORDS_FOR_TRAIN(self)` (`app/app/settings_dynamic.py:742`): Obtém o valor da configuração `CENTROIDS_MIN_RECORDS_FOR_TRAIN`.
  - `DynamicSettings.CENTROIDS_MAX_HISTORY(self)` (`app/app/settings_dynamic.py:746`): Obtém o valor da configuração `CENTROIDS_MAX_HISTORY`.
  - `DynamicSettings.CENTROIDS_HOURLY_REFRESH_ENABLED(self)` (`app/app/settings_dynamic.py:750`): Obtém o valor da configuração `CENTROIDS_HOURLY_REFRESH_ENABLED`.
  - `DynamicSettings.CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH(self)` (`app/app/settings_dynamic.py:754`): Obtém o valor da configuração `CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH`.
  - `DynamicSettings.NSGA_UPDATE_INTERVAL_S(self)` (`app/app/settings_dynamic.py:760`): Obtém o valor da configuração `NSGA_UPDATE_INTERVAL_S`.
  - `DynamicSettings.NSGA_LOOKBACK_MINUTES(self)` (`app/app/settings_dynamic.py:764`): Obtém o valor da configuração `NSGA_LOOKBACK_MINUTES`.
  - `DynamicSettings.NSGA_LOOKBACK_MAXROWS(self)` (`app/app/settings_dynamic.py:768`): Obtém o valor da configuração `NSGA_LOOKBACK_MAXROWS`.
  - `DynamicSettings.METAOPT_REPS(self)` (`app/app/settings_dynamic.py:772`): Obtém o valor da configuração `METAOPT_REPS`.
  - `DynamicSettings.METAOPT_TRIALS(self)` (`app/app/settings_dynamic.py:776`): Obtém o valor da configuração `METAOPT_TRIALS`.
  - `DynamicSettings.NSGA_W_QUALITY(self)` (`app/app/settings_dynamic.py:782`): Obtém o valor da configuração `NSGA_W_QUALITY`.
  - `DynamicSettings.NSGA_W_LATENCY(self)` (`app/app/settings_dynamic.py:786`): Obtém o valor da configuração `NSGA_W_LATENCY`.
  - `DynamicSettings.NSGA_W_COST(self)` (`app/app/settings_dynamic.py:790`): Obtém o valor da configuração `NSGA_W_COST`.
  - `DynamicSettings.NSGA_W_ALIGNMENT(self)` (`app/app/settings_dynamic.py:794`): Obtém o valor da configuração `NSGA_W_ALIGNMENT`.
  - `DynamicSettings.NSGA_CONVERGENCE_HISTORY_SIZE(self)` (`app/app/settings_dynamic.py:800`): Obtém o valor da configuração `NSGA_CONVERGENCE_HISTORY_SIZE`.
  - `DynamicSettings.CASCADE_WARNING_THRESHOLD(self)` (`app/app/settings_dynamic.py:804`): Obtém o valor da configuração `CASCADE_WARNING_THRESHOLD`.
  - `DynamicSettings.CASCADE_CRITICAL_THRESHOLD(self)` (`app/app/settings_dynamic.py:808`): Obtém o valor da configuração `CASCADE_CRITICAL_THRESHOLD`.
  - `DynamicSettings.RISK_FACTOR_SOTA_HIGH_UQ(self)` (`app/app/settings_dynamic.py:814`): Obtém o valor da configuração `RISK_FACTOR_SOTA_HIGH_UQ`.
  - `DynamicSettings.RISK_FACTOR_LOCAL_HIGH_UQ(self)` (`app/app/settings_dynamic.py:818`): Obtém o valor da configuração `RISK_FACTOR_LOCAL_HIGH_UQ`.
  - `DynamicSettings.RISK_FACTOR_LOCAL_LOW_UQ(self)` (`app/app/settings_dynamic.py:822`): Obtém o valor da configuração `RISK_FACTOR_LOCAL_LOW_UQ`.
  - `DynamicSettings.RISK_FACTOR_ADAPT_ENABLED(self)` (`app/app/settings_dynamic.py:826`): Obtém o valor da configuração `RISK_FACTOR_ADAPT_ENABLED`.
  - `DynamicSettings.RISK_FACTOR_ADAPT_RATE(self)` (`app/app/settings_dynamic.py:830`): Obtém o valor da configuração `RISK_FACTOR_ADAPT_RATE`.
  - `DynamicSettings.ADAPTIVE_TIMEOUT_ENABLED(self)` (`app/app/settings_dynamic.py:834`): Obtém o valor da configuração `ADAPTIVE_TIMEOUT_ENABLED`.
  - `DynamicSettings.ADAPTIVE_TIMEOUT_MULTIPLIER(self)` (`app/app/settings_dynamic.py:838`): Obtém o valor da configuração `ADAPTIVE_TIMEOUT_MULTIPLIER`.
  - `DynamicSettings.ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER(self)` (`app/app/settings_dynamic.py:842`): Obtém o valor da configuração `ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER`.
  - `DynamicSettings.MIN_TIMEOUT(self)` (`app/app/settings_dynamic.py:846`): Obtém o valor da configuração `MIN_TIMEOUT`.
  - `DynamicSettings.MAX_TIMEOUT(self)` (`app/app/settings_dynamic.py:850`): Obtém o valor da configuração `MAX_TIMEOUT`.
  - `DynamicSettings.META_OPT_ENABLED(self)` (`app/app/settings_dynamic.py:854`): Obtém o valor da configuração `META_OPT_ENABLED`.
  - `DynamicSettings.META_OPT_SCHEDULE_HOUR(self)` (`app/app/settings_dynamic.py:858`): Obtém o valor da configuração `META_OPT_SCHEDULE_HOUR`.
  - `DynamicSettings.META_OPT_SCHEDULED_TRIALS(self)` (`app/app/settings_dynamic.py:862`): Obtém o valor da configuração `META_OPT_SCHEDULED_TRIALS`.
  - `DynamicSettings.DRIFT_THRESHOLD(self)` (`app/app/settings_dynamic.py:868`): Obtém o valor da configuração `DRIFT_THRESHOLD`.
  - `DynamicSettings.DRIFT_WINDOW_SIZE(self)` (`app/app/settings_dynamic.py:872`): Obtém o valor da configuração `DRIFT_WINDOW_SIZE`.
  - `DynamicSettings.USER_FEEDBACK_WEIGHT(self)` (`app/app/settings_dynamic.py:876`): Obtém o valor da configuração `USER_FEEDBACK_WEIGHT`.
  - `DynamicSettings.AB_TESTING_ENABLED(self)` (`app/app/settings_dynamic.py:882`): Obtém o valor da configuração `AB_TESTING_ENABLED`.
  - `DynamicSettings.CACHE_THRESHOLD_MIN(self)` (`app/app/settings_dynamic.py:888`): Obtém o valor da configuração `CACHE_THRESHOLD_MIN`.
  - `DynamicSettings.CACHE_THRESHOLD_MAX(self)` (`app/app/settings_dynamic.py:892`): Obtém o valor da configuração `CACHE_THRESHOLD_MAX`.
  - `DynamicSettings.CACHE_HIT_RATE_TARGET(self)` (`app/app/settings_dynamic.py:896`): Obtém o valor da configuração `CACHE_HIT_RATE_TARGET`.
  - `DynamicSettings.CACHE_THRESHOLD_ADAPT_ENABLED(self)` (`app/app/settings_dynamic.py:900`): Obtém o valor da configuração `CACHE_THRESHOLD_ADAPT_ENABLED`.
  - `DynamicSettings.PREDICTOR_VALIDATION_ENABLED(self)` (`app/app/settings_dynamic.py:906`): Obtém o valor da configuração `PREDICTOR_VALIDATION_ENABLED`.
  - `DynamicSettings.PREDICTOR_BRIER_SCORE_THRESHOLD(self)` (`app/app/settings_dynamic.py:910`): Obtém o valor da configuração `PREDICTOR_BRIER_SCORE_THRESHOLD`.
  - `DynamicSettings.PREDICTOR_CALIBRATION_WINDOW(self)` (`app/app/settings_dynamic.py:914`): Obtém o valor da configuração `PREDICTOR_CALIBRATION_WINDOW`.
  - `DynamicSettings.UQ_CALIBRATION_ENABLED(self)` (`app/app/settings_dynamic.py:920`): Obtém o valor da configuração `UQ_CALIBRATION_ENABLED`.
  - `DynamicSettings.UQ_QUALITY_GAP_RELAX(self)` (`app/app/settings_dynamic.py:924`): Obtém o valor da configuração `UQ_QUALITY_GAP_RELAX`.
  - `DynamicSettings.UQ_QUALITY_GAP_TIGHTEN(self)` (`app/app/settings_dynamic.py:928`): Obtém o valor da configuração `UQ_QUALITY_GAP_TIGHTEN`.
  - `DynamicSettings.JUDGE_CALIBRATION_ENABLED(self)` (`app/app/settings_dynamic.py:934`): Obtém o valor da configuração `JUDGE_CALIBRATION_ENABLED`.
  - `DynamicSettings.JUDGE_CACHE_AGREEMENT_TARGET(self)` (`app/app/settings_dynamic.py:938`): Obtém o valor da configuração `JUDGE_CACHE_AGREEMENT_TARGET`.
  - `DynamicSettings.CIRCUIT_BREAKER_FAIL_MAX(self)` (`app/app/settings_dynamic.py:944`): Obtém o valor da configuração `CIRCUIT_BREAKER_FAIL_MAX`.
  - `DynamicSettings.CIRCUIT_BREAKER_RESET_TIMEOUT(self)` (`app/app/settings_dynamic.py:948`): Obtém o valor da configuração `CIRCUIT_BREAKER_RESET_TIMEOUT`.
  - `DynamicSettings.CIRCUIT_BREAKER_LOCAL_FAIL_MAX(self)` (`app/app/settings_dynamic.py:952`): Obtém o valor da configuração `CIRCUIT_BREAKER_LOCAL_FAIL_MAX`.
  - `DynamicSettings.CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT(self)` (`app/app/settings_dynamic.py:956`): Obtém o valor da configuração `CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT`.
  - `DynamicSettings.MAX_CONCURRENT_REQUESTS(self)` (`app/app/settings_dynamic.py:962`): Obtém o valor da configuração `MAX_CONCURRENT_REQUESTS`.
  - `DynamicSettings.BACKPRESSURE_ENABLED(self)` (`app/app/settings_dynamic.py:966`): Obtém o valor da configuração `BACKPRESSURE_ENABLED`.
  - `DynamicSettings.EMERGENCY_FALLBACK_MODELS(self)` (`app/app/settings_dynamic.py:972`): Executa a responsabilidade descrita por este método.

## `app/app/sparse_index.py`

Resumo do arquivo: sparse_index.py — Gerenciador de Índice BM25 (Busca por Palavras-Chave)

### Classes e métodos

- Classe `SparseIndex` (`app/app/sparse_index.py:25`): Classe `SparseIndex`: organiza responsabilidades de sparse index.
  - `SparseIndex.__init__(self)` (`app/app/sparse_index.py:27`): Inicializa estado interno necessário para uso da classe.
  - `SparseIndex._tokenize(self, text)` (`app/app/sparse_index.py:35`): Tokenização simples para o BM25 (lowercase + split).
  - `SparseIndex.add_document(self, doc_id, text)` (`app/app/sparse_index.py:42`): Adiciona um documento ao índice em memória.
  - `SparseIndex.commit(self)` (`app/app/sparse_index.py:57`): Reconstrói o índice BM25 e salva no disco.
  - `SparseIndex.search(self, query, top_k)` (`app/app/sparse_index.py:75`): Retorna [(doc_id, score), ...]
  - `SparseIndex.get_text(self, doc_id)` (`app/app/sparse_index.py:100`): Recupera o texto original pelo ID.
  - `SparseIndex._save(self)` (`app/app/sparse_index.py:108`): Executa save.
  - `SparseIndex._load(self)` (`app/app/sparse_index.py:121`): Executa load.

## `app/app/tasks.py`

Resumo do arquivo: Celery tasks for background processing.

### Funções de módulo

- `_get_or_create_event_loop()` (`app/app/tasks.py:38`): Get or create a persistent event loop for this worker thread.
- `run_async(coro)` (`app/app/tasks.py:56`): Run an async coroutine in the worker's persistent event loop.
- `on_worker_process_init(**kwargs)` (`app/app/tasks.py:71`): Initialize event loop when worker process starts.
- `on_worker_process_shutdown(**kwargs)` (`app/app/tasks.py:78`): Clean up event loop when worker process shuts down.
- `task_process_feedback(self, query, answer, chosen_model, modality, latency_s, cost_val, image_b64, raw_payload, prompt_tokens, completion_tokens)` (`app/app/tasks.py:95`): Executa o feedback loop (Juízes, Bandit Update, Logging) em background via Celery.
- `task_execute_eval_run(self, run_id, modality, use_cache, max_tokens, temperature)` (`app/app/tasks.py:141`): Execute an eval run asynchronously and persist per-prompt metrics.

## `app/app/umrag.py`

Resumo do arquivo: umrag.py — Unified Multimodal RAG

### Funções de módulo

- `_unit(x)` (`app/app/umrag.py:63`): Executa unit.
- `_embed_for_rag(query, modality, image_b64)` (`app/app/umrag.py:70`): Executa embed for rag.
- `_extract_docs_from(res)` (`app/app/umrag.py:107`): Executa extract docs from.
- `build_augmented_prompt(query, modality, image_b64, top_k)` (`app/app/umrag.py:123`): Estrategia unificada (C):
- `add_document(doc_id, text, metadata, modality, image_b64)` (`app/app/umrag.py:187`): Compatível com chamadas antigas.
- `health()` (`app/app/umrag.py:217`): Executa health.

## `app/app/update_nsga_best_params.py`

Resumo do arquivo: update_nsga_best_params.py  (VERSÃO MULTIMODAL)

### Funções de módulo

- `init_tables()` (`app/app/update_nsga_best_params.py:108`): Executa init tables.
- `load_best_trial(modality)` (`app/app/update_nsga_best_params.py:127`): Executa load best trial.
- `update_best_params(modality, row)` (`app/app/update_nsga_best_params.py:163`): Executa update best params.
- `compute_model_weights(modality)` (`app/app/update_nsga_best_params.py:208`): Executa compute model weights.
- `persist_weights(modality, weights)` (`app/app/update_nsga_best_params.py:247`): Executa persist weights.

## `app/app/user_feedback.py`

Resumo do arquivo: user_feedback.py — User Feedback Processing

### Funções de módulo

- `rating_to_quality(rating)` (`app/app/user_feedback.py:73`): Convert 1-5 star rating to 0-10 quality score.
- `get_quality_from_feedback(request)` (`app/app/user_feedback.py:87`): Extract quality score from feedback request.
- `blend_quality(user_quality, original_quality)` (`app/app/user_feedback.py:113`): Blend user feedback quality with original quality.
- `process_feedback(request)` (`app/app/user_feedback.py:126`): Process user feedback and update bandit/metrics.
- `_persist_feedback(request, user_quality, blended_quality, reward)` (`app/app/user_feedback.py:185`): Persist feedback to database.
- `get_feedback_stats(model, hours)` (`app/app/user_feedback.py:254`): Get feedback statistics for a model or all models.

### Classes e métodos

- Classe `FeedbackType` (`app/app/user_feedback.py:32`): Supported feedback types.
- Classe `UserFeedbackRequest` (`app/app/user_feedback.py:47`): Request model for user feedback submission.
- Classe `ProcessedFeedback` (`app/app/user_feedback.py:63`): Processed feedback result.

## `app/app/utils/bkp.redis_client.py`

Resumo do arquivo: Módulo principal: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `get_redis(max_wait_s)` (`app/app/utils/bkp.redis_client.py:19`): Retorna cliente Redis síncrono.

## `app/app/utils/pricing.py`

Resumo do arquivo: pricing.py - Model Cost Calculator with Redis Caching (Quick Win #4)

### Funções de módulo

- `_get_rds()` (`app/app/utils/pricing.py:27`): Executa get rds.
- `_refresh_pricing_from_db()` (`app/app/utils/pricing.py:40`): Load pricing data from database.
- `_refresh_pricing()` (`app/app/utils/pricing.py:54`): Refresh pricing cache from Redis or DB.
- `invalidate_pricing_cache()` (`app/app/utils/pricing.py:90`): Invalidate pricing cache (call on settings hot-reload).
- `get_model_cost(model, input_tokens, output_tokens)` (`app/app/utils/pricing.py:106`): Calculate total cost in USD.

## `app/app/utils/redis_client.py`

Resumo do arquivo: redis_client.py — Redis Client with Connection Pooling

### Funções de módulo

- `_create_pool()` (`app/app/utils/redis_client.py:45`): Create a Redis connection pool.
- `_ensure_pool_initialized()` (`app/app/utils/redis_client.py:71`): Initialize Redis pool once and return it.
- `_connect_once()` (`app/app/utils/redis_client.py:80`): Attempt a single Redis connection without sleeping.
- `ensure_redis_connected(max_wait_s, min_retry_interval_s)` (`app/app/utils/redis_client.py:95`): Ensure Redis connection is available.
- `get_redis(max_wait_s)` (`app/app/utils/redis_client.py:144`): Get a Redis client from the connection pool.
- `get_redis_async_safe()` (`app/app/utils/redis_client.py:162`): Get Redis client without blocking (for use in async contexts).
- `check_redis_health()` (`app/app/utils/redis_client.py:175`): Check Redis health and return detailed status.
- `redis_pipeline()` (`app/app/utils/redis_client.py:221`): Context manager for Redis pipeline operations.
- `close_redis()` (`app/app/utils/redis_client.py:244`): Close Redis connections and pool. Call on application shutdown.

## `app/app/utils/text_splitter.py`

Resumo do arquivo: Módulo principal: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `split_into_sentences(text)` (`app/app/utils/text_splitter.py:14`): Executa split into sentences.
- `_cosine_similarity(v1, v2)` (`app/app/utils/text_splitter.py:19`): Executa cosine similarity.
- `semantic_chunking(text, similarity_threshold_percentile, max_chunk_size, min_chunk_size)` (`app/app/utils/text_splitter.py:26`): Divide o texto dinamicamente baseado na mudança de tópico.

## `app/app/utils/token_utils.py`

Resumo do arquivo: token_utils.py — Token Counting Utilities

### Funções de módulo

- `_get_encoder(model_name)` (`app/app/utils/token_utils.py:26`): Get tiktoken encoder for a model with LRU cache.
- `count_tokens(text, model_name)` (`app/app/utils/token_utils.py:53`): Count tokens with precision using tiktoken for OpenAI models,
- `get_encoder_cache_info()` (`app/app/utils/token_utils.py:86`): Get information about the encoder cache.
- `clear_encoder_cache()` (`app/app/utils/token_utils.py:97`): Clear the encoder cache.

## `app/app/utils/uncertainty.py`

Resumo do arquivo: app/app/utils/uncertainty.py

### Funções de módulo

- `_cosine_similarity(v1, v2)` (`app/app/utils/uncertainty.py:24`): Calcula similaridade de cosseno entre dois vetores numpy.
- `get_uncertainty_score(query_text, modality)` (`app/app/utils/uncertainty.py:32`): Calcula o score de incerteza para uma query.

## `app/app/vectorstore.py`

Resumo do arquivo: vectorstore.py — RAG Multimodal com Versionamento e Auto-Healing

### Funções de módulo

- `_sanitize_model_name(model_name)` (`app/app/vectorstore.py:53`): Transforma 'nomic-ai/nomic-embed-text-v1.5' em 'nomic_ai_nomic_embed_text_v1_5'.
- `_get_versioned_collection_name(base_name, modality)` (`app/app/vectorstore.py:66`): Gera o nome da coleção atrelado ao modelo configurado no settings.
- `_connect_local()` (`app/app/vectorstore.py:85`): Inicializa cliente persistente do ChromaDB.
- `get_chroma_client()` (`app/app/vectorstore.py:102`): Lazy initializer for Chroma client.
- `_ensure_list_of_floats(vec)` (`app/app/vectorstore.py:119`): Converte embedding para list[float].
- `_safe_metadata(meta)` (`app/app/vectorstore.py:128`): Executa safe metadata.
- `_normalize_modality(modality)` (`app/app/vectorstore.py:133`): Executa normalize modality.
- `_collection_for_modality(modality)` (`app/app/vectorstore.py:143`): Retorna o nome versionado da coleção baseado na modalidade.
- `_get_or_create_sync(name, metadata)` (`app/app/vectorstore.py:159`): Wrapper síncrono para o client do Chroma.
- `get_or_create_collection_async(name, metadata)` (`app/app/vectorstore.py:164`): Cria ou recupera uma coleção de forma assíncrona (threadpool).
- `init_vectorstore()` (`app/app/vectorstore.py:188`): Cria coleções versionadas no boot.
- `_insert_embedding_sync(collection_name, doc_id, text, embedding, metadata)` (`app/app/vectorstore.py:211`): Executa insert embedding sync.
- `add_document(modality, doc_id, text, image_b64, metadata)` (`app/app/vectorstore.py:253`): Insere documento multimodal completo na coleção versionada correta E no índice BM25.
- `_query_embedding_sync(collection_name, embedding, n_results)` (`app/app/vectorstore.py:298`): Executa query embedding sync.
- `query_embedding(modality, embedding, n_results)` (`app/app/vectorstore.py:323`): Consulta embeddings no Chroma.
- `reset_collections()` (`app/app/vectorstore.py:343`): Apaga todas as coleções.
- `health_async()` (`app/app/vectorstore.py:352`): Executa health async.
- `reset_vectorstore_runtime_state()` (`app/app/vectorstore.py:361`): Reset lazy client state (primarily for tests/dev).

## `app/__init__.py`

Resumo do arquivo: Módulo `app/__init__.py`: descreve responsabilidades e integrações deste arquivo.

## `app/advanced_analytics.py`

Resumo do arquivo: advanced_analytics.py — Advanced Scientific Validation for Thesis

### Funções de módulo

- `load_latest_data()` (`app/advanced_analytics.py:33`): Carrega latest data.
- `analyze_regret(df)` (`app/advanced_analytics.py:44`): Executa analyze regret.
- `analyze_shap(df)` (`app/advanced_analytics.py:110`): Executa analyze shap.
- `analyze_decontamination(df)` (`app/advanced_analytics.py:166`): Executa analyze decontamination.

## `app/audit_anomalies.py`

Resumo do arquivo: audit_anomalies.py — Auditoria de Discrepâncias (Sincronizado)

### Funções de módulo

- `load_latest_data()` (`app/audit_anomalies.py:18`): Carrega latest data.
- `debug_parser(dataset_name, model_output)` (`app/audit_anomalies.py:26`): Lógica idêntica ao check_correctness do benchmark_thesis.py
- `main()` (`app/audit_anomalies.py:48`): Executa main.

## `app/benchmark_thesis.py`

Resumo do arquivo: benchmark_thesis.py — Phase 1: Generation & Performance (NO JUDGE)

### Funções de módulo

- `temporary_setting(key, value)` (`app/benchmark_thesis.py:96`): Aplica uma configuração temporária durante execução de contexto.
- `force_switch_ollama_model(target_model_name)` (`app/benchmark_thesis.py:109`): Força troca de modelo ativo no Ollama e realiza warmup.
- `check_correctness(dataset_name, model_output, reference)` (`app/benchmark_thesis.py:141`): Compara saída do modelo com referência usando regra por dataset.
- `format_mmlu(example)` (`app/benchmark_thesis.py:164`): Formata item do MMLU para prompt de múltipla escolha.
- `format_gsm8k(example)` (`app/benchmark_thesis.py:169`): Formata item do GSM8K para raciocínio passo a passo.
- `format_hellaswag(example)` (`app/benchmark_thesis.py:172`): Formata item do HellaSwag para seleção de continuação.
- `format_humaneval(example)` (`app/benchmark_thesis.py:176`): Formata item do HumanEval para tarefa de completude de código.
- `format_truthfulqa(example)` (`app/benchmark_thesis.py:179`): Formata item do TruthfulQA para resposta factual concisa.
- `format_arc(example)` (`app/benchmark_thesis.py:182`): Formata item do ARC-Challenge para múltipla escolha.
- `format_bbh(example)` (`app/benchmark_thesis.py:188`): Formata item do BBH para cadeia de raciocínio.
- `load_datasets()` (`app/benchmark_thesis.py:192`): Carrega e amostra datasets usados no benchmark de tese.
- `run_frugal_cascade(query)` (`app/benchmark_thesis.py:221`): Executa cascata local->SOTA com heurística frugal de escalonamento.
- `estimate_fallback_cost(query, answer)` (`app/benchmark_thesis.py:255`): Estima custo de fallback SOTA com aproximação por tokens.
- `evaluate_interaction(mode_label, task, run_id)` (`app/benchmark_thesis.py:261`): Avalia uma interação de benchmark em um modo específico.
- `run_benchmark_suite()` (`app/benchmark_thesis.py:367`): Executa benchmark completo e persiste progresso em checkpoint.

### Classes e métodos

- Classe `RateLimiter` (`app/benchmark_thesis.py:68`): Controla taxa de chamadas assíncronas por janela de tempo.
  - `RateLimiter.__init__(self, max_calls_per_minute)` (`app/benchmark_thesis.py:70`): Inicializa limite de chamadas por minuto.
  - `RateLimiter.wait(self)` (`app/benchmark_thesis.py:77`): Aguarda até que nova chamada possa ser executada com segurança.

## `app/calculate_savings.py`

Resumo do arquivo: calculate_savings.py

### Funções de módulo

- `run_analysis()` (`app/calculate_savings.py:29`): Executa analysis.

## `app/evaluate_results.py`

Resumo do arquivo: evaluate_results.py — Phase 2: Batch Evaluation (Parallelized)

### Funções de módulo

- `get_judge_score(query, answer, reference)` (`app/evaluate_results.py:30`): Chama o Ollama para julgar. Suporta extração de <think> e Score.
- `process_batch()` (`app/evaluate_results.py:94`): Executa process batch.

## `app/experiment_oml_standard.py`

Resumo do arquivo: experiment_oml_standard.py — OML Comparison with Statistical Validation

### Funções de módulo

- `load_data()` (`app/experiment_oml_standard.py:37`): Carrega data.
- `get_model(name)` (`app/experiment_oml_standard.py:71`): Obtém model.
- `run_kfold_for_model(model_name, X, y, k_folds)` (`app/experiment_oml_standard.py:92`): Executa kfold for model.
- `analyze_statistics(final_results)` (`app/experiment_oml_standard.py:131`): Realiza testes de Friedman e Wilcoxon para determinar o melhor algoritmo.
- `main()` (`app/experiment_oml_standard.py:201`): Executa main.

## `app/populate_vectorstore.py`

Resumo do arquivo: populate_vectorstore.py (Versão Final: OCR + Deduplicação + Metadados)

### Funções de módulo

- `generate_deterministic_id(filename, chunk_index, content)` (`app/populate_vectorstore.py:64`): Gera um Hash SHA-256 único para o fragmento.
- `chunk_text(text, chunk_size, overlap)` (`app/populate_vectorstore.py:77`): Divide texto longo em fragmentos com sobreposição.
- `_perform_ocr(path)` (`app/populate_vectorstore.py:97`): Converte páginas do PDF em imagens e roda Tesseract.
- `load_text_file(path)` (`app/populate_vectorstore.py:123`): Carrega text file.
- `load_pdf_file(path)` (`app/populate_vectorstore.py:134`): Lê PDF. Tenta extração direta (rápida).
- `gather_documents(folder)` (`app/populate_vectorstore.py:164`): Varre a pasta e retorna {nome_arquivo: [chunks]}.
- `summarize_text_async(text, model)` (`app/populate_vectorstore.py:200`): Gera título e resumo usando o LLM via providers_async.
- `populate_vectorstore()` (`app/populate_vectorstore.py:243`): Executa populate vectorstore.

## `app/prestart_vectorstore.py`

Resumo do arquivo: prestart_vectorstore.py

### Funções de módulo

- `detect_embedding_dim(model_name)` (`app/prestart_vectorstore.py:35`): Detecta automaticamente a dimensão do modelo de embedding.

## `app/sensitivity_runner.py`

Resumo do arquivo: sensitivity_runner.py — Sensitivity Analysis for Thesis (ROBUST FIXED)

### Funções de módulo

- `run_benchmark_iteration(threshold)` (`app/sensitivity_runner.py:32`): Executa benchmark iteration.
- `plot_sensitivity(results)` (`app/sensitivity_runner.py:104`): Executa plot sensitivity.

## `app/statistical_validation.py`

Resumo do arquivo: statistical_validation.py — Validação Estatística (Sincronizado com Benchmark Final)

### Funções de módulo

- `load_latest_data()` (`app/statistical_validation.py:40`): Carrega latest data.
- `calculate_cohens_d(x, y)` (`app/statistical_validation.py:68`): Executa calculate cohens d.
- `format_p_value(p)` (`app/statistical_validation.py:77`): Executa format p value.
- `analyze_metric(df, metric_col, metric_name_en)` (`app/statistical_validation.py:82`): Executa analyze metric.
- `generate_latex_text(results)` (`app/statistical_validation.py:139`): Executa generate latex text.
- `main()` (`app/statistical_validation.py:207`): Executa main.

## `alembic/env.py`

Resumo do arquivo: Módulo `alembic/env.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `run_migrations_offline()` (`alembic/env.py:35`): Executa migrations offline.
- `run_migrations_online()` (`alembic/env.py:49`): Executa migrations online.

## `alembic/versions/0002_add_performance_indices.py`

Resumo do arquivo: Add performance indices for frequently queried tables

### Funções de módulo

- `upgrade()` (`alembic/versions/0002_add_performance_indices.py:25`): Add performance indices.
- `downgrade()` (`alembic/versions/0002_add_performance_indices.py:68`): Remove performance indices.

## `tests/conftest.py`

Resumo do arquivo: Módulo `tests/conftest.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `mock_dependencies(monkeypatch)` (`tests/conftest.py:22`): Mocka automaticamente conexões externas para TODOS os testes.

## `tests/locustfile.py`

Resumo do arquivo: Módulo `tests/locustfile.py`: descreve responsabilidades e integrações deste arquivo.

### Classes e métodos

- Classe `RouterUser` (`tests/locustfile.py:2592`): Standard user sending random queries.
  - `RouterUser.send_query(self)` (`tests/locustfile.py:2598`): Standard query task.
  - `RouterUser.health_check(self)` (`tests/locustfile.py:2610`): Health check task - lower weight.
  - `RouterUser.metrics(self)` (`tests/locustfile.py:2615`): Metrics endpoint task - lowest weight.
- Classe `RAGUser` (`tests/locustfile.py:2620`): User that primarily uses RAG-enabled queries.
  - `RAGUser.send_rag_query(self)` (`tests/locustfile.py:2627`): RAG-enabled query task.
- Classe `HighTemperatureUser` (`tests/locustfile.py:2640`): Creative user with high temperature settings.
  - `HighTemperatureUser.send_creative_query(self)` (`tests/locustfile.py:2647`): High temperature query for creative tasks.
- Classe `BurstUser` (`tests/locustfile.py:2664`): Simulates burst traffic patterns.
  - `BurstUser.burst_queries(self)` (`tests/locustfile.py:2671`): Rapid burst of queries.
- Classe `LongRunningUser` (`tests/locustfile.py:2683`): User that sends complex, long-running queries.
  - `LongRunningUser.send_complex_query(self)` (`tests/locustfile.py:2690`): Complex query requiring more processing.
- Classe `MixedWorkloadUser` (`tests/locustfile.py:2708`): Simulates realistic mixed workload with varying request types.
  - `MixedWorkloadUser.simple_query(self)` (`tests/locustfile.py:2715`): Simple, short queries - most common.
  - `MixedWorkloadUser.medium_query(self)` (`tests/locustfile.py:2732`): Medium complexity queries.
  - `MixedWorkloadUser.complex_query(self)` (`tests/locustfile.py:2743`): Complex queries - least common.
- Classe `APIVersionUser` (`tests/locustfile.py:2754`): Tests versioned API endpoints.
  - `APIVersionUser.v1_query(self)` (`tests/locustfile.py:2761`): Test v1 API endpoint.
  - `APIVersionUser.v1_health(self)` (`tests/locustfile.py:2772`): Test v1 health endpoint.

## `tests/test_ab_testing.py`

Resumo do arquivo: Tests for A/B testing infrastructure.

### Classes e métodos

- Classe `TestVariant` (`tests/test_ab_testing.py:10`): Test suite for Variant dataclass.
  - `TestVariant.test_variant_creation(self)` (`tests/test_ab_testing.py:13`): Test creating a variant.
- Classe `TestExperiment` (`tests/test_ab_testing.py:24`): Test suite for Experiment dataclass.
  - `TestExperiment.test_experiment_to_dict(self)` (`tests/test_ab_testing.py:27`): Test experiment serialization.
  - `TestExperiment.test_experiment_from_dict(self)` (`tests/test_ab_testing.py:49`): Test experiment deserialization.
- Classe `TestABTestManager` (`tests/test_ab_testing.py:71`): Test suite for ABTestManager.
  - `TestABTestManager.ab_manager(self)` (`tests/test_ab_testing.py:75`): Create a fresh A/B test manager instance.
  - `TestABTestManager.test_hash_to_bucket_deterministic(self, ab_manager)` (`tests/test_ab_testing.py:87`): Test that hashing is deterministic.
  - `TestABTestManager.test_hash_to_bucket_distribution(self, ab_manager)` (`tests/test_ab_testing.py:94`): Test that hashing distributes across buckets.
  - `TestABTestManager.test_create_experiment(self, ab_manager)` (`tests/test_ab_testing.py:102`): Test creating an experiment.
  - `TestABTestManager.test_start_experiment(self, ab_manager)` (`tests/test_ab_testing.py:124`): Test starting an experiment.
  - `TestABTestManager.test_pause_experiment(self, ab_manager)` (`tests/test_ab_testing.py:142`): Test pausing an experiment.
  - `TestABTestManager.test_complete_experiment(self, ab_manager)` (`tests/test_ab_testing.py:160`): Test completing an experiment.
  - `TestABTestManager.test_get_assignment_not_running(self, ab_manager)` (`tests/test_ab_testing.py:179`): Test that assignment returns None for non-running experiment.
  - `TestABTestManager.test_get_assignment_consistent(self, ab_manager)` (`tests/test_ab_testing.py:198`): Test that assignment is consistent for same user.
  - `TestABTestManager.test_list_experiments_filter_by_status(self, ab_manager)` (`tests/test_ab_testing.py:219`): Test listing experiments filtered by status.
  - `TestABTestManager.test_delete_experiment(self, ab_manager)` (`tests/test_ab_testing.py:242`): Test deleting an experiment.
  - `TestABTestManager.test_delete_nonexistent_experiment(self, ab_manager)` (`tests/test_ab_testing.py:259`): Test deleting non-existent experiment returns False.
- Classe `TestGetABTestManager` (`tests/test_ab_testing.py:265`): Test the get_ab_test_manager factory function.
  - `TestGetABTestManager.test_returns_singleton(self)` (`tests/test_ab_testing.py:268`): Test that get_ab_test_manager returns singleton instance.

## `tests/test_autonomous.py`

Resumo do arquivo: test_autonomous.py — Tests for Phase 5 Autonomous Behavior Improvements

### Classes e métodos

- Classe `TestAdaptiveRiskFactors` (`tests/test_autonomous.py:22`): Tests for adaptive risk factor tuning.
  - `TestAdaptiveRiskFactors.test_risk_factor_settings_exist(self)` (`tests/test_autonomous.py:25`): Verify risk factor settings are defined with defaults.
  - `TestAdaptiveRiskFactors.test_risk_factor_bounds(self)` (`tests/test_autonomous.py:33`): Risk factors should be within reasonable bounds.
  - `TestAdaptiveRiskFactors.test_risk_factors_used_in_strategy(self)` (`tests/test_autonomous.py:46`): Verify router_strategy uses dynamic risk factors.
- Classe `TestPredictorValidation` (`tests/test_autonomous.py:71`): Tests for online predictor validation and calibration.
  - `TestPredictorValidation.test_predictor_initialization(self)` (`tests/test_autonomous.py:74`): Test predictor initializes with validation tracking.
  - `TestPredictorValidation.test_record_outcome(self)` (`tests/test_autonomous.py:85`): Test recording prediction outcomes.
  - `TestPredictorValidation.test_brier_score_calculation(self)` (`tests/test_autonomous.py:99`): Test Brier score computation.
  - `TestPredictorValidation.test_brier_score_random(self)` (`tests/test_autonomous.py:116`): Test Brier score for random predictions.
  - `TestPredictorValidation.test_calibration_metrics(self)` (`tests/test_autonomous.py:130`): Test getting calibration metrics.
- Classe `TestAdaptiveCacheThreshold` (`tests/test_autonomous.py:152`): Tests for adaptive semantic cache threshold.
  - `TestAdaptiveCacheThreshold.test_cache_threshold_settings(self)` (`tests/test_autonomous.py:155`): Verify cache threshold settings exist.
  - `TestAdaptiveCacheThreshold.test_get_cache_threshold_dynamic(self)` (`tests/test_autonomous.py:164`): Test dynamic threshold retrieval.
  - `TestAdaptiveCacheThreshold.test_get_cache_hit_rate(self)` (`tests/test_autonomous.py:171`): Test hit rate calculation.
  - `TestAdaptiveCacheThreshold.test_cache_stats(self)` (`tests/test_autonomous.py:183`): Test cache statistics retrieval.
- Classe `TestUQCalibration` (`tests/test_autonomous.py:199`): Tests for uncertainty quantification calibration.
  - `TestUQCalibration.test_uq_settings_exist(self)` (`tests/test_autonomous.py:202`): Verify UQ calibration settings exist.
  - `TestUQCalibration.test_uncertainty_threshold_bounds(self)` (`tests/test_autonomous.py:210`): Uncertainty threshold should be within bounds.
- Classe `TestJudgeCalibration` (`tests/test_autonomous.py:222`): Tests for judge calibration system.
  - `TestJudgeCalibration.test_judge_calibration_settings(self)` (`tests/test_autonomous.py:225`): Verify judge calibration settings exist.
  - `TestJudgeCalibration.test_calibrate_judges_disabled(self)` (`tests/test_autonomous.py:233`): Test calibrate_judges when disabled.
- Classe `TestAutonomousMetrics` (`tests/test_autonomous.py:255`): Tests for autonomous behavior Prometheus metrics.
  - `TestAutonomousMetrics.test_predictor_metrics_exist(self)` (`tests/test_autonomous.py:258`): Verify predictor metrics are defined.
  - `TestAutonomousMetrics.test_cache_metrics_exist(self)` (`tests/test_autonomous.py:270`): Verify cache threshold metrics are defined.
  - `TestAutonomousMetrics.test_uq_metrics_exist(self)` (`tests/test_autonomous.py:282`): Verify UQ calibration metrics are defined.
  - `TestAutonomousMetrics.test_judge_calibration_metrics_exist(self)` (`tests/test_autonomous.py:294`): Verify judge calibration metrics are defined.
  - `TestAutonomousMetrics.test_risk_factor_metrics_exist(self)` (`tests/test_autonomous.py:306`): Verify risk factor metrics are defined.
- Classe `TestAutonomousIntegration` (`tests/test_autonomous.py:317`): Integration tests for autonomous behavior.
  - `TestAutonomousIntegration.test_uncertainty_score_in_metadata(self)` (`tests/test_autonomous.py:320`): Verify uncertainty_score is included in route response metadata.
  - `TestAutonomousIntegration.test_settings_snapshot_includes_autonomous(self)` (`tests/test_autonomous.py:329`): Verify settings snapshot includes autonomous behavior settings.

## `tests/test_bandits.py`

Resumo do arquivo: Módulo `tests/test_bandits.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_compute_reward_logic()` (`tests/test_bandits.py:6`): Testa se a função de recompensa respeita os limites [0, 1].
- `test_dynamic_epsilon()` (`tests/test_bandits.py:18`): Testa se a taxa de exploração aumenta quando há poucos dados.

## `tests/test_bandits_more.py`

Resumo do arquivo: Módulo `tests/test_bandits_more.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_numpy_helpers()` (`tests/test_bandits_more.py:33`): Testa numpy helpers.
- `test_centroid_matrix_cache_basic()` (`tests/test_bandits_more.py:49`): Testa centroid matrix cache basic.
- `test_new_centroid_id_and_nearest(monkeypatch)` (`tests/test_bandits_more.py:69`): Testa new centroid id and nearest.
- `test_dynamic_epsilon_and_choosers(monkeypatch)` (`tests/test_bandits_more.py:92`): Testa dynamic epsilon and choosers.
- `test_meta_strategy_and_combine(monkeypatch)` (`tests/test_bandits_more.py:119`): Testa meta strategy and combine.
- `test_select_model_and_snapshot(monkeypatch)` (`tests/test_bandits_more.py:144`): Testa select model and snapshot.
- `test_bandit_update_and_reward(monkeypatch)` (`tests/test_bandits_more.py:170`): Testa bandit update and reward.
- `test_centroids_online_update_and_label(monkeypatch)` (`tests/test_bandits_more.py:215`): Testa centroids online update and label.

### Classes e métodos

- Classe `_DummyMetric` (`tests/test_bandits_more.py:7`): Classe `_DummyMetric`: concentra responsabilidades de test bandits more.
  - `_DummyMetric.labels(self, **_kwargs)` (`tests/test_bandits_more.py:9`): Executa labels.
  - `_DummyMetric.inc(self, *_args, **_kwargs)` (`tests/test_bandits_more.py:13`): Executa inc.
  - `_DummyMetric.observe(self, *_args, **_kwargs)` (`tests/test_bandits_more.py:17`): Executa observe.
- Classe `_FakeRedis` (`tests/test_bandits_more.py:22`): Classe `_FakeRedis`: concentra responsabilidades de test bandits more.
  - `_FakeRedis.__init__(self, value)` (`tests/test_bandits_more.py:24`): Inicializa estado interno necessário para uso da classe.
  - `_FakeRedis.get(self, _key)` (`tests/test_bandits_more.py:28`): Executa get.

## `tests/test_cascade_detector.py`

Resumo do arquivo: Tests for cascade failure detection.

### Classes e métodos

- Classe `TestCascadeDetector` (`tests/test_cascade_detector.py:10`): Test suite for CascadeDetector.
  - `TestCascadeDetector.cascade_detector(self)` (`tests/test_cascade_detector.py:14`): Create a fresh cascade detector instance.
  - `TestCascadeDetector.test_severity_normal(self, cascade_detector)` (`tests/test_cascade_detector.py:22`): Test normal severity when no models are failing.
  - `TestCascadeDetector.test_severity_warning(self, cascade_detector)` (`tests/test_cascade_detector.py:30`): Test warning severity when 30% of models are failing.
  - `TestCascadeDetector.test_severity_critical(self, cascade_detector)` (`tests/test_cascade_detector.py:38`): Test critical severity when 50% of models are failing.
  - `TestCascadeDetector.test_severity_emergency(self, cascade_detector)` (`tests/test_cascade_detector.py:46`): Test emergency severity when 80% of models are failing.
  - `TestCascadeDetector.test_get_emergency_fallback_not_emergency(self, cascade_detector)` (`tests/test_cascade_detector.py:54`): Test that fallback returns None when not in emergency mode.
  - `TestCascadeDetector.test_get_status_structure(self, cascade_detector)` (`tests/test_cascade_detector.py:63`): Test that get_status returns expected structure.
  - `TestCascadeDetector.test_check_and_log_warnings_emergency(self, cascade_detector)` (`tests/test_cascade_detector.py:77`): Test that emergency warnings are logged.
  - `TestCascadeDetector.test_thresholds_values(self, cascade_detector)` (`tests/test_cascade_detector.py:89`): Test that thresholds have expected values.
- Classe `TestGetCascadeDetector` (`tests/test_cascade_detector.py:96`): Test the get_cascade_detector factory function.
  - `TestGetCascadeDetector.test_returns_singleton(self)` (`tests/test_cascade_detector.py:99`): Test that get_cascade_detector returns singleton instance.

## `tests/test_chaos.py`

Resumo do arquivo: test_chaos.py — Chaos Testing for Failure Scenarios

### Classes e métodos

- Classe `TestCircuitBreakerChaos` (`tests/test_chaos.py:20`): Tests circuit breaker behavior under failure conditions.
  - `TestCircuitBreakerChaos.test_circuit_breaker_opens_after_failures(self)` (`tests/test_chaos.py:23`): Circuit breaker should open after threshold failures.
  - `TestCircuitBreakerChaos.test_circuit_breaker_half_open_after_timeout(self)` (`tests/test_chaos.py:45`): Circuit breaker should transition to half-open after timeout.
  - `TestCircuitBreakerChaos.test_reset_breaker(self)` (`tests/test_chaos.py:80`): Admin should be able to manually reset a circuit breaker.
- Classe `TestRateLimitChaos` (`tests/test_chaos.py:105`): Tests rate limiting under heavy load.
  - `TestRateLimitChaos.test_rate_limit_exceeded(self)` (`tests/test_chaos.py:109`): Rate limiter should block requests over threshold.
  - `TestRateLimitChaos.test_rate_limit_window_expiry(self)` (`tests/test_chaos.py:128`): Rate limit should reset after window expires.
  - `TestRateLimitChaos.test_rate_limit_cleanup(self)` (`tests/test_chaos.py:151`): Cleanup should remove old entries to prevent memory bloat.
- Classe `TestTimeoutChaos` (`tests/test_chaos.py:175`): Tests timeout handling.
  - `TestTimeoutChaos.test_request_timeout_handling(self)` (`tests/test_chaos.py:179`): System should handle timeouts gracefully.
  - `TestTimeoutChaos.test_slow_provider_timeout(self)` (`tests/test_chaos.py:191`): Slow providers should timeout appropriately.
- Classe `TestCacheFailureChaos` (`tests/test_chaos.py:202`): Tests cache failure scenarios.
  - `TestCacheFailureChaos.test_l1_cache_graceful_degradation(self)` (`tests/test_chaos.py:205`): System should work even if L1 cache fails.
  - `TestCacheFailureChaos.test_l1_cache_full_eviction(self)` (`tests/test_chaos.py:220`): Cache should evict oldest entries when full.
  - `TestCacheFailureChaos.test_redis_unavailable_fallback(self)` (`tests/test_chaos.py:239`): System should fallback gracefully when Redis is unavailable.
- Classe `TestProviderFailureChaos` (`tests/test_chaos.py:249`): Tests provider failure scenarios.
  - `TestProviderFailureChaos.test_provider_rate_limit_error(self)` (`tests/test_chaos.py:253`): System should handle provider rate limit errors.
  - `TestProviderFailureChaos.test_provider_auth_error(self)` (`tests/test_chaos.py:266`): System should handle provider auth errors.
  - `TestProviderFailureChaos.test_fallback_chain_execution(self)` (`tests/test_chaos.py:279`): Fallback chain should try alternative models.
- Classe `TestConcurrencyChaos` (`tests/test_chaos.py:296`): Tests concurrent request handling.
  - `TestConcurrencyChaos.test_request_deduplication_under_load(self)` (`tests/test_chaos.py:300`): Request deduplicator should handle concurrent identical requests.
  - `TestConcurrencyChaos.test_parallel_requests_different_queries(self)` (`tests/test_chaos.py:330`): Different queries should be processed independently.
- Classe `TestHealthCheckChaos` (`tests/test_chaos.py:369`): Tests health check under degraded conditions.
  - `TestHealthCheckChaos.test_partial_health_check(self)` (`tests/test_chaos.py:373`): System should report degraded status when some components fail.
  - `TestHealthCheckChaos.test_complete_failure_health(self)` (`tests/test_chaos.py:399`): System should report unhealthy when all components fail.
- Classe `TestErrorRecoveryChaos` (`tests/test_chaos.py:423`): Tests error recovery scenarios.
  - `TestErrorRecoveryChaos.test_error_categorization(self)` (`tests/test_chaos.py:426`): Errors should be categorized correctly.
  - `TestErrorRecoveryChaos.test_error_response_creation(self)` (`tests/test_chaos.py:440`): Error responses should be user-friendly.
- Classe `TestMemoryPressureChaos` (`tests/test_chaos.py:459`): Tests behavior under memory pressure.
  - `TestMemoryPressureChaos.test_large_cache_entries(self)` (`tests/test_chaos.py:462`): Cache should handle large entries gracefully.
  - `TestMemoryPressureChaos.test_deduplicator_cleanup(self)` (`tests/test_chaos.py:477`): Deduplicator should clean up stale requests.

## `tests/test_correlation.py`

Resumo do arquivo: test_correlation.py — Tests for correlation ID infrastructure

### Classes e métodos

- Classe `TestCorrelationIdGeneration` (`tests/test_correlation.py:23`): Tests for correlation ID generation.
  - `TestCorrelationIdGeneration.test_generate_correlation_id_returns_uuid(self)` (`tests/test_correlation.py:26`): Generated ID should be a valid UUID4 string.
  - `TestCorrelationIdGeneration.test_generate_correlation_id_is_unique(self)` (`tests/test_correlation.py:34`): Each generated ID should be unique.
- Classe `TestCorrelationIdContextVar` (`tests/test_correlation.py:41`): Tests for correlation ID context variable.
  - `TestCorrelationIdContextVar.test_set_and_get_correlation_id(self)` (`tests/test_correlation.py:44`): Should be able to set and get correlation ID.
  - `TestCorrelationIdContextVar.test_set_correlation_id_generates_new_if_none(self)` (`tests/test_correlation.py:56`): Should generate new ID if None is passed.
  - `TestCorrelationIdContextVar.test_clear_correlation_id(self)` (`tests/test_correlation.py:69`): Should clear the correlation ID.
  - `TestCorrelationIdContextVar.test_get_correlation_id_returns_none_when_not_set(self)` (`tests/test_correlation.py:76`): Should return None when no ID is set.
- Classe `TestCorrelationIdContext` (`tests/test_correlation.py:85`): Tests for CorrelationIdContext context manager.
  - `TestCorrelationIdContext.test_context_manager_sets_id(self)` (`tests/test_correlation.py:88`): Context manager should set correlation ID.
  - `TestCorrelationIdContext.test_context_manager_generates_id_if_none(self)` (`tests/test_correlation.py:96`): Context manager should generate ID if not provided.
  - `TestCorrelationIdContext.test_context_manager_restores_previous_id(self)` (`tests/test_correlation.py:105`): Context manager should restore previous ID on exit.
  - `TestCorrelationIdContext.test_nested_context_managers(self)` (`tests/test_correlation.py:117`): Nested context managers should work correctly.
- Classe `TestAsyncCorrelationPropagation` (`tests/test_correlation.py:137`): Tests for async correlation ID propagation.
  - `TestAsyncCorrelationPropagation.test_correlation_propagates_in_async_context(self)` (`tests/test_correlation.py:141`): Correlation ID should propagate through async calls.
  - `TestAsyncCorrelationPropagation.test_correlation_isolated_in_concurrent_tasks(self)` (`tests/test_correlation.py:157`): Each concurrent task should have isolated correlation ID.
- Classe `TestCorrelationIdHeader` (`tests/test_correlation.py:176`): Tests for correlation ID HTTP header constant.
  - `TestCorrelationIdHeader.test_header_name_is_correct(self)` (`tests/test_correlation.py:179`): Header name should be X-Correlation-ID.

## `tests/test_coverage_boosters.py`

Resumo do arquivo: Módulo `tests/test_coverage_boosters.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_uncertainty_helpers_and_paths(monkeypatch)` (`tests/test_coverage_boosters.py:11`): Testa uncertainty helpers and paths.
- `test_query_service_helpers_and_insert(monkeypatch)` (`tests/test_coverage_boosters.py:47`): Testa query service helpers and insert.
- `test_sparse_index_core(monkeypatch)` (`tests/test_coverage_boosters.py:105`): Testa sparse index core.
- `test_tasks_event_loop_and_task(monkeypatch)` (`tests/test_coverage_boosters.py:132`): Testa tasks event loop and task.
- `test_online_predictor_metrics_and_calibration(monkeypatch, tmp_path)` (`tests/test_coverage_boosters.py:150`): Testa online predictor metrics and calibration.

## `tests/test_db_module.py`

Resumo do arquivo: Módulo `tests/test_db_module.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_get_db_url_with_config()` (`tests/test_db_module.py:10`): Testa get db url with config.
- `test_engine_singleton_close_and_lazy(monkeypatch)` (`tests/test_db_module.py:18`): Testa engine singleton close and lazy.
- `test_engine_failure_and_pool_stats_error(monkeypatch)` (`tests/test_db_module.py:91`): Testa engine failure and pool stats error.

## `tests/test_drift_detector.py`

Resumo do arquivo: Tests for query distribution drift detection.

### Classes e métodos

- Classe `TestCosineDistance` (`tests/test_drift_detector.py:11`): Test suite for cosine_distance function.
  - `TestCosineDistance.test_identical_vectors(self)` (`tests/test_drift_detector.py:14`): Test that identical vectors have distance 0.
  - `TestCosineDistance.test_orthogonal_vectors(self)` (`tests/test_drift_detector.py:23`): Test that orthogonal vectors have distance 1.
  - `TestCosineDistance.test_opposite_vectors(self)` (`tests/test_drift_detector.py:32`): Test that opposite vectors have distance 2.
  - `TestCosineDistance.test_zero_vector(self)` (`tests/test_drift_detector.py:41`): Test that zero vectors return neutral distance.
- Classe `TestQueryDriftDetector` (`tests/test_drift_detector.py:51`): Test suite for QueryDriftDetector.
  - `TestQueryDriftDetector.drift_detector(self)` (`tests/test_drift_detector.py:55`): Create a fresh drift detector instance.
  - `TestQueryDriftDetector.test_record_query_builds_baseline(self, drift_detector)` (`tests/test_drift_detector.py:67`): Test that initial queries build baseline.
  - `TestQueryDriftDetector.test_record_query_updates_total(self, drift_detector)` (`tests/test_drift_detector.py:76`): Test that recording updates total query count.
  - `TestQueryDriftDetector.test_drift_not_detected_similar_queries(self, drift_detector)` (`tests/test_drift_detector.py:85`): Test that similar queries don't trigger drift.
  - `TestQueryDriftDetector.test_get_status_structure(self, drift_detector)` (`tests/test_drift_detector.py:104`): Test that get_status returns expected structure.
  - `TestQueryDriftDetector.test_reset_baseline(self, drift_detector)` (`tests/test_drift_detector.py:115`): Test that reset_baseline clears data.
  - `TestQueryDriftDetector.test_force_baseline_update_not_enough_samples(self, drift_detector)` (`tests/test_drift_detector.py:127`): Test that force update fails with few samples.
  - `TestQueryDriftDetector.test_force_baseline_update_with_samples(self, drift_detector)` (`tests/test_drift_detector.py:133`): Test that force update succeeds with enough samples.
- Classe `TestGetDriftDetector` (`tests/test_drift_detector.py:145`): Test the get_drift_detector factory function.
  - `TestGetDriftDetector.test_returns_singleton(self)` (`tests/test_drift_detector.py:148`): Test that get_drift_detector returns singleton instance.

## `tests/test_embeddings.py`

Resumo do arquivo: test_embeddings.py — Tests for Embeddings Module

### Classes e métodos

- Classe `TestEmbeddingL1Cache` (`tests/test_embeddings.py:17`): Tests for the L1 in-memory embedding cache.
  - `TestEmbeddingL1Cache.test_cache_initialization(self)` (`tests/test_embeddings.py:20`): Test that cache initializes with correct defaults.
  - `TestEmbeddingL1Cache.test_cache_set_and_get(self)` (`tests/test_embeddings.py:29`): Test basic set and get operations.
  - `TestEmbeddingL1Cache.test_cache_miss_returns_none(self)` (`tests/test_embeddings.py:42`): Test that cache miss returns None.
  - `TestEmbeddingL1Cache.test_cache_eviction_on_maxsize(self)` (`tests/test_embeddings.py:52`): Test that oldest entries are evicted when maxsize is reached.
  - `TestEmbeddingL1Cache.test_cache_stats(self)` (`tests/test_embeddings.py:66`): Test that cache statistics are tracked correctly.
  - `TestEmbeddingL1Cache.test_cache_lru_behavior(self)` (`tests/test_embeddings.py:83`): Test LRU behavior - recently used items are kept.
- Classe `TestHashText` (`tests/test_embeddings.py:103`): Tests for the text hashing function.
  - `TestHashText.test_hash_text_deterministic(self)` (`tests/test_embeddings.py:106`): Test that same input produces same hash.
  - `TestHashText.test_hash_text_different_texts(self)` (`tests/test_embeddings.py:118`): Test that different texts produce different hashes.
  - `TestHashText.test_hash_text_different_models(self)` (`tests/test_embeddings.py:129`): Test that same text with different models produces different hashes.
- Classe `TestNorm` (`tests/test_embeddings.py:141`): Tests for vector normalization.
  - `TestNorm.test_norm_normalizes_vector(self)` (`tests/test_embeddings.py:144`): Test that _norm normalizes a vector to unit length.
  - `TestNorm.test_norm_zero_vector(self)` (`tests/test_embeddings.py:153`): Test that _norm handles zero vectors.
- Classe `TestEmbedText` (`tests/test_embeddings.py:164`): Tests for the main embed_text function.
  - `TestEmbedText.test_embed_text_empty_string(self)` (`tests/test_embeddings.py:167`): Test that empty string returns minimal vector.
  - `TestEmbedText.test_embed_text_none(self)` (`tests/test_embeddings.py:175`): Test that None returns minimal vector.
  - `TestEmbedText.test_embed_text_whitespace_only(self)` (`tests/test_embeddings.py:183`): Test that whitespace-only string returns minimal vector.
  - `TestEmbedText.test_embed_text_returns_list(self, mock_embed)` (`tests/test_embeddings.py:192`): Test that embed_text returns a list of floats.
  - `TestEmbedText.test_embed_text_caches_result(self, mock_embed, _mock_save_cache, _mock_load_cache)` (`tests/test_embeddings.py:209`): Test that results are cached.
- Classe `TestEmbedImage` (`tests/test_embeddings.py:229`): Tests for image embedding function.
  - `TestEmbedImage.test_embed_image_returns_minimal_vector(self)` (`tests/test_embeddings.py:232`): Test that embed_image returns a minimal vector (feature disabled).
- Classe `TestEmbedMultimodal` (`tests/test_embeddings.py:241`): Tests for multimodal embedding function.
  - `TestEmbedMultimodal.test_embed_multimodal_returns_dict(self, mock_embed_text)` (`tests/test_embeddings.py:245`): Test that embed_multimodal returns a dictionary.
  - `TestEmbedMultimodal.test_embed_multimodal_text_only(self, mock_embed_text)` (`tests/test_embeddings.py:259`): Test embed_multimodal with text only.
- Classe `TestGetEmbeddingCacheStats` (`tests/test_embeddings.py:270`): Tests for cache statistics function.
  - `TestGetEmbeddingCacheStats.test_get_embedding_cache_stats_returns_dict(self)` (`tests/test_embeddings.py:273`): Test that get_embedding_cache_stats returns a dictionary.
- Classe `TestLocalCpuEmbed` (`tests/test_embeddings.py:286`): Tests for local CPU embedding generation.
  - `TestLocalCpuEmbed.test_local_cpu_embed_adds_prefix_for_nomic(self, mock_get_model)` (`tests/test_embeddings.py:290`): Test that Nomic models get search_query prefix.
  - `TestLocalCpuEmbed.test_local_cpu_embed_returns_list(self, mock_get_model)` (`tests/test_embeddings.py:307`): Test that _local_cpu_embed returns a list.
  - `TestLocalCpuEmbed.test_local_cpu_embed_raises_when_no_model(self, mock_get_model)` (`tests/test_embeddings.py:320`): Test that _local_cpu_embed raises when model unavailable.

## `tests/test_health_components.py`

Resumo do arquivo: Módulo `tests/test_health_components.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_component_health_checks_success_paths(monkeypatch)` (`tests/test_health_components.py:29`): Testa component health checks success paths.
- `test_component_health_error_paths_and_cache(monkeypatch)` (`tests/test_health_components.py:103`): Testa component health error paths and cache.
- `test_component_to_dict_and_liveness()` (`tests/test_health_components.py:150`): Testa component to dict and liveness.

### Classes e métodos

- Classe `_Response` (`tests/test_health_components.py:11`): Classe `_Response`: concentra responsabilidades de test health components.
  - `_Response.__init__(self, data, fail)` (`tests/test_health_components.py:13`): Inicializa estado interno necessário para uso da classe.
  - `_Response.raise_for_status(self)` (`tests/test_health_components.py:18`): Executa raise for status.
  - `_Response.json(self)` (`tests/test_health_components.py:23`): Executa json.

## `tests/test_health_readiness_mode.py`

Resumo do arquivo: Módulo `tests/test_health_readiness_mode.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_readiness_strict_requires_both(monkeypatch)` (`tests/test_health_readiness_mode.py:15`): Testa readiness strict requires both.
- `test_readiness_degraded_accepts_one_dependency(monkeypatch)` (`tests/test_health_readiness_mode.py:46`): Testa readiness degraded accepts one dependency.
- `test_readiness_defaults_to_strict(monkeypatch)` (`tests/test_health_readiness_mode.py:77`): Testa readiness defaults to strict.

## `tests/test_integration_pipeline.py`

Resumo do arquivo: test_integration_pipeline.py — Integration tests for the full request pipeline

### Funções de módulo

- `client()` (`tests/test_integration_pipeline.py:18`): Create a test client for the FastAPI app.

### Classes e métodos

- Classe `TestQueryEndpoint` (`tests/test_integration_pipeline.py:57`): Tests for the /query endpoint.
  - `TestQueryEndpoint.test_query_returns_valid_response(self, client)` (`tests/test_integration_pipeline.py:60`): Basic query should return a valid response structure.
  - `TestQueryEndpoint.test_query_includes_correlation_id(self, client)` (`tests/test_integration_pipeline.py:75`): Response should include a correlation ID.
  - `TestQueryEndpoint.test_query_propagates_incoming_correlation_id(self, client)` (`tests/test_integration_pipeline.py:93`): When client sends X-Correlation-ID, it should be used.
  - `TestQueryEndpoint.test_empty_query_returns_400(self, client)` (`tests/test_integration_pipeline.py:107`): Empty query should return 400 or 422 error.
  - `TestQueryEndpoint.test_query_with_image_sets_vision_modality(self, client)` (`tests/test_integration_pipeline.py:119`): Query with image should use vision modality.
  - `TestQueryEndpoint.test_query_with_rag_enabled(self, client)` (`tests/test_integration_pipeline.py:154`): Query with RAG enabled should work correctly.
- Classe `TestHealthEndpoint` (`tests/test_integration_pipeline.py:172`): Tests for the /health endpoint.
  - `TestHealthEndpoint.test_health_returns_ok(self, client)` (`tests/test_integration_pipeline.py:175`): Health endpoint should return status and components.
- Classe `TestMetricsEndpoint` (`tests/test_integration_pipeline.py:189`): Tests for the /metrics endpoint.
  - `TestMetricsEndpoint.test_metrics_returns_prometheus_format(self, client)` (`tests/test_integration_pipeline.py:192`): Metrics endpoint should return Prometheus format.
- Classe `TestErrorHandling` (`tests/test_integration_pipeline.py:204`): Tests for error handling in the API.
  - `TestErrorHandling.test_router_error_returns_500(self, client)` (`tests/test_integration_pipeline.py:207`): When router fails, should return 500.
  - `TestErrorHandling.test_missing_query_field_returns_422(self, client)` (`tests/test_integration_pipeline.py:219`): Missing required field should return 422.
- Classe `TestResponseStructure` (`tests/test_integration_pipeline.py:231`): Tests for validating response structure.
  - `TestResponseStructure.test_response_has_all_required_fields(self, client)` (`tests/test_integration_pipeline.py:234`): Response should have all required fields from QueryResponse schema.
  - `TestResponseStructure.test_candidates_is_list(self, client)` (`tests/test_integration_pipeline.py:257`): Candidates field should be a list.

## `tests/test_judges.py`

Resumo do arquivo: Módulo `tests/test_judges.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_heuristic_score_valid()` (`tests/test_judges.py:9`): Avalia se o score heurístico retorna dentro do intervalo esperado.
- `test_heuristic_score_empty()` (`tests/test_judges.py:16`): Respostas vazias devem gerar score 0.
- `test_judge_answer_empty_response()` (`tests/test_judges.py:23`): Se a resposta for vazia, deve retornar score 0.
- `test_llm_based_score_mock(monkeypatch)` (`tests/test_judges.py:31`): Substitui o modelo real por mock e verifica conversão do score.
- `test_parse_score_from_text()` (`tests/test_judges.py:49`): Testa a extração de números de texto de resposta.

## `tests/test_judges_extra.py`

Resumo do arquivo: Módulo `tests/test_judges_extra.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_verdict_cache_and_helpers()` (`tests/test_judges_extra.py:10`): Testa verdict cache and helpers.
- `test_choose_two_and_extract_verdict(monkeypatch)` (`tests/test_judges_extra.py:27`): Testa choose two and extract verdict.
- `test_get_rag_context_describe_and_meta(monkeypatch)` (`tests/test_judges_extra.py:48`): Testa get rag context describe and meta.
- `test_llm_pair_score_and_judge_answer_modes(monkeypatch)` (`tests/test_judges_extra.py:82`): Testa llm pair score and judge answer modes.
- `test_judge_calibration_functions(monkeypatch)` (`tests/test_judges_extra.py:137`): Testa judge calibration functions.

## `tests/test_main_admin.py`

Resumo do arquivo: Módulo `tests/test_main_admin.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `_set_admin_token(monkeypatch, value)` (`tests/test_main_admin.py:9`): Executa set admin token.
- `_set_settings_map(monkeypatch, mapping)` (`tests/test_main_admin.py:20`): Executa set settings map.
- `test_safe_parse_json_variants()` (`tests/test_main_admin.py:27`): Testa safe parse json variants.
- `test_require_admin(monkeypatch)` (`tests/test_main_admin.py:36`): Testa require admin.
- `test_admin_settings_endpoints(monkeypatch)` (`tests/test_main_admin.py:48`): Testa admin settings endpoints.
- `test_circuit_breaker_admin(monkeypatch)` (`tests/test_main_admin.py:72`): Testa circuit breaker admin.
- `test_cascade_status_admin(monkeypatch)` (`tests/test_main_admin.py:95`): Testa cascade status admin.
- `test_feedback_endpoints(monkeypatch)` (`tests/test_main_admin.py:105`): Testa feedback endpoints.
- `test_experiment_admin_endpoints(monkeypatch)` (`tests/test_main_admin.py:126`): Testa experiment admin endpoints.
- `test_experiments_disabled_paths(monkeypatch)` (`tests/test_main_admin.py:173`): Testa experiments disabled paths.

## `tests/test_main_lifecycle.py`

Resumo do arquivo: Módulo `tests/test_main_lifecycle.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_preload_ollama_models_download_flow(monkeypatch)` (`tests/test_main_lifecycle.py:8`): Testa preload ollama models download flow.
- `test_startup_event_executes_warmup_and_shutdown(monkeypatch)` (`tests/test_main_lifecycle.py:77`): Testa startup event executes warmup and shutdown.
- `test_health_and_versioned_endpoints(monkeypatch)` (`tests/test_main_lifecycle.py:128`): Testa health and versioned endpoints.

## `tests/test_main_provider_errors.py`

Resumo do arquivo: Módulo `tests/test_main_provider_errors.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `_stabilize_settings_get(monkeypatch)` (`tests/test_main_provider_errors.py:15`): Executa stabilize settings get.
- `test_route_query_maps_provider_timeout_to_504(monkeypatch)` (`tests/test_main_provider_errors.py:23`): Testa route query maps provider timeout to 504.
- `test_route_query_maps_provider_rate_limit_to_429(monkeypatch)` (`tests/test_main_provider_errors.py:44`): Testa route query maps provider rate limit to 429.
- `test_route_query_maps_provider_unavailable_to_502(monkeypatch)` (`tests/test_main_provider_errors.py:65`): Testa route query maps provider unavailable to 502.
- `test_route_query_maps_circuit_open_to_503(monkeypatch)` (`tests/test_main_provider_errors.py:86`): Testa route query maps circuit open to 503.
- `test_startup_rejects_empty_admin_token(monkeypatch)` (`tests/test_main_provider_errors.py:107`): Testa startup rejects empty admin token.

## `tests/test_metrics_collector.py`

Resumo do arquivo: Módulo `tests/test_metrics_collector.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_ensure_table_and_persist_sample_success(monkeypatch)` (`tests/test_metrics_collector.py:35`): Testa ensure table and persist sample success.
- `test_persist_sample_swallows_exceptions(monkeypatch)` (`tests/test_metrics_collector.py:64`): Testa persist sample swallows exceptions.
- `test_update_model_metrics_ema_and_snapshot_copy(monkeypatch)` (`tests/test_metrics_collector.py:74`): Testa update model metrics ema and snapshot copy.

### Classes e métodos

- Classe `_Conn` (`tests/test_metrics_collector.py:8`): Classe `_Conn`: concentra responsabilidades de test metrics collector.
  - `_Conn.__init__(self)` (`tests/test_metrics_collector.py:10`): Inicializa estado interno necessário para uso da classe.
  - `_Conn.execute(self, stmt, params)` (`tests/test_metrics_collector.py:14`): Executa execute.
- Classe `_Ctx` (`tests/test_metrics_collector.py:20`): Classe `_Ctx`: concentra responsabilidades de test metrics collector.
  - `_Ctx.__init__(self, conn)` (`tests/test_metrics_collector.py:22`): Inicializa estado interno necessário para uso da classe.
  - `_Ctx.__enter__(self)` (`tests/test_metrics_collector.py:26`): Executa enter.
  - `_Ctx.__exit__(self, exc_type, exc, tb)` (`tests/test_metrics_collector.py:30`): Executa exit.

## `tests/test_model_registry.py`

Resumo do arquivo: Módulo `tests/test_model_registry.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `_fresh_registry()` (`tests/test_model_registry.py:6`): Executa fresh registry.
- `test_model_config_properties_and_cost()` (`tests/test_model_registry.py:12`): Testa model config properties and cost.
- `test_registry_get_list_and_filters()` (`tests/test_model_registry.py:30`): Testa registry get list and filters.
- `test_registry_get_or_default_and_fallback_chain()` (`tests/test_model_registry.py:52`): Testa registry get or default and fallback chain.
- `test_registry_cheapest_and_convenience_helpers()` (`tests/test_model_registry.py:87`): Testa registry cheapest and convenience helpers.

## `tests/test_performance.py`

Resumo do arquivo: test_performance.py - Testes de Performance para Quick Wins

### Classes e métodos

- Classe `TestEMAHistoryCache` (`tests/test_performance.py:20`): Testes para o cache EMA com TTL e LRU.
  - `TestEMAHistoryCache.test_ema_cache_basic_operations(self)` (`tests/test_performance.py:23`): Testa operações básicas de get/set no cache EMA.
  - `TestEMAHistoryCache.test_ema_cache_lru_eviction(self)` (`tests/test_performance.py:41`): Testa evicção LRU quando cache está cheio.
  - `TestEMAHistoryCache.test_ema_cache_ttl_expiration(self)` (`tests/test_performance.py:58`): Testa expiração por TTL.
  - `TestEMAHistoryCache.test_ema_cache_size_tracking(self)` (`tests/test_performance.py:74`): Testa tracking de tamanho do cache.
  - `TestEMAHistoryCache.test_ema_cache_cleanup_expired(self)` (`tests/test_performance.py:85`): Testa cleanup de entradas expiradas.
- Classe `TestEmbeddingL1Cache` (`tests/test_performance.py:103`): Testes para o cache L1 de embeddings.
  - `TestEmbeddingL1Cache.test_embedding_cache_basic(self)` (`tests/test_performance.py:106`): Testa operações básicas do cache de embeddings.
  - `TestEmbeddingL1Cache.test_embedding_cache_miss_tracking(self)` (`tests/test_performance.py:121`): Testa tracking de hits e misses.
  - `TestEmbeddingL1Cache.test_embedding_cache_ttl(self)` (`tests/test_performance.py:139`): Testa TTL do cache de embeddings.
- Classe `TestVerdictCache` (`tests/test_performance.py:152`): Testes para o cache de verdicts de juízes.
  - `TestVerdictCache.test_verdict_cache_basic(self)` (`tests/test_performance.py:155`): Testa operações básicas do cache de verdicts.
  - `TestVerdictCache.test_verdict_cache_key_generation(self)` (`tests/test_performance.py:171`): Testa que diferentes queries/answers geram chaves diferentes.
  - `TestVerdictCache.test_verdict_cache_stats(self)` (`tests/test_performance.py:189`): Testa estatísticas do cache de verdicts.
- Classe `TestCentroidMatrixCache` (`tests/test_performance.py:205`): Testes para o cache de matriz de centróides.
  - `TestCentroidMatrixCache.test_centroid_matrix_basic(self)` (`tests/test_performance.py:208`): Testa operações básicas do cache de matriz de centróides.
  - `TestCentroidMatrixCache.test_centroid_matrix_nearest_accuracy(self)` (`tests/test_performance.py:231`): Testa precisão do nearest neighbor.
  - `TestCentroidMatrixCache.test_centroid_matrix_staleness(self)` (`tests/test_performance.py:253`): Testa detecção de cache stale.
- Classe `TestConnectionPoolConfig` (`tests/test_performance.py:269`): Testes para verificar configuração do pool de conexões.
  - `TestConnectionPoolConfig.test_pool_settings_applied(self)` (`tests/test_performance.py:272`): Verifica que as configurações do pool estão corretas.
- Classe `TestPerformanceMetrics` (`tests/test_performance.py:285`): Testes para verificar que as métricas de performance existem.
  - `TestPerformanceMetrics.test_metrics_exist(self)` (`tests/test_performance.py:288`): Verifica que as novas métricas estão definidas.
- Classe `TestCacheIntegration` (`tests/test_performance.py:314`): Testes de integração para os caches.
  - `TestCacheIntegration.test_get_verdict_cache_stats_function(self)` (`tests/test_performance.py:317`): Testa função de estatísticas do cache de verdicts.
  - `TestCacheIntegration.test_get_embedding_cache_stats_function(self)` (`tests/test_performance.py:328`): Testa função de estatísticas do cache de embeddings.
- Classe `TestThreadSafety` (`tests/test_performance.py:340`): Testes de thread safety para os caches.
  - `TestThreadSafety.test_ema_cache_thread_safety(self)` (`tests/test_performance.py:343`): Testa que o cache EMA é thread-safe.
  - `TestThreadSafety.test_embedding_cache_thread_safety(self)` (`tests/test_performance.py:379`): Testa que o cache de embeddings é thread-safe.

## `tests/test_pricing_extra.py`

Resumo do arquivo: Módulo `tests/test_pricing_extra.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_refresh_pricing_from_db_success_and_error(monkeypatch)` (`tests/test_pricing_extra.py:36`): Testa refresh pricing from db success and error.
- `test_refresh_pricing_redis_hit_and_fallback(monkeypatch)` (`tests/test_pricing_extra.py:46`): Testa refresh pricing redis hit and fallback.
- `test_get_model_cost_paths(monkeypatch)` (`tests/test_pricing_extra.py:86`): Testa get model cost paths.

### Classes e métodos

- Classe `_Conn` (`tests/test_pricing_extra.py:10`): Classe `_Conn`: concentra responsabilidades de test pricing extra.
  - `_Conn.__init__(self, rows)` (`tests/test_pricing_extra.py:12`): Inicializa estado interno necessário para uso da classe.
  - `_Conn.execute(self, *_a, **_k)` (`tests/test_pricing_extra.py:16`): Executa execute.
- Classe `_Ctx` (`tests/test_pricing_extra.py:21`): Classe `_Ctx`: concentra responsabilidades de test pricing extra.
  - `_Ctx.__init__(self, rows)` (`tests/test_pricing_extra.py:23`): Inicializa estado interno necessário para uso da classe.
  - `_Ctx.__enter__(self)` (`tests/test_pricing_extra.py:27`): Executa enter.
  - `_Ctx.__exit__(self, exc_type, exc, tb)` (`tests/test_pricing_extra.py:31`): Executa exit.

## `tests/test_prometheus_setup.py`

Resumo do arquivo: Módulo `tests/test_prometheus_setup.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_setup_prometheus_and_registry(monkeypatch, tmp_path)` (`tests/test_prometheus_setup.py:8`): Testa setup prometheus and registry.
- `test_prometheus_metrics(monkeypatch)` (`tests/test_prometheus_setup.py:26`): Testa prometheus metrics.

## `tests/test_providers.py`

Resumo do arquivo: Módulo `tests/test_providers.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_heuristic_quality_estimate()` (`tests/test_providers.py:8`): Test heuristic quality estimation from text.
- `test_estimate_tokens()` (`tests/test_providers.py:31`): Test token estimation from text.
- `test_call_model_timeout_handling()` (`tests/test_providers.py:44`): Test that call_model handles timeouts gracefully.
- `test_call_model_returns_dict()` (`tests/test_providers.py:52`): Test that call_model returns expected structure.

## `tests/test_providers_async_core_reliability.py`

Resumo do arquivo: Módulo `tests/test_providers_async_core_reliability.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `_close_breakers()` (`tests/test_providers_async_core_reliability.py:13`): Executa close breakers.
- `test_openai_provider_generate_success_with_reasoning_model(monkeypatch)` (`tests/test_providers_async_core_reliability.py:20`): Testa openai provider generate success with reasoning model.
- `test_anthropic_provider_generate_success(monkeypatch)` (`tests/test_providers_async_core_reliability.py:57`): Testa anthropic provider generate success.
- `test_gemini_provider_generate_success(monkeypatch)` (`tests/test_providers_async_core_reliability.py:83`): Testa gemini provider generate success.
- `test_close_http_client_and_render_metrics(monkeypatch)` (`tests/test_providers_async_core_reliability.py:111`): Testa close http client and render metrics.
- `test_ensure_ollama_model_sync_paths(monkeypatch)` (`tests/test_providers_async_core_reliability.py:134`): Testa ensure ollama model sync paths.
- `test_call_model_error_categories_timeout_and_unavailable(monkeypatch)` (`tests/test_providers_async_core_reliability.py:176`): Testa call model error categories timeout and unavailable.

## `tests/test_providers_async_extra.py`

Resumo do arquivo: Módulo `tests/test_providers_async_extra.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_timeout_quality_tokens_and_factory(monkeypatch)` (`tests/test_providers_async_extra.py:13`): Testa timeout quality tokens and factory.
- `test_call_model_success_and_error_categories(monkeypatch)` (`tests/test_providers_async_extra.py:46`): Testa call model success and error categories.
- `test_ensure_ollama_model_async_paths(monkeypatch)` (`tests/test_providers_async_extra.py:107`): Testa ensure ollama model async paths.

## `tests/test_providers_ensure_ollama.py`

Resumo do arquivo: test_providers_ensure_ollama.py — Extensive tests for _ensure_ollama_model

### Classes e métodos

- Classe `TestEnsureOllamaModelBasic` (`tests/test_providers_ensure_ollama.py:20`): Basic functionality tests for _ensure_ollama_model.
  - `TestEnsureOllamaModelBasic.test_model_already_exists_exact_match(self)` (`tests/test_providers_ensure_ollama.py:23`): Should return True when model already exists with exact name.
  - `TestEnsureOllamaModelBasic.test_model_already_exists_partial_match(self)` (`tests/test_providers_ensure_ollama.py:38`): Should return True when model name is substring of existing model.
  - `TestEnsureOllamaModelBasic.test_model_not_found_pull_success(self)` (`tests/test_providers_ensure_ollama.py:55`): Should pull model and return True when model doesn't exist.
  - `TestEnsureOllamaModelBasic.test_model_not_found_pull_failure(self)` (`tests/test_providers_ensure_ollama.py:74`): Should return False when pull fails.
  - `TestEnsureOllamaModelBasic.test_empty_models_list(self)` (`tests/test_providers_ensure_ollama.py:91`): Should handle empty models list and attempt pull.
- Classe `TestEnsureOllamaModelErrorHandling` (`tests/test_providers_ensure_ollama.py:110`): Error handling tests for _ensure_ollama_model.
  - `TestEnsureOllamaModelErrorHandling.test_tags_endpoint_timeout(self)` (`tests/test_providers_ensure_ollama.py:113`): Should return False on tags endpoint timeout.
  - `TestEnsureOllamaModelErrorHandling.test_tags_endpoint_connection_error(self)` (`tests/test_providers_ensure_ollama.py:122`): Should return False on connection error.
  - `TestEnsureOllamaModelErrorHandling.test_pull_endpoint_timeout(self)` (`tests/test_providers_ensure_ollama.py:131`): Should return False on pull endpoint timeout.
  - `TestEnsureOllamaModelErrorHandling.test_tags_endpoint_non_200_status(self)` (`tests/test_providers_ensure_ollama.py:145`): Should attempt pull even if tags endpoint returns non-200.
  - `TestEnsureOllamaModelErrorHandling.test_malformed_json_response(self)` (`tests/test_providers_ensure_ollama.py:162`): Should handle malformed JSON gracefully.
  - `TestEnsureOllamaModelErrorHandling.test_missing_models_key_in_response(self)` (`tests/test_providers_ensure_ollama.py:175`): Should handle missing 'models' key in response.
  - `TestEnsureOllamaModelErrorHandling.test_model_without_name_field(self)` (`tests/test_providers_ensure_ollama.py:193`): Should handle models without 'name' field.
  - `TestEnsureOllamaModelErrorHandling.test_generic_exception(self)` (`tests/test_providers_ensure_ollama.py:216`): Should handle unexpected exceptions gracefully.
- Classe `TestEnsureOllamaModelThreadSafety` (`tests/test_providers_ensure_ollama.py:226`): Thread safety and event loop independence tests.
  - `TestEnsureOllamaModelThreadSafety.test_runs_in_thread_without_event_loop(self)` (`tests/test_providers_ensure_ollama.py:229`): Should work correctly when called from a thread without event loop.
  - `TestEnsureOllamaModelThreadSafety.test_runs_in_multiple_threads_concurrently(self)` (`tests/test_providers_ensure_ollama.py:245`): Should work correctly when called from multiple threads.
  - `TestEnsureOllamaModelThreadSafety.test_works_with_asyncio_to_thread(self)` (`tests/test_providers_ensure_ollama.py:262`): Should work correctly when called via asyncio.to_thread.
  - `TestEnsureOllamaModelThreadSafety.test_fire_and_forget_pattern(self)` (`tests/test_providers_ensure_ollama.py:277`): Should work with fire-and-forget pattern used in router_core.py.
- Classe `TestEnsureOllamaModelIntegration` (`tests/test_providers_ensure_ollama.py:295`): Integration tests with router_core patterns.
  - `TestEnsureOllamaModelIntegration.test_ollama_model_prefix_handling(self)` (`tests/test_providers_ensure_ollama.py:299`): Should handle model names with 'ollama/' prefix stripped.
  - `TestEnsureOllamaModelIntegration.test_multiple_concurrent_ensures(self)` (`tests/test_providers_ensure_ollama.py:317`): Should handle multiple concurrent ensure calls.
- Classe `TestEnsureOllamaModelLogging` (`tests/test_providers_ensure_ollama.py:338`): Tests for logging behavior.
  - `TestEnsureOllamaModelLogging.test_logs_download_on_pull(self)` (`tests/test_providers_ensure_ollama.py:341`): Should log when downloading a new model.
  - `TestEnsureOllamaModelLogging.test_logs_warning_on_failure(self)` (`tests/test_providers_ensure_ollama.py:361`): Should log warning when operation fails.
- Classe `TestEnsureOllamaModelAsync` (`tests/test_providers_ensure_ollama.py:373`): Tests for the async version _ensure_ollama_model_async.
  - `TestEnsureOllamaModelAsync.test_async_model_exists(self)` (`tests/test_providers_ensure_ollama.py:377`): Async version should return True when model exists.
  - `TestEnsureOllamaModelAsync.test_async_pull_model(self)` (`tests/test_providers_ensure_ollama.py:409`): Async version should pull model when not found.

## `tests/test_providers_reliability.py`

Resumo do arquivo: test_providers_reliability.py — Tests for provider reliability patterns

### Classes e métodos

- Classe `TestCircuitBreaker` (`tests/test_providers_reliability.py:15`): Tests for circuit breaker functionality.
  - `TestCircuitBreaker.test_cloud_breaker_configuration(self)` (`tests/test_providers_reliability.py:18`): Cloud breaker should have correct configuration.
  - `TestCircuitBreaker.test_local_breaker_configuration(self)` (`tests/test_providers_reliability.py:26`): Local breaker should have correct configuration.
  - `TestCircuitBreaker.test_breaker_states(self)` (`tests/test_providers_reliability.py:34`): Circuit breaker should have correct state transitions.
- Classe `TestRetryLogic` (`tests/test_providers_reliability.py:62`): Tests for retry configuration.
  - `TestRetryLogic.test_retryable_errors_includes_timeout(self)` (`tests/test_providers_reliability.py:65`): RETRYABLE_ERRORS should include timeout errors.
  - `TestRetryLogic.test_retry_strategy_configuration(self)` (`tests/test_providers_reliability.py:72`): COMMON_RETRY_STRATEGY should have correct configuration.
- Classe `TestProviderFactory` (`tests/test_providers_reliability.py:81`): Tests for ProviderFactory.
  - `TestProviderFactory.test_factory_returns_correct_provider_for_openai(self)` (`tests/test_providers_reliability.py:84`): Factory should return OpenAIProvider for openai/ prefix.
  - `TestProviderFactory.test_factory_returns_correct_provider_for_ollama(self)` (`tests/test_providers_reliability.py:92`): Factory should return OllamaProvider for ollama/ prefix.
  - `TestProviderFactory.test_factory_returns_ollama_for_no_prefix(self)` (`tests/test_providers_reliability.py:99`): Factory should default to OllamaProvider for models without prefix.
  - `TestProviderFactory.test_factory_raises_for_unknown_namespace(self)` (`tests/test_providers_reliability.py:106`): Factory should raise ValueError for unknown namespace.
- Classe `TestOllamaProvider` (`tests/test_providers_reliability.py:114`): Tests for OllamaProvider.
  - `TestOllamaProvider.test_ollama_provider_generates_response(self)` (`tests/test_providers_reliability.py:118`): OllamaProvider should generate a response.
  - `TestOllamaProvider.test_ollama_provider_extracts_reasoning(self)` (`tests/test_providers_reliability.py:154`): OllamaProvider should extract <think> tags from response.
- Classe `TestCallModelWrapper` (`tests/test_providers_reliability.py:187`): Tests for the call_model wrapper function.
  - `TestCallModelWrapper.test_call_model_returns_text_and_meta(self)` (`tests/test_providers_reliability.py:191`): call_model should return (text, metadata) tuple.
  - `TestCallModelWrapper.test_call_model_handles_circuit_breaker_open(self)` (`tests/test_providers_reliability.py:220`): call_model should raise ProviderCircuitOpenError for open breaker.
  - `TestCallModelWrapper.test_call_model_handles_general_exception(self)` (`tests/test_providers_reliability.py:238`): call_model should raise ProviderCallError for generic failures.
- Classe `TestHeuristicQuality` (`tests/test_providers_reliability.py:255`): Tests for heuristic_quality_estimate function.
  - `TestHeuristicQuality.test_empty_text_returns_zero(self)` (`tests/test_providers_reliability.py:258`): Empty text should return 0.0 quality.
  - `TestHeuristicQuality.test_longer_text_has_higher_quality(self)` (`tests/test_providers_reliability.py:265`): Longer text should have higher quality score.
  - `TestHeuristicQuality.test_punctuation_adds_bonus(self)` (`tests/test_providers_reliability.py:274`): Text with punctuation should have higher quality.
  - `TestHeuristicQuality.test_quality_is_bounded(self)` (`tests/test_providers_reliability.py:283`): Quality score should be between 0 and 10.
- Classe `TestReasoningModelDetection` (`tests/test_providers_reliability.py:291`): Tests for reasoning model detection.
  - `TestReasoningModelDetection.test_phi4_is_reasoning_model(self)` (`tests/test_providers_reliability.py:294`): phi4 should be detected as a reasoning model.
  - `TestReasoningModelDetection.test_deepseek_r1_is_reasoning_model(self)` (`tests/test_providers_reliability.py:302`): deepseek-r1 should be detected as a reasoning model.
  - `TestReasoningModelDetection.test_gpt4_is_not_reasoning_model(self)` (`tests/test_providers_reliability.py:310`): gpt-4 should not be detected as a reasoning model.

## `tests/test_rag_local_extra.py`

Resumo do arquivo: Módulo `tests/test_rag_local_extra.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_rag_local_helpers_and_visual_query_cache(monkeypatch)` (`tests/test_rag_local_extra.py:9`): Testa rag local helpers and visual query cache.
- `test_compute_embedding_and_fusion_paths(monkeypatch)` (`tests/test_rag_local_extra.py:47`): Testa compute embedding and fusion paths.
- `test_build_prompt_add_document_and_health(monkeypatch)` (`tests/test_rag_local_extra.py:75`): Testa build prompt add document and health.

## `tests/test_rag_logic.py`

Resumo do arquivo: Módulo `tests/test_rag_logic.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_rrf_logic()` (`tests/test_rag_logic.py:6`): Testa a fusão de rankings.
- `test_rrf_empty()` (`tests/test_rag_logic.py:29`): Testa listas vazias.

## `tests/test_rag_router.py`

Resumo do arquivo: Módulo `tests/test_rag_router.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_chunk_text_basic()` (`tests/test_rag_router.py:12`): Testa chunk text basic.
- `test_extract_text_from_pdf_error(monkeypatch)` (`tests/test_rag_router.py:20`): Testa extract text from pdf error.
- `test_summarize_text_success_and_fallback(monkeypatch)` (`tests/test_rag_router.py:29`): Testa summarize text success and fallback.
- `test_add_doc_txt_success(monkeypatch)` (`tests/test_rag_router.py:51`): Testa add doc txt success.
- `test_add_doc_validation_and_empty(monkeypatch)` (`tests/test_rag_router.py:76`): Testa add doc validation and empty.
- `test_ingest_text_success_and_fail(monkeypatch)` (`tests/test_rag_router.py:90`): Testa ingest text success and fail.

## `tests/test_redis_client.py`

Resumo do arquivo: test_redis_client.py — Tests for Redis Client Module

### Classes e métodos

- Classe `TestRedisClientConfiguration` (`tests/test_redis_client.py:17`): Tests for Redis client configuration.
  - `TestRedisClientConfiguration.test_default_configuration_values(self)` (`tests/test_redis_client.py:20`): Test that default configuration values are set correctly.
  - `TestRedisClientConfiguration.test_configuration_from_environment(self)` (`tests/test_redis_client.py:31`): Test that configuration can be read from environment variables.
- Classe `TestRedisClientConnection` (`tests/test_redis_client.py:47`): Tests for Redis connection functionality.
  - `TestRedisClientConnection.test_get_redis_function_exists(self)` (`tests/test_redis_client.py:50`): Test that get_redis function exists and is callable.
  - `TestRedisClientConnection.test_get_redis_async_safe_function_exists(self)` (`tests/test_redis_client.py:56`): Test that get_redis_async_safe function exists and is callable.
  - `TestRedisClientConnection.test_get_redis_async_safe_is_none_safe(self)` (`tests/test_redis_client.py:62`): Test get_redis_async_safe handles None client gracefully.
- Classe `TestRedisHealthCheck` (`tests/test_redis_client.py:72`): Tests for Redis health check functionality.
  - `TestRedisHealthCheck.test_check_redis_health_returns_dict(self)` (`tests/test_redis_client.py:75`): Test check_redis_health returns a dictionary with expected keys.
  - `TestRedisHealthCheck.test_check_redis_health_has_expected_fields(self)` (`tests/test_redis_client.py:87`): Test health check returns expected fields.
  - `TestRedisHealthCheck.test_check_redis_health_measures_latency(self, mock_get_redis)` (`tests/test_redis_client.py:99`): Test health check measures ping latency.
- Classe `TestRedisPipeline` (`tests/test_redis_client.py:114`): Tests for Redis pipeline operations.
  - `TestRedisPipeline.test_redis_pipeline_context_manager(self, mock_get_redis)` (`tests/test_redis_client.py:118`): Test redis_pipeline context manager works correctly.
  - `TestRedisPipeline.test_redis_pipeline_raises_when_unavailable(self, mock_get_redis)` (`tests/test_redis_client.py:134`): Test redis_pipeline raises RuntimeError when Redis unavailable.
- Classe `TestRedisCleanup` (`tests/test_redis_client.py:145`): Tests for Redis cleanup functionality.
  - `TestRedisCleanup.test_close_redis_function_exists(self)` (`tests/test_redis_client.py:148`): Test close_redis function exists and is callable.
  - `TestRedisCleanup.test_close_redis_can_be_called_multiple_times(self)` (`tests/test_redis_client.py:154`): Test close_redis can be called multiple times without error.
- Classe `TestRedisPoolCreation` (`tests/test_redis_client.py:164`): Tests for Redis pool creation.
  - `TestRedisPoolCreation.test_create_pool_with_correct_parameters(self, mock_pool_cls)` (`tests/test_redis_client.py:168`): Test _create_pool creates pool with correct parameters.
  - `TestRedisPoolCreation.test_create_pool_returns_none_on_error(self, mock_pool_cls)` (`tests/test_redis_client.py:184`): Test _create_pool returns None when pool creation fails.

## `tests/test_reliability_core_extra.py`

Resumo do arquivo: Módulo `tests/test_reliability_core_extra.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_circuit_breaker_manager_config_status_reset(monkeypatch)` (`tests/test_reliability_core_extra.py:12`): Testa circuit breaker manager config status reset.
- `test_request_deduplicator_compute_cleanup_and_stats()` (`tests/test_reliability_core_extra.py:32`): Testa request deduplicator compute cleanup and stats.
- `test_execute_with_fallback_success_and_fail(monkeypatch)` (`tests/test_reliability_core_extra.py:51`): Testa execute with fallback success and fail.
- `test_model_health_helpers(monkeypatch)` (`tests/test_reliability_core_extra.py:110`): Testa model health helpers.
- `test_cascade_detector_status_and_warnings(monkeypatch)` (`tests/test_reliability_core_extra.py:131`): Testa cascade detector status and warnings.

## `tests/test_reranker_module.py`

Resumo do arquivo: Módulo `tests/test_reranker_module.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_get_reranker_model_and_rerank_paths(monkeypatch)` (`tests/test_reranker_module.py:6`): Testa get reranker model and rerank paths.
- `test_get_reranker_model_load_error(monkeypatch)` (`tests/test_reranker_module.py:44`): Testa get reranker model load error.

## `tests/test_risk_tuner.py`

Resumo do arquivo: Tests for adaptive risk factor management.

### Classes e métodos

- Classe `TestPerformanceRecord` (`tests/test_risk_tuner.py:10`): Test suite for PerformanceRecord dataclass.
  - `TestPerformanceRecord.test_avg_quality_empty(self)` (`tests/test_risk_tuner.py:13`): Test avg_quality returns default when no samples.
  - `TestPerformanceRecord.test_avg_quality_with_samples(self)` (`tests/test_risk_tuner.py:20`): Test avg_quality calculation with samples.
  - `TestPerformanceRecord.test_success_rate_empty(self)` (`tests/test_risk_tuner.py:27`): Test success_rate returns default when no samples.
  - `TestPerformanceRecord.test_success_rate_with_samples(self)` (`tests/test_risk_tuner.py:34`): Test success_rate calculation with samples.
  - `TestPerformanceRecord.test_to_dict_and_from_dict(self)` (`tests/test_risk_tuner.py:41`): Test serialization and deserialization.
- Classe `TestAdaptiveRiskTuner` (`tests/test_risk_tuner.py:59`): Test suite for AdaptiveRiskTuner.
  - `TestAdaptiveRiskTuner.risk_tuner(self)` (`tests/test_risk_tuner.py:63`): Create a fresh risk tuner instance.
  - `TestAdaptiveRiskTuner.test_get_model_type_sota(self, risk_tuner)` (`tests/test_risk_tuner.py:71`): Test SOTA model detection.
  - `TestAdaptiveRiskTuner.test_get_model_type_local(self, risk_tuner)` (`tests/test_risk_tuner.py:77`): Test local model detection.
  - `TestAdaptiveRiskTuner.test_record_outcome_disabled(self, risk_tuner)` (`tests/test_risk_tuner.py:83`): Test that recording is skipped when disabled.
  - `TestAdaptiveRiskTuner.test_record_outcome_updates_performance(self, risk_tuner)` (`tests/test_risk_tuner.py:91`): Test that recording updates performance data.
  - `TestAdaptiveRiskTuner.test_calculate_adjustment_not_enough_samples(self, risk_tuner)` (`tests/test_risk_tuner.py:102`): Test that adjustment returns current factor when not enough samples.
  - `TestAdaptiveRiskTuner.test_calculate_adjustment_clamps_to_bounds(self, risk_tuner)` (`tests/test_risk_tuner.py:111`): Test that adjustment respects bounds.
  - `TestAdaptiveRiskTuner.test_get_status_structure(self, risk_tuner)` (`tests/test_risk_tuner.py:128`): Test that get_status returns expected structure.
  - `TestAdaptiveRiskTuner.test_reset_clears_data(self, risk_tuner)` (`tests/test_risk_tuner.py:143`): Test that reset clears all performance data.
- Classe `TestGetRiskTuner` (`tests/test_risk_tuner.py:154`): Test the get_risk_tuner factory function.
  - `TestGetRiskTuner.test_returns_singleton(self)` (`tests/test_risk_tuner.py:157`): Test that get_risk_tuner returns singleton instance.

## `tests/test_router_core.py`

Resumo do arquivo: test_router_core.py — Unit tests for router_core.py

### Classes e métodos

- Classe `TestRouteAndAnswer` (`tests/test_router_core.py:17`): Tests for the route_and_answer function.
  - `TestRouteAndAnswer.test_cache_hit_returns_cached_response(self)` (`tests/test_router_core.py:21`): When cache hits, should return cached response without calling model.
  - `TestRouteAndAnswer.test_cache_miss_calls_model(self)` (`tests/test_router_core.py:47`): When cache misses, should call the selected model.
  - `TestRouteAndAnswer.test_cache_disabled_skips_cache_check(self)` (`tests/test_router_core.py:83`): When use_cache=False, should skip cache lookup.
  - `TestRouteAndAnswer.test_image_input_sets_vision_modality(self)` (`tests/test_router_core.py:117`): When image is provided with text modality, should switch to vision.
  - `TestRouteAndAnswer.test_model_error_returns_error_message(self)` (`tests/test_router_core.py:153`): When model call fails, route should propagate the exception.
- Classe `TestDynamicWeights` (`tests/test_router_core.py:173`): Tests for dynamic strategy weights.
  - `TestDynamicWeights.test_get_dynamic_strategy_weights_returns_dict(self)` (`tests/test_router_core.py:176`): Should return a dictionary with quality, latency, and cost weights.
- Classe `TestRAGIntegration` (`tests/test_router_core.py:194`): Tests for RAG (Retrieval Augmented Generation) integration.
  - `TestRAGIntegration.test_rag_enabled_augments_prompt(self)` (`tests/test_router_core.py:198`): When use_rag=True, should augment the prompt with RAG context.

## `tests/test_router_core_extra.py`

Resumo do arquivo: Módulo `tests/test_router_core_extra.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_ema_history_cache_and_batch_queue(monkeypatch)` (`tests/test_router_core_extra.py:11`): Testa ema history cache and batch queue.
- `test_load_ema_from_db_and_start_stop_services(monkeypatch)` (`tests/test_router_core_extra.py:39`): Testa load ema from db and start stop services.
- `test_route_and_answer_dedup_and_timeout(monkeypatch)` (`tests/test_router_core_extra.py:104`): Testa route and answer dedup and timeout.
- `test_process_background_feedback_branches(monkeypatch)` (`tests/test_router_core_extra.py:138`): Testa process background feedback branches.

## `tests/test_router_core_internal.py`

Resumo do arquivo: Módulo `tests/test_router_core_internal.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `_mock_settings()` (`tests/test_router_core_internal.py:9`): Executa mock settings.
- `test_internal_cache_hit(monkeypatch)` (`tests/test_router_core_internal.py:21`): Testa internal cache hit.
- `test_internal_full_flow_with_pricing_fallback(monkeypatch)` (`tests/test_router_core_internal.py:35`): Testa internal full flow with pricing fallback.
- `test_internal_rag_fallback_when_augmented_prompt_fails(monkeypatch)` (`tests/test_router_core_internal.py:73`): Testa internal rag fallback when augmented prompt fails.
- `test_internal_handles_non_dict_metadata(monkeypatch)` (`tests/test_router_core_internal.py:100`): Testa internal handles non dict metadata.
- `test_internal_fallback_when_all_candidates_blocked(monkeypatch)` (`tests/test_router_core_internal.py:119`): Testa internal fallback when all candidates blocked.

## `tests/test_router_maintenance_service.py`

Resumo do arquivo: Módulo `tests/test_router_maintenance_service.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_create_background_threads()` (`tests/test_router_maintenance_service.py:6`): Testa create background threads.

## `tests/test_router_strategy.py`

Resumo do arquivo: Módulo `tests/test_router_strategy.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `mock_bandit_data()` (`tests/test_router_strategy.py:22`): Executa mock bandit data.
- `test_router_prefers_quality(mock_bandit_data)` (`tests/test_router_strategy.py:28`): Se o peso da qualidade for alto, deve escolher o GPT-5.
- `test_router_prefers_cost(mock_bandit_data)` (`tests/test_router_strategy.py:36`): Se o peso do custo for alto, deve escolher o Gemma.
- `test_vision_filter_logic(mock_bandit_data)` (`tests/test_router_strategy.py:44`): Se a modalidade for visão, deve filtrar modelos de texto.
- `test_uncertainty_trigger(mock_bandit_data)` (`tests/test_router_strategy.py:54`): Se a incerteza for alta, deve penalizar modelos locais (risk_factor).

## `tests/test_runtime_state.py`

Resumo do arquivo: Módulo `tests/test_runtime_state.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_reset_runtime_state_calls_all(monkeypatch)` (`tests/test_runtime_state.py:6`): Testa reset runtime state calls all.

## `tests/test_schemas.py`

Resumo do arquivo: test_schemas.py — Tests for Pydantic Schema Validation

### Classes e métodos

- Classe `TestModalityEnum` (`tests/test_schemas.py:16`): Tests for Modality enum.
  - `TestModalityEnum.test_modality_values(self)` (`tests/test_schemas.py:19`): Test that Modality enum has expected values.
- Classe `TestQueryRequest` (`tests/test_schemas.py:28`): Tests for QueryRequest schema validation.
  - `TestQueryRequest.test_valid_minimal_request(self)` (`tests/test_schemas.py:31`): Test minimal valid request.
  - `TestQueryRequest.test_valid_full_request(self)` (`tests/test_schemas.py:42`): Test request with all optional fields.
  - `TestQueryRequest.test_query_required(self)` (`tests/test_schemas.py:65`): Test that query is required.
  - `TestQueryRequest.test_query_min_length(self)` (`tests/test_schemas.py:75`): Test query minimum length constraint.
  - `TestQueryRequest.test_query_whitespace_only_rejected(self)` (`tests/test_schemas.py:85`): Test that whitespace-only query is rejected by validator.
  - `TestQueryRequest.test_query_max_length(self)` (`tests/test_schemas.py:94`): Test query maximum length constraint.
  - `TestQueryRequest.test_query_strips_whitespace(self)` (`tests/test_schemas.py:106`): Test that query validator strips whitespace.
  - `TestQueryRequest.test_modality_validation_valid(self)` (`tests/test_schemas.py:114`): Test valid modality values.
  - `TestQueryRequest.test_modality_validation_invalid(self)` (`tests/test_schemas.py:122`): Test invalid modality value is rejected.
  - `TestQueryRequest.test_modality_case_insensitive(self)` (`tests/test_schemas.py:131`): Test that modality validation is case insensitive.
  - `TestQueryRequest.test_max_tokens_boundaries(self)` (`tests/test_schemas.py:141`): Test max_tokens boundary validation.
  - `TestQueryRequest.test_temperature_boundaries(self)` (`tests/test_schemas.py:161`): Test temperature boundary validation.
  - `TestQueryRequest.test_timeout_boundaries(self)` (`tests/test_schemas.py:181`): Test timeout_seconds boundary validation.
  - `TestQueryRequest.test_images_list_max_length(self)` (`tests/test_schemas.py:201`): Test images list max length constraint.
  - `TestQueryRequest.test_rag_modality_validation(self)` (`tests/test_schemas.py:215`): Test rag_modality validation.
- Classe `TestJudgeScore` (`tests/test_schemas.py:228`): Tests for JudgeScore schema.
  - `TestJudgeScore.test_valid_judge_score(self)` (`tests/test_schemas.py:231`): Test valid JudgeScore creation.
  - `TestJudgeScore.test_minimal_judge_score(self)` (`tests/test_schemas.py:245`): Test minimal JudgeScore with only required fields.
- Classe `TestCandidateResult` (`tests/test_schemas.py:257`): Tests for CandidateResult schema.
  - `TestCandidateResult.test_valid_candidate_result(self)` (`tests/test_schemas.py:260`): Test valid CandidateResult creation.
  - `TestCandidateResult.test_minimal_candidate_result(self)` (`tests/test_schemas.py:279`): Test minimal CandidateResult.
  - `TestCandidateResult.test_candidate_with_judge_scores(self)` (`tests/test_schemas.py:290`): Test CandidateResult with judge scores.
- Classe `TestRouteDecision` (`tests/test_schemas.py:306`): Tests for RouteDecision schema.
  - `TestRouteDecision.test_valid_route_decision(self)` (`tests/test_schemas.py:309`): Test valid RouteDecision creation.
  - `TestRouteDecision.test_minimal_route_decision(self)` (`tests/test_schemas.py:324`): Test minimal RouteDecision.
- Classe `TestQueryResponse` (`tests/test_schemas.py:336`): Tests for QueryResponse schema.
  - `TestQueryResponse.test_valid_query_response(self)` (`tests/test_schemas.py:339`): Test valid QueryResponse creation.
  - `TestQueryResponse.test_response_with_image(self)` (`tests/test_schemas.py:356`): Test QueryResponse with image output.
  - `TestQueryResponse.test_response_with_candidates(self)` (`tests/test_schemas.py:375`): Test QueryResponse with candidate results.
- Classe `TestSchemaJsonSerialization` (`tests/test_schemas.py:392`): Tests for JSON serialization of schemas.
  - `TestSchemaJsonSerialization.test_query_request_to_json(self)` (`tests/test_schemas.py:395`): Test QueryRequest JSON serialization.
  - `TestSchemaJsonSerialization.test_query_response_to_json(self)` (`tests/test_schemas.py:407`): Test QueryResponse JSON serialization.
- Classe `TestSchemaEdgeCases` (`tests/test_schemas.py:424`): Tests for edge cases and special scenarios.
  - `TestSchemaEdgeCases.test_unicode_query(self)` (`tests/test_schemas.py:427`): Test Unicode characters in query.
  - `TestSchemaEdgeCases.test_emoji_in_query(self)` (`tests/test_schemas.py:435`): Test emoji characters in query.
  - `TestSchemaEdgeCases.test_newlines_in_query(self)` (`tests/test_schemas.py:443`): Test newlines in query are preserved (after strip).
  - `TestSchemaEdgeCases.test_payload_accepts_dict(self)` (`tests/test_schemas.py:451`): Test that payload field accepts dict.
  - `TestSchemaEdgeCases.test_payload_accepts_string(self)` (`tests/test_schemas.py:462`): Test that payload field accepts string.
  - `TestSchemaEdgeCases.test_very_long_answer(self)` (`tests/test_schemas.py:470`): Test that very long answers are accepted.

## `tests/test_semantic_cache.py`

Resumo do arquivo: Módulo `tests/test_semantic_cache.py`: descreve responsabilidades e integrações deste arquivo.

### Classes e métodos

- Classe `TestL1Cache` (`tests/test_semantic_cache.py:15`): Tests for the L1Cache class.
  - `TestL1Cache.test_l1_cache_store_and_get(self)` (`tests/test_semantic_cache.py:18`): L1 cache should store and retrieve values.
  - `TestL1Cache.test_l1_cache_returns_none_for_missing_key(self)` (`tests/test_semantic_cache.py:28`): L1 cache should return None for missing keys.
  - `TestL1Cache.test_l1_cache_respects_ttl(self)` (`tests/test_semantic_cache.py:36`): L1 cache should expire entries after TTL.
  - `TestL1Cache.test_l1_cache_evicts_oldest_when_full(self)` (`tests/test_semantic_cache.py:51`): L1 cache should evict oldest entries when maxsize exceeded.
  - `TestL1Cache.test_l1_cache_updates_lru_on_access(self)` (`tests/test_semantic_cache.py:65`): L1 cache should move accessed items to end (LRU).
  - `TestL1Cache.test_l1_cache_stats_tracks_hits_and_misses(self)` (`tests/test_semantic_cache.py:84`): L1 cache should track hits and misses correctly.
  - `TestL1Cache.test_l1_cache_clear(self)` (`tests/test_semantic_cache.py:100`): L1 cache clear should remove all entries.
- Classe `TestGlobalL1Cache` (`tests/test_semantic_cache.py:114`): Tests for the global L1 cache instance.
  - `TestGlobalL1Cache.test_global_l1_cache_exists(self)` (`tests/test_semantic_cache.py:117`): Global L1 cache instance should exist.
  - `TestGlobalL1Cache.test_get_l1_cache_stats_returns_dict(self)` (`tests/test_semantic_cache.py:122`): get_l1_cache_stats should return statistics dictionary.
- Classe `TestHashConsistency` (`tests/test_semantic_cache.py:132`): Tests for hash functions.
  - `TestHashConsistency.test_hash_consistency(self)` (`tests/test_semantic_cache.py:135`): O hash deve ser determinístico.
- Classe `TestCacheLogic` (`tests/test_semantic_cache.py:144`): Tests for cache hit/miss logic.
  - `TestCacheLogic.test_cache_hit_logic(self)` (`tests/test_semantic_cache.py:148`): Simula um HIT no cache vetorial.
  - `TestCacheLogic.test_cache_miss_logic(self)` (`tests/test_semantic_cache.py:173`): Simula um MISS no cache (distância alta).
  - `TestCacheLogic.test_l1_cache_hit_skips_chroma(self)` (`tests/test_semantic_cache.py:195`): L1 cache hit should skip ChromaDB lookup.

## `tests/test_semantic_cache_extra.py`

Resumo do arquivo: Módulo `tests/test_semantic_cache_extra.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_make_embedding_and_normalize_paths(monkeypatch)` (`tests/test_semantic_cache_extra.py:12`): Testa make embedding and normalize paths.
- `test_check_cache_branches(monkeypatch)` (`tests/test_semantic_cache_extra.py:32`): Testa check cache branches.
- `test_store_cache_and_hit_rate_tuning(monkeypatch)` (`tests/test_semantic_cache_extra.py:100`): Testa store cache and hit rate tuning.

## `tests/test_settings.py`

Resumo do arquivo: Módulo `tests/test_settings.py`: descreve responsabilidades e integrações deste arquivo.

### Classes e métodos

- Classe `TestLRUCache` (`tests/test_settings.py:8`): Test the LRU cache implementation.
  - `TestLRUCache.test_cache_basic_operations(self)` (`tests/test_settings.py:11`): Test basic get/set operations.
  - `TestLRUCache.test_cache_eviction(self)` (`tests/test_settings.py:22`): Test LRU eviction when cache is full.
  - `TestLRUCache.test_cache_ttl_expiry(self)` (`tests/test_settings.py:41`): Test TTL expiry.
  - `TestLRUCache.test_cache_clear(self)` (`tests/test_settings.py:55`): Test cache clear operation.
- Classe `TestDynamicSettings` (`tests/test_settings.py:68`): Test the DynamicSettings class.
  - `TestDynamicSettings.test_settings_defaults(self)` (`tests/test_settings.py:71`): Test that default values are accessible.
  - `TestDynamicSettings.test_settings_get_method(self)` (`tests/test_settings.py:79`): Test the get method with default values.
  - `TestDynamicSettings.test_settings_snapshot(self)` (`tests/test_settings.py:87`): Test snapshot method returns dict.

## `tests/test_smoke.py`

Resumo do arquivo: test_smoke.py — Smoke Tests for LLM Router

### Funções de módulo

- `run_single_smoke_test(query, domain, min_length, timeout)` (`tests/test_smoke.py:120`): Run a single smoke test query.
- `run_smoke_tests(queries, parallel, timeout_per_query)` (`tests/test_smoke.py:193`): Run all smoke tests.
- `test_smoke_technology()` (`tests/test_smoke.py:271`): Test technology/programming query.
- `test_smoke_math()` (`tests/test_smoke.py:285`): Test math query.
- `test_smoke_science()` (`tests/test_smoke.py:299`): Test science query.
- `test_smoke_full_suite()` (`tests/test_smoke.py:314`): Run full smoke test suite (slow).

## `tests/test_token_utils.py`

Resumo do arquivo: test_token_utils.py — Tests for Token Utilities

### Classes e métodos

- Classe `TestCountTokens` (`tests/test_token_utils.py:12`): Tests for the count_tokens function.
  - `TestCountTokens.test_count_tokens_empty_string(self)` (`tests/test_token_utils.py:15`): Test that empty string returns 0 tokens.
  - `TestCountTokens.test_count_tokens_none_returns_zero(self)` (`tests/test_token_utils.py:22`): Test that None input returns 0 tokens.
  - `TestCountTokens.test_count_tokens_heuristic_for_non_openai(self)` (`tests/test_token_utils.py:29`): Test heuristic token counting for non-OpenAI models.
  - `TestCountTokens.test_count_tokens_heuristic_for_gemma(self)` (`tests/test_token_utils.py:40`): Test heuristic for Gemma model.
  - `TestCountTokens.test_count_tokens_uses_tiktoken_for_gpt(self)` (`tests/test_token_utils.py:50`): Test that tiktoken is used for GPT models.
  - `TestCountTokens.test_count_tokens_uses_tiktoken_for_o1(self)` (`tests/test_token_utils.py:61`): Test that tiktoken is used for o1 models.
  - `TestCountTokens.test_count_tokens_minimum_one(self)` (`tests/test_token_utils.py:70`): Test that count_tokens returns at least 1 for non-empty text.
- Classe `TestEncoderCache` (`tests/test_token_utils.py:80`): Tests for the encoder LRU cache.
  - `TestEncoderCache.test_get_encoder_cache_info(self)` (`tests/test_token_utils.py:83`): Test that cache info is available.
  - `TestEncoderCache.test_clear_encoder_cache(self)` (`tests/test_token_utils.py:95`): Test that cache can be cleared.
  - `TestEncoderCache.test_cache_hit_after_repeated_calls(self)` (`tests/test_token_utils.py:113`): Test that repeated calls result in cache hits.
  - `TestEncoderCache.test_cache_respects_maxsize(self)` (`tests/test_token_utils.py:134`): Test that cache respects the maximum size limit.
- Classe `TestEncoderFallback` (`tests/test_token_utils.py:146`): Tests for encoder fallback behavior.
  - `TestEncoderFallback.test_fallback_when_tiktoken_unavailable(self, mock_tiktoken)` (`tests/test_token_utils.py:150`): Test fallback when tiktoken is not available.
  - `TestEncoderFallback.test_heuristic_fallback_on_error(self)` (`tests/test_token_utils.py:165`): Test that heuristic is used when tiktoken fails.
- Classe `TestTokenCountingAccuracy` (`tests/test_token_utils.py:178`): Tests for token counting accuracy.
  - `TestTokenCountingAccuracy.test_longer_text_more_tokens(self)` (`tests/test_token_utils.py:181`): Test that longer text produces more tokens.
  - `TestTokenCountingAccuracy.test_special_characters_counted(self)` (`tests/test_token_utils.py:193`): Test that special characters are handled.
  - `TestTokenCountingAccuracy.test_unicode_text_handled(self)` (`tests/test_token_utils.py:202`): Test that unicode text is handled correctly.
  - `TestTokenCountingAccuracy.test_emoji_counted(self)` (`tests/test_token_utils.py:211`): Test that emojis are counted correctly.

## `tests/test_user_feedback.py`

Resumo do arquivo: Tests for user feedback processing.

### Classes e métodos

- Classe `TestFeedbackQualityMapping` (`tests/test_user_feedback.py:10`): Test suite for feedback to quality mapping.
  - `TestFeedbackQualityMapping.test_rating_to_quality(self)` (`tests/test_user_feedback.py:13`): Test rating to quality conversion.
  - `TestFeedbackQualityMapping.test_thumbs_up_quality(self)` (`tests/test_user_feedback.py:23`): Test thumbs up maps to high quality.
  - `TestFeedbackQualityMapping.test_thumbs_down_quality(self)` (`tests/test_user_feedback.py:32`): Test thumbs down maps to low quality.
- Classe `TestGetQualityFromFeedback` (`tests/test_user_feedback.py:42`): Test suite for get_quality_from_feedback function.
  - `TestGetQualityFromFeedback.test_thumbs_up(self)` (`tests/test_user_feedback.py:45`): Test getting quality from thumbs up feedback.
  - `TestGetQualityFromFeedback.test_thumbs_down(self)` (`tests/test_user_feedback.py:61`): Test getting quality from thumbs down feedback.
  - `TestGetQualityFromFeedback.test_rating(self)` (`tests/test_user_feedback.py:77`): Test getting quality from rating feedback.
  - `TestGetQualityFromFeedback.test_rating_missing_value(self)` (`tests/test_user_feedback.py:94`): Test that rating without value raises error.
  - `TestGetQualityFromFeedback.test_explicit_quality(self)` (`tests/test_user_feedback.py:111`): Test getting quality from explicit quality feedback.
- Classe `TestBlendQuality` (`tests/test_user_feedback.py:129`): Test suite for blend_quality function.
  - `TestBlendQuality.test_blend_quality_default_weight(self)` (`tests/test_user_feedback.py:132`): Test quality blending with default weight.
  - `TestBlendQuality.test_blend_quality_full_user_weight(self)` (`tests/test_user_feedback.py:144`): Test quality blending with full user weight.
  - `TestBlendQuality.test_blend_quality_full_original_weight(self)` (`tests/test_user_feedback.py:155`): Test quality blending with full original weight.
- Classe `TestProcessFeedback` (`tests/test_user_feedback.py:167`): Test suite for process_feedback function.
  - `TestProcessFeedback.test_process_feedback_returns_result(self)` (`tests/test_user_feedback.py:170`): Test that process_feedback returns ProcessedFeedback.
  - `TestProcessFeedback.test_process_feedback_updates_bandit(self)` (`tests/test_user_feedback.py:200`): Test that process_feedback calls bandit_update.
- Classe `TestUserFeedbackRequest` (`tests/test_user_feedback.py:227`): Test suite for UserFeedbackRequest validation.
  - `TestUserFeedbackRequest.test_valid_thumbs_up(self)` (`tests/test_user_feedback.py:230`): Test valid thumbs up request.
  - `TestUserFeedbackRequest.test_valid_rating(self)` (`tests/test_user_feedback.py:242`): Test valid rating request.
  - `TestUserFeedbackRequest.test_rating_out_of_range(self)` (`tests/test_user_feedback.py:255`): Test that rating must be 1-5.
  - `TestUserFeedbackRequest.test_quality_out_of_range(self)` (`tests/test_user_feedback.py:268`): Test that quality must be 0-10.

## `tests/test_utils.py`

Resumo do arquivo: Módulo `tests/test_utils.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_pricing_logic()` (`tests/test_utils.py:7`): Testa pricing logic.
- `test_pricing_fallback()` (`tests/test_utils.py:15`): Testa pricing fallback.
- `test_settings_defaults()` (`tests/test_utils.py:21`): Garante que configurações críticas têm defaults.

## `tests/test_vectorstore_extra.py`

Resumo do arquivo: Módulo `tests/test_vectorstore_extra.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `test_vectorstore_helpers_and_collection_name(monkeypatch)` (`tests/test_vectorstore_extra.py:12`): Testa vectorstore helpers and collection name.
- `test_get_or_create_collection_async_with_versioning(monkeypatch)` (`tests/test_vectorstore_extra.py:34`): Testa get or create collection async with versioning.
- `test_insert_embedding_sync_and_query_sync_paths(monkeypatch)` (`tests/test_vectorstore_extra.py:53`): Testa insert embedding sync and query sync paths.
- `test_add_query_reset_and_health(monkeypatch)` (`tests/test_vectorstore_extra.py:140`): Testa add query reset and health.

## `tests/test_vision.py`

Resumo do arquivo: Módulo `tests/test_vision.py`: descreve responsabilidades e integrações deste arquivo.

### Funções de módulo

- `encode_image_optimized(image_path)` (`tests/test_vision.py:22`): Lê a imagem, converte para RGB, redimensiona se necessário e retorna Base64.
- `send_request(image_path)` (`tests/test_vision.py:52`): Envia uma única imagem para o Router.
- `main()` (`tests/test_vision.py:101`): Executa main.

