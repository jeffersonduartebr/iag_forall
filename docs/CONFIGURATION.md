# Guia de Configuração

## Princípios
1. Configuração em camadas: env -> Redis -> DB -> defaults.
2. Parte das configurações é dinâmica (hot-reload) via `settings_dynamic`.
3. Configurações críticas devem ser explícitas em produção.

## Variáveis críticas (obrigatórias em produção)
- `ADMIN_TOKEN`
- `DB_PASS`
- `MYSQL_ROOT_PASSWORD`
- `REDIS_PASSWORD`

## Variáveis recomendadas para governança (novas)
- `AB_TESTING_ENABLED`
- `REQUEST_TIMEOUT_SECONDS`
- `MAX_CONCURRENT_REQUESTS`
- `BACKPRESSURE_ENABLED`

## Banco e cache
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASS`, `DB_NAME`
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`

## Providers
- `OLLAMA_HOST`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `OLLAMA_CONCURRENCY_LIMIT`

## Roteamento e qualidade
- `MAX_TOKENS_DEFAULT`, `TEMPERATURE_DEFAULT`
- `REQUEST_TIMEOUT_SECONDS`, `REQUEST_DEDUP_ENABLED`
- `BANDIT_EPSILON`
- `NSGA_W_QUALITY`, `NSGA_W_LATENCY`, `NSGA_W_COST`, `NSGA_W_ALIGNMENT`

## Resiliência
- `CIRCUIT_BREAKER_FAIL_MAX`, `CIRCUIT_BREAKER_RESET_TIMEOUT`
- `CIRCUIT_BREAKER_LOCAL_FAIL_MAX`, `CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT`
- `BACKPRESSURE_ENABLED`, `MAX_CONCURRENT_REQUESTS`
- `ADAPTIVE_TIMEOUT_ENABLED`, `MIN_TIMEOUT`, `MAX_TIMEOUT`

## Cache e RAG
- `CACHE_THRESHOLD`, `CACHE_TTL_DAYS`
- `CACHE_THRESHOLD_ADAPT_ENABLED`, `CACHE_HIT_RATE_TARGET`
- `RAG_DATA_DIR`, `RERANK_ENABLED`, `RERANK_MODEL`

## Quais mudanças são dinâmicas?
Regra prática:
1. Configuração lida via `settings_dynamic.get(...)` no caminho de execução tende a ser dinâmica.
2. Configuração capturada apenas no import do módulo pode exigir restart.

## Alteração em runtime
Exemplo:
```bash
curl -X PUT http://localhost:8000/admin/settings \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"BANDIT_EPSILON":"0.10"}'
```

## Governança e cota por tenant (MVP)
Além de settings dinâmicos globais, o sistema agora suporta:
1. Limite diário e mensal por `tenant_id`.
2. Acúmulo de uso mensal (`requests`, `tokens`, `cost_usd`).
3. Bloqueio de requisição quando limite é excedido.
4. RBAC simples por usuário (`rbac_user_roles`) para governança, política e eval.
5. Execução assíncrona de eval via Celery (`task_execute_eval_run`).
6. Relatório de significância estatística por run de avaliação.

Observação:
- A governança por tenant é aplicada quando `tenant_id` é enviado no `POST /query`.

## Perfis sugeridos
### Perfil mais barato
- Aumentar peso de custo (`NSGA_W_COST`).
- Reduzir exploração (`BANDIT_EPSILON`).

### Perfil mais rápido
- Aumentar peso de latência (`NSGA_W_LATENCY`).
- Ajustar `MAX_CONCURRENT_REQUESTS` e timeout.

### Perfil mais conservador
- Timeout mais alto para modelos complexos.
- Circuit breaker mais sensível.

## Checklist de produção
1. Segredos não versionados.
2. Health checks e métricas ativos.
3. Limites de concorrência calibrados.
4. Alertas básicos em erro/latência.
