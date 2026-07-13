# Roadmap de Refatoração — Limite de 500 SLOC

O projeto impõe **máx. 500 SLOC por arquivo** (`app/app/` e `tests/`), verificado por
`scripts/check_file_length.py` no CI e no pre-commit. Os arquivos que já excediam o
limite foram **congelados** (grandfathered) em `scripts/sloc_baseline.json` no modo
ratchet: eles **não podem crescer**, apenas encolher.

Este documento orienta a redução gradual desse baseline — cada divisão é uma **PR
isolada**, com a suíte verde e `python3 scripts/check_file_length.py --update`
baixando o teto do arquivo após a redução.

## Como reduzir um arquivo do baseline
1. Extrair coesão em módulo(s) novo(s) ≤500 SLOC, mantendo a API pública via reexport.
2. Rodar `PYTHONPATH=app pytest -q tests -m "not integration and not slow"`.
3. Rodar `python3 scripts/check_file_length.py --update` e commitar `sloc_baseline.json`.
4. Quando o arquivo cair para ≤500, sua entrada some do baseline automaticamente.

## Violadores atuais (prioridade por tamanho)

| Arquivo | SLOC | Estratégia de divisão sugerida |
|---|---:|---|
| `app/app/providers_async.py` | 1532 | Pacote `providers/`: `base` (BaseProvider, LLMResponse, factory), `openai`, `anthropic`, `gemini`, `ollama`, `http` (client/retry/circuit breaker/timeout). Reexportar `call_model`, `ProviderFactory`, `LLMResponse` de `providers_async` para não quebrar imports. |
| `app/app/roadmap_features.py` | 805 | ✅ Diretório de especialistas extraído para `roadmap_experts.py` (1046→805, baseline ratcheteado). Próximo: submódulos de governança `governance/budgets.py`, `governance/policies.py`, `governance/rbac.py`, `governance/audit.py`, `governance/reviews.py`. |
| `app/app/observability.py` | 917 | Separar `metrics_defs` (definições Prometheus) de `logging_setup` (structlog/render) e `helpers`. |
| `app/app/settings_dynamic.py` | 858 | Extrair grupos de properties (providers, routing, cache/RAG, resiliência) em mixins/módulos por domínio. |
| `app/app/openrouter_explorer.py` | 856 | Separar seleção/pool, promoção automática, e persistência (stats Redis/DB). |
| `app/app/bandits.py` | 841 | Separar algoritmos (epsilon-greedy/UCB1/Thompson) da meta-política e do estado/persistência. |
| `app/app/nsga_weights_updater.py` | 784 | Separar setup DEAP/otimização do tuning de UQ e do tuning de estratégia. |
| `app/app/judges.py` | 727 | Já há `judges/` (heuristic/llm/judge); mover o restante de `judges.py` para lá e reexportar. |
| `app/app/services/query_runtime.py` | 682 | Extrair `reliability enrichment` e `runtime profile` para módulos próprios em `services/`. |
| `app/app/services/router_execution.py` | 551 | Extrair montagem de candidatos e o bloco de RAG/prompt. |
| `app/app/router_core.py` | 520 | Mover o mapa de deps (`_build_route_deps`) e wiring de background para `services/`. |
| `app/app/main.py` | 514 | Extrair helpers de lifecycle/warmup e o `v1_router` para módulos dedicados. |
| `app/app/reliability.py` | 507 | Separar circuit breakers de deduplicação e fallback. |
| `tests/test_router_services_extracted.py` | 729 | Dividir por subsistema testado. |
| `tests/test_query_runtime.py` | 522 | Dividir em `test_query_runtime_profile.py` e `test_query_runtime_process.py`. |
| `tests/test_providers_reliability.py` | 515 | Dividir por provedor/cenário. |

> Nota: `app/app/provider_tools.py` (novo, tool calling) está em ~500 SLOC — manter
> abaixo do limite ao evoluí-lo.
