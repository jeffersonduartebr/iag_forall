# Roadmap de Refatoração — Limite de 200 SLOC

O projeto impõe **máx. 200 SLOC por arquivo** (500 até set/2026, 300 até set/2026) (`app/app/` e `tests/`), verificado por
`scripts/check_file_length.py` no CI e no pre-commit. Os arquivos que já excediam o
limite foram **congelados** (grandfathered) em `scripts/sloc_baseline.json` no modo
ratchet: eles **não podem crescer**, apenas encolher.

**SLOC = linhas lógicas** (um `NEWLINE` por statement via `tokenize`), contadas no
arquivo em disco. Linhas em branco, comentários e continuações (um statement quebrado
em várias linhas físicas) **não contam**. Isso torna a métrica **invariante ao
`ruff format`** — que só muda a quebra física — então os dois hooks de pre-commit
(`ruff-format` e este) nunca mais entram em conflito. O check é Python puro (não precisa
de ruff em tempo de check). O baseline é gravado sobre a contagem lógica do arquivo
**formatado** (`--init`/`--update`), de modo que um `ruff format` posterior nunca
estoura o teto (formatar só *divide* one-liners compostos, nunca funde).

> **Nota histórica:** ao migrar de contagem *física* para *lógica* (jul/2026), os
> violadores caíram de **16 → 5 arquivos** — a maioria das "violações" era apenas
> quebra de linha, não complexidade real.

## Como reduzir um arquivo do baseline
1. Extrair coesão em módulo(s) novo(s) ≤200 SLOC, mantendo a API pública via reexport.
2. Rodar `PYTHONPATH=app pytest -q tests -m "not integration and not slow"`.
3. Rodar `python3 scripts/check_file_length.py --update` (com `ruff` disponível) e
   commitar `sloc_baseline.json`.
4. Quando o arquivo cair para ≤200, sua entrada some do baseline automaticamente.

## Violadores atuais (SLOC lógico, prioridade por risco)

Limite reduzido de 300 para **200** em set/2026, junto com a complexidade ciclomática
(de ≤ 20 para **CC < 15**, `ruff` C901). Os arquivos abaixo ficaram congelados no
baseline: **44 arquivos, 2874 SLOC de excesso**. O terceiro portão,
CRAP ≤ 30 por função (`scripts/crap_report.py`), **não tem grandfathering nenhum** —
`crap_baseline.json` é `{}`, logo o limite já vale para todas as funções do repo.

O baseline é uma dívida declarada, não uma isenção: nenhum destes arquivos pode
crescer, e cada um que encolher aperta o próprio teto.

| Arquivo | SLOC | Excesso | Estratégia de divisão sugerida |
|---|---:|---:|---|
| `app/app/config/settings_properties.py` | 382 | +182 | Declarativo (propriedades tipadas): baixa prioridade. |
| `app/app/provider_tools.py` | 378 | +178 | Um módulo por formato de provider (OpenAI, Anthropic, Gemini), mantendo o reexport. |
| `app/app/providers/_ollama.py` | 364 | +164 | Separar montagem de payload, parse de resposta/tool calls e política de retry. |
| `app/app/settings_dynamic.py` | 326 | +126 | Separar o listener de hot-reload (pub/sub) e o prime em lote do cache. |
| `app/app/roadmap_features.py` | 325 | +125 | Seguir o padrão de `roadmap_experts.py`: eval runs, políticas e RBAC em módulos próprios. |
| `app/app/middleware/rate_limit.py` | 324 | +124 | Separar `RateLimitStore` (Redis/memória) do middleware. |
| `app/app/reliability.py` | 322 | +122 | Separar circuit breakers, deduplicação e detector de cascata. |
| `app/app/schemas.py` | 304 | +104 | Declarativo (modelos Pydantic): dividir por domínio se crescer. |
| `app/app/main.py` | 295 | +95 | Mover lifespan (startup/shutdown) e middlewares para `app/app/bootstrap/`. |
| `app/app/openrouter_explorer.py` | 284 | +84 | — |
| `app/app/bandits.py` | 284 | +84 | — |
| `app/app/providers/_infra.py` | 272 | +72 | — |
| `app/app/query_jobs.py` | 269 | +69 | — |
| `app/app/judges.py` | 268 | +68 | — |
| `app/app/nsga_weights_updater.py` | 267 | +67 | — |
| `app/app/router_core.py` | 249 | +49 | — |
| `app/app/semantic_cache.py` | 246 | +46 | Separar a camada L1 em memória da busca L2 no Chroma. |
| `app/app/ab_testing.py` | 245 | +45 | — |
| `app/app/rag_local.py` | 242 | +42 | — |
| `app/app/model_registry.py` | 235 | +35 | — |
| `app/app/observability.py` | 229 | +29 | — |
| `app/app/services/router_stages.py` | 228 | +28 | — |
| `app/app/online_predictor.py` | 224 | +24 | — |
| `app/app/vectorstore.py` | 223 | +23 | — |
| `app/app/openrouter_exploration_policy.py` | 222 | +22 | — |
| `app/app/services/feedback_stages.py` | 216 | +16 | — |
| `app/app/services/centroid_store.py` | 211 | +11 | — |
| `app/app/services/query_runtime.py` | 207 | +7 | — |
| `app/app/services/query_http.py` | 203 | +3 | — |

**Testes** (15 arquivos). Um arquivo de teste grande divide-se por componente,
não por classe — o objetivo é que uma falha aponte para um assunto só.

| Arquivo | SLOC | Excesso |
|---|---:|---:|
| `tests/test_chaos.py` | 304 | +104 |
| `tests/test_providers_ensure_ollama.py` | 292 | +92 |
| `tests/locustfile.py` | 291 | +91 |
| `tests/test_providers_async_extra.py` | 275 | +75 |
| `tests/test_schemas.py` | 273 | +73 |
| `tests/test_tool_calling.py` | 263 | +63 |
| `tests/test_router_execution_impl.py` | 263 | +63 |
| `tests/test_judge_usurpation.py` | 259 | +59 |
| `tests/test_semantic_cache_extra.py` | 255 | +55 |
| `tests/test_judges_extra.py` | 255 | +55 |
| `tests/test_performance.py` | 254 | +54 |
| `tests/test_query_runtime.py` | 232 | +32 |
| `tests/test_main_lifecycle.py` | 210 | +10 |
| `tests/test_settings_dynamic_runtime.py` | 203 | +3 |
| `tests/test_embeddings.py` | 201 | +1 |

### ✅ Concluídos (ramo refactor/perf-quality, set/2026)
- `bandits.py` 534→281 → `services/bandit_stats_store.py`, `services/centroid_store.py`.
- `openrouter_explorer.py` 606→284 → `openrouter_exploration_{policy,state}.py`, `openrouter_shadow.py`.
- `providers/_implementations.py` 424→6 → um módulo por provider em `providers/`.
- `judges.py` 486→242 → `services/judge_{calibration,cache,selection,context}.py`.
- `nsga_weights_updater.py` 388→266 → `nsga_core.py`, `nsga_calibration.py`.
- `rag_local.py` 301→242 → `services/retrieval_assembly.py`.
- `services/query_runtime.py` 430→207 → `services/query_profile.py`, `services/query_reliability.py`.
- `services/router_execution.py` e `services/router_feedback.py` → estágios em
  `services/router_stages.py`, `services/router_provider_stage.py`, `services/feedback_stages.py`.
- Testes: `test_router_services_extracted.py` → `test_router_{execution,feedback}_impl.py`;
  `test_providers_reliability.py` → + `test_providers_ollama.py`;
  `test_query_runtime.py` → + `test_query_profile_runtime.py`.

### ✅ Concluídos (roadmap #19)
- `nsga_weights_updater.py` 561→370 → `services/nsga_tuning.py` + `services/nsga_metrics.py`.
- `settings_dynamic.py` 708→332 → `config/settings_properties.py` (mixin) + `config/settings_env.py`.
- `providers_async.py` 1168→170 → pacote `providers/{_infra,_ollama,_implementations}.py`;
  facade com `__getattr__` (PEP 562) reexporta tudo. Símbolos test-patchados roteados
  via `_pa.` para preservar os `monkeypatch(pa, ...)` sem alterar testes.

> Saíram do baseline com a métrica lógica (agora ≤500): `observability.py`,
> `roadmap_features.py`, `judges.py`, `services/query_runtime.py`,
> `services/router_execution.py`, `router_core.py`, `main.py`, `reliability.py` e os
> testes grandes. Continuam alvos de coesão/legibilidade, mas não bloqueiam mais o CI.
