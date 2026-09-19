# Roadmap de Refatoração — Limite de 300 SLOC

O projeto impõe **máx. 300 SLOC por arquivo** (era 500 até set/2026) (`app/app/` e `tests/`), verificado por
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
1. Extrair coesão em módulo(s) novo(s) ≤300 SLOC, mantendo a API pública via reexport.
2. Rodar `PYTHONPATH=app pytest -q tests -m "not integration and not slow"`.
3. Rodar `python3 scripts/check_file_length.py --update` (com `ruff` disponível) e
   commitar `sloc_baseline.json`.
4. Quando o arquivo cair para ≤300, sua entrada some do baseline automaticamente.

## Violadores atuais (SLOC lógico, prioridade por risco)

Limite reduzido de 500 para 300 em set/2026 (ramo `refactor/perf-quality`); os arquivos
abaixo ficaram congelados no baseline. Portões relacionados: complexidade ciclomática
≤ 20 por função (`ruff` C901) e CRAP ≤ 30 por função (`scripts/crap_report.py`, com
ratchet em `scripts/crap_baseline.json`).

| Arquivo | SLOC | Estratégia de divisão sugerida |
|---|---:|---|
| `app/app/provider_tools.py` | 378 | Um módulo por formato de provider (OpenAI, Anthropic, Gemini), mantendo o reexport. |
| `app/app/providers/_ollama.py` | 364 | Separar montagem de payload, parse de resposta/tool calls e política de retry. |
| `app/app/main.py` | 357 | Mover lifespan (startup/shutdown) e middlewares para `app/app/bootstrap/`. |
| `app/app/roadmap_features.py` | 325 | Seguir o padrão de `roadmap_experts.py`: eval runs, políticas e RBAC em módulos próprios. |
| `app/app/middleware/rate_limit.py` | 324 | Separar `RateLimitStore` (Redis/memória) do middleware. |
| `app/app/settings_dynamic.py` | 326 | Separar o listener de hot-reload (pub/sub) e o prime em lote do cache. |
| `app/app/reliability.py` | 322 | Separar circuit breakers, deduplicação e detector de cascata. |
| `tests/test_chaos.py` | 304 | Dividir por componente (breakers, rate limit, dedup, timeouts). |
| `app/app/config/settings_properties.py` | 382 | Declarativo (propriedades tipadas): baixa prioridade. |
| `app/app/schemas.py` | 304 | Declarativo (modelos Pydantic): dividir por domínio se crescer. |

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
