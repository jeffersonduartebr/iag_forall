# Roadmap de Refatoração — Limite de 500 SLOC

O projeto impõe **máx. 500 SLOC por arquivo** (`app/app/` e `tests/`), verificado por
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
1. Extrair coesão em módulo(s) novo(s) ≤500 SLOC, mantendo a API pública via reexport.
2. Rodar `PYTHONPATH=app pytest -q tests -m "not integration and not slow"`.
3. Rodar `python3 scripts/check_file_length.py --update` (com `ruff` disponível) e
   commitar `sloc_baseline.json`.
4. Quando o arquivo cair para ≤500, sua entrada some do baseline automaticamente.

## Violadores atuais (SLOC lógico, prioridade por tamanho)

| Arquivo | SLOC | Estratégia de divisão sugerida |
|---|---:|---|
| `app/app/openrouter_explorer.py` | 606 | Separar seleção/pool, promoção automática, e persistência (stats Redis/DB). |
| `app/app/bandits.py` | 577 | Separar algoritmos (epsilon-greedy/UCB1/Thompson) da meta-política e do estado/persistência. |

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
