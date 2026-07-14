# Roadmap de Produto — `iag_forall`

Roadmap de evolução do sistema, numerado na mesma convenção dos commits (`roadmap #N`).
Os itens **#1–#12** (trilha de *tool-calling*) estão **concluídos**; os novos itens começam
em **#13**. Cada item vira uma **PR isolada** com a suíte verde, e o commit referencia o
número (`feat(...): ... (roadmap #13)`).

**Legenda de status:** ✅ concluído · 🔜 próximo (curto prazo) · ⏳ planejado · 🧭 contínuo

---

## Contexto: as duas trilhas do projeto

O sistema amadureceu em duas frentes que ainda não se encontraram:

1. **Plataforma de produção** — API de roteamento (NSGA-II + bandits + UQ), RAG multimodal,
   tool-calling, governança por tenant, OpenAI-compat e `admin-ui/` (React). Madura.
2. **Aparato de pesquisa** (tese *Universidade Sintética*, ver `PUBLICATIONS_ROADMAP.md`) —
   existe, mas como scripts soltos na raiz: `adversarial_university_full_suite.py`
   (Provocador + matriz 30×30 + Kruskal-Wallis + Spearman + ANOVA), gravando em
   `thesis_results/` local, **fora** da infra de `eval_runs` / `experiment_manifest` /
   `admin-ui`.

O objetivo central deste roadmap é **fazer as duas trilhas convergirem**: produtizar o
aparato de pesquisa sobre a infra que já existe e fechar as lacunas da visão da tese, sem
parar o hardening da plataforma.

---

## ✅ Concluído (#1–#12) — trilha de *tool-calling*

| # | Entrega | Commit |
|--:|---------|--------|
| 1 | Streaming real de tokens do provider em `/query/stream` | `14e6098` |
| 2 | Reward + métricas para turnos de tool-calling | `466f022` |
| 3 | Unificação da fonte de verdade de *model capabilities* | `0a6610e` |
| 4 | Accounting preciso de token/custo em turnos com ferramentas | `c3f273f` |
| 5 | `response_format` / JSON mode entre providers | `6296752` |
| 6 | Harness de eval de tool-calling estilo BFCL | `26c7391` |
| 7 | Pass-through e roteamento de *native / server-side tools* | `725af90` |
| 8 | Extração do diretório de especialistas (`roadmap_experts.py`) | `e185bfa` |
| 9 | Correção do anti-padrão de isolamento de módulo em testes | `3ad2461` |
| 10 | Migração de campos opcionais Pydantic para forma `Annotated` | `cdc13d3` |
| 11 | Governança por tenant: allowlist + cost cap + audit | `5effef9` |
| 12 | Testes de integração + contrato para tool-calling | `33c21a2` |

---

## 🔜 Trilha A — Produtizar a pesquisa (curto prazo)

> Maior alavancagem: o experimento que sustenta a publicação (N=900) roda hoje como script
> à parte. Trazê-lo para dentro da plataforma dá **reprodutibilidade, tenancy e auditoria**
> imediatas, reaproveitando infra que já existe.

### #13 — Orquestrador de suíte adversária como serviço 🔜
- **Objetivo:** portar `adversarial_university_full_suite.py` para `app/app/services/`
  gravando resultados em `eval_runs` / `eval_run_results` (já em `roadmap_features.py`),
  amarrado ao `services/experiment_manifest.py` (git hash + snapshot de config).
- **Estado atual:** script na raiz escreve em `thesis_results/` local; sem retomada,
  sem tenancy, sem trilha de auditoria.
- **Entregáveis:** `services/adversarial_suite.py` (geração Provocador → Tutor → Auditor),
  persistência via `create_eval_run` / `add_eval_result`, execução assíncrona via Celery.
- **Critério de pronto:** rodar a matriz 30×30 por API produz um `eval_run` retomável e
  auditável; script legado passa a ser um *thin wrapper* que chama o serviço.
- **Depende de:** — (base já existe)

### #14 — Endpoints + painel de red-teaming no admin-ui 🔜
- **Objetivo:** expor `POST /admin/experiments/adversarial` e um `AdversarialSuitePanel.tsx`
  (ao lado de `ExpertKappaPanel.tsx`) com ASR por tópico, matriz 30×30 e testes estatísticos.
- **Estado atual:** resultados só existem como PNGs/CSVs em `thesis_results/`.
- **Entregáveis:** rotas em `api/eval_routes.py` (ou novo `api/experiment_routes.py`),
  componente React + gráfico (reusar `ExplorationScatterChart.tsx` como base).
- **Critério de pronto:** operador dispara e acompanha um run de red-teaming pelo painel,
  com ASR e p-values renderizados.
- **Depende de:** #13.

### #15 — Consolidar a camada estatística em `academic_stats` ✅
- **Objetivo:** unificar a estatística que estava espalhada. Kruskal-Wallis/ANOVA viviam em
  `adversarial_university_full_suite.py`; Spearman estava duplicado em
  `services/uq_calibration.py`; um `_welch_ttest` morto em `roadmap_features.py`.
- **Entregue:** `services/academic_stats.py` ganhou `spearman`, `kruskal_wallis` e
  `anova_oneway` (fonte única, com fallback `scipy_unavailable`). `uq_calibration`
  passou a delegar; `_welch_ttest` morto removido; a suíte adversária consome
  `academic_stats` e **não depende mais de `statsmodels`**.
- **Testes:** `tests/test_academic_stats.py` (spearman/kruskal/anova + casos de amostra
  insuficiente). Suíte verde.

---

## ⏳ Trilha B — Fechar a visão da tese (médio prazo)

### #16 — Integração Moodle real ⏳
- **Objetivo:** conector bidirecional com Moodle (webhook de entrada + push de resposta),
  o ambiente de teste definido em `PUBLICATIONS_ROADMAP.md`.
- **Estado atual:** `services/query_webhooks.py` é um callback genérico de job assíncrono,
  **não** específico do Moodle.
- **Entregáveis:** `services/moodle_connector.py` (verificação de assinatura, mapeamento
  curso/tópico → tenant/tema), rota de ingestão, entrega da resposta do Tutor.
- **Critério de pronto:** pergunta postada no Moodle percorre Tutor→Auditor e retorna
  ao fórum; interação registrada com `correlation_id`.
- **Depende de:** #13 (loop orquestrado).

### #17 — Loop de governança adversária online (fechado) ✅
- **Correção de premissa:** o loop **Tutor → Auditor → aprendizado online (River + bandits)**
  **já estava fechado para tráfego orgânico** em `services/router_feedback.py`. A lacuna real
  era o **lado adversário desconectado** e a **falta de memória por cluster + escalonamento por risco**.
- **Entregue:** `services/adversarial_governance.py` com três capacidades:
  1. **Memória de risco por cluster** de conhecimento (Redis + fallback em memória):
     `record_adversarial_outcome` / `get_cluster_risk` (ASR por cluster, flag `high_risk`).
  2. **Fechamento do loop no lado adversário:** o verdito do Auditor agora retroalimenta os
     mesmos `bandit_update`/`compute_reward` e retreina o `online_predictor` — igual ao tráfego
     orgânico. Ligado na suíte (`adversarial_university_full_suite.process_duel`).
  3. **Escalonamento por risco:** `suggest_escalation` (cluster `high_risk` **ou** UQ ≥
     `UNCERTAINTY_THRESHOLD` → candidato mais forte), integrado ao seletor em
     `services/router_execution.py` via `advgov_escalate` (helper enxuto p/ respeitar o SLOC).
- **Config:** grupo `adversarial_governance` no catálogo (`ADVGOV_*`), **desligado por padrão**
  (`ADVGOV_ENABLED=0`) — produção intacta até habilitar.
- **Testes:** `tests/test_adversarial_governance.py` (14 casos: memória de cluster, wiring do
  loop, escalonamento por UQ/risco/preferência, gating). Suíte verde.
- **Nota:** quando o serviço do #13 existir, a mesma chamada `record_adversarial_outcome`
  migra do script para o serviço — a API já está pronta para isso.

### #18 — Export do dataset de *Hard Negatives* ⏳
- **Objetivo:** pipeline de exportação versionado/anonimizado das 900 interações
  (contribuição científica declarada), com *datasheet*.
- **Estado atual:** `benchmark_catalog.py` cura entradas com campo `attack_strategy`, mas não
  há export reprodutível.
- **Entregáveis:** exportador JSONL + datasheet (proveniência, git hash, licença) reusando
  `experiment_manifest.py`; anonimização de PII.
- **Critério de pronto:** artefato de dataset reproduzível a partir de um `eval_run`.
- **Depende de:** #13.

---

## 🧭 Trilha C — Hardening de plataforma (contínuo)

### #19 — Backlog de refatoração SLOC 🧭
- **Objetivo:** reduzir os maiores violadores de `docs/SLOC_REFACTOR_ROADMAP.md` (ratchet).
- **Prioridades:** `providers_async.py` (1532 → pacote `providers/`), `observability.py` (917),
  `settings_dynamic.py` (858), e o split de governança de `roadmap_features.py` em
  `governance/{budgets,policies,rbac,audit,reviews}.py`.
- **Critério de pronto:** cada divisão baixa o teto via `check_file_length.py --update`,
  API pública preservada por reexport, suíte verde.
- **Depende de:** — (cada arquivo é uma PR independente).

### #20 — Segurança / hardening ⏳
- **Objetivo:** endurecer superfície de produção.
- **Entregáveis:** rotação de segredos/API keys por tenant, rate-limit no gateway, e revisão
  do `TRUST_HEADER_ROLES` (roles por header quando habilitado — risco se exposto sem gateway
  confiável; ver `roadmap_features.check_access`).
- **Critério de pronto:** checklist de `docs/OPERATIONS.md` estendido e verificado.
- **Depende de:** — .

---

## Ordem sugerida

```
#15 ──▶ #13 ──▶ #14        (Trilha A: base estatística → serviço → painel)
              └▶ #16, #17, #18   (Trilha B, sobre o loop orquestrado)
#19, #20  em paralelo, contínuos (Trilha C)
```

**Ponto de partida recomendado:** #15 + #13 — maior retorno com risco baixo, porque
reaproveita `eval_runs`, `experiment_manifest` e `academic_stats`, e transforma o resultado
científico que já existe em algo rastreável e auditável na própria plataforma.
