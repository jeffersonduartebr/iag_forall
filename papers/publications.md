# Plano de Submissão (24 Meses) — Linha de Papers em Aprendizagem Online/Decisão

## Resumo
Plano para **4 papers de alto impacto** em decisão online, encadeados por dependência científica e de artefatos.
Ordem proposta:
1. Safe MO Contextual Bandit (base algorítmica)
2. Non-Stationary/Drift-Aware Routing
3. Risk-Aware Bandit com Guardrails/Governança
4. Meta-Policy (bandit sobre bandits)

Objetivo: produzir uma trilha coerente de publicações com incrementalidade clara, reutilização de infraestrutura experimental e resultados publicáveis em venues A/A*.

## 1. Carteira de Papers (escopo final)

## Paper 1 — Safe Multi-Objective Contextual Bandit under Budget Constraints
- **Pergunta**: Como otimizar qualidade mantendo restrições de custo/latência em tempo real?
- **Contribuição principal**:
  1. Reward vetorial (qualidade, custo, latência).
  2. Otimização online com constraints (budget feasibility).
  3. Regret multiobjetivo com violação de orçamento como métrica adicional.
- **Baseline**:
  1. Epsilon-greedy
  2. UCB
  3. Thompson
  4. Política fixa
- **Entregável científico**: algoritmo + prova/argumento de convergência empírica + benchmark reproducível.

## Paper 2 — Drift-Aware Online Routing for Non-Stationary LLM Ecosystems
- **Pergunta**: Como manter desempenho quando distribuição/fornecedores mudam?
- **Contribuição principal**:
  1. Detecção de drift no contexto + qualidade/custo.
  2. Mecanismo de adaptação (windowing, decay adaptativo, reset parcial).
  3. Métrica de recovery time pós-shock.
- **Baseline**:
  1. Bandit estacionário
  2. SW-UCB/Discounted-UCB (ou equivalentes)
- **Entregável científico**: protocolo de stress (shocks controlados) + análise robusta de recuperação.

## Paper 3 — Risk-Aware Online Decision with Guardrails and Governance Constraints
- **Pergunta**: Como incorporar segurança/compliance na decisão online sem colapsar utilidade?
- **Contribuição principal**:
  1. Penalização de risco dinâmica no reward.
  2. Restrições de guardrail/quota como constraints de decisão.
  3. Fronteira utilidade vs risco operacional.
- **Baseline**:
  1. MO bandit sem risco
  2. Heurísticas estáticas de bloqueio
- **Entregável científico**: avaliação adversarial + análise de incidentes evitados.

## Paper 4 — Meta-Policy Selection: A Bandit-over-Bandits for Adaptive Routing
- **Pergunta**: Qual política de decisão usar em cada contexto?
- **Contribuição principal**:
  1. Meta-controlador que escolhe política base (UCB/TS/MO-safe/risk-aware).
  2. Sinais de contexto/instabilidade/incerteza para seleção de política.
  3. Generalização cross-domain.
- **Baseline**:
  1. Melhor política fixa global
  2. Seleção heurística
- **Entregável científico**: ganho de robustez e menor sensibilidade a hiperparâmetros.

## 2. Cronograma de Submissão (24 meses)

## Fase A (M1–M6): Paper 1
1. M1–M2: formalização do algoritmo + desenho experimental congelado.
2. M3–M4: execução principal + ablações.
3. M5: escrita, revisão interna, pacote de artefatos.
4. M6: submissão.

## Fase B (M5–M11): Paper 2 (sobrepõe parcialmente)
1. M5–M6: construção de cenários de drift.
2. M7–M9: experimentos estacionário vs não-estacionário.
3. M10: escrita.
4. M11: submissão.

## Fase C (M10–M17): Paper 3
1. M10–M12: modelagem de risco e integração com guardrails.
2. M13–M15: avaliação adversarial + robustez.
3. M16: escrita.
4. M17: submissão.

## Fase D (M16–M24): Paper 4
1. M16–M19: meta-policy e integração.
2. M20–M22: experimentos cross-domain.
3. M23: escrita final de alto impacto.
4. M24: submissão.

## 3. Venues-alvo (prioridade)
1. **Paper 1**: NeurIPS/ICLR (track datasets&benchmarks/systems) ou KDD.
2. **Paper 2**: WWW, KDD, CIKM, AISTATS (dependendo foco teórico vs aplicado).
3. **Paper 3**: FAccT, AIES, IEEE S&P workshops (AI safety/applications), ou ACL Industry.
4. **Paper 4**: NeurIPS/ICML workshops + conferência principal (KDD/WWW) após maturidade.

## 4. Interface pública e artefatos (decision-complete)

## APIs/Tipos a congelar para os 4 papers
1. `eval_runs` e `eval_run_results` como schema oficial de experimento.
2. Endpoint de significância por run: `GET /admin/evals/runs/{run_id}/significance`.
3. Metadados obrigatórios por amostra:
   1. `policy_id`
   2. `algorithm_id`
   3. `seed`
   4. `dataset`
   5. `task_id`
   6. `replicate_id`
4. Export canônico:
   1. `results_raw.csv`
   2. `results_aggregated.csv`
   3. `stats_report.json`
   4. `experiment_manifest.json` (config e hashes)

## Requisitos de reprodutibilidade
1. Seed global e seed por réplica.
2. Hash de configuração de política.
3. Versionamento de prompts/dataset split.
4. Snapshot de modelos/preços usados na execução.

## 5. Plano experimental transversal (para todos os papers)
1. **Réplicas**: mínimo 10 réplicas por condição.
2. **Pareamento**: comparação por task-id entre políticas.
3. **Testes estatísticos**:
   1. Friedman (global)
   2. Wilcoxon pareado (post-hoc)
   3. Holm (primário), FDR-BH (secundário)
   4. Efeito (Cohen’s d)
4. **Métricas obrigatórias**:
   1. qualidade
   2. custo
   3. latência
   4. utilidade composta
   5. incidentes de risco (paper 3)
   6. tempo de recuperação pós-drift (paper 2)

## 6. Testes e cenários por paper

## Paper 1
1. Budget apertado, médio e folgado.
2. Carga baixa e alta.
3. Ablation de pesos multiobjetivo.

## Paper 2
1. Drift abrupto de custo.
2. Drift de qualidade por provedor.
3. Mudança de distribuição de tarefas.

## Paper 3
1. Prompt injection e exfiltration.
2. Conteúdo sensível com guardrail.
3. Quotas por tenant/usuário sob estresse.

## Paper 4
1. Domínios heterogêneos (QA, raciocínio, código).
2. Mudança de domínio no meio da execução.
3. Robustez a hiperparâmetros.

## 7. Riscos e mitigação
1. **Risco**: dependência de APIs externas muda custo/latência.
- **Mitigação**: janelas de execução balanceadas + snapshot de preço por run.

2. **Risco**: variância alta entre réplicas.
- **Mitigação**: mais réplicas, bootstrap estratificado, intervalos de confiança.

3. **Risco**: contribuição parecer incremental demais.
- **Mitigação**: separar claramente contribuição algorítmica por paper e incluir ablações fortes.

4. **Risco**: sobreposição excessiva entre papers.
- **Mitigação**: escopo e hipótese exclusivos por paper; tabelas de novelty matrix no apêndice interno.

## 8. Critérios de “go/no-go” por submissão
1. Pelo menos 1 ganho estatisticamente significativo em métrica primária vs baseline forte.
2. Tamanho de efeito não-trivial (definir limiar interno por paper, ex.: |d| ≥ 0.2).
3. Reprodutibilidade validada (rerun parcial com desvio dentro de faixa aceitável).
4. Artefatos completos e executáveis (manifest + scripts + outputs).

## 9. Assumptions e defaults adotados
1. Horizonte de planejamento: **24 meses**.
2. Número de papers: **4**.
3. A infraestrutura atual do projeto é base experimental principal.
4. O paper de metodologia já está concluído e não entra como objetivo desta trilha.
5. Prioridade de impacto: inovação algorítmica + validação em sistema real.
