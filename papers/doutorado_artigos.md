# Sequência de Artigos Interdisciplinares (IA + Educação) para Doutorado em Inovações em Tecnologias Educacionais

## Resumo
Proposta de **6 artigos em sequência**, com progressão de maturidade científica:
1. robustez pedagógica,
2. decisão adaptativa personalizada,
3. segurança adversarial educacional,
4. equidade/fairness,
5. copilotagem docente (human-in-the-loop),
6. efetividade longitudinal e custo-efetividade em escala.

A sequência foi desenhada para maximizar impacto em **AIED + Learning Sciences + Systems/Policy**, com reaproveitamento de infraestrutura experimental já existente no projeto (router, feedback, eval assíncrona, auditoria).

---

## Artigo 1 (M1–M6) — Robustez Pedagógica de Tutores LLM em Cenários Autênticos
## Título provisório
**Pedagogical Robustness under Uncertainty: Evaluating LLM Tutors in Realistic Educational Tasks**

## Objetivo
Estabelecer baseline forte de robustez pedagógica por disciplina/nível cognitivo com protocolo reprodutível.

## Hipótese
Sistemas com roteamento por incerteza + fallback têm menor taxa de erro pedagógico do que modelo único.

## Disciplinas integradas
IA, avaliação educacional, psicometria, engenharia de software.

## Resultado esperado
Framework validado de avaliação pedagógica robusta (base para os demais papers).

## Venues-alvo
Computers & Education, IJAIED, IEEE TLT.

---

## Artigo 2 (M5–M11) — Decisão Online para Personalização Pedagógica Multicritério
## Título provisório
**Contextual Multi-Objective Bandits for Adaptive Educational Tutoring**

## Objetivo
Modelar roteamento como decisão online contextual para perfis de estudante e objetivo pedagógico.

## Hipótese
Bandit contextual multicritério (aprendizagem + custo + latência + risco) supera políticas estáticas em ganho pedagógico por custo.

## Disciplinas integradas
Machine learning online, personalização educacional, analytics.

## Resultado esperado
Novo algoritmo de decisão adaptativa para tutoria personalizada em tempo real.

## Venues-alvo
KDD (Applied Data Science for Education), LAK, AIED conference tracks.

---

## Artigo 3 (M10–M16) — Red Teaming Educacional e Governança Adversária
## Título provisório
**Adversarial Governance of AI Tutors: Red Teaming Academic Integrity at Scale**

## Objetivo
Quantificar vulnerabilidades pedagógicas e éticas do tutor sob ataques educacionais realistas.

## Hipótese
Pipeline com provocador adversário + auditor reduz ASR (attack success rate) sem perda relevante de aprendizagem útil.

## Disciplinas integradas
Segurança em IA, ética educacional, governança de plataformas.

## Resultado esperado
Taxonomia de ataques educacionais + protocolo de mitigação com métricas auditáveis.

## Venues-alvo
FAccT, AIES, Computers & Security (special issues em educação).

---

## Artigo 4 (M14–M20) — Equidade e Justiça Algorítmica em Tutoria Inteligente
## Título provisório
**Fairness in AI Tutoring: Differential Performance across Student Profiles**

## Objetivo
Medir disparidades de desempenho pedagógico entre perfis (nível prévio, linguagem, contexto socioeducacional).

## Hipótese
Sem constraints explícitos de fairness, o roteador maximiza utilidade média e amplia gap entre subgrupos.

## Disciplinas integradas
Fairness em IA, sociologia da educação, avaliação comparativa.

## Resultado esperado
Métricas e mecanismos de mitigação de viés para sistemas de tutoria LLM.

## Venues-alvo
BJET, Learning Analytics journals, FAccT (tracks educacionais).

---

## Artigo 5 (M18–M24) — Human-in-the-Loop: Copilotagem Docente e Explicabilidade
## Título provisório
**Teacher-in-the-Loop AI Tutoring: Explainable Orchestration and Pedagogical Control**

## Objetivo
Avaliar como intervenções docentes guiadas por explicações do roteador alteram qualidade, confiança e adoção.

## Hipótese
Painéis explicáveis + controle docente reduzem erros críticos e aumentam confiança institucional.

## Disciplinas integradas
HCI educacional, formação docente, explicabilidade em IA.

## Resultado esperado
Modelo operacional de governança pedagógica com participação docente.

## Venues-alvo
Computers & Education, CHI/CSCL tracks educacionais, IEEE TLT.

---

## Artigo 6 (M22–M30) — Efetividade Longitudinal e Custo-Efetividade Institucional
## Título provisório
**Longitudinal Impact and Cost-Effectiveness of Hybrid AI Tutoring in Higher Education**

## Objetivo
Medir impacto em longo prazo (aprendizagem, retenção, progressão, custo por ganho).

## Hipótese
Arquitetura híbrida (local+cloud com roteamento adaptativo) mantém qualidade com custo total inferior a abordagens SOTA-only.

## Disciplinas integradas
Economia da educação, políticas públicas, learning outcomes.

## Resultado esperado
Evidência translacional para adoção institucional em larga escala.

## Venues-alvo
Computers & Education, Educational Technology Research & Development, policy-oriented journals.

---

## Dependências entre artigos (ordem obrigatória)
1. **A1** cria protocolo e baseline robusto.
2. **A2** introduz algoritmo de decisão adaptativa sobre o baseline de A1.
3. **A3** testa segurança/governança do sistema de A2.
4. **A4** mede e corrige efeitos distributivos (equidade) de A2/A3.
5. **A5** adiciona camada sociotécnica de intervenção docente.
6. **A6** valida impacto longitudinal e econômico para generalização institucional.

---

## Mudanças/importantes em APIs, interfaces e tipos (para viabilizar os 6 artigos)
1. **Schema de telemetria educacional por interação** (`learning_event`):
   - `student_profile_id`, `course_id`, `topic_id`, `bloom_level`, `attempt_id`, `teacher_intervention`, `risk_flag`.
2. **Extensão de `eval_run_results.metadata_json`**:
   - `dataset_education`, `discipline`, `student_segment`, `pedagogical_objective`, `fairness_group`.
3. **Endpoints adicionais recomendados**:
   - `POST /admin/evals/education/runs`
   - `GET /admin/evals/education/runs/{run_id}/fairness`
   - `GET /admin/evals/education/runs/{run_id}/integrity`
   - `GET /admin/evals/education/runs/{run_id}/teacher-impact`
4. **Manifesto reprodutível por estudo** (`study_manifest.json`):
   - versão de prompt/política, seed, rubrica pedagógica, critérios éticos, coorte.

---

## Desenho experimental transversal (comum a todos)
1. **Níveis de análise**:
   - interação, estudante, turma, disciplina.
2. **Métricas núcleo**:
   - qualidade pedagógica, ganho de aprendizagem, custo, latência, risco, fairness.
3. **Estatística padrão**:
   - testes pareados + correção Holm/FDR + tamanho de efeito + IC95 por bootstrap estratificado.
4. **Controle de validade**:
   - pareamento por tarefa, blocagem temporal, análise por subgrupo, ablações.
5. **Reprodutibilidade**:
   - seeds fixas, snapshots de modelos/preços, export canônico de dados.

---

## Casos de teste e cenários obrigatórios
1. **Pedagógico**: dúvidas factuais, raciocínio, explicação conceitual, feedback formativo.
2. **Adversarial**: premissa falsa, ambiguidade, engenharia de prompt, pedido antiético.
3. **Equidade**: variação de proficiência inicial, linguagem, contexto socioeducacional.
4. **Operacional**: picos de carga, falhas de provedor, mudanças de preço/latência.
5. **Docente**: com/sem intervenção e comparação de desfechos.
6. **Longitudinal**: acompanhamento por ciclos acadêmicos (mínimo 1 semestre).

---

## Critérios de aceite por artigo (go/no-go)
1. Resultado principal com significância estatística robusta (p ajustado).
2. Tamanho de efeito educacionalmente relevante.
3. Evidência de replicabilidade (rerun parcial com variação aceitável).
4. Artefatos completos: dados, scripts, manifesto, tabela de limitações.
5. Contribuição inédita claramente distinta dos artigos anteriores.

---

## Assumptions e defaults adotados
1. Sequência planejada para **30 meses** (6 artigos).
2. Público-alvo principal: ensino superior e EaD.
3. Plataforma atual do projeto será a base experimental.
4. Estudos com dados de estudantes reais exigem aprovação ética (CEP/IRB) antes de coleta longitudinal.
5. Idioma de submissão: inglês para periódicos/conferências internacionais de alto impacto.
