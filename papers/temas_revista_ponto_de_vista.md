# Cinco temas de artigo para a *Revista Ponto de Vista* (RPV/UFV)

Proposta de pauta editorial construída a partir de (a) o perfil e as publicações recentes da
*Revista Ponto de Vista* e (b) as capacidades já implementadas neste repositório — o roteador
multiobjetivo de LLMs que serve como **ferramenta/infraestrutura experimental** do trabalho.

> **Nota de método.** O portal `periodicos.ufv.br` está bloqueado pela política de egresso do
> ambiente em que este documento foi produzido; o sumário da edição e o escopo do periódico foram
> obtidos por busca web indireta. Antes da submissão, **confirme o sumário e as normas diretamente
> em <https://periodicos.ufv.br/RPV/issue/view/793>** — a lista abaixo pode estar incompleta
> (seções como "outros olhares", "destaque é" e resenhas podem não ter sido capturadas).

---

## 1. Leitura do periódico

| Dimensão | Observação |
|---|---|
| Título / ISSN | *Revista Ponto de Vista* — ISSN 1983-2656 (online) |
| Editora | COLUNI — Colégio de Aplicação da UFV |
| Escopo | Multidisciplinar, **com foco em ensino e educação**, de todas as áreas do conhecimento |
| Idioma | Português |
| Periodicidade | Semestral |
| Avaliação | Duplo-cega, dois pareceristas; **sem taxas** (nem submissão, nem publicação) |
| Seções | Artigos científicos; artigos de **iniciação científica**; **relatos de experiência**; resenhas; "outros olhares"; "destaque é" |
| Edição analisada | v. 15, n. 1 (2026) — `issue/view/793`, publicada em 01/03/2026 |

### Sumário da edição v. 15, n. 1 (2026)

1. *Ações extensionistas voltadas para educação ambiental através do ensino remoto síncrono* — Macedo, Vieira & Corrêa (p. 01–15)
2. *A educação física da Base Nacional Comum Curricular: reflexões e tensões* — Rezende, Novaes, Telles & Triani (p. 01–19)
3. *Ensino remoto emergencial catalisa inovação pedagógica e reflexão crítica na formação inicial de professores de química* — Rodrigues, Tavares, Matos, Soares & Malheiros (p. 01–14)
4. *Integruno: um jogo matemático para ensino de cálculo integral* — Pinto & Wasques (p. 01–20)
5. *Novo Ensino Médio: condição do trabalho docente de sociologia em Minas Gerais* — Bigão, Oliveira & Diana (p. 01–18)
6. *A formação do ferroviário e a escola profissional ferroviária em Divinópolis-MG (1941–1972)* — Silva & Giarola (p. 01–20)
7. *Trajetórias de estudantes universitários surdos e com deficiência auditiva com a educação física e o esporte* — Duarte, Santos, Oliveira & Abreu (p. 01–18)

### Publicações recentes sobre IA no mesmo periódico

- *Para além dos algoritmos: a formação humanística nos cursos de inteligência artificial no Brasil* (v. 14, n. 2, 2025)
- *Análise de conteúdo sobre inteligência artificial e educação em veículos de imprensa no Brasil*
- *Desafios e perspectivas da educação com o avanço da inteligência artificial*
- *Uma inteligência artificial na educação para além do modelo behaviorista*

### Padrões editoriais inferidos (e o que eles impõem à nossa escrita)

1. **Artigos curtos**: 14 a 21 páginas. Nada de *paper* de conferência de 12 páginas em duas colunas com 6 tabelas de ablação.
2. **Contexto brasileiro e público**: BNCC, Novo Ensino Médio, rede estadual de MG, extensão universitária, formação inicial de professores. O universal abstrato não conversa com esta revista; o município, a escola e a disciplina, sim.
3. **Predomínio qualitativo**: análise documental, análise de conteúdo, relato de experiência, entrevistas. Quantitativo é bem-vindo, mas precisa ser **legível por quem não é da computação**.
4. **A IA já entrou na pauta — mas como objeto de crítica, não como artefato**. Os quatro artigos de IA acima são *sobre* IA (discurso da imprensa, modelo behaviorista, formação humanística). **Nenhum apresenta um sistema construído e medido.** É exatamente aí que este repositório tem algo inédito a oferecer para o periódico: evidência empírica de primeira mão, produzida por um sistema aberto, escrita em português e endereçada a professores.
5. **Consequência editorial**: em todos os cinco temas abaixo, a métrica é meio, não fim. O texto se organiza em torno da pergunta pedagógica; NSGA-II, bandits e ECE aparecem em uma seção metodológica enxuta e num apêndice.

---

## 2. O que o repositório já oferece como base empírica

| Capacidade | Artefato | Uso científico |
|---|---|---|
| Roteamento multiobjetivo (custo, latência, qualidade) | `router_strategy.py`, `nsga_weights_updater.py` | Trade-off explícito e mensurável entre gasto e qualidade |
| Decisão online adaptativa | `bandits.py` (ε-greedy, UCB1, Thompson) | Aprendizagem com o uso, sem re-treino de modelo |
| Quantificação de incerteza + calibração | `services/uq_calibration.py` (ECE, correlação de posto) | "O sistema sabe quando não sabe?" — central para uso escolar |
| Juízes LLM com consenso | `judges.py`, `judges/` | Avaliação automática de qualidade de resposta em escala |
| Governança adversária em malha fechada | `services/adversarial_governance.py` | Memória de risco por *cluster* de conhecimento + escalonamento |
| Portal de especialistas humanos | `services/expert_review.py`, `api/expert_routes.py` | Fila de revisão docente e **kappa de Cohen** humano×IA |
| Estatística acadêmica | `services/academic_stats.py` | Holm-Bonferroni, Cohen's *d*, *bootstrap* CI, kappa |
| Análise de custo contrafactual | `services/roi_analytics.py` | Quanto custaria se tudo fosse para um modelo premium |
| Catálogo de benchmark | `data/benchmark_queries/` — 34 temas, ~4.780 consultas | Disciplinas escolares + `adversarial` + `multimodal` |
| Detecção de complexidade | `services/query_complexity.py` (`simple`→`expert`) | Proxy operacional de demanda cognitiva |
| Reprodutibilidade | `experiment_manifest.json`, seeds, snapshot de preços | Requisito de aceite para qualquer um dos cinco temas |
| Inferência local | Ollama no `docker-compose.yml` | Viabilidade sem depender de API paga — argumento de política pública |

O catálogo de temas já cobre componentes curriculares reconhecíveis pela escola brasileira:
História, História do Brasil, Geografia, Geografia do Brasil, Física, Química, Matemática,
Matemática Avançada, Biologia e Saúde, Literatura e Gramática, Filosofia e Sociologia, Arte e
Música, Direito e Política, Economia, Astronomia — além de `adversarial`
(premissa-falsa, contradição, cenário-impossível, ambiguidade, falácia, antiético) e `multimodal`.

---

## 3. Os cinco temas

### Tema 1 — Robustez pedagógica por componente curricular

**Título provisório:** *Onde a IA erra mais? Robustez de um tutor automatizado por componente
curricular e nível de complexidade da pergunta*

- **Seção sugerida:** Artigo científico.
- **Pergunta:** A taxa de erro de um tutor de IA é homogênea entre componentes curriculares, ou existem disciplinas sistematicamente mais frágeis? A incerteza declarada pelo sistema antecipa esses erros?
- **Diálogo com a edição:** conversa direto com o artigo 3 (formação inicial em química) e o artigo 2 (BNCC): a pergunta "que competência da BNCC a IA sustenta e qual ela não sustenta" é exatamente a que a revista já está fazendo por outros meios.
- **Método:** execução do catálogo de benchmark estratificado por tema × complexidade (`simple`/`medium`/`hard`/`expert`); qualidade avaliada por consenso de juízes com validação humana amostral; Kruskal-Wallis entre disciplinas com pós-teste corrigido por Holm; correlação de Spearman entre incerteza (UQ) e erro; ECE para calibração.
- **Dados:** ~4.780 consultas do catálogo, ≥10 réplicas por condição, `experiment_manifest.json` por execução.
- **Resultado esperado:** um mapa de fragilidade por disciplina — provavelmente pior em cálculo simbólico, cadeias causais em História e interpretação de texto literário — e evidência de que a incerteza é um sinal utilizável (ou não) como semáforo para o professor.
- **Por que interessa à RPV:** entrega ao docente uma resposta acionável ("em que conteúdo eu não posso soltar a mão do aluno com o chatbot"), não uma métrica de *leaderboard*.
- **Falta construir:** mapeamento explícito dos temas do catálogo para componentes/competências da BNCC; amostra de validação humana com professores da área.

---

### Tema 2 — *Red teaming* pedagógico e integridade acadêmica

**Título provisório:** *Quando o aluno testa a máquina: taxonomia de perguntas adversariais e a
integridade de tutores de IA no ensino*

- **Seção sugerida:** Artigo científico.
- **Pergunta:** Quais estratégias de pergunta (premissa falsa, cenário impossível, ambiguidade, falácia, pedido antiético) fazem um tutor de IA falhar pedagogicamente — isto é, concordar com o falso, inventar referência ou realizar a tarefa pelo aluno em vez de ensinar?
- **Diálogo com a edição:** é a contrapartida empírica do artigo de v. 14 n. 2 sobre formação humanística e do artigo sobre "IA para além do behaviorismo": mostra, com dados, o que acontece quando o sistema é otimizado só para agradar.
- **Método:** matriz tema × estratégia de ataque a partir de `data/benchmark_queries/adversarial.jsonl` (6 subtópicos); métrica primária **ASR (attack success rate)** com definição pedagógica de "sucesso do ataque"; ANOVA/Kruskal-Wallis entre estratégias; ablação com e sem `adversarial_governance` ativo, medindo se a memória de risco por *cluster* reduz o ASR sem degradar a resposta útil.
- **Contribuição diferencial:** uma **taxonomia de ataques em linguagem docente** — "premissa falsa" vira "o aluno afirma como fato algo que não aconteceu"; e um conjunto público de casos difíceis reutilizável por outros grupos.
- **Risco a declarar:** publicar exemplos de ataques bem-sucedidos é um risco de duplo uso. Mitigação: liberar a taxonomia e as estatísticas; liberar os *prompts* brutos sob solicitação acadêmica.
- **Falta construir:** rubrica de "falha pedagógica" validada por pares humanos (distinta de "resposta errada"); anotação de concordância entre anotadores.

---

### Tema 3 — Custo, soberania e viabilidade em rede pública

**Título provisório:** *Quanto custa um tutor de IA para uma rede pública? Custo-efetividade de
uma arquitetura híbrida local-nuvem para escolas brasileiras*

- **Seção sugerida:** Artigo científico (com forte viés de política educacional).
- **Pergunta:** É possível manter qualidade pedagógica aceitável rodando a maior parte das interações em modelos **locais**, em hardware acessível, reservando a nuvem paga apenas para as consultas de alta incerteza? Qual a economia real e qual o preço em qualidade?
- **Diálogo com a edição:** é o mesmo gênero de preocupação estrutural dos artigos 5 (condições de trabalho docente no Novo Ensino Médio) e 1 (extensão e ensino remoto): infraestrutura, orçamento e desigualdade de acesso.
- **Método:** três políticas comparadas sob o mesmo *benchmark* — (i) tudo em modelo premium de nuvem, (ii) tudo local via Ollama, (iii) roteamento multiobjetivo com escalonamento por incerteza. Métricas: custo por consulta útil (`roi_analytics.py`, baseline contrafactual configurável), qualidade, latência, e **custo por resposta aceitável** (não custo bruto). Análise de sensibilidade a preço de API e a orçamento apertado/folgado.
- **Resultado esperado:** uma curva de decisão — "com X% das consultas escalonadas, você mantém Y% da qualidade premium por Z% do custo" — traduzida em uma tabela que um secretário municipal de educação consegue ler.
- **Por que é forte para esta revista:** é o argumento de **soberania tecnológica e equidade orçamentária**, temas com tração editorial evidente e quase nenhuma evidência quantitativa publicada em português.
- **Falta construir:** medição de consumo energético e de hardware mínimo (a promessa "hardware de consumo" precisa de número, não de adjetivo).

---

### Tema 4 — O professor como juiz: concordância humano–IA

**Título provisório:** *Quem avalia o avaliador? Concordância entre juízes automáticos e
professores na correção de respostas geradas por IA*

- **Seção sugerida:** Artigo científico (metodológico) — alternativamente iniciação científica, se conduzido com bolsistas de licenciatura.
- **Pergunta:** Um "juiz LLM" concorda com professores da área ao julgar a qualidade pedagógica de uma resposta? Em que tipo de item a concordância desaba?
- **Diálogo com a edição:** é a pergunta de avaliação educacional clássica — validade e fidedignidade de instrumento — aplicada a um instrumento novo. Encaixa na tradição de artigos de avaliação da escrita/aprendizagem que a revista já publica.
- **Método:** `services/expert_review.py` já entrega fila de revisão por tema e cálculo de **kappa de Cohen**; painel de N professores por área avaliando amostra estratificada com rubrica pedagógica explícita (correção factual, adequação ao nível, indução ao raciocínio *vs* entrega da resposta pronta); kappa humano×humano como teto e kappa humano×juiz-IA como medida; análise dos itens de discordância.
- **Contribuição diferencial:** este é o artigo **metodológico de base** — sem ele, os temas 1, 2 e 3 dependem de um avaliador não validado. Publicá-lo primeiro fortalece toda a série.
- **Falta construir:** recrutamento e termo de consentimento dos professores; rubrica em português validada; interface de anotação minimamente amigável (o portal existe, a usabilidade para docente não-técnico não foi testada).

---

### Tema 5 — Relato de experiência: formação docente e acessibilidade

**Título provisório:** *Um roteador de IA aberto na formação continuada de professores: relato de
experiência sobre curadoria docente, linguagem acessível e limites do apoio automatizado*

- **Seção sugerida:** **Relato de experiência** (a seção de entrada mais rápida da revista, e a de menor exigência de N).
- **Pergunta:** O que muda na prática docente quando o professor tem controle explícito sobre o sistema — vê a incerteza, escolhe escalonar, corrige a resposta e essa correção realimenta o modelo?
- **Diálogo com a edição:** ecoa o artigo 1 (extensão + ensino remoto), o artigo 3 (formação de professores) e, na dimensão de acessibilidade, o artigo 7 (trajetórias de estudantes surdos e com deficiência auditiva) — aqui pela via da **equidade linguística**: desempenho do sistema com português simplificado, estrutura sintática reduzida, variação regional e vocabulário escolar.
- **Método:** oficina de extensão com professores (rede pública ou COLUNI/UFV); registro sistemático de uso; questionário pré/pós de confiança e adoção; análise qualitativa das intervenções docentes; medição quantitativa complementar do desempenho por registro linguístico da pergunta (mesma pergunta reescrita em três registros — acadêmico, escolar, simplificado — comparando qualidade e incerteza).
- **Resultado esperado:** evidência de que a reescrita da pergunta em registro simplificado degrada a resposta — uma desigualdade invisível que penaliza justamente quem tem menos domínio da norma culta — e um protocolo de mediação docente para compensá-la.
- **Por que é o tema de menor risco de entrada:** aceita N pequeno, é escrito no gênero que a revista mais publica, e produz o material humano que os temas 1–4 não capturam.
- **Falta construir:** aprovação ética; conjunto pareado de perguntas em três registros linguísticos; instrumento de coleta.

---

## 4. Ordem de submissão recomendada

```
Tema 5 (relato)  ──►  entrada rápida no periódico, gera material humano
      │
Tema 4 (juízes)  ──►  valida o instrumento de avaliação
      │
      ├──►  Tema 1 (robustez por disciplina)
      ├──►  Tema 2 (red teaming pedagógico)
      └──►  Tema 3 (custo e política pública)
```

O Tema 4 é dependência metodológica dos Temas 1–3: enquanto o juiz automático não estiver
validado contra professores, as métricas de qualidade dos outros três artigos são contestáveis em
parecer.

---

## 5. Condições transversais (valem para os cinco)

1. **Ética.** A RPV é publicada pelo **Colégio de Aplicação da UFV** — o contexto natural do
   periódico inclui **estudantes menores de idade**. Qualquer coleta com alunos exige aprovação de
   CEP, TCLE e assentimento; dados de menores não devem transitar por APIs de nuvem sem base legal
   e anonimização. Os Temas 1–3 foram desenhados para rodar **sem dados de estudantes reais**
   (catálogo sintético), justamente para não depender do trâmite ético.
2. **Idioma e altura do texto.** Português, foco pedagógico, jargão técnico confinado a uma seção
   de método e a um apêndice. O README do repositório já tem a seção "Entenda Rapidamente" escrita
   para leitores sem formação em TI — ela é o registro correto para estes artigos.
3. **Extensão.** Mirar 15–20 páginas, coerente com a edição analisada.
4. **Reprodutibilidade.** Publicar `experiment_manifest.json`, seeds, *snapshot* de preços e a
   versão dos *prompts*. É diferencial competitivo num periódico onde quase nenhum artigo de IA
   traz artefato executável.
5. **Conflito de interesse.** Declarar que os autores desenvolvem a ferramenta avaliada e que os
   dados foram gerados por ela — e mitigar com o Tema 4 (validação humana independente) e com
   *baselines* fortes, não com uma nota de rodapé.
6. **Relação com a agenda internacional.** Estes cinco temas são a via nacional e em português da
   trilha descrita em `papers/doutorado_artigos.md`; não competem com ela, alimentam-na. Cuidado
   com autoplágio: recortes, dados e argumentos devem ser distintos dos artigos internacionais.

---

## 6. Fontes consultadas

- Revista Ponto de Vista — <https://periodicos.ufv.br/RPV>
- Edição v. 15, n. 1 (2026) — <https://periodicos.ufv.br/RPV/issue/view/793>
- Sobre a revista / foco e escopo — <https://periodicos.ufv.br/RPV/about>
- Normas de submissão — <https://periodicos.ufv.br/RPV/about/submissions>
- COLUNI/UFV — <https://coluni.ufv.br/revista-ponto-de-vista/>
- ISSN 1983-2656 — <https://portal.issn.org/resource/ISSN/1983-2656>
