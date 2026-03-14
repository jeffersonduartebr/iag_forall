# Algoritmos e Estratégias

## Objetivo do sistema
Escolher o melhor modelo para cada requisição balanceando qualidade, latência e custo.

## Diagrama do pipeline algorítmico
Objetivo: resumir como as estratégias se encadeiam no caminho de decisão do roteador.

```mermaid
flowchart LR
    A[Consulta recebida]
    B[Workload class e runtime hints]
    C[Cache semântico]
    D[Retrieval gate<br/>no/light/full]
    E[UQ e top-K candidatos]
    F[Bandit e decisão final]
    G[Provider com timeout e sync deadline]
    H[Resposta com provenance]
    I[Feedback e aprendizado]

    A --> B --> C
    C -->|cache hit| H --> I
    C -->|cache miss| D --> E --> F --> G --> H --> I
```

Nota: o diagrama mostra a ordem lógica principal; detalhes internos de cada algoritmo continuam nas seções abaixo.

## Visão para Engenharia de IA
Objetivo: destacar os mecanismos que afetam escolha de modelo, reward e aprendizagem online.

```mermaid
flowchart TD
    A[Consulta]
    B[Workload class + retrieval mode]
    C[Features de contexto]
    D[UQ]
    E[Top-K candidatos]
    F[Bandit / política online]
    G[Modelo escolhido]
    H[Resposta + confidence + verification]
    I[Judge / heurística]
    J[Reward]
    K[Atualização EMA e bandit]

    A --> B --> C --> D --> E --> F --> G --> H
    H --> I --> J --> K
    K --> F
```

Nota: este diagrama é voltado a engenharia de IA; ele foca aprendizagem e decisão adaptativa, não topologia de serviços.

## 1. Seleção de candidatos
Arquivo principal: `app/app/router_strategy.py`

Resumo:
1. Recebe lista de modelos candidatos.
2. Aplica pesos/heurísticas por modalidade e contexto.
3. Devolve subconjunto (ex.: top-2) para decisão final.

## 2. Bandits (decisão online)
Arquivo principal: `app/app/bandits.py`

Estratégias usadas:
1. **Epsilon-greedy**: explora com probabilidade `epsilon`.
2. **UCB1**: favorece melhor média + incerteza de amostragem.
3. **Thompson Sampling**: usa distribuição para explorar/explotar.

Meta-estratégia:
- O sistema pode combinar estratégias e manter estatísticas por contexto.

## 3. Incerteza da consulta (UQ)
Arquivos: `app/app/utils/uncertainty.py`, `app/app/router_core.py`

Uso:
1. Estimar dificuldade/risco da consulta.
2. Ajustar tendência de escolha para modelos mais robustos quando necessário.

## 4. Cache semântico
Arquivo principal: `app/app/semantic_cache.py`

Camadas:
1. L1 em memória (exato, TTL curto).
2. L2 vetorial (similaridade semântica no Chroma).

Benefício:
- Redução de custo e latência para consultas recorrentes/semelhantes.

## 5. RAG (recuperação aumentada)
Arquivos: `app/app/rag_local.py`, `app/app/vectorstore.py`, `app/app/reranker.py`

Fluxo:
1. Decide entre `no_retrieval`, `light_retrieval` e `full_retrieval`.
2. Recupera contexto relevante apenas quando o gate indicar ganho esperado.
3. Reordena (reranker) quando habilitado e quando houver candidatos suficientes.
4. Enriquecer prompt antes da inferência, preservando provenance estruturada.

## 6. Fallback e circuit breaker
Arquivos: `app/app/reliability.py`, `app/app/providers_async.py`

Objetivo:
1. Evitar indisponibilidade total por falha de um único provider/modelo.
2. Encadear modelos alternativos automaticamente.
3. Cortar fallbacks tardios quando o orçamento restante do deadline síncrono já não comporta nova tentativa.

## 7. Aprendizado por feedback
Arquivos: `app/app/tasks.py`, `app/app/user_feedback.py`, `app/app/router_core.py`

Resumo:
1. Coleta feedback de qualidade.
2. Atualiza estatísticas históricas e bandit.
3. Ajusta comportamento futuro do roteador.

## Limitações atuais
1. Heurísticas dependem da qualidade dos dados históricos.
2. Mudanças drásticas de carga podem exigir recalibração manual.
3. Configuração ruim de timeout/concorrência pode mascarar ganhos do algoritmo.
