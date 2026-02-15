# Algoritmos e Estratégias

## Objetivo do sistema
Escolher o melhor modelo para cada requisição balanceando qualidade, latência e custo.

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
1. Recupera contexto relevante.
2. Reordena (reranker) quando habilitado.
3. Enriquecer prompt antes da inferência.

## 6. Fallback e circuit breaker
Arquivos: `app/app/reliability.py`, `app/app/providers_async.py`

Objetivo:
1. Evitar indisponibilidade total por falha de um único provider/modelo.
2. Encadear modelos alternativos automaticamente.

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
