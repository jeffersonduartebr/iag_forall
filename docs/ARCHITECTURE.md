# Arquitetura do Sistema

## Visão geral
O sistema roteia consultas para modelos de linguagem com foco em equilíbrio entre:
1. Qualidade da resposta.
2. Latência.
3. Custo.

## Componentes principais
1. **Camada HTTP (FastAPI)**
- Arquivo: `app/app/main.py`.
- Responsável por endpoints, middlewares, lifecycle e despacho de tarefas assíncronas.

2. **Core de roteamento**
- Arquivo: `app/app/router_core.py`.
- Faz cache check, estimativa de incerteza, seleção de candidatos, chamada de provider e retorno final.

3. **Estratégia de decisão**
- Arquivos: `app/app/router_strategy.py`, `app/app/bandits.py`, `app/app/online_predictor.py`.
- Combina pesos multiobjetivo + política online.

4. **Providers**
- Arquivo: `app/app/providers_async.py`.
- Integra provedores externos e locais com timeout, retry e circuit breaker.

5. **RAG e cache semântico**
- Arquivos: `app/app/rag_local.py`, `app/app/vectorstore.py`, `app/app/semantic_cache.py`, `app/app/embeddings.py`.
- Faz recuperação contextual e acelera respostas repetidas/semelhantes.

6. **Persistência e estado operacional**
- MariaDB: logs/estatísticas/configurações dinâmicas.
- Redis: estado de runtime, cache auxiliar e coordenação.
- ChromaDB: embeddings e documentos vetoriais.

7. **Observabilidade**
- Arquivos: `app/app/observability.py`, `app/app/health.py`, `app/app/metrics_collector.py`.
- Exporta métricas, saúde de componentes e rastreabilidade.

## Fluxo de requisição (`POST /query`)
1. Request entra na API e passa por middlewares.
2. Core normaliza modalidade e tenta cache semântico.
3. Se não houver cache hit, calcula incerteza e seleciona modelos candidatos.
4. Bandit escolhe modelo final.
5. Provider é chamado com resiliência (retry/circuit/fallback).
6. Resposta é retornada e processamento de feedback é disparado em background.

## Fluxo de feedback assíncrono
1. Salva metadados da resposta.
2. Avalia qualidade (heurística/juízes quando habilitado).
3. Atualiza bandit e indicadores EMA.
4. Opcionalmente armazena entrada no cache semântico.

## Decisões arquiteturais importantes
1. **Separar caminho síncrono e assíncrono**
- Síncrono: latência da resposta ao cliente.
- Assíncrono: aprendizagem e manutenção de estado.

2. **Degradação graciosa**
- Se Redis falhar, partes do sistema degradam para estratégias locais.
- Se provider falhar, fallback chain tenta modelos alternativos.

3. **Configuração dinâmica**
- Variáveis podem ser alteradas em runtime via `settings_dynamic`.

## Limites e riscos operacionais
1. Cardinalidade de métricas muito alta pode custar caro no Prometheus.
2. Concurrency de API desbalanceada com pool de DB gera saturação.
3. Timeouts inadequados para modelos lentos causam `504` frequente.

## Onde investigar cada incidente
1. Timeouts/falhas de modelo: `providers_async.py`, `reliability.py`.
2. Seleção de modelo inesperada: `router_core.py`, `router_strategy.py`, `bandits.py`.
3. Problemas de configuração: `settings_dynamic.py`, `/admin/settings`.
4. Latência geral alta: `main.py`, `db.py`, `middleware/backpressure.py`.
