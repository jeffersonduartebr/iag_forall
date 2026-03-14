# Arquitetura do Sistema

## Visão geral
O sistema roteia consultas para modelos de linguagem com foco em equilíbrio entre:
1. Qualidade da resposta.
2. Latência.
3. Custo.

## Mapa deste documento
Objetivo: separar rapidamente a camada de explicação para professores da camada técnica detalhada.

```mermaid
flowchart LR
    A[Visão geral]
    B[Visão para professores]
    C[Documentação técnica detalhada]
    D[Fluxos e componentes]
    E[Riscos e investigação]

    A --> B --> C --> D --> E
```

## Visão Para Professores
Esta seção explica o sistema sem depender de nomes de arquivos, bancos ou serviços internos.

### Fluxo principal em linguagem simples
Objetivo: mostrar como uma solicitação do professor vira uma resposta de apoio.

```mermaid
flowchart LR
    A[Professor envia uma pergunta<br/>ou tarefa]
    B[Sistema entende o tipo<br/>de ajuda necessária]
    C[Sistema busca informações<br/>e exemplos úteis]
    D[Sistema prepara uma resposta<br/>ou sugestão]
    E[Professor revisa e decide<br/>como usar]

    A --> B --> C --> D --> E
```

Legenda:
- `Professor envia uma pergunta`: por exemplo, pedir explicação, atividade, resumo ou apoio de aula.
- `Sistema entende`: tenta identificar a intenção do pedido.
- `Busca informações`: consulta conteúdos e respostas anteriores relevantes.
- `Prepara uma resposta`: organiza uma sugestão útil para o contexto.
- `Professor revisa e decide`: a resposta pode ser ajustada, reaproveitada ou descartada.

### Melhoria contínua em linguagem simples
Objetivo: mostrar que o sistema tenta aprender com o uso, sem retirar o papel pedagógico do professor.

```mermaid
flowchart TD
    A[Professor recebe a resposta]
    B[Professor usa, adapta<br/>ou não aproveita]
    C[Sistema registra sinais<br/>do que funcionou]
    D[Sistema tenta oferecer<br/>respostas melhores no futuro]

    A --> B --> C --> D
```

Mensagem principal:
- o sistema é um apoio ao trabalho docente;
- a decisão pedagógica continua sendo humana;
- o histórico de uso ajuda a melhorar sugestões futuras.

### Limites e uso responsável
Objetivo: deixar explícito que a resposta da IA precisa de revisão humana.

```mermaid
flowchart LR
    A[Pessoa faz uma pergunta]
    B[IA sugere uma resposta]
    C[Pessoa revisa o conteúdo]
    D[Pessoa ajusta ou descarta]
    E[Uso responsável]

    A --> B --> C --> D --> E
```

## Documentação Técnica Detalhada

## Visão para Desenvolvedores de TI
Objetivo: mostrar os blocos operacionais da stack e suas responsabilidades.

```mermaid
flowchart TD
    A[Cliente HTTP]
    B[API FastAPI]
    C[Middlewares e admissao]
    D[Serviços de roteamento]
    J[Query Jobs + Redis]
    P[Providers e Ollama]
    E[Redis, MariaDB, ChromaDB]
    F[Celery Worker]
    G[Prometheus, Grafana, Loki]

    A --> B --> C --> D
    C --> J
    D --> P
    D --> E
    B --> F
    F --> E
    B --> G
    F --> G
```

Nota: este diagrama e para backend e operacao; ele prioriza componentes implantaveis e fluxos entre servicos.

## Visão para Engenharia de IA
Objetivo: mostrar o pipeline de decisão e atualização do roteador.

```mermaid
flowchart LR
    A[Consulta]
    B[Workload class e runtime hints]
    C[Cache, retrieval gate e RAG]
    D[UQ e bandit]
    E[Modelo escolhido]
    F[Resposta + confidence + provenance]
    G[Judge e reward]
    H[Aprendizado online]

    A --> B --> C --> D --> E --> F --> G --> H
```

Nota: este diagrama abstrai detalhes de container e infraestrutura para focar seleção, qualidade e feedback.

## Diagrama de componentes
Objetivo: mostrar como os módulos centrais se conectam no caminho de decisão e operação.

```mermaid
flowchart TD
    Main[main.py<br/>FastAPI e lifecycle]
    Router[router_core.py<br/>fluxo síncrono e feedback]
    QueryJobs[query_jobs.py<br/>fila de overflow e polling]
    Strategy[router_strategy.py / bandits.py / online_predictor.py<br/>decisão multiobjetivo]
    Providers[providers_async.py<br/>timeouts, retries, circuit breaker]
    Cache[semantic_cache.py<br/>L1 + busca semântica]
    Rag[rag_local.py / vectorstore.py / embeddings.py<br/>recuperação contextual]
    Settings[settings_dynamic.py<br/>configuração em runtime]
    Obs[observability.py / health.py / metrics_collector.py]
    Data[(MariaDB / Redis / ChromaDB)]
    Workers[tasks.py / Celery / serviços de background]

    Main --> Router
    Main --> QueryJobs
    Main --> Settings
    Main --> Obs
    Router --> Cache
    Router --> Rag
    Router --> Strategy
    Router --> Providers
    Router --> Settings
    Router --> Obs
    Cache --> Data
    Rag --> Data
    Providers --> Data
    QueryJobs --> Data
    QueryJobs --> Workers
    Workers --> Router
    Workers --> Data
    Workers --> Obs
```

Nota: caixas agrupam módulos próximos para manter o diagrama legível; detalhes de classes e helpers ficam em `docs/MODULE_INDEX.md`.

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
1. Request entra na API e passa por correlação, backpressure e limitador adaptativo.
2. O runtime classifica a consulta (`workload_class`) e monta hints como timeout e deadline.
3. Se houver pressão operacional, a query pode ser deferida para fila assíncrona e responder `202`.
4. Se seguir no caminho síncrono, tenta cache semântico, retrieval seletivo e seleção de candidatos.
5. O provider é chamado com timeout por workload, deadline total e corte antecipado de fallback.
6. A resposta pública volta com sinais de confiabilidade, proveniência e diagnóstico opcional.
7. O feedback assíncrono continua sendo disparado em background após a resposta.

## Sequência de `POST /query`
Objetivo: mostrar o caminho síncrono principal e onde entram cache, seleção e fallback.

```mermaid
sequenceDiagram
    participant C as Cliente
    participant API as FastAPI /query
    participant MW as Middlewares
    participant QR as query_runtime
    participant RC as router_core.route_and_answer
    participant SC as semantic_cache
    participant RG as rag_local
    participant ST as router_strategy + bandit
    participant PR as providers_async
    participant QJ as query_jobs
    participant DB as Redis / MariaDB / ChromaDB
    participant CW as Celery task_process_feedback

    C->>API: POST /query
    API->>MW: correlation + backpressure + adaptive limiter
    alt deferida por overload
        MW->>QJ: enqueue query job
        QJ->>DB: persistir job/status
        QJ-->>API: job_id + poll_url
        API-->>C: 202 Accepted
    else caminho síncrono
        API->>QR: process_query_request(...)
        QR->>RC: route_and_answer(...)
        QR->>QR: workload_class + provider_timeout + sync_deadline
        RC->>SC: check_cache(query, modality, image)
        SC->>DB: L1 Redis / L2 Chroma lookup
        DB-->>SC: hit ou miss
        alt cache hit
            SC-->>RC: resposta em cache
            RC-->>QR: resultado final
        else cache miss
            RC->>RG: retrieval gate + light/full RAG
            RG->>DB: embeddings / vector search
            DB-->>RG: contexto relevante
            RG-->>RC: contexto + provenance
            RC->>ST: UQ, candidatos e fallback
            ST-->>RC: modelo alvo + fallback chain
            RC->>PR: chamar provider com timeout por workload
            PR->>PR: retry / circuit breaker / fallback condicionado ao deadline
            PR-->>RC: resposta do modelo
            RC-->>QR: payload final + metadados
        end
        QR-->>API: QueryResponse
        API->>CW: delay(feedback payload)
        API-->>C: 200 response
    end
```

Nota: o diagrama destaca os dois caminhos principais atuais: resposta síncrona e deferimento assíncrono.

## Sequência de polling dos jobs de query
Objetivo: mostrar o fluxo da query deferida até o resultado final.

```mermaid
sequenceDiagram
    participant C as Cliente
    participant API as FastAPI
    participant QJ as query_jobs
    participant CW as Celery worker
    participant QR as query_runtime
    participant DB as Redis / MariaDB

    C->>API: GET /query/jobs/{job_id}
    API->>QJ: get_query_job_status(job_id)
    QJ->>DB: carregar status
    DB-->>QJ: queued/running/completed/failed
    QJ-->>API: QueryJobStatusResponse
    API-->>C: status atual
    CW->>QJ: consumir job enfileirado
    QJ->>QR: processar query
    QR-->>QJ: QueryResponse final
    QJ->>DB: persistir resultado
    C->>API: GET /query/jobs/{job_id}/result
    API->>QJ: get_query_job_result(job_id)
    QJ->>DB: carregar resultado
    DB-->>QJ: QueryResponse
    QJ-->>API: resposta pronta
    API-->>C: 200 + resultado final
```

## Fluxo de feedback assíncrono
1. Salva metadados da resposta.
2. Avalia qualidade (heurística/juízes quando habilitado).
3. Atualiza bandit e indicadores EMA.
4. Opcionalmente armazena entrada no cache semântico.

## Sequência do feedback assíncrono
Objetivo: mostrar o que acontece após a resposta ser entregue ao cliente.

```mermaid
sequenceDiagram
    participant API as FastAPI
    participant CW as Celery worker
    participant RC as router_core.process_background_feedback
    participant JD as judges / heurísticas
    participant BT as bandits / EMA
    participant SC as semantic_cache.store_cache
    participant DB as MariaDB / Redis / ChromaDB
    participant OB as observability

    API->>CW: task_process_feedback.delay(...)
    CW->>RC: process_background_feedback(...)
    RC->>DB: persistir metadados da resposta
    RC->>JD: avaliar qualidade se habilitado
    JD-->>RC: score / sinais auxiliares
    RC->>BT: atualizar bandit e métricas históricas
    BT->>DB: persistir estado e sinais
    opt resposta elegível para cache semântico
        RC->>SC: store_cache(query, answer, ...)
        SC->>DB: gravar L1/L2 cache
    end
    RC->>OB: registrar latência, contadores e falhas
    RC-->>CW: finaliza tarefa
```

Nota: o worker pode degradar parcialmente em falhas de Redis ou judges; a intenção aqui é destacar a ordem das responsabilidades.

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
