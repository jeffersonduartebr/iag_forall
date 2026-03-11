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

## Documentação Técnica Detalhada

## Diagrama de componentes
Objetivo: mostrar como os módulos centrais se conectam no caminho de decisão e operação.

```mermaid
flowchart TD
    Main[main.py<br/>FastAPI e lifecycle]
    Router[router_core.py<br/>fluxo síncrono e feedback]
    Strategy[router_strategy.py / bandits.py / online_predictor.py<br/>decisão multiobjetivo]
    Providers[providers_async.py<br/>timeouts, retries, circuit breaker]
    Cache[semantic_cache.py<br/>L1 + busca semântica]
    Rag[rag_local.py / vectorstore.py / embeddings.py<br/>recuperação contextual]
    Settings[settings_dynamic.py<br/>configuração em runtime]
    Obs[observability.py / health.py / metrics_collector.py]
    Data[(MariaDB / Redis / ChromaDB)]
    Workers[tasks.py / Celery / serviços de background]

    Main --> Router
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
1. Request entra na API e passa por middlewares.
2. Core normaliza modalidade e tenta cache semântico.
3. Se não houver cache hit, calcula incerteza e seleciona modelos candidatos.
4. Bandit escolhe modelo final.
5. Provider é chamado com resiliência (retry/circuit/fallback).
6. Resposta é retornada e processamento de feedback é disparado em background.

## Sequência de `POST /query`
Objetivo: mostrar o caminho síncrono principal e onde entram cache, seleção e fallback.

```mermaid
sequenceDiagram
    participant C as Cliente
    participant API as FastAPI /query
    participant RC as router_core.route_and_answer
    participant SC as semantic_cache
    participant RG as rag_local
    participant ST as router_strategy + bandit
    participant PR as providers_async
    participant DB as Redis / MariaDB / ChromaDB
    participant CW as Celery task_process_feedback

    C->>API: POST /query
    API->>RC: route_and_answer(...)
    RC->>SC: check_cache(query, modality, image)
    SC->>DB: L1 Redis / L2 Chroma lookup
    DB-->>SC: hit ou miss
    alt cache hit
        SC-->>RC: resposta em cache
        RC-->>API: resultado final
        API-->>C: 200 response
    else cache miss
        RC->>RG: recuperar contexto se RAG habilitado
        RG->>DB: embeddings / vector search
        DB-->>RG: contexto relevante
        RG-->>RC: contexto consolidado
        RC->>ST: estimar incerteza e escolher candidatos
        ST-->>RC: modelo alvo + fallback chain
        RC->>PR: chamar provider com timeout global
        PR->>PR: retry / circuit breaker / fallback
        PR-->>RC: resposta do modelo
        RC-->>API: payload final + metadados
        API->>CW: delay(feedback payload)
        API-->>C: 200 response
    end
```

Nota: o diagrama foca o happy path e a decisão principal; branches administrativos e streaming ficam fora para evitar sobrecarga visual.

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
