# Guia de Performance e Latência

Trilha de otimização de desempenho, numerada como `perf #N` (paralela à trilha
`roadmap #N` de produto). Cada item é uma PR isolada, com testes verdes e —
quando muda comportamento em runtime — **desligado por padrão** via feature flag,
no mesmo padrão conservador do `ADVGOV_ENABLED=0`.

**Princípio:** a chamada ao LLM domina o tempo de resposta (segundos vs. ms do
resto do pipeline). As maiores alavancas são, nesta ordem: **evitar a chamada**
(cache), **cortar a cauda** (hedging), e **remover contenção** no caminho Python
(threadpool de CPU, pool de DB).

---

## Onde o tempo é gasto — breakdown por estágio (perf #21)

Cada resposta agora carrega um breakdown por estágio em
`diagnostics.stage_timings_ms` (ms por estágio: `cache_lookup`, `precheck`,
`selection`, `retrieval`, `provider_call`, `postprocess`). Serve para triar uma
request lenta isolada, complementando o histograma agregado
`router_stage_latency_seconds` (label `stage`) no Prometheus/Grafana.

Comece **sempre** por aqui: confirme em qual estágio o tempo está antes de
otimizar. Na prática `provider_call` domina — o que aponta para as alavancas
abaixo.

---

## Cortar a cauda — requests especulativos / hedged (perf #22)

Padrão *tied-request* de *The Tail at Scale*: dispara o modelo primário e, se ele
não responder dentro de um atraso, lança **um backup em paralelo** e usa o
primeiro que terminar, cancelando o perdedor. Diferente da fallback chain
sequencial (que só reage após o timeout estourar), o hedging ataca
proativamente a fração lenta do tráfego.

| Flag | Default | Efeito |
|---|---|---|
| `REQUEST_HEDGING_ENABLED` | `0` | Liga o hedging. |
| `REQUEST_HEDGE_DELAY_MS` | `0` | Atraso fixo (ms) antes do backup. `0` ⇒ derivado da latência EMA. |
| `REQUEST_HEDGE_EMA_FACTOR` | `1.3` | Quando `DELAY_MS=0`: dispara backup em `EMA × fator`. |
| `REQUEST_HEDGE_MAX_PARALLEL` | `2` | Máximo de chamadas concorrentes. |

**Custo:** até `MAX_PARALLEL` chamadas de provider na fração de tráfego que
sofre hedge — troca custo por p95/p99 menor. Não se aplica a turnos com tools
ou multi-turn, nem quando não há backup distinto. Métrica: `router_hedge_total`
(`outcome` = launched/primary_win/backup_win/all_failed).

---

## Remover contenção no caminho Python

### Pool de CPU dedicado para embeddings (perf #23)

`SentenceTransformer.encode()` é CPU-bound e segura o GIL. Rodá-lo no executor
default do asyncio — o mesmo pool que serve I/O bloqueante via
`asyncio.to_thread` (reads de DB, Redis) — deixa um burst de embeddings
*starvar* o I/O (head-of-line blocking). Os embeddings do hot path
(cache semântico, RAG) agora rodam num pool **separado e pequeno**.

| Env | Default | Efeito |
|---|---|---|
| `EMBED_CPU_THREADS` | `clamp(2..4, cpu_count)` | Tamanho do pool de CPU. |

Pequeno de propósito: como o encode segura o GIL, threads extras não somam
throughput de CPU — o objetivo é **isolar do I/O**, não paralelizar CPU.

### Cache semântico: normalização de query (perf #24)

Um acerto de cache pula o LLM inteiro — a maior economia por request. A query é
canonicalizada (casefold + colapso de espaços) antes do hash L1 e do embedding
L2, então variações triviais ("What is  2+2?" vs "what is 2+2?") compartilham a
mesma entrada.

| Flag | Default | Efeito |
|---|---|---|
| `SEMANTIC_CACHE_NORMALIZE_ENABLED` | `1` | Liga a normalização (simétrica em lookup e store). |

### Pool de DB: engine único + sizing por env (perf #25)

Consolidado para **um** pool de conexões (`db.get_engine()`); o `db_manager`
deixou de abrir um segundo engine. O sizing é configurável por env para casar
com o número de workers e o `max_connections` do MariaDB, sem mudança de código.

| Env | Default | Efeito |
|---|---|---|
| `DB_POOL_SIZE` | `10` | Conexões base no pool. |
| `DB_MAX_OVERFLOW` | `5` | Conexões extras em burst. |
| `DB_POOL_RECYCLE` | `300` | Recicla conexões (s). |
| `DB_POOL_TIMEOUT` | `60` | Espera por conexão (s). |

---

## Já presente no sistema

- **Streaming real de tokens** (`roadmap #1`, `/query/stream`): reduz o
  *time-to-first-token*. Migrar clientes do `/query` síncrono é ganho percebido.
- **Peso de latência do NSGA** (`NSGA_W_LATENCY`): já é setting dinâmico via
  `/admin/settings` — subi-lo enviesa o roteador para modelos mais rápidos.
- **Timeout adaptativo** (`adaptive_timeout.py`, EMA por modelo) e **gate de RAG**
  (`light`/`full` com budgets de token) já cortam overhead condicionalmente.
- **Reuso de cliente httpx** com pool (200 conexões / 50 keepalive).

## Deferido (requer infra não validável aqui)

- **Embeddings via ONNX/quantização ou GPU:** 2–4× no encode em CPU / offload
  para a GPU do Ollama. O pool de CPU dedicado (perf #23) já mitiga a contenção;
  este é o próximo passo de teto de latência do retrieval.
- **Driver de DB async (`asyncmy`/`aiomysql`):** remove os reads de DB do
  threadpool. Toca todo o caminho de leitura — merece PR dedicada com DB real.
