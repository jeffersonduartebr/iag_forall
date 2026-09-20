# Auditoria de robustez, auditabilidade e governança

**Data:** 2026-09-20 · **Alvo:** `app/app/` no commit `723d729` · **Método:** três varreduras
independentes do código, seguidas de verificação manual dos achados de maior severidade.

## Como ler este documento

Cada achado traz **evidência com ficheiro:linha** e a **consequência concreta** — o que um
utilizador ou um leitor da tese veria acontecer. Achados sem consequência observável não
entraram.

Os achados marcados **✅ verificado** foram confirmados linha a linha durante a redação
deste documento. Os marcados **⚠️ reportado** vieram das varreduras e não foram
re-confirmados; trate-os como hipóteses fortes, não como facto, e confirme antes de agir.

A ordenação é por severidade, não por esforço. A tabela final ordena por rácio
impacto/esforço.

---

## 1. Auditabilidade — o que a tese não consegue provar hoje

Esta secção é a mais grave do documento. Não são bugs de funcionamento: o sistema corre
perfeitamente bem com todos eles. São bugs de **prova** — afirmações do artefacto que os
dados não sustentam.

### A1 — O `query_log` é apagado ao fim de 7 dias, em silêncio ✅ verificado

```
app/app/router_core.py:180   LOG_RETENTION_DAYS = _safe_setting_int("QUERY_LOG_RETENTION_DAYS", 7)
app/app/router_core.py:396   DELETE FROM query_log WHERE created_at < (NOW() - INTERVAL :d DAY)
app/app/router_core.py:399   except Exception:
app/app/router_core.py:400       pass
```

**Consequência.** A tabela que sustenta toda a análise empírica tem uma janela de sete
dias. Uma experiência que corra durante um mês perde as três primeiras semanas enquanto
está a decorrer, e ninguém repara, porque a limpeza corre numa thread de fundo, diária.
O `except Exception: pass` da linha 399 não tem `logger` nenhum: se a limpeza falhar
todos os dias durante um ano, o único sintoma é o disco a encher.

Note-se a assimetria: `ema_history_log` tem retenção de **30 dias** e regista quantas
linhas apagou (`router_core.py:328,342`). O log de queries, que é o dado primário, tem
menos retenção e nenhuma observabilidade.

**Correção.** `QUERY_LOG_RETENTION_DAYS` explícito e alto (ou `0` = nunca) nos ambientes
de experiência, e trocar o `pass` por um `logger.warning` com a contagem, como o irmão já
faz.

### A2 — As internas da decisão nunca são persistidas ✅ verificado

```
app/app/services/router_stages.py:115           "pareto_front": [],
app/app/services/router_stages.py:119           "candidates": [],
app/app/services/router_provider_stage.py:286   "pareto_front": [],
app/app/services/router_provider_stage.py:294   "candidates": [],
```

Os dois caminhos de roteamento — cache hit e execução normal — emitem ambos os campos
**literalmente vazios**. `query_response_builder.py:42` lê
`result.get("candidates", []) or []`, que é sempre `[]`.

**Consequência.** Não existe forma de reconstruir *por que* um modelo foi escolhido. A
frente de Pareto que dá nome à contribuição do trabalho nunca é gravada, e o conjunto de
candidatos considerados também não. Uma pergunta de banca do tipo *"mostre-me uma decisão
em que o NSGA-II preferiu um modelo mais caro, e porquê"* não tem resposta possível com
os dados existentes.

**Correção.** Preencher os dois campos no `router_provider_stage` a partir do estado que
já existe em memória nesse ponto, e gravá-los. É o achado com maior retorno científico do
documento.

### A3 — `correlation_id` nunca chega ao `query_log` ⚠️ reportado

O identificador que atravessa os logs estruturados não é escrito na tabela, o que impede
cruzar uma linha de `query_log` com o rasto de execução que a produziu.

### A4 — O snapshot `frozen_policy` é *write-only* ⚠️ reportado

A política congelada é gravada e nunca lida de volta por nenhum caminho de análise, o que
torna impossível afirmar sob que política uma linha foi produzida.

### A5 — Não existe tabela de auditoria de settings ⚠️ reportado

`/admin/settings` altera o comportamento do router em runtime (Redis + MariaDB) e não fica
registo de quem mudou o quê, quando, nem de qual era o valor anterior. Qualquer resultado
experimental é, por isso, não reprodutível: não há como saber que configuração estava
ativa no instante de uma linha.

Isto agrava-se com **C6**: com o MariaDB em baixo, um override persistido reverte em
silêncio para o default de código — ou seja, a configuração ativa pode mudar sozinha, sem
que nada o registe.

---

## 2. Segurança

### S1 — Fuga entre tenants na cache semântica ✅ verificado

```
app/app/semantic_cache.py:361   "tenant_id": tenant_ns,                       # store: sempre grava
app/app/semantic_cache.py:263   where={"tenant_id": tenant_ns} if tenant_ns != "global" else None
```

`store_cache` grava **sempre** o `tenant_id` — um tenant real, ou `"global"` quando o
pedido não traz nenhum. `check_cache` filtra por esse campo, **exceto** quando o pedido
não traz tenant: aí `where` fica `None` e a busca por vizinho mais próximo alcança as
linhas de todos os tenants.

**Consequência.** Num deployment multi-tenant, um pedido sem `tenant_id` — um teste, uma
chamada interna, um cliente mal configurado — pode receber como resposta em cache o
conteúdo gerado para um cliente pago. É uma fuga de dados entre clientes através de um
canal que ninguém inspeciona.

**Correção.** Uma linha: `where={"tenant_id": tenant_ns}` sempre. Em instalações
single-tenant (o default, onde tudo é `"global"`) o comportamento não muda; em
multi-tenant, fecha a fuga.

### S2 — `REQUIRE_API_AUTH` desligado por omissão ✅ verificado

```
app/app/config/settings_catalog.py:13   "REQUIRE_API_AUTH": "0",
```

Um deployment que não leia a documentação expõe `/query` sem autenticação nenhuma.
Default inseguro é uma escolha de governança, não um detalhe: o valor seguro devia ser o
default e o inseguro devia exigir uma afirmação explícita.

### S3 — Chaves de API em texto plano ✅ verificado (ausência de cifra)

```
app/app/config/settings_catalog.py:114   "OPENROUTER_API_KEY": "",
```

As chaves atravessam as três camadas de settings (env → Redis → MariaDB) sem cifra —
não há `Fernet` nem equivalente em `settings_dynamic.py`. Um dump da base de dados ou um
`KEYS *` no Redis entrega as credenciais de todos os providers.

### S4 — Promoção automática de modelos sem aprovação humana ⚠️ reportado

Modelos são promovidos a candidatos com ~15 amostras e sem nenhum passo de aprovação
(`feedback_stages.py:241` regista a promoção já feita). Um modelo que tenha tido sorte em
15 pedidos entra a servir tráfego real.

---

## 3. Caminhos que reportam sucesso sem o ter

O padrão comum desta secção: uma excepção é apanhada, o erro é registado num log que
ninguém lê, e a função devolve o valor que significa "correu bem".

### F1 — `POST /rag/ingest` devolve 200 com o ChromaDB em baixo ✅ verificado

```
app/app/vectorstore.py:268-287   except Exception as e:  →  só logger.error
app/app/vectorstore.py:335       return True             ←  incondicional
app/app/routers/rag_router.py:206-217   success = await add_document(...); if not success: raise HTTPException(500)
```

Como `add_document` devolve `True` sempre, o `raise HTTPException(500)` do endpoint é
**código morto**.

**Consequência.** Um utilizador carrega um PDF, recebe
`HTTP 200 {"fragments_inserted": 47}`, e nada foi indexado. As queries RAG seguintes
respondem sem grounding e ninguém investiga, porque o upload disse que correu bem. O
campo `fragments_inserted` conta iterações do ciclo, não documentos gravados
(`rag_router.py:147-159`).

**Correção.** `_insert_embedding_sync` devolve `bool` e `add_document` propaga. ~10
linhas; fecha o achado por completo.

### F2 — Uma consulta pode apagar o corpus RAG inteiro ✅ verificado

```
app/app/embeddings.py:265        return [0.0] * 768   # falha total de embedding
app/app/embeddings.py:224,297    return [0.0]         # comprimento 1
app/app/vectorstore.py:352       return col.query(**kwargs)              ← caminho de LEITURA
app/app/vectorstore.py:356-359   if "dimension" in msg and "match" in msg:
                                     get_chroma_client().delete_collection(collection_name)
```

O "auto-healing" por dimensão incompatível existe nos dois caminhos: escrita
(`vectorstore.py:271-283`) e **leitura** (`vectorstore.py:356-362`).

**Consequência.** Quando o modelo de embeddings local falha a carregar — exatamente o que
o commit `d6f0f61` ("embedding local funcionando de novo (pin do transformers)") descreve
— `embed_text` devolve um vetor de zeros com a dimensão errada. A **primeira consulta**
seguinte apanha o erro de dimensão e **apaga a coleção**. Uma operação de leitura destrói
dados do utilizador, e o único aviso aparece depois do estrago.

**Correção.** Remover o `delete_collection` do caminho de leitura: uma consulta sinaliza e
devolve vazio, nunca destrói. O auto-healing na escrita é defensável; na leitura não é.

### F3 — `db_init` sai com código 0 mesmo com o alembic falhado ✅ verificado

```yaml
# docker-compose.yml:68-77
command: >
  bash -c "
    echo '⏳ Executando Alembic migrations...';
    alembic upgrade head;
    echo '⏳ Executando db_manager.py...';
    python -m app.db_manager;
    echo '✅ Banco pronto!';
  "
```

Separadores `;` em vez de `&&`, e **o último comando é um `echo`**, que devolve sempre 0.
A API depende de `db_init: condition: service_completed_successfully`
(`docker-compose.yml:111-112`), condição que fica satisfeita sempre.

**Consequência.** O `alembic upgrade head` falha, o container imprime "✅ Banco pronto!",
sai 0, e a API arranca contra um esquema parcialmente migrado. Os `INSERT` que dependem
das colunas novas falham em runtime e são engolidos por **D1**.

**Correção.** `set -e` e `&&`; mover o `echo` final para antes do último comando real, ou
terminar com o comando real.

### F4 — O ciclo de feedback engole tudo e reporta `SUCCESS` ⚠️ reportado

`services/router_feedback.py:60-70` envolve todo o pipeline num `except Exception`, e cada
estágio já tem o seu próprio. Com o MariaDB em baixo, a tarefa Celery termina em
`SUCCESS`, nenhuma linha de `query_log` é escrita, nenhum posterior de bandit é
persistido, nenhuma linha de `ema_history`. O ciclo de aprendizagem para em silêncio.

O que salva este caso é o contador `feedback_task_failures_total{stage=...}`, que **existe**
— desde que haja um alerta ligado a ele. Não há.

---

## 4. Disponibilidade

### D1 — Um único `cloud_breaker` partilhado pelos quatro providers ✅ verificado

```
app/app/providers/_infra.py:164   cloud_breaker = pybreaker.CircuitBreaker(fail_max=CB_FAIL_MAX, ...)
app/app/providers/_openai.py:41      @cloud_breaker   (OpenAI)
app/app/providers/_openai.py:152     @cloud_breaker   (OpenRouter)
app/app/providers/_anthropic.py:35   @cloud_breaker
app/app/providers/_gemini.py:129     @cloud_breaker
```

Uma instância só, para os quatro. E a ordem dos decoradores é:

```python
@COMMON_RETRY_STRATEGY   # ← por fora
@cloud_breaker           # ← por dentro
async def generate(...)
```

O tenacity chama o breaker 5 vezes, e **cada tentativa conta uma falha no breaker**, cujo
`fail_max` é 5.

**Consequência.** Uma única requisição ao OpenAI que falhe de forma retryable esgota as
tentativas, abre o breaker global, e durante 60 s **todas** as chamadas a Anthropic,
Gemini e OpenRouter levantam `ProviderCircuitOpenError` → HTTP 503, mesmo estando os três
perfeitamente saudáveis. Um provider em baixo derruba os quatro.

Agravante: `RETRYABLE_ERRORS` (`_infra.py:122-135`) inclui `httpx.HTTPStatusError` e
`anthropic.APIStatusError`, classes-base de **todos** os 4xx. Uma chave inválida (401) é
retentada 5 vezes com backoff até 60 s — ~2 minutos por requisição — e abre o breaker
global. **Um erro de configuração vira uma indisponibilidade de toda a nuvem.**

**Correção.** Um breaker por provider, e inverter a ordem dos decoradores para que o
breaker conte uma falha por requisição e não cinco. Filtrar `RETRYABLE_ERRORS` por código
de estado em vez de por classe.

### D2 — Sem fallback nem hedging por omissão ✅ verificado

```
app/app/services/router_provider_stage.py:84    _safe_setting_bool("REQUEST_HEDGING_ENABLED", False)
app/app/services/router_provider_stage.py:204   _safe_setting_bool("REQUEST_FALLBACK_ENABLED", False)
app/app/config/settings_catalog.py:252          "REQUEST_HEDGING_ENABLED": "0",
```

`REQUEST_HEDGING_ENABLED` está no catálogo, a `0`. **`REQUEST_FALLBACK_ENABLED` não está
no catálogo de todo** — nem no `.env.example`, nem no compose. Logo o default é `False` e
não há sequer como o descobrir a partir da configuração.

**Consequência.** Se o modelo escolhido pelo bandit falhar, o pedido devolve 502/503/504
ao utilizador **com cinco outros modelos configurados e saudáveis**. Toda a cadeia de
fallback (`reliability.execute_with_fallback`, `registry.get_fallback_chain`) está escrita
e testada, e nunca corre no caminho de pedido.

Consequência de segunda ordem: o `CascadeDetector` conta breakers do
`ModelCircuitBreakerManager`, que sem `execute_with_fallback` nunca são alimentados. **O
detetor de cascata nunca dispara.**

**Correção.** Declarar `REQUEST_FALLBACK_ENABLED` no catálogo com valor `1`. É o melhor
retorno por linha de código do documento inteiro: a funcionalidade já existe, já está
testada, está apenas desligada por uma chave em falta.

### D3 — O engine da base de dados fica envenenado para sempre ✅ verificado

```
app/app/db.py:108-111   if _engine_initialized:
                            raise RuntimeError("Database engine initialization failed previously")
app/app/db.py:113       _engine_initialized = True      ← antes do try
```

`_engine_initialized` nunca é reposto, exceto em `close_engine()`.

**Consequência.** Se o MariaDB estiver 2 segundos indisponível no arranque do processo —
um restart, um rolling deploy — a primeira chamada a `get_engine()` falha e **todas** as
seguintes, pelo resto da vida do processo, levantam `RuntimeError`. O MariaDB recupera, a
API não. O `/health` mostra `mariadb: unhealthy` com a mensagem acima até alguém
reiniciar o container.

O gatilho é fácil: `settings_dynamic.py:125-129` executa DDL **ao importar o módulo**,
dentro de um `try/except` que só faz `logger.warning`.

**Correção.** Permitir reinicialização com backoff em vez do `RuntimeError` permanente.

### D4 — Os SDKs de nuvem ignoram o `timeout_seconds` ⚠️ reportado

O router calcula e passa `timeout_seconds`, mas só o adaptador Ollama o lê. `_openai.py:70`,
`_anthropic.py:73` e `_gemini.py:104,119` chamam os SDKs sem timeout → defaults de 600 s.

O deadline global **é** aplicado (`router_facade.py:62,74`), portanto o cliente não fica
pendurado. Mas a chamada ao Gemini corre em `asyncio.to_thread`, e **threads não são
canceláveis**: continua a bloquear até 600 s, a consumir um slot do executor por omissão
(16 threads). Com o Gemini lento, 16 pedidos drenam o executor e **todo** o
`asyncio.to_thread` da aplicação — guardrails, budget, políticas, embeddings — fica em
fila. Lentidão de um provider vira paragem global.

### D5 — MariaDB em baixo → `/query` devolve 500 cru ⚠️ reportado

`services/query_runtime.py:271-278` propaga a excepção de `check_tenant_budget` /
`get_active_policy`, e não existe `@app.exception_handler` nenhum no projeto. O LLM não
precisa do MariaDB para nada; o pedido falha numa pré-verificação de governança.

### D6 — Sem timeout de leitura no PyMySQL ✅ verificado

`db.py:119-125` cria o engine com `poolclass`, `pool_pre_ping` e a config de pool, e
**sem `connect_args` nenhum**. O default do PyMySQL é
`read_timeout=None`: um MariaDB que aceita a ligação mas não responde bloqueia a thread
indefinidamente. Contraste: `correlation_metrics.py:98` define `socket_connect_timeout=2`
para o Redis.

### D7 — Backpressure e rate limit são *fail-open* ⚠️ reportado

`utils/redis_distributed.py:74-91` admite o pedido quando o Redis falha, e
`middleware/backpressure.py:153-159` não usa o semáforo local como alternativa. Com
`BACKPRESSURE_REDIS_ENABLED=1` e o Redis em baixo, `MAX_CONCURRENT_REQUESTS` deixa de
existir e nenhum 503 é devolvido.

---

## 5. Consistência de dados

### C1 — Idempotência é *fail-open* e não reserva ⚠️ reportado

`redis_distributed.py:105-113` devolve `None` em qualquer excepção, e
`query_http.py:56-92` só escreve **depois** da resposta pronta. A verificação
`cached.get("status") == "completed"` sugere um desenho de duas fases que nunca foi
implementado. Resultado: com o Redis em baixo, ou com dois pedidos concorrentes, a mesma
query executa duas vezes e é **cobrada duas vezes**.

### C2 — Bandit: Redis e MariaDB sem transação nem reconciliação ⚠️ reportado

`bandits.py:559-566` escreve nos dois sem atomicidade. Redis OK + DB falhado passa
despercebido (a leitura prefere Redis) **até o Redis ser limpo** — aí os posteriores
aprendidos regridem para o último estado persistido com sucesso, em silêncio. O routing
simplesmente volta a decisões piores, sem sintoma.

### C3 — `EMABatchQueue` limpa a fila antes de persistir ⚠️ reportado

`services/router_state.py:107` faz `self._queue.clear()`; a linha 110 é que persiste. Um
blip de 2 s no MariaDB perde até 500 observações de EMA.

### C4 — Consumo de tenant contabilizado depois de invalidar a cache ⚠️ reportado

`roadmap_features.py:341` invalida, `:344-367` escreve. Se o `INSERT` falhar, o utilizador
foi servido, o LLM foi pago, e o consumo não foi contabilizado: **o orçamento do tenant
pode ser ultrapassado durante uma indisponibilidade do DB**.

### C5 — Retries que duplicam custo real ⚠️ reportado

`tasks.py:149-190` (`task_execute_eval_run`) repete **todas** as chamadas LLM do run se
falhar a meio. E a composição `REQUEST_MAX_RETRIES` (1) × tenacity (5) × retries internos
do SDK OpenAI (2) permite que **uma requisição do utilizador gere até 15 chamadas ao
upstream**.

### C6 — Settings persistidas revertem em silêncio para o default de código ⚠️ reportado

`settings_dynamic.py:229-241` (`_get_from_db`) devolve `None` em qualquer falha, e
`config/settings_sources.py:73-79` cai então para env → defaults.

**Consequência.** Com o MariaDB em baixo, um override que o operador persistiu via
`/admin/settings` — por exemplo desativar um modelo avariado — **reverte para o default
de código sem aviso nenhum**. O modelo desativado volta a servir tráfego. Combinado com
**A5** (não há auditoria de settings), não fica registo de que a configuração mudou nem
de quando voltou.

---

## 6. Esquema e arranque

### E1 — Duas fontes de verdade para o esquema ⚠️ reportado

`db_manager.SCHEMA_DEFINITIONS` (14 tabelas) **e** as migrações alembic. Nenhum teste
verifica que concordam, e o histórico mostra que já divergiram (`085d07f`, `a372aa9`).
Existe `tests/test_migration_0006.py`, que cobre uma migração — não a consistência entre
as duas definições.

### E2 — Sem verificação de versão de esquema no arranque ⚠️ reportado

`main.py:398-512` valida o `ADMIN_TOKEN` e as settings críticas (fail-fast, correto) mas
não compara `alembic current` com `head`. A API serve tráfego contra um esquema
desatualizado sem aviso; a degradação só aparece como linhas em falta.

---

## 7. Lacunas de teste

Três testes em `tests/test_chaos.py` passam sempre, por construção — ✅ os três
verificados linha a linha:

| Linhas | Problema |
|---|---|
| `293-308` | `test_fallback_chain_execution` define `failing_execute` e depois **não a chama, nem faz assert nenhum**; termina em dois comentários ("Actual execution would require model registry setup"). |
| `249-256` | `test_redis_unavailable_fallback` faz patch de `get_redis` para devolver `None` e depois afirma que `get_redis()` devolve `None`. Testa o `patch`, não o sistema. |
| `262-275` | `test_provider_rate_limit_error` classifica uma `RateLimitError` definida localmente (que o classificador não pode reconhecer) e afirma `retry is True or category == ErrorCategory.UNKNOWN` — a disjunção cobre os dois resultados possíveis. |

Sem cobertura nenhuma: **F1**, **F2**, **D1** (o acoplamento entre providers, não a
configuração), **D3** (a irreversibilidade, não a falha), **D4**, **D5**, **F3**, **E1**,
**C2**, **S1**.

O que **está** bem coberto, e merece registo: breakers e half-open, deduplicação, hedging,
mapeamento de erro de provider → código HTTP, backpressure, tarefas de fundo
(`utils/background.py` é exemplar — referência forte, excepções observadas, `drain()` no
shutdown), throttling de login, e readiness em modo strict/degraded.

---

## 8. As correções com melhor rácio impacto/esforço

| # | Correção | Esforço | O que fecha |
|---|---|---|---|
| 1 | Declarar `REQUEST_FALLBACK_ENABLED: "1"` no catálogo | **1 linha** | D2 — e reativa o detetor de cascata |
| 2 | `where={"tenant_id": tenant_ns}` sempre | **1 linha** | S1, fuga entre clientes |
| 3 | Remover `delete_collection` do caminho de leitura | ~5 linhas | F2, destruição de dados por consulta |
| 4 | `add_document` devolver o resultado real | ~10 linhas | F1, e revive o 500 morto |
| 5 | Preencher `candidates` e `pareto_front` | ~20 linhas | A2, o maior ganho científico |
| 6 | Um breaker por provider + inverter decoradores | ~20 linhas | D1, indisponibilidade em cascata |
| 7 | `connect_args` com timeouts + reinicialização do engine | ~15 linhas | D3, D6 |
| 8 | `set -e` e `&&` no `db_init` | 1 linha | F3 |
| 9 | `QUERY_LOG_RETENTION_DAYS` explícito + log na limpeza | ~3 linhas | A1 |

As correções 1, 2, 3, 4 e 8 somam **menos de 20 linhas** e fecham uma fuga de dados entre
clientes, uma destruição de dados por operação de leitura, dois caminhos que mentem sobre
sucesso e uma indisponibilidade evitável. São o ponto de partida óbvio.

A correção 5 não resolve nenhum bug de funcionamento e é, ainda assim, a mais importante
para a tese: sem ela, a contribuição central do trabalho não é auditável a partir dos
dados que o sistema grava.
