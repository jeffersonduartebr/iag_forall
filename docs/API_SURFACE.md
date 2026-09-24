# Superfície da API — o que um frontend pode chamar

**Data:** 2026-09-20 · **Alvo:** 96 rotas na app principal + 5 no worker NSGA.

Este documento responde a uma pergunta: **que endpoints devem ser expostos a um
browser?** Não é a mesma pergunta que "quais existem" nem "quais estão
autenticados". Um endpoint pode estar correctamente protegido e mesmo assim não
pertencer a um frontend — porque gasta dinheiro sem tecto, porque devolve um
fluxo infinito, ou porque a sua reversão exige saber o valor anterior.

## Os três níveis

| Nível | Significado | Nº |
|---|---|---:|
| **A** | Um browser pode chamar. Seguro e útil num portal de operador ou de perito. | 70 |
| **B** | Só servidor a servidor. Exige um segredo que o browser não deve guardar, ou é uma API de máquina. | 7 |
| **C** | Nunca expor. Sondas de operação, ou perigoso sem um fluxo de confirmação. | 24 |

O nível **não** substitui a autenticação. Tudo em `/admin/*` exige credenciais;
o nível diz se, *tendo-as*, faz sentido pôr o botão no ecrã.

---

## 1. O que foi corrigido

Cinco defeitos apareceram ao fazer esta análise. Não são exóticos: são erros de
ordem ou de precedência, que se lêem como correctos até se perguntar quem chega
à linha seguinte.

### 1.1 O portal de peritos confiava numa identidade escolhida pelo cliente

`app/app/api/expert_routes.py` — `_expert_id` lia o cabeçalho `X-User-Id`
**antes** de consultar o JWT:

```python
if x_user_id and str(x_user_id).strip():
    return str(x_user_id).strip()[:128]      # ← o cliente escolhe
jwt_user, _, _ = _roles_from_jwt(authorization)
```

Um perito autenticado só tinha de enviar o id de um colega para agir como ele:
ler o perfil, o email e o telefone, ler e escrever as avaliações — e sobretudo
gravar **as etiquetas humanas que calibram o juiz**, que é o dado de validação
mais valioso do sistema.

O cabeçalho continua a ser aceite, porque é assim que serviços internos e a
federação de identidade se apresentam, mas só quando não há JWT e só se
`TRUST_HEADER_ROLES=1`. Um cabeçalho que o cliente escolhe não pode ter
precedência sobre uma identidade assinada.

### 1.2 Seis handlers de avaliação carregavam a corrida antes de autenticar

`app/app/api/eval_routes.py` — `get_eval_run(run_id)` corria antes de
`require_admin_or_role(...)`. Isso dava duas coisas a um chamador anónimo:

- **Um oráculo de existência.** 404 significa "não existe", 401 significa
  "existe, mas não és tu". Dá para enumerar ids de corridas sem credenciais.
- **Uma consulta à base de dados por pedido**, sem autenticação nenhuma.

Trocar as duas linhas não chega: a autorização é por tenant e o tenant só se
conhece depois do lookup. `_authorized_run` faz as duas fases — autentica sem
âmbito, carrega, e só então autoriza com o tenant da corrida.

### 1.3 `/v1/query` descartava o contexto de autenticação

`app/app/main.py` — `v1_route_query` aceitava `auth` e chamava
`route_query(req, request)`. A jusante, isso significa orçamento de tenant vazio
e nenhum dono registado num job em fila, para quem entrasse por `/v1`.

### 1.4 `/admin/models/pricing` devolvia a excepção da base de dados

`detail=f"Pricing unavailable: {exc}"` — a excepção do SQLAlchemy traz o DSN:
host, utilizador e nome da base de dados. Fica no log, onde é útil; sai da
resposta, onde é topologia interna entregue a quem chamou.

### 1.5 `/admin/dashboard/series` aceitava qualquer `step`

Ia sem validação para o `query_range` do Prometheus. Não é uma expressão PromQL
— as cinco consultas são constantes — mas `step=1s` sobre 24 h pede 86 400
pontos por série, cinco séries de cada vez, e é o chamador que escolhe.

---

## 2. Duas propriedades que o teste agora fixa

### 2.1 A autenticação é uma dependência (corrigido)

Cada handler chamava `_auth(...)` ou `resolve_admin_session(...)` ele próprio —
uma linha que tinha de ser repetida 85 vezes e que, faltando, deixava a rota
**pública**. Nada no sistema de tipos, no router ou no diff de revisão apontava
para isso, e a análise estática também não resolvia: alguns handlers delegavam
num helper que autenticava, o que produzia falsos positivos.

Passou a ser uma dependência do FastAPI:

- **Ao nível do router** onde a autorização é uniforme — `admin_routes`,
  `admin_dashboard_routes` e `admin_models_routes`. Aí não há nada para
  esquecer: uma rota acrescentada amanhã fica protegida por construção.
- **Por rota**, com `require_roles(...)`, onde os papéis variam — eval,
  governança e o portal de peritos.
- **`admin_token_only`** para as quatro rotas que rejeitam JWT de propósito
  (as primitivas de RBAC e as estatísticas de feedback). Dar-lhes uma
  dependência própria evita que sejam silenciosamente alargadas.

Seis handlers mantêm uma chamada no corpo, e não é duplicação: a dependência
autentica e verifica os papéis **sem âmbito**, e a chamada interior acrescenta
o âmbito do tenant, que só se conhece depois de ler o caminho ou carregar a
corrida. O comentário no código diz isso, para que ninguém apague uma das duas.

`tests/test_admin_surface_auth.py` enumera as rotas a partir da própria app —
**285 asserções** — e exige 401/403 em todas.

### 2.2 A validação já não precede a autenticação (corrigido)

O FastAPI valida a entrada antes de o handler correr, por isso um POST com
corpo inválido devolvia 422 **sem nunca chegar às credenciais**. Um chamador
anónimo conseguia assim enumerar os campos obrigatórios de toda a superfície de
admin.

As dependências são resolvidas **antes** de os erros de validação acumulados
serem levantados, por isso o 401 chega primeiro. Verificado com uma app mínima:
a mesma rota responde **401 como dependência** e **422 como primeira linha**. O
teste fixa-o em todas as rotas — `test_the_body_is_never_validated_before_the_credentials`.

### 2.3 `PUT /admin/settings` está dividido (corrigido)

Aceitava qualquer chave do catálogo. Quem tivesse uma **sessão** de admin podia
pôr `REQUIRE_API_AUTH=0`, `TRUST_HEADER_ROLES=1`, `ADMIN_UI_CORS_ORIGINS=*` ou
rodar o `JWT_SECRET` — desligar a autenticação da instalação inteira a partir
do browser. Uma sessão roubada bastava.

São agora dois endpoints, e a divisão é de **autorização**, não de
comportamento: o que conta como conhecido, o que exige reinício e como o valor
é serializado são idênticos dos dois lados.

| Endpoint | Chaves | Credencial | Nível |
|---|---|---|---|
| `PUT /admin/settings` | as 220 operacionais | sessão de admin ou token | **A** |
| `PUT /admin/settings/security` | as 13 de segurança | **só** o `ADMIN_TOKEN` mestre | **C** |

Cada um recusa as chaves do outro, e recusa o *batch inteiro* quando vêm
misturadas: aplicar metade seria a pior das três hipóteses.

A classificação é **domínio `auth` ∪ credencial**, e a segunda metade é a que
uma lista escrita à mão esqueceria — `REDIS_PASSWORD` vive no domínio `redis`.
O predicado de credencial é o mesmo que a redacção, a cifra em repouso e a
auditoria já usam; uma quarta lista divergiria das outras três no dia em que
alguém acrescentasse uma chave.

**Defeito encontrado ao fazer isto:** esse predicado casava os marcadores como
*substring*, não como sufixo, e por isso escondia do operador oito definições
puramente operacionais como se fossem segredos — `MAX_TOKENS_DEFAULT`, os três
`RAG_*_CONTEXT_TOKEN_BUDGET`, `REWARD_LATENCY_TOKENS_PER_S`,
`REWARD_DEFAULT_COMPLETION_TOKENS`, `ROUTER_SIMPLE_QUERY_MAX_TOKENS` e
`RAG_SIMPLE_QUERY_BYPASS_ENABLED`, esta última porque "BYPASS" contém "PASS".
Nenhuma era visível em `/admin/settings`. A regra passou a ser por sufixo, com
a lista explícita a continuar a ser a autoridade; verificado que **nenhuma**
chave passa a ser redigida com a mudança.

### 2.4 A allowlist de CORS vale em runtime (corrigido)

O `CORSMiddleware` do Starlette lê `allow_origins` uma vez, no construtor, e a
stack de middleware é montada durante o import de `main`. Mudar
`ADMIN_UI_CORS_ORIGINS` persistia o valor, invalidava as caches, publicava o
reload — e não tinha efeito nenhum até alguém reiniciar o container. O catálogo
marcava-a `requires_restart`, o que era honesto e era também uma admissão.

Só uma decisão depende da lista: `is_allowed_origin`, que o Starlette chama em
cada resposta quando a middleware foi construída com uma lista explícita.
Sobrepor esse método chega, e deixa tudo o resto — `Vary: Origin`, as
credenciais, a pré-computação dos cabeçalhos de preflight — exactamente como a
biblioteca o escreveu.

A classe base é construída de propósito com uma lista **vazia**. É isso que põe
`allow_all_origins` a falso, que é o que encaminha cada pedido por
`is_allowed_origin` e define `Vary: Origin` — o cabeçalho que impede uma cache
partilhada de servir a resposta de uma origem a outra. Uma allowlist dinâmica
sem ele seria um bug de envenenamento de cache.

Degrada para o valor de arranque, não para uma lista vazia: um Redis em baixo
não pode transformar-se em "nenhuma origem é permitida", que partiria o
frontend inteiro por causa de uma falha de infraestrutura.

A chave é agora `runtime_safe` **e** continua classificada como de segurança —
é o par certo: muda sem reinício, mas só por `PUT /admin/settings/security`,
com o token mestre. O browser não pode alargar a sua própria política de CORS.

---

## 3. Nível C — nunca expor

### Operação e sondas
`GET /health`, `GET /v1/health`, `GET /healthz`, `GET /ready`, `GET /metrics`.

`/health` é **não autenticado** e devolve, por componente, a string de erro da
sonda — que em falha de base de dados ou Redis carrega host e utilizador.
`/metrics` fica aberto quando `METRICS_TOKEN` está vazio, que é o default, e as
suas labels incluem ids de tenant e nomes de modelo.

### Worker NSGA (app separada, porta 9999)
`POST /run/{modality}`, `POST /calibration/run`, `GET /calibration/status`,
`GET /metrics`, `GET /health` — **zero autenticação em todas**.

`POST /run/{modality}` reescreve os pesos `NSGA_W_*`, o limiar de incerteza e as
quotas de recompensa do bandit **para toda a plataforma**. Mitigado por o
compose não publicar a porta: só é alcançável de dentro da rede Docker. Isso
torna-o um problema de movimento lateral, não de exposição directa — mas
qualquer container nessa rede reescreve a política de routing.

### Destrutivo ou irreversível sem confirmação
- `POST /admin/runtime/reset` — apaga os singletons de provider, reliability,
  router, **bandit** e vectorstore. O estado de routing aprendido perde-se, e
  atinge só o worker que servir o pedido: a frota fica inconsistente.
- `DELETE /admin/experiments/{id}` — sem desfazer.
- `POST /admin/evals/tasks/{id}/cancel` — revoga **qualquer** tarefa Celery pelo
  id, não só as de avaliação; com `terminate=true` mata o processo filho.
- `PUT /admin/settings/security` — as 13 chaves que decidem *quem entra*.
  Exige o `ADMIN_TOKEN` mestre e rejeita a sessão de admin, pela mesma razão
  que as rotas de RBAC o fazem: o browser não deve guardar a credencial que
  permite mudar a política de acesso. (`PUT /admin/settings`, com o resto das
  chaves, é nível A — ver §5.)
- `POST /admin/policies/{version}/activate` e
  `POST /admin/evals/runs/{id}/execute` — o primeiro muda o routing de 100% do
  tráfego, o segundo gasta sem tecto.
- `GET /admin/logs/stream` — o único endpoint que entrega uma string do browser
  a uma linguagem de consulta de backend (LogQL) e devolve um fluxo que nunca
  termina.

### Documentação
`/openapi.json`, `/docs`, `/docs/oauth2-redirect`, `/redoc` — desligados quando
`ENV=production`. **Confirmar que `ENV` está mesmo definido no deployment:** o
default é `development`, e estas rotas publicam o mapa inteiro da superfície de
admin.

---

## 4. Nível B — só servidor a servidor

| Endpoint | Porquê |
|---|---|
| `POST /v1/chat/completions` | API de compatibilidade OpenAI: é para clientes, não para o painel. |
| `POST /rag/add_doc`, `POST /rag/ingest`, `POST /rag/delete` | Escrevem na base de conhecimento. Sem tenant, ela é **partilhada por todas as queries**. Com um token vinculado a tenant, `/rag/ingest` carimba `tenant_id` (o do cliente é descartado) e prefixa o `doc_id` com o tenant, de modo que um tenant não sobrescreve outro. `/rag/delete` remove um escopo e exige tenant ou admin. Sem tenant, o `doc_id` continua sendo escolhido pelo chamador (e sobrescreve) e o `metadata` é arbitrário. |
| `GET /feedback/stats`, `POST /admin/rbac/{grants,revokes}`, `GET /admin/rbac/roles` | `require_admin` sem `authorization`: aceitam **só o `ADMIN_TOKEN` em bruto**, não o JWT de admin. Um browser teria de guardar o token mestre. Os RBAC são além disso primitivas de escalada de privilégio. |

---

## 5. Nível A — pode ir para o browser

70 endpoints: autenticação, dashboard (menos `logs/stream`), modelos e
exploração OpenRouter, peritos, avaliações (menos `execute` e `tasks/cancel`),
governança (menos RBAC e activação de política), `POST /feedback`, e a
superfície de produto `/query*` e `/v1/query*`.

Vale a pena saber três coisas antes de construir sobre eles.

**Limites de resposta.** `GET /admin/models/openrouter` devolve o catálogo
inteiro (~300 modelos, centenas de KB) sem parâmetro de limite.
`GET /admin/evals/runs/{id}/results` tem `limit` com **default 2000** e tecto de
20 000. `GET /admin/experts/accounts` e `GET /admin/budgets` não têm limite.

**Visibilidade entre tenants.** `GET /admin/dashboard/summary`,
`/admin/quotas/usage`, `/admin/budgets` e `/admin/reviews` mostram todos os
tenants. `GET /admin/dashboard/roi` aceita um `tenant_id` escolhido pelo
chamador, não ligado à sessão.

**Onde está o PII.** `GET /admin/reviews` é a maior densidade de dados pessoais
da API: prompts e respostas de utilizadores finais, de todos os tenants.
`GET /admin/experts/accounts` devolve email e telefone de cada perito.
`GET /admin/audit/events` carrega emails no `metadata` — mas, desde a auditoria
de robustez, os valores de segredos vão redigidos.

**Dois endpoints de nível A gastam dinheiro:**
`POST /admin/experts/preview-answer` faz uma chamada real ao modelo com o cache
forçado a desligado, e `POST /feedback` altera a recompensa do bandit — com
`REQUIRE_API_AUTH=0`, que é o default, é um vector de envenenamento da
recompensa aberto a qualquer um.

---

---

## 6. O que fica por decidir

Nada. Os itens desta secção foram fechados: a autenticação é uma dependência
(§2.1), a validação já não a precede (§2.2), `PUT /admin/settings` está
dividido (§2.3) e a allowlist de CORS vale em runtime (§2.4).
