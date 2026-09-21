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
- `PUT /admin/settings` — o maior raio de dano da API. Qualquer chave do
  catálogo, persistida e propagada a todos os processos. Inclui
  `REQUIRE_API_AUTH`, `TRUST_HEADER_ROLES`, `ADMIN_UI_CORS_ORIGINS`,
  `JWT_SECRET`, `API_KEYS`. Reversível só se souberes o valor anterior — o que,
  desde a auditoria de robustez, o `audit_log` regista.
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
| `POST /rag/add_doc`, `POST /rag/ingest` | Escrevem na base de conhecimento **partilhada por todas as queries**, e não há endpoint de remoção: o envenenamento é irreversível pela API. `/rag/ingest` aceita `doc_id` escolhido pelo chamador (sobrescreve) e um `metadata` arbitrário. |
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

## 6. O que fica por decidir

**`PUT /admin/settings` merece ser dividido.** Uma lista permitida de chaves
operacionais seria nível A; as do domínio `auth` continuariam C. Hoje é tudo ou
nada, e a UI de admin já o usa.

**`ADMIN_UI_CORS_ORIGINS` exige reinício.** O middleware CORS é montado no
import de `main.py`, por isso está marcado `requires_restart` no catálogo.
Rotulá-lo `runtime_safe` faria a UI prometer uma coisa que não acontece.
