# Operações — Produção em larga escala (sem Kubernetes)

Guia operacional para deploy multi-réplica do `iag_forall` com Docker Compose ou VMs.

## Checklist de bootstrap (produção)

| Variável | Valor em produção |
|----------|-------------------|
| `ENV` | `production` |
| `REQUIRE_API_AUTH` | `1` |
| `API_KEYS` ou `JWT_SECRET` | obrigatório |
| `METRICS_TOKEN` | obrigatório |
| `ROADMAP_AUTO_DDL` | `0` (migrations via Alembic) |
| `REDIS_REQUIRED_IN_PRODUCTION` | `1` |
| `ENFORCE_TENANT_BINDING` | `1` |
| `ADMIN_TOKEN` | valor forte, não default |

## Rede e TLS

- Expor na LAN apenas **API `:8000`** e **Admin UI `:8082`**.
- Demais serviços (MariaDB, Redis, Prometheus, Grafana, Loki) em `127.0.0.1`.
- Terminar TLS na borda com `deploy/nginx-api.conf` (template incluído).
- Admin UI preferencialmente em VPN ou rede interna.

## Migrations

```bash
docker compose run --rm db_init
# ou
cd app && alembic upgrade head
```

Nunca use `ROADMAP_AUTO_DDL=1` em produção.

## Backups

```bash
export DB_PASS=...
export REDIS_PASSWORD=...
./scripts/backup.sh
```

Restore testado mensalmente:

```bash
./scripts/restore.sh 20260704_120000
```

**RPO sugerido:** 1h (backup diário + Redis AOF). **RTO sugerido:** 4h.

## Escala horizontal (stateless)

| Componente | Config |
|------------|--------|
| Chroma compartilhado | `CHROMA_HOST=<host>` `CHROMA_PORT=8000` |
| Backpressure global | `BACKPRESSURE_REDIS_ENABLED=1` |
| Rate limit tenant | `TENANT_RATE_LIMIT_ENABLED=1` (middleware sempre ativo) |
| Idempotência | header `Idempotency-Key` + Redis |
| Jobs 202 | webhook via `webhook_url` no `QueryRequest` |

Subir N réplicas da API apontando ao mesmo Redis/MariaDB/Chroma.

## Observabilidade

- Prometheus: `http://127.0.0.1:9091`
- Alertmanager: `http://127.0.0.1:9093`
- Grafana: `http://127.0.0.1:3001`
- Loki: restrito a localhost; colocar auth no reverse proxy em produção
- Métricas API: `GET /metrics` com header `X-Metrics-Token`
- Tracing opcional: `OTEL_EXPORTER_OTLP_ENDPOINT` + `OTEL_SERVICE_NAME`

## Probes

| Probe | Endpoint | Critério |
|-------|----------|----------|
| Liveness | `/healthz` | processo vivo |
| Readiness | `/ready` | Redis + MariaDB (`READINESS_MODE=strict`) |

## API compatível OpenAI

`POST /v1/chat/completions` — adapter para clientes SDK OpenAI.

Contrato nativo continua em `POST /query`.

## Rotação de segredos

1. Gerar novo `JWT_SECRET` / API keys
2. Atualizar `.env` e reiniciar API
3. Para admin: rotacionar `ADMIN_TOKEN` e `ADMIN_UI_PASSWORD`
4. `METRICS_TOKEN`: atualizar Prometheus scrape config se necessário

## Incidentes (MTTR alvo < 30 min P1)

1. Verificar `/ready` e `/health`
2. Grafana dashboards `llm_router_operations`
3. Logs Loki: filtrar `correlation_id`
4. Circuit breakers abertos → fallback models / reduzir carga
5. Fila jobs → escalar workers Celery

## CI/CD

- PR: unit tests + lint + mypy + smoke integration
- Staging: `docker compose up -d --build` + Locust baseline
- Produção: migrations → backup → rolling restart API
