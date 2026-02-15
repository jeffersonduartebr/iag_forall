# Onboarding Técnico (Estagiário)

Este guia foi feito para alguém que está entrando agora no projeto e precisa começar a contribuir com segurança.

## Objetivo da primeira semana
Ao final da semana você deve conseguir:
1. Subir o ambiente local.
2. Executar e depurar `POST /query`.
3. Alterar uma configuração dinâmica com segurança.
4. Entender o caminho de feedback assíncrono.
5. Abrir PR com mudança pequena + teste + documentação.

## Dia 1 - Ambiente e arquitetura
1. Leia `README.md` (seções de quickstart e fluxo da requisição).
2. Leia `docs/ARCHITECTURE.md`.
3. Suba a stack:
```bash
docker compose up -d --build
```
4. Teste saúde e consulta básica.

Checklist:
- API responde em `/health`.
- Consulta básica retorna `200`.
- Você consegue identificar modelo escolhido na resposta.

## Dia 2 - API e contratos
1. Leia `docs/API.md`.
2. Abra `app/app/main.py` e localize:
- Endpoint `/query`.
- Endpoints de health.
- Endpoints administrativos.
3. Entenda os schemas em `app/app/schemas.py`.

Exercício:
- Adicione uma consulta com imagem base64 e entenda mudança de modalidade.

## Dia 3 - Roteamento e providers
1. Leia `app/app/router_core.py`.
2. Leia `app/app/providers_async.py`.
3. Leia `app/app/reliability.py`.

Foco:
- Cache hit vs execução normal.
- Fallback chain.
- Tratamento de timeout/rate-limit/circuit-open.

## Dia 4 - Dados e configurações
1. Leia `app/app/settings_dynamic.py`.
2. Leia `docs/CONFIGURATION.md`.
3. Leia `app/app/db.py`, `app/app/query_service.py`, `app/app/metrics_collector.py`.

Exercício:
- Alterar uma configuração via `/admin/settings` e observar efeito.

## Dia 5 - Observabilidade e feedback
1. Leia `app/app/observability.py` e `app/app/health.py`.
2. Leia `app/app/tasks.py` e `app/app/user_feedback.py`.
3. Execute uma sequência de consultas e acompanhe métricas.

Exercício final:
- Documentar um bug encontrado em formato: sintoma -> hipótese -> evidência -> ação.

## Regras de contribuição
1. Mudou comportamento? Atualize docstring + `docs/*.md`.
2. Toda mudança deve ter evidência mínima (teste, log, ou reprodução).
3. PR pequena e focada.

## Dicionário rápido
- Bandit: política online de escolha entre modelos candidatos.
- NSGA: ponderação multiobjetivo para qualidade/custo/latência.
- UQ: estimativa de incerteza da consulta.
- Fallback chain: sequência de modelos alternativos quando o principal falha.
- EMA: suavização de métricas históricas por modelo/modalidade.
