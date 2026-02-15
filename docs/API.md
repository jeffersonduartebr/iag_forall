# Documentação da API

Base URL local:
```text
http://localhost:8000
```

## Autenticação administrativa
Endpoints administrativos exigem header:
```text
X-Admin-Token: <token>
```

## Endpoints principais

## `POST /query`
Roteia uma consulta para o melhor modelo disponível.

### Payload mínimo
```json
{
  "query": "Explique cache semântico.",
  "modality": "text"
}
```

### Campos suportados
- `query` (string, obrigatório)
- `modality` (`text|vision|multimodal`, opcional, default `text`)
- `system_prompt` (string)
- `max_tokens` (int)
- `temperature` (float)
- `image_b64` (string)
- `images` (array de base64)
- `enable_rag_for_answer` (bool)
- `enable_rag_for_image` (bool)
- `rag_modality` (`text|vision|multimodal`)
- `use_cache` (bool)
- `timeout_seconds` (int)

### Resposta (estrutura)
- `answer`: texto final
- `model`: modelo usado
- `modality`: modalidade final
- `route`: decisão de roteamento
- `candidates`: candidatos avaliados
- `correlation_id`: id de rastreio
- `payload`: payload bruto tratado

### Erros comuns
- `400`: payload inválida
- `429`: rate limit ou provider rate-limit
- `502`: falha de provider
- `503`: circuit breaker aberto/backpressure
- `504`: timeout

## `GET /health`
Health check completo dos componentes.

## `GET /healthz`
Liveness check simples.

## `GET /ready`
Readiness check (dependências críticas).

## `GET /metrics`
Métricas Prometheus.

## Endpoints administrativos
- `GET /admin/settings`
- `PUT /admin/settings`
- `GET /admin/circuit-breakers`
- `POST /admin/circuit-breakers/{model_name}/reset`
- `GET /admin/cascade-status`
- `POST /admin/runtime/reset`

## Feedback e A/B
- `POST /feedback`
- `GET /feedback/stats`
- Endpoints de experimento A/B em `/admin/experiments*`

## Exemplo de chamada com `curl`
```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Qual a diferença entre UCB1 e Thompson Sampling?","modality":"text"}' | jq
```

## Boas práticas para clientes
1. Enviar `correlation-id` quando possível.
2. Tratar `429/503/504` com retry exponencial no cliente.
3. Evitar payloads gigantes sem necessidade.
4. Definir timeout de cliente coerente com `timeout_seconds` enviado.
