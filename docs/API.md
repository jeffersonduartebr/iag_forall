# API Documentation

This document describes all API endpoints for the Multi-Objective LLM Router.

## Base URL

```
http://localhost:8000
```

## Authentication

Admin endpoints require the `X-Admin-Token` header:

```bash
curl -H "X-Admin-Token: your-admin-token" http://localhost:8000/admin/settings
```

---

## Endpoints

### Query Endpoint

#### POST /query

Route a query to the optimal LLM and return the response.

**Request:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "modality": "text",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | The query text |
| `modality` | string | No | `text` | Query type: `text`, `vision`, `multimodal` |
| `system_prompt` | string | No | - | System prompt to prepend |
| `max_tokens` | integer | No | 2000 | Maximum response tokens |
| `temperature` | float | No | 0.55 | Generation temperature |
| `image_b64` | string | No | - | Base64-encoded image (for vision) |
| `images` | array | No | - | Array of base64 images |
| `enable_rag_for_answer` | boolean | No | false | Enable RAG augmentation |
| `enable_rag_for_image` | boolean | No | false | Enable image RAG |
| `rag_modality` | string | No | `text` | RAG search modality |
| `use_cache` | boolean | No | true | Use semantic cache |
| `timeout_seconds` | integer | No | 120 | Request timeout override |

**Response:**

```json
{
  "answer": "The capital of France is Paris.",
  "model": "ollama/phi4:latest",
  "modality": "text",
  "image_output_b64": null,
  "correlation_id": "abc123-def456",
  "route": {
    "chosen_model": "ollama/phi4:latest",
    "modality_selected": "text",
    "is_multimodal_route": false,
    "objectives": {
      "latency": 1.23,
      "cost": 0.0001,
      "uncertainty": 0.35
    },
    "pareto_front": [],
    "explanation": "Selected ollama/phi4:latest (UQ=0.35)"
  },
  "candidates": [],
  "payload": {}
}
```

**Status Codes:**

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Invalid request (missing query) |
| 503 | System at capacity (backpressure) |
| 504 | Request timeout |
| 500 | Internal server error |

---

### Vision Query Example

```bash
# Encode image to base64
IMAGE_B64=$(base64 -w0 image.jpg)

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"What is in this image?\",
    \"modality\": \"vision\",
    \"image_b64\": \"$IMAGE_B64\"
  }"
```

---

### Health Endpoints

#### GET /health

Deep health check with all component statuses.

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": 1706500000.123,
  "duration_ms": 45.67,
  "cached": false,
  "components": {
    "redis": {
      "name": "redis",
      "healthy": true,
      "latency_ms": 1.23,
      "details": {"pool_size": 10}
    },
    "mariadb": {
      "name": "mariadb",
      "healthy": true,
      "latency_ms": 5.67
    },
    "chromadb": {
      "name": "chromadb",
      "healthy": true,
      "latency_ms": 12.34,
      "details": {"collections": 4}
    },
    "ollama": {
      "name": "ollama",
      "healthy": true,
      "latency_ms": 8.90,
      "details": {"models_loaded": 3, "models": ["phi4:latest", "gemma3:4b"]}
    },
    "circuit_breakers": {
      "name": "circuit_breakers",
      "healthy": true,
      "details": {"total": 5, "open": 0, "open_models": []}
    }
  },
  "summary": {
    "healthy": 5,
    "total": 5
  }
}
```

**Status Codes:**

| Code | Status Value | Description |
|------|--------------|-------------|
| 200 | `healthy` | All components operational |
| 200 | `degraded` | Some components failing |
| 503 | `unhealthy` | Critical components down |

#### GET /healthz

Kubernetes liveness probe.

```bash
curl http://localhost:8000/healthz
```

**Response:**

```json
{
  "status": "alive",
  "timestamp": 1706500000.123
}
```

#### GET /ready

Kubernetes readiness probe.

```bash
curl http://localhost:8000/ready
```

**Response:**

```json
{
  "status": "ready",
  "timestamp": 1706500000.123,
  "redis": true,
  "database": true
}
```

---

### Metrics Endpoint

#### GET /metrics

Prometheus metrics in OpenMetrics format.

```bash
curl http://localhost:8000/metrics
```

**Response:** (text/plain)

```
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total 12345

# HELP api_latency_seconds Request latency
# TYPE api_latency_seconds histogram
api_latency_seconds_bucket{le="0.1"} 5000
api_latency_seconds_bucket{le="0.5"} 10000
...
```

---

### Feedback Endpoint

#### POST /feedback

Submit user feedback for a model response.

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "abc123",
    "model": "ollama/phi4:latest",
    "thumbs_up": true
  }'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query_id` | string | No | Query identifier |
| `model` | string | Yes | Model that generated response |
| `query` | string | No | Original query text |
| `answer` | string | No | Model's response |
| `thumbs_up` | boolean | No | Positive feedback |
| `thumbs_down` | boolean | No | Negative feedback |
| `rating` | integer | No | 1-5 star rating |
| `explicit_quality` | float | No | 0-10 quality score |

**Response:**

```json
{
  "status": "accepted",
  "user_quality": 8.5,
  "blended_quality": 7.8,
  "model": "ollama/phi4:latest",
  "reward": 0.7234
}
```

#### GET /feedback/stats

Get feedback statistics.

```bash
curl "http://localhost:8000/feedback/stats?model=ollama/phi4:latest&hours=24"
```

---

### Admin Endpoints

All admin endpoints require `X-Admin-Token` header.

#### GET /admin/settings

Get current system settings.

```bash
curl http://localhost:8000/admin/settings \
  -H "X-Admin-Token: your-token"
```

#### PUT /admin/settings

Update system settings.

```bash
curl -X PUT http://localhost:8000/admin/settings \
  -H "X-Admin-Token: your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "BANDIT_EPSILON": "0.15",
    "NSGA_W_QUALITY": "1.5"
  }'
```

#### GET /admin/circuit-breakers

Get circuit breaker status for all models.

```bash
curl http://localhost:8000/admin/circuit-breakers \
  -H "X-Admin-Token: your-token"
```

**Response:**

```json
{
  "circuit_breakers": [
    {
      "model": "openai/gpt-4o",
      "state": "closed",
      "fail_counter": 0,
      "fail_max": 5,
      "reset_timeout": 60
    },
    {
      "model": "ollama/phi4:latest",
      "state": "open",
      "fail_counter": 5,
      "fail_max": 3,
      "reset_timeout": 30
    }
  ],
  "timestamp": 1706500000.123
}
```

#### POST /admin/circuit-breakers/{model_name}/reset

Reset a specific circuit breaker.

```bash
curl -X POST "http://localhost:8000/admin/circuit-breakers/ollama%2Fphi4:latest/reset" \
  -H "X-Admin-Token: your-token"
```

#### GET /admin/cascade-status

Get cascade failure detection status.

```bash
curl http://localhost:8000/admin/cascade-status \
  -H "X-Admin-Token: your-token"
```

**Response:**

```json
{
  "severity": 0,
  "severity_name": "normal",
  "failed_model_ratio": 0.1,
  "failed_models": 1,
  "total_models": 10,
  "is_emergency_mode": false,
  "is_degraded": false,
  "emergency_fallback": null,
  "thresholds": {
    "warning": 0.3,
    "critical": 0.5,
    "emergency": 0.8
  }
}
```

---

### A/B Testing Endpoints

#### GET /admin/experiments

List all A/B experiments.

```bash
curl "http://localhost:8000/admin/experiments?status=running" \
  -H "X-Admin-Token: your-token"
```

#### POST /admin/experiments

Create a new experiment.

```bash
curl -X POST http://localhost:8000/admin/experiments \
  -H "X-Admin-Token: your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "phi4-vs-gemma",
    "description": "Compare Phi-4 against Gemma 3",
    "control_model": "ollama/gemma3:4b",
    "treatment_model": "ollama/phi4:latest",
    "traffic_percentage": 0.1
  }'
```

#### POST /admin/experiments/{id}/start

Start an experiment.

```bash
curl -X POST http://localhost:8000/admin/experiments/exp-123/start \
  -H "X-Admin-Token: your-token"
```

#### POST /admin/experiments/{id}/pause

Pause an experiment.

```bash
curl -X POST http://localhost:8000/admin/experiments/exp-123/pause \
  -H "X-Admin-Token: your-token"
```

#### POST /admin/experiments/{id}/complete

Complete an experiment.

```bash
curl -X POST http://localhost:8000/admin/experiments/exp-123/complete \
  -H "X-Admin-Token: your-token"
```

#### GET /admin/experiments/{id}/results

Get experiment results.

```bash
curl http://localhost:8000/admin/experiments/exp-123/results \
  -H "X-Admin-Token: your-token"
```

#### DELETE /admin/experiments/{id}

Delete an experiment.

```bash
curl -X DELETE http://localhost:8000/admin/experiments/exp-123 \
  -H "X-Admin-Token: your-token"
```

---

### RAG Endpoints

#### POST /rag/add

Add a document to the RAG vector store.

```bash
curl -X POST http://localhost:8000/rag/add \
  -H "Content-Type: application/json" \
  -d '{
    "modality": "text",
    "doc_id": "doc-123",
    "text": "This is the document content.",
    "metadata": {"source": "manual", "category": "faq"}
  }'
```

#### POST /rag/search

Search the RAG vector store.

```bash
curl -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I reset my password?",
    "modality": "text",
    "top_k": 5
  }'
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": {
    "error_id": "err-abc123",
    "category": "provider_timeout",
    "message": "Request timed out after 120 seconds",
    "timestamp": 1706500000.123,
    "correlation_id": "corr-xyz789"
  }
}
```

### Error Categories

| Category | HTTP Code | Description |
|----------|-----------|-------------|
| `validation_error` | 400 | Invalid request data |
| `authentication_error` | 401 | Invalid admin token |
| `not_found` | 404 | Resource not found |
| `rate_limit_exceeded` | 429 | Too many requests |
| `backpressure` | 503 | System at capacity |
| `provider_timeout` | 504 | LLM request timeout |
| `provider_error` | 502 | LLM provider error |
| `circuit_open` | 503 | Circuit breaker open |
| `internal_error` | 500 | Unexpected error |

---

## Rate Limiting

The API implements Redis-based rate limiting:

- **Default**: 100 requests per minute per IP
- **Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

When exceeded:

```json
{
  "detail": "Rate limit exceeded. Retry after 60 seconds.",
  "retry_after": 60
}
```

---

## Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | `application/json` |
| `X-Admin-Token` | Admin only | Admin authentication |
| `X-Correlation-ID` | No | Request tracing ID (auto-generated if missing) |
| `Accept-Encoding` | No | `gzip` for compressed responses |

---

## Response Headers

| Header | Description |
|--------|-------------|
| `X-Correlation-ID` | Request tracing ID |
| `X-RateLimit-Limit` | Rate limit max |
| `X-RateLimit-Remaining` | Remaining requests |
| `X-RateLimit-Reset` | Reset timestamp |
| `Content-Encoding` | `gzip` if compressed |

---

## SDK Examples

### Python

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000")

# Simple query
response = client.post("/query", json={
    "query": "What is machine learning?",
    "modality": "text"
})
result = response.json()
print(result["answer"])

# With image
import base64
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.post("/query", json={
    "query": "Describe this image",
    "modality": "vision",
    "image_b64": image_b64
})
```

### JavaScript

```javascript
// Simple query
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'What is machine learning?',
    modality: 'text'
  })
});
const result = await response.json();
console.log(result.answer);
```

### cURL with Timeout

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  --max-time 180 \
  -d '{
    "query": "Complex reasoning question...",
    "modality": "text",
    "timeout_seconds": 180
  }'
```
