# ==============================
#   STAGE 1 - Base Image
# ==============================
FROM python:3.11-slim AS base

# Evita criação de .pyc e melhora logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instala dependências básicas do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia os arquivos da aplicação
COPY ./app /app/app

# ==============================
#   STAGE 2 - Dependências
# ==============================
FROM base AS deps

# uv: resolução e instalação a partir de um lock com hash por pacote.
# A lista solta que existia aqui divergia de requirements.txt (pydantic 2.12.0
# contra o pin real), e uma imagem que instala outras versões não reproduz nada.
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv
COPY ./requirements.lock /app/requirements.lock

RUN uv pip install --system --no-cache --require-hashes -r /app/requirements.lock

# ==============================
#   STAGE 3 - Execução
# ==============================
FROM deps AS runtime

RUN mkdir -p /tmp/prom && chmod 777 /tmp/prom
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prom

WORKDIR /app

# Define variáveis padrão
ENV PYTHONPATH=/app \
    LOG_LEVEL=INFO

# Exposição da porta da API
EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
