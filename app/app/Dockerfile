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
COPY ./requirements.txt /app/requirements.txt

# ==============================
#   STAGE 2 - Dependências
# ==============================
FROM base AS deps

# Força atualização e instalação dos pacotes necessários
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        pydantic==2.12.0 \
        pydantic-settings==2.5.2 \
        litellm \
        numpy \
        onnxruntime \
        prometheus_client \
        aiohttp \
        httpx

# ==============================
#   STAGE 3 - Execução
# ==============================
FROM deps AS runtime

WORKDIR /app

# Define variáveis padrão
ENV PYTHONPATH=/app \
    LOG_LEVEL=INFO

# Exposição da porta da API
EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
