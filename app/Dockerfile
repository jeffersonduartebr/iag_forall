# =========================================================
# 🧱 Base
# =========================================================
FROM python:3.11-slim AS base

LABEL maintainer="Jefferson Duarte <jefferson.igorbr@gmail.com>"
LABEL description="Router LLMs + Painel + NSGA-II + Observabilidade"

# Evita prompts interativos
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# =========================================================
# 📁 Copia os arquivos da aplicação
# =========================================================
COPY . /app

# Instala dependências básicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmariadb-dev \
    pkg-config \
    curl \
    vim \
    tesseract-ocr \
    tesseract-ocr-por \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# 📦 Instala dependências Python
# =========================================================
# uv: resolução e instalação a partir de um lock com hash por pacote.
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv
COPY requirements.lock /app/requirements.lock

RUN uv pip install --system --no-cache --require-hashes -r /app/requirements.lock

# =========================================================
# 🧩 Configuração de ambiente
# =========================================================
ENV PATH="/app:${PATH}"
ENV PYTHONPATH="/app"

ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prom \
    PYTHONUNBUFFERED=1 \
    CHROMA_TELEMETRY_ENABLED=false

# Cria diretório compartilhado de métricas
RUN mkdir -p /tmp/prom && chmod -R 777 /tmp/prom
# =========================================================
# 🛠 Prestart: inicialização de banco e Redis
# =========================================================
RUN echo '#!/bin/bash\n\
    echo "🚀 Executando prestart: inicializando banco e Redis..."\n\
    python /app/app/db_manager.py || echo "⚠️ Falha na inicialização automática do banco"\n\
    echo "✅ Prestart finalizado."\n\
    exec "$@"' > /app/prestart.sh && chmod +x /app/prestart.sh

# =========================================================
# 🔧 Healthcheck básico
# =========================================================
HEALTHCHECK --interval=10s --timeout=5s --retries=5 CMD curl -f http://localhost:8000/health || exit 1

# =========================================================
# 🧩 Entrada padrão
# =========================================================
ENTRYPOINT ["/app/prestart.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
