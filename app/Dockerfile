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
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# 📦 Instala dependências Python
# =========================================================
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir gunicorn && \
    pip install --no-cache-dir python-dotenv

# =========================================================
# 🧩 Configuração de ambiente
# =========================================================
ENV PATH="/app:${PATH}"
ENV PYTHONPATH="/app"

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
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
