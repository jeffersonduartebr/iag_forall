#!/bin/bash
# Objective: Shell utility for prestart nsga.
# =========================================================
# 🚀 PRESTART NSGA-II
# ---------------------------------------------------------
# Script de inicialização do container nsga_updater.
# - Aguarda Redis e MariaDB ficarem acessíveis.
# - Lê a lista de modelos do arquivo .env (CANDIDATE_MODELS_LIST).
# - Popula a chave Redis `nsga:candidate_models` se ela ainda não existir.
# - Em seguida, inicia o serviço principal (nsga_weights_updater.py).
# =========================================================

set -e

# ---------------------------------------------------------
# 🔧 Carrega variáveis do .env (se existir)
# ---------------------------------------------------------
if [ -f ".env" ]; then
  echo "📦 Carregando variáveis do .env..."
  # Exporta apenas variáveis com formato KEY=VALUE (ignora comentários)
  export $(grep -v '^#' .env | xargs)
else
  echo "⚠️  Arquivo .env não encontrado em /app. Continuando com variáveis de ambiente existentes."
fi

# ---------------------------------------------------------
# 🔍 Definições básicas
# ---------------------------------------------------------
REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-SenhaForte}"
DB_HOST="${DB_HOST:-mariadb}"
DB_PORT="${DB_PORT:-3306}"

# ---------------------------------------------------------
# 🕒 Aguarda Redis e MariaDB ficarem disponíveis
# ---------------------------------------------------------
echo "⏳ Aguardando Redis e MariaDB ficarem prontos..."
until nc -z "$REDIS_HOST" "$REDIS_PORT"; do
  echo "   → Redis ainda não disponível em ${REDIS_HOST}:${REDIS_PORT}"
  sleep 2
done

until nc -z "$DB_HOST" "$DB_PORT"; do
  echo "   → MariaDB ainda não disponível em ${DB_HOST}:${DB_PORT}"
  sleep 2
done

echo "✅ Redis e MariaDB disponíveis."

# ---------------------------------------------------------
# 🧩 Lê lista de modelos do .env
# ---------------------------------------------------------
if [ -z "$CANDIDATE_MODELS_LIST" ]; then
  echo "⚠️  Variável CANDIDATE_MODELS_LIST não encontrada no ambiente."
  echo "🔧 Usando lista padrão para inicialização."
  CANDIDATE_MODELS_LIST='["ollama/deepseek-r1:8b","ollama/phi4:14b","ollama/mistral-nemo:12b","openai/gpt-4o-mini","gemini/gemini-1.5-flash"]'
else
  echo "✅ Lista de modelos carregada do .env:"
  echo "   $CANDIDATE_MODELS_LIST"
fi

# ---------------------------------------------------------
# 💾 Popula o Redis (chave nsga:candidate_models)
# ---------------------------------------------------------
EXISTS=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" get nsga:candidate_models || true)

if [ -z "$EXISTS" ]; then
  echo "🧠 Nenhum modelo detectado no Redis — populando nsga:candidate_models..."
  redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" \
    set nsga:candidate_models "$CANDIDATE_MODELS_LIST" >/dev/null 2>&1 \
    && echo "✅ Modelos adicionados ao Redis com sucesso."
else
  echo "✅ Modelos já existem no Redis — nenhuma ação necessária."
fi

# ---------------------------------------------------------
# 🚀 Inicia o serviço Python (NSGA-II Updater)
# ---------------------------------------------------------
echo "🚀 Iniciando nsga_weights_updater.py..."
exec "$@"
