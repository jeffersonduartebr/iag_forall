#!/bin/bash
# ============================================================
# 🚀 prestart_nsga_populate.sh
# Inicializador do container NSGA-II
# Jefferson Duarte (IAG ForAll)
# ============================================================

echo "🔧 [NSGA-INIT] Aguardando serviços..."

# Espera o banco de dados
until nc -z ${DB_HOST:-mariadb} 3306; do
  echo "⏳ Aguardando MariaDB em ${DB_HOST:-mariadb}:3306..."
  sleep 3
done

# Espera o Redis
until nc -z ${REDIS_HOST:-redis} ${REDIS_PORT:-6379}; do
  echo "⏳ Aguardando Redis em ${REDIS_HOST:-redis}:${REDIS_PORT:-6379}..."
  sleep 3
done

echo "✅ Serviços disponíveis. Iniciando verificação de configuração Redis..."

# Testa conexão ao Redis
if ! redis-cli -h "${REDIS_HOST:-redis}" -p "${REDIS_PORT:-6379}" -a "${REDIS_PASSWORD:-SenhaForte}" ping >/dev/null 2>&1; then
  echo "❌ Falha ao conectar ao Redis (${REDIS_HOST:-redis}:${REDIS_PORT:-6379})."
  exit 1
fi

# Verifica se há lista de modelos
MODELS_EXIST=$(redis-cli -h "${REDIS_HOST:-redis}" -p "${REDIS_PORT:-6379}" -a "${REDIS_PASSWORD:-SenhaForte}" get nsga:candidate_models)

if [ -z "$MODELS_EXIST" ]; then
  echo "⚠️ Nenhum modelo encontrado em Redis. Populando chave 'nsga:candidate_models'..."
  redis-cli -h "${REDIS_HOST:-redis}" -p "${REDIS_PORT:-6379}" -a "${REDIS_PASSWORD:-SenhaForte}" \
    set nsga:candidate_models '["ollama/deepseek-r1:8b","openai/gpt-4o-mini","gemini/gemini-1.5-flash","claude/haiku-3","mistral/mistral-nemo"]'
else
  echo "✅ Modelos já configurados no Redis."
fi

echo "🚀 Executando inicialização do banco..."
python /app/app/db_manager.py || echo "⚠️ Banco já sincronizado."

echo "🧠 Iniciando NSGA-II..."
exec "$@"
