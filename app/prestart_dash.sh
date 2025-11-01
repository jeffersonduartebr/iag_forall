#!/bin/bash
echo "🚀 Inicializando Painel Dash..."
echo "🔍 Verificando Redis e Banco de Dados..."

# Aguarda Redis
until nc -z -v -w30 $REDIS_HOST $REDIS_PORT
do
  echo "⏳ Aguardando Redis em $REDIS_HOST:$REDIS_PORT..."
  sleep 2
done

# Aguarda Banco de Dados
until nc -z -v -w30 $DB_HOST 3306
do
  echo "⏳ Aguardando MariaDB em $DB_HOST:3306..."
  sleep 2
done

echo "✅ Dependências disponíveis. Iniciando aplicação Dash..."
exec "$@"
