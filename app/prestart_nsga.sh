#!/bin/bash
# ============================================================
# 🚀 prestart_nsga_populate.sh
# Inicializador do container NSGA-II (IAG ForAll)
# Jefferson Duarte — IFRN / IAG ForAll
# ============================================================

# ---------- Configuração visual e função de log ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # Sem cor

log() {
  echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
  echo -e "${GREEN}✅ $1${NC}"
}

warn() {
  echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
  echo -e "${RED}❌ $1${NC}"
}

# ---------- Variáveis com fallback ----------
DB_HOST=${DB_HOST:-mariadb}
REDIS_HOST=${REDIS_HOST:-redis}
REDIS_PORT=${REDIS_PORT:-6379}
REDIS_PASSWORD=${REDIS_PASSWORD:-SenhaForte}

# ---------- Início ----------
log "🔧 [NSGA-INIT] Iniciando validações de infraestrutura..."

# ---------- Espera por MariaDB ----------
log "⏳ Aguardando conexão com MariaDB (${DB_HOST}:3306)..."
until nc -z "$DB_HOST" 3306 >/dev/null 2>&1; do
  warn "MariaDB ainda não está pronto. Tentando novamente..."
  sleep 3
done
success "MariaDB disponível em ${DB_HOST}:3306"

# Exibe versão do MariaDB (quando possível)
DB_VERSION=$(mysql -h "$DB_HOST" -u "${DB_USER:-router_user}" -p"${DB_PASS:-router_pass}" -e "SELECT VERSION();" 2>/dev/null | tail -n 1)
if [ -n "$DB_VERSION" ]; then
  log "📦 Versão do MariaDB detectada: ${DB_VERSION}"
else
  warn "Não foi possível obter a versão do MariaDB (sem acesso direto ou cliente ausente)."
fi

# ---------- Espera por Redis ----------
log "⏳ Aguardando conexão com Redis (${REDIS_HOST}:${REDIS_PORT})..."
until nc -z "$REDIS_HOST" "$REDIS_PORT" >/dev/null 2>&1; do
  warn "Redis ainda não respondeu. Tentando novamente..."
  sleep 3
done
success "Redis disponível em ${REDIS_HOST}:${REDIS_PORT}"

log "🔐 Testando autenticação Redis diretamente..."
AUTH_TEST=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" ping 2>&1)
if [[ "$AUTH_TEST" == *"WRONGPASS"* ]]; then
  error "Senha incorreta para o Redis! Verifique variável REDIS_PASSWORD (${REDIS_PASSWORD})."
  exit 1
elif [[ "$AUTH_TEST" != *"PONG"* ]]; then
  error "Redis inacessível ou outro erro: $AUTH_TEST"
  exit 1
else
  success "Autenticação Redis confirmada via teste direto."
fi


# Exibe versão do Redis (quando possível)
REDIS_VERSION=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" info server 2>/dev/null | grep -m1 "redis_version:" | cut -d':' -f2)
if [ -n "$REDIS_VERSION" ]; then
  log "📦 Versão do Redis detectada: ${REDIS_VERSION}"
else
  warn "Não foi possível obter a versão do Redis (autenticação ou conexão falhou)."
fi

# ---------- Teste de autenticação Redis ----------
log "🔍 Verificando autenticação Redis..."
if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" ping >/dev/null 2>&1; then
  error "Falha ao autenticar no Redis (${REDIS_HOST}:${REDIS_PORT})."
  warn "Verifique se a senha Redis (${REDIS_PASSWORD}) está correta e consistente entre serviços."
  exit 1
else
  success "Autenticação Redis confirmada."
fi

# ---------- População inicial de modelos ----------
log "🧩 Verificando chave Redis: nsga:candidate_models"
MODELS_EXIST=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" get nsga:candidate_models)

if [ -z "$MODELS_EXIST" ]; then
  warn "Nenhum modelo encontrado. Criando configuração inicial..."
  redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" \
    set nsga:candidate_models '["ollama/deepseek-r1:8b","openai/gpt-4o-mini","gemini/gemini-1.5-flash","claude/haiku-3","mistral/mistral-nemo"]' \
    >/dev/null 2>&1 && success "Chave 'nsga:candidate_models' criada com sucesso."
else
  success "Modelos já configurados em Redis."
fi
# ---------- Espera até o banco estar acessível ----------
until mysqladmin ping -h"$DB_HOST" -u"$DB_USER" -p"$DB_PASS" --silent; do
  echo "⏳ Aguardando MariaDB..."
  sleep 3
done

# ---------- Inicialização do banco de dados ----------
log "🚀 Executando verificação/migração do banco..."
python /app/app/db_manager.py >/dev/null 2>&1
if [ $? -eq 0 ]; then
  success "Banco inicializado ou já sincronizado."
else
  warn "Banco já estava sincronizado anteriormente."
fi

# ---------- Inicializa o serviço principal ----------
log "🧠 Iniciando NSGA-II..."
exec "$@"
