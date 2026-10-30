#!/usr/bin/env bash
# Sobe o ARISTO com a superficie do contrato RBIE em URL local estavel.
#
#   scripts/rbie/subir_roteador.sh
#
# Variaveis uteis:
#   ARISTO_PORTA=8000                 porta do uvicorn (igual a config.yaml)
#   CONFIG_EXPERIMENTO=<caminho>      config.yaml do experimento
#   ENV_EXPERIMENTO=<caminho>         .env com a OPENROUTER_API_KEY
#   OPENROUTER_BASE_URL=<url>         modo seco (servidor falso); sem esta
#                                     variavel o roteador fala com o OpenRouter
#   ARISTO_CICLO_S_TESTE=5            atalho do ciclo, so vale em modo seco
#
# Idempotente: reusa os conteineres, reaplica as migracoes e reinicia o uvicorn.
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORTA="${ARISTO_PORTA:-8000}"
CONFIG_EXPERIMENTO="${CONFIG_EXPERIMENTO:-/data/desenvolvimento/doutorado-publicacoes/rbie-roteamento-feedback-enem/experimentos/codigo/config.yaml}"
ENV_EXPERIMENTO="${ENV_EXPERIMENTO:-/data/desenvolvimento/doutorado-publicacoes/.env}"
DIR_ESTADO="$RAIZ/state/rbie"
CNT_REDIS="aristo-rbie-redis"
CNT_PG="aristo-rbie-pg"
PORTA_REDIS="${ARISTO_REDIS_PORT:-6380}"
PORTA_PG="${ARISTO_PG_PORT:-55432}"
SENHA_PG="${ARISTO_RBIE_DB_PASS:-rbie_local_dev}"
mkdir -p "$DIR_ESTADO"

# 1. Chaves: carregadas do .env do experimento, nunca impressas.
if [ -f "$ENV_EXPERIMENTO" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_EXPERIMENTO"
  set +a
fi

# 2. Servicos locais (Redis e PostgreSQL) em conteineres.
if ! docker inspect "$CNT_REDIS" >/dev/null 2>&1; then
  docker run -d --name "$CNT_REDIS" -p "127.0.0.1:${PORTA_REDIS}:6379" redis:7-alpine >/dev/null
elif [ "$(docker inspect -f '{{.State.Running}}' "$CNT_REDIS")" != "true" ]; then
  docker start "$CNT_REDIS" >/dev/null
fi
if ! docker inspect "$CNT_PG" >/dev/null 2>&1; then
  docker run -d --name "$CNT_PG" \
    -e POSTGRES_DB=routerdb -e POSTGRES_USER=router_user -e POSTGRES_PASSWORD="$SENHA_PG" \
    -p "127.0.0.1:${PORTA_PG}:5432" postgres:16-alpine >/dev/null
elif [ "$(docker inspect -f '{{.State.Running}}' "$CNT_PG")" != "true" ]; then
  docker start "$CNT_PG" >/dev/null
fi
for _ in $(seq 1 60); do
  if docker exec "$CNT_PG" pg_isready -U router_user -d routerdb >/dev/null 2>&1; then break; fi
  sleep 1
done

# 3. Ambiente do processo (nenhuma chave e gravada em arquivo).
export ENV=development
export ADMIN_TOKEN="${ARISTO_ADMIN_TOKEN:-aristo-rbie-local-admin}"
export ADMIN_UI_USERNAME=admin
export ADMIN_UI_PASSWORD="${ARISTO_ADMIN_TOKEN:-aristo-rbie-local-admin}"
export JWT_SECRET="${ARISTO_JWT_SECRET:-aristo-rbie-local-jwt}"
export REQUIRE_API_AUTH=0
export DB_DIALECT=postgresql
export DB_HOST=127.0.0.1
export DB_PORT="$PORTA_PG"
export DB_USER=router_user
export DB_PASS="$SENHA_PG"
export DB_NAME=routerdb
export REDIS_HOST=127.0.0.1
export REDIS_PORT="$PORTA_REDIS"
export REDIS_DB=0
export REDIS_PASSWORD=
export STATE_DIR="$RAIZ/state"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export READINESS_MODE=lenient
export CANDIDATE_MODELS_LIST='["qwen/qwen3.8-flash","google/gemma-4-26b-a4b-it","qwen/qwen3.8-27b","google/gemini-3.8-flash","anthropic/claude-sonnet-5.5","google/gemini-4-argon"]'
export CANDIDATE_VISION_MODELS_LIST='[]'
export CANDIDATE_MULTIMODAL_MODELS_LIST='[]'
export ARISTO_CONFIG_EFETIVA="$DIR_ESTADO/config_efetiva.json"
export ARISTO_ESTADO_DIR="$DIR_ESTADO"
export ARISTO_COMMIT="$(git -C "$RAIZ" rev-parse HEAD)"

# 4. Migracoes do banco (idempotente).
(cd "$RAIZ" && .venv/bin/alembic -c alembic_pg.ini upgrade head >/dev/null 2>&1) || true

# 5. Configuracao efetiva gerada do config.yaml do experimento.
.venv/bin/python "$RAIZ/scripts/rbie/gerar_config_efetiva.py" \
  --config "$CONFIG_EXPERIMENTO" --saida "$ARISTO_CONFIG_EFETIVA" >/dev/null

# 6. Reinicia o uvicorn (um worker: o estado vive no processo).
if [ -f "$DIR_ESTADO/uvicorn.pid" ]; then
  kill "$(cat "$DIR_ESTADO/uvicorn.pid")" >/dev/null 2>&1 || true
  sleep 1
fi
if command -v ss >/dev/null 2>&1; then
  PIDS="$(ss -ltnp 2>/dev/null | awk -v p=":${PORTA}\$" '$4 ~ p {print $0}' | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)"
  for pid in $PIDS; do kill "$pid" >/dev/null 2>&1 || true; done
fi
(
  cd "$RAIZ/app"
  nohup ../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$PORTA" --log-level warning \
    >> "$DIR_ESTADO/uvicorn.log" 2>&1 &
  echo $! > "$DIR_ESTADO/uvicorn.pid"
)

# 7. Espera /saude.
for _ in $(seq 1 90); do
  if curl -sf -m 2 "http://127.0.0.1:${PORTA}/saude" >/dev/null 2>&1; then
    echo "ARISTO no ar em http://127.0.0.1:${PORTA}"
    curl -s "http://127.0.0.1:${PORTA}/saude"
    echo
    echo "modo: $([ -n "${OPENROUTER_BASE_URL:-}" ] && echo "seco (${OPENROUTER_BASE_URL})" || echo "OpenRouter real")"
    echo "log: $DIR_ESTADO/uvicorn.log"
    exit 0
  fi
  sleep 1
done
echo "falha ao subir o ARISTO; ultimas linhas do log:" >&2
tail -40 "$DIR_ESTADO/uvicorn.log" >&2
exit 1
