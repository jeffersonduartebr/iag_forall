#!/usr/bin/env bash
# Derruba o ARISTO do contrato RBIE.
#
#   scripts/rbie/derrubar_roteador.sh            # so o uvicorn (mantem Redis/PG)
#   scripts/rbie/derrubar_roteador.sh --tudo     # tambem remove os conteineres
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR_ESTADO="$RAIZ/state/rbie"
CNT_REDIS="aristo-rbie-redis"
CNT_PG="aristo-rbie-pg"

if [ -f "$DIR_ESTADO/uvicorn.pid" ]; then
  kill "$(cat "$DIR_ESTADO/uvicorn.pid")" >/dev/null 2>&1 || true
  rm -f "$DIR_ESTADO/uvicorn.pid"
fi
pkill -f "uvicorn app.main:app --host 127.0.0.1 --port" >/dev/null 2>&1 || true
echo "uvicorn do ARISTO derrubado"

if [ "${1:-}" = "--tudo" ]; then
  docker rm -f "$CNT_REDIS" "$CNT_PG" >/dev/null 2>&1 || true
  echo "conteineres $CNT_REDIS e $CNT_PG removidos"
fi
