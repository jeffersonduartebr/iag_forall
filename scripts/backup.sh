#!/usr/bin/env bash
# Backup MariaDB, Redis, and Chroma volumes for disaster recovery.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$ROOT_DIR/backups}"
STAMP="$(date +%Y%m%d_%H%M%S)"
TARGET="$BACKUP_DIR/$STAMP"
mkdir -p "$TARGET"

: "${DB_HOST:=mariadb}"
: "${DB_PORT:=3306}"
: "${DB_USER:=router_user}"
: "${DB_NAME:=routerdb}"

echo "[backup] writing to $TARGET"

if command -v mysqldump >/dev/null 2>&1; then
  mysqldump -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"${DB_PASS:?DB_PASS required}" \
    --single-transaction --routines --triggers "$DB_NAME" \
    | gzip > "$TARGET/mariadb.sql.gz"
  echo "[backup] MariaDB dump ok"
else
  echo "[backup] mysqldump not found — skip DB dump"
fi

if docker ps --format '{{.Names}}' | grep -q '^redis_router$'; then
  docker exec redis_router redis-cli -a "${REDIS_PASSWORD:-}" --no-auth-warning BGSAVE || true
  docker cp redis_router:/data/dump.rdb "$TARGET/redis-dump.rdb" 2>/dev/null || true
  echo "[backup] Redis RDB ok (if present)"
fi

if [ -d "$ROOT_DIR/chromadb-data" ]; then
  tar -czf "$TARGET/chromadb-data.tar.gz" -C "$ROOT_DIR" chromadb-data
  echo "[backup] Chroma volume ok"
fi

echo "$STAMP" > "$BACKUP_DIR/LATEST"
echo "[backup] done -> $TARGET"
