#!/usr/bin/env bash
# Restore PostgreSQL (and Chroma) from a backup archive produced by scripts/backup.sh.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <backup_dir_or_stamp>"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="$1"
if [ ! -d "$INPUT" ]; then
  INPUT="$ROOT_DIR/backups/$1"
fi
if [ ! -d "$INPUT" ]; then
  echo "Backup not found: $INPUT"
  exit 1
fi

: "${DB_HOST:=127.0.0.1}"
: "${DB_PORT:=5433}"
: "${DB_USER:=router_user}"
: "${DB_NAME:=routerdb}"

DUMP="$INPUT/postgres.sql.gz"
if [ -f "$DUMP" ]; then
  gunzip -c "$DUMP" | PGPASSWORD="${DB_PASS:?DB_PASS required}" psql -v ON_ERROR_STOP=1 -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"
  echo "[restore] PostgreSQL restored from $DUMP"
else
  echo "[restore] no PostgreSQL dump in $INPUT"
fi

CHROMA="$INPUT/chromadb-data.tar.gz"
if [ -f "$CHROMA" ]; then
  rm -rf "$ROOT_DIR/chromadb-data"
  tar -xzf "$CHROMA" -C "$ROOT_DIR"
  echo "[restore] Chroma volume restored"
fi

echo "[restore] complete"
