#!/usr/bin/env bash
# Restore MariaDB from a backup archive produced by scripts/backup.sh.
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

: "${DB_HOST:=mariadb}"
: "${DB_PORT:=3306}"
: "${DB_USER:=router_user}"
: "${DB_NAME:=routerdb}"

DUMP="$INPUT/mariadb.sql.gz"
if [ -f "$DUMP" ]; then
  gunzip -c "$DUMP" | mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"${DB_PASS:?DB_PASS required}" "$DB_NAME"
  echo "[restore] MariaDB restored from $DUMP"
else
  echo "[restore] no MariaDB dump in $INPUT"
fi

CHROMA="$INPUT/chromadb-data.tar.gz"
if [ -f "$CHROMA" ]; then
  rm -rf "$ROOT_DIR/chromadb-data"
  tar -xzf "$CHROMA" -C "$ROOT_DIR"
  echo "[restore] Chroma volume restored"
fi

echo "[restore] complete"
