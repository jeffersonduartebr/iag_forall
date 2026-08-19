#!/bin/sh
# Objective: Shell utility for init chroma.
set -e
chroma run --path /data --host 0.0.0.0 --port 8000 &
sleep 5
chroma-cli tenants create default_tenant || true
chroma-cli databases create default_database || true
wait
