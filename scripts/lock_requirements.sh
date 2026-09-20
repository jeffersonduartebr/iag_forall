#!/usr/bin/env bash
# Objective: Regenerate every dependency lock from its .txt source, with hashes.
#
# Os ficheiros .txt sao a fonte editavel; os .lock sao o que as imagens e o CI
# instalam. Editar um .txt sem correr este script deixa os dois em desacordo, e
# o `uv pip sync --require-hashes` do CI passa a instalar a versao antiga sem
# se queixar. Correr isto e a unica forma suportada de mexer em dependencias.
#
# Uso:
#   scripts/lock_requirements.sh            # regenera todos
#   scripts/lock_requirements.sh app/requirements.txt   # so um
#
# Requer uv (https://docs.astral.sh/uv/). Python 3.11 e fixado explicitamente
# porque o lock tem de ser resolvido para a versao das imagens, nao para a do
# portatil de quem corre o comando.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_VERSION=3.11

SOURCES=(
  requirements-dev.txt
  app/requirements.txt
  app/requirements-db.txt
  app/requirements.correlation.txt
  app/requirements.dash.txt
  app/requirements.metaopt.txt
  app/requirements.nsga.txt
)

if ! command -v uv >/dev/null 2>&1; then
  echo "uv nao encontrado. Instala com: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [ "$#" -gt 0 ]; then
  SOURCES=("$@")
fi

for src in "${SOURCES[@]}"; do
  out="${src%.txt}.lock"
  echo "==> ${src} -> ${out}"
  uv pip compile "${src}" \
    --generate-hashes \
    --python-version "${PYTHON_VERSION}" \
    -o "${out}"
done

echo
echo "Locks regenerados. Confirma o diff antes de fazer commit:"
echo "  git diff -- '*.lock'"
