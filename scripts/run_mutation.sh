#!/usr/bin/env bash
# Objective: Run mutation testing (mutmut) on the numeric core in a throwaway workspace.
# O mutmut deriva o nome do módulo do caminho do arquivo e só remove o prefixo "src.";
# como os testes importam "app.*" (pacote em app/app), o workspace expõe o pacote em
# src/app. Configuração em [tool.mutmut] do pyproject.toml.
#
# Uso: scripts/run_mutation.sh [argumentos do "mutmut run"]
#      MUTMUT_WORKDIR=/tmp/mut scripts/run_mutation.sh --max-children 8
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
WORK="${MUTMUT_WORKDIR:-$ROOT/.mutmut-work}"
MUTMUT="${MUTMUT:-mutmut}"

rm -rf "$WORK"
mkdir -p "$WORK/src"
# tar em vez de cp -r: __pycache__ pode pertencer a outro usuário (containers).
tar -C "$ROOT/app" --exclude=__pycache__ -cf - app | tar -C "$WORK/src" -xf -
tar -C "$ROOT" --exclude=__pycache__ -cf - tests pytest.ini pyproject.toml | tar -C "$WORK" -xf -

cd "$WORK"
unset PYTHONPATH
"$MUTMUT" run "$@" || status=$?
"$MUTMUT" results > "$WORK/mutmut-results.txt" || true
echo "Resultados: $WORK/mutmut-results.txt"
exit "${status:-0}"
