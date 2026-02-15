# Workflow de Documentação

Este documento define como manter a documentação técnica do projeto consistente para novos contribuidores (especialmente estagiários).

## Objetivo
1. Garantir rastreabilidade entre código e documentação.
2. Facilitar onboarding e manutenção.
3. Evitar docstrings genéricas ou desatualizadas.

## Fontes oficiais de verdade
1. Código-fonte em `app/app`.
2. Contratos HTTP em `docs/API.md`.
3. Configuração de runtime em `docs/CONFIGURATION.md`.
4. Arquitetura e módulos em `docs/ARCHITECTURE.md` e `docs/MODULE_INDEX.md`.

## Catálogos automáticos
Gerados pelo script `scripts/generate_docs_catalog.py`:
1. `docs/FILE_CATALOG.md`
2. `docs/METHOD_CATALOG.md`
3. `docs/DOCSTRING_BACKLOG.md`

## Procedimento obrigatório em cada mudança de código
1. Atualizar docstring dos métodos e classes alterados.
2. Atualizar os documentos temáticos impactados (`API`, `CONFIGURATION`, etc.).
3. Executar:
```bash
python3 scripts/generate_docs_catalog.py
```
4. Revisar `docs/DOCSTRING_BACKLOG.md` e reduzir backlog quando possível.

## Critérios mínimos de qualidade de docstring
1. Explicar responsabilidade do método e contexto de uso.
2. Explicar parâmetros e retornos relevantes.
3. Evitar frases genéricas como “Resumo do comportamento desta função”.
4. Indicar efeitos colaterais importantes (I/O, DB, fila, métricas).

## Checklist de revisão para PR
1. Mudança de comportamento está refletida em docstring?
2. Endpoints alterados estão documentados em `docs/API.md`?
3. Novas variáveis de configuração estão em `docs/CONFIGURATION.md`?
4. Catálogos automáticos foram regenerados?
