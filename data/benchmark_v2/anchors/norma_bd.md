# Norma Interna de Padronizacao de Esquemas de Banco de Dados

Instituto Tecnico do Vale (ITV) - documento interno. Versao 1.0.

Este documento e ficticio e existe apenas para avaliar a dependencia de recuperacao de contexto (RAG). Os valores aqui definidos nao correspondem a nenhuma instituicao real.

## Secao 2.1

Toda tabela transacional deve usar o prefixo tb_ e nome no singular, em minusculas.

## Secao 2.3

Chaves primarias substitutas devem ser do tipo BIGINT sem sinal, com incremento automatico.

## Secao 3.2

Toda tabela deve conter os campos de auditoria criado_em, atualizado_em e criado_por, sendo os dois primeiros obrigatoriamente NOT NULL.

## Secao 5.1

Os registros da tabela de log de auditoria devem ser retidos por 18 meses antes do expurgo.

## Secao 2.5

Esquemas novos devem ser entregues no minimo na 3FN; desnormalizacoes exigem justificativa formal de desempenho aprovada pelo comite de dados.

## Secao 4.2

Cada tabela pode ter no maximo 6 indices secundarios sem aprovacao do comite de dados.

## Secao 2.4

Chaves estrangeiras devem ser nomeadas no padrao fk_<tabela_origem>_<tabela_destino>.

## Secao 6.1

A janela de manutencao programada dos bancos de producao ocorre aos domingos, das 2h as 5h.
