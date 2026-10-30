# -*- coding: utf-8 -*-
"""Superficie HTTP do contrato do experimento RBIE sobre o ARISTO.

Rotas aditivas em ``/saude``, ``/configuracao``, ``/estados`` e ``/rotear``.
Nao substituem ``/query`` nem compartilham estado com ele: cada estado do
contrato mantem a sua propria populacao, os seus centroides, as suas
estatisticas, o seu meta-bandit, o seu controlador, o seu historico e o seu
estado dos geradores aleatorios. Ver ``experimentos/codigo/roteador/CONTRATO.md``
no repositorio do experimento.
"""
