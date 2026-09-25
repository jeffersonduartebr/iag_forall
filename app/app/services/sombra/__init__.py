# Objective: Shadow execution for the thesis (Caso 1): paired evaluation of every admissible candidate.
"""Execução em sombra.

For a sample of requests, after the chosen configuration's answer is delivered, every other admissible candidate
receives the same final prompt and the same retrieved context, and all answers are scored by the same judge panel.
Only scores, cost, latency, tokens and a SHA-256 of each text are stored (``shadow_evaluations``); the texts are
discarded. Nothing here feeds the routing policy. Distinct from ``openrouter_shadow`` (catalogue curation).
"""
