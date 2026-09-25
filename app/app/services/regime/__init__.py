# Objective: Exploration regime of the Caso 1 protocol (explicit draw, per-participant cap, regeneration).
"""Regime de exploração do protocolo do Caso 1.

For tenants in ``REGIME_EXPLORACAO_TENANTS`` the route is chosen by one explicit, auditable draw instead of the
meta-bandit vote, so the assignment probability of every delivered answer is known exactly:

- exploitation = the candidate with the highest posterior mean reward in the request's context (NSGA-II score
  breaks ties); probability ``1 - eps``;
- exploration = a uniform draw over the other admissible candidates plus the OpenRouter catalogue pool;
  probability ``eps / K`` for each of the K arms;
- ``eps = REGIME_EPSILON`` (0.15) while the participant's exploratory share over the last
  ``REGIME_JANELA_EPISODIOS`` (20) episodes, counting this request, stays <= ``REGIME_TETO`` (0.15); else 0;
- an exploratory answer that fails the uncertainty check is regenerated with the exploitation configuration.
The bandit keeps learning from feedback as before; only the choice rule differs for these tenants.
"""
