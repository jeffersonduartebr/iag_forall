# Objective: Provider-level chaos — classification, fallback chain, rate limits.
"""Split out of ``test_chaos.py``, which had grown past the size limit.

Two of the tests in this class used to pass unconditionally: one defined a
failing executor, never called it, and ended in two comments; the other
asserted a disjunction that covered both possible outcomes. That is how the
fallback chain could be broken in three ways at once — the flag off, a
Tornado-only ``call_async`` raising NameError, and circuit breakers that never
counted a failure — without a single red test.
"""

import pytest


class TestProviderFailureChaos:
    """Tests provider failure scenarios."""

    @pytest.mark.asyncio
    async def test_provider_rate_limit_error(self):
        """System should handle provider rate limit errors."""
        from app.error_handling import ErrorCategory, ErrorSeverity, classify_exception

        class RateLimitError(Exception):
            """Represent `RateLimitError` within this module.

The class groups the state and behavior required for RateLimitError."""
            pass

        category, severity, retry = classify_exception(RateLimitError())
        # A disjunção que estava aqui (`retry is True or category == UNKNOWN`)
        # cobria os dois resultados possíveis, portanto não podia falhar. O
        # classificador reconhece pelo nome da classe, e um rate limit é
        # exactamente o caso em que repetir vale a pena.
        assert category == ErrorCategory.PROVIDER_RATE_LIMIT
        assert severity == ErrorSeverity.WARNING
        assert retry is True

    @pytest.mark.asyncio
    async def test_provider_auth_error(self):
        """System should handle provider auth errors."""
        from app.error_handling import ErrorCategory, classify_exception

        class AuthenticationError(Exception):
            """Represent `AuthenticationError` within this module.

The class groups the state and behavior required for AuthenticationError."""
            pass

        category, severity, retry = classify_exception(AuthenticationError())
        assert category == ErrorCategory.PROVIDER_AUTH_ERROR
        assert retry is False  # Auth errors shouldn't be retried

    @pytest.mark.asyncio
    async def test_fallback_chain_execution(self, monkeypatch):
        """The chain must keep trying until a model answers.

        This test used to define ``failing_execute``, never call it, and end in
        two comments — it passed unconditionally. That is how the chain could
        be broken in three separate ways at once (flag off, Tornado-only
        ``call_async``, breakers that never counted) without a red test.
        """
        from types import SimpleNamespace

        from app import reliability as rel

        attempts = []

        async def failing_execute(model: str):
            attempts.append(model)
            if len(attempts) < 3:
                raise RuntimeError(f"Model {model} failed")
            return f"Success with {model}"

        chain = [SimpleNamespace(full_name="fb/1"), SimpleNamespace(full_name="fb/2")]
        registry = SimpleNamespace(
            get_fallback_chain=lambda m, max_depth=3: chain,
            get=lambda m: None,  # sem config própria: usa os defaults do breaker
        )
        monkeypatch.setattr(rel, "get_model_registry", lambda: registry)

        result = await rel.execute_with_fallback("primary/m", failing_execute, max_fallbacks=2)

        assert result.success is True
        assert attempts == ["primary/m", "fb/1", "fb/2"]
        assert result.model_used == "fb/2"
        assert len(result.errors) == 2
