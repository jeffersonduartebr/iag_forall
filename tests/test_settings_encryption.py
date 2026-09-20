# Objective: Provider credentials must not sit in plaintext in MariaDB and Redis.
"""API keys travelled through the settings layers and were stored in the clear.

A database dump or a ``KEYS *`` on Redis handed over the credentials for every
provider at once. The admin API already redacted them on the way *out*; nothing
protected them at rest.

The properties that make the layer safe to deploy are the ones tested here:
without a key nothing changes, plaintext written before the key still reads,
and a decryption failure degrades instead of taking the router down.
"""

import pytest
from app.config import settings_encryption as enc


@pytest.fixture(autouse=True)
def fresh_cipher():
    enc.reset_cipher_cache()
    yield
    enc.reset_cipher_cache()


@pytest.fixture
def with_key(monkeypatch):
    from cryptography.fernet import Fernet

    monkeypatch.setenv("SETTINGS_ENCRYPTION_KEY", Fernet.generate_key().decode())
    enc.reset_cipher_cache()


# ---------------------------------------------------------------------------
# Which settings are covered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "ADMIN_TOKEN", "JWT_SECRET", "DB_PASS"]
)
def test_a_credential_is_recognised(key):
    assert enc.is_secret(key)


@pytest.mark.parametrize("key", ["NSGA_W_QUALITY", "BANDIT_EPSILON", "CANDIDATE_MODELS_LIST"])
def test_an_ordinary_setting_is_not(key):
    """Encrypting everything would make the table unreadable for no gain."""
    assert not enc.is_secret(key)


# ---------------------------------------------------------------------------
# Opt-in: no key, no change
# ---------------------------------------------------------------------------


def test_without_a_key_nothing_is_encrypted(monkeypatch):
    monkeypatch.delenv("SETTINGS_ENCRYPTION_KEY", raising=False)
    enc.reset_cipher_cache()
    assert enc.encrypt("OPENROUTER_API_KEY", "sk-segredo") == "sk-segredo"


def test_a_malformed_key_disables_the_cipher_instead_of_crashing(monkeypatch):
    """The effect is the same as having no key — which is the prior state."""
    monkeypatch.setenv("SETTINGS_ENCRYPTION_KEY", "isto-não-é-uma-chave-fernet")
    enc.reset_cipher_cache()
    assert enc.encrypt("OPENROUTER_API_KEY", "sk-segredo") == "sk-segredo"


# ---------------------------------------------------------------------------
# With a key
# ---------------------------------------------------------------------------


def test_a_credential_round_trips(with_key):
    stored = enc.encrypt("OPENROUTER_API_KEY", "sk-segredo")
    assert stored.startswith(enc.PREFIX)
    assert "sk-segredo" not in stored
    assert enc.decrypt(stored) == "sk-segredo"


def test_an_ordinary_setting_stays_readable(with_key):
    assert enc.encrypt("NSGA_W_QUALITY", "1.0") == "1.0"


def test_an_empty_value_is_left_alone(with_key):
    assert enc.encrypt("OPENROUTER_API_KEY", "") == ""


def test_encrypting_twice_does_not_nest(with_key):
    once = enc.encrypt("OPENROUTER_API_KEY", "sk-segredo")
    assert enc.encrypt("OPENROUTER_API_KEY", once) == once


# ---------------------------------------------------------------------------
# Backwards and forwards compatibility
# ---------------------------------------------------------------------------


def test_plaintext_written_before_the_key_still_reads(with_key):
    """Introducing a key must not require a migration."""
    assert enc.decrypt("sk-guardada-em-claro") == "sk-guardada-em-claro"


def test_a_value_that_cannot_be_decrypted_degrades(monkeypatch, with_key):
    """A rotated key must not turn into an outage; the caller fails later with
    a clearer error when the credential is rejected."""
    assert enc.decrypt(enc.PREFIX + "lixo-que-não-decifra") == enc.PREFIX + "lixo-que-não-decifra"


def test_ciphertext_found_without_a_key_is_returned_as_is(monkeypatch):
    monkeypatch.delenv("SETTINGS_ENCRYPTION_KEY", raising=False)
    enc.reset_cipher_cache()
    assert enc.decrypt(enc.PREFIX + "abc") == enc.PREFIX + "abc"


def test_none_survives_decryption():
    assert enc.decrypt(None) is None


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_the_write_path_encrypts_before_both_backends():
    import inspect

    from app import settings_dynamic as sd

    source = inspect.getsource(sd.DynamicSettings.set)
    assert "stored = encrypt(key, value)" in source
    assert '{"k": key, "v": stored}' in source
    assert "_set_to_redis(key, stored)" in source


def test_every_read_path_decrypts():
    import inspect

    from app import settings_dynamic as sd

    for fn in (sd._get_from_redis, sd._get_from_db, sd._all_from_db, sd._many_from_redis):
        assert "decrypt(" in inspect.getsource(fn), fn.__name__
