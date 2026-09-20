# Objective: Encrypt secret settings before they reach MariaDB and Redis.
"""Encryption at rest for the settings that are credentials.

Provider API keys travel through the three settings layers — env, Redis,
MariaDB — and were stored in plaintext in the last two. A database dump or a
``KEYS *`` on Redis handed over the credentials for every provider at once.
The admin API already redacts them on the way *out* (``secrets_redaction``);
nothing protected them at rest.

The design is deliberately conservative, because the failure mode of a broken
encryption layer is an unreachable router:

- **Opt-in.** With no ``SETTINGS_ENCRYPTION_KEY`` configured, nothing changes.
  A deployment that has not set up key management keeps working exactly as
  before, and a key can be introduced without a migration.
- **Transparent on read.** Ciphertext carries the ``enc:v1:`` prefix. Anything
  without it is read as plaintext, so values written before the key existed
  keep resolving, and the store can hold both at once while it rotates.
- **Never fatal.** A value that cannot be decrypted — the key was rotated
  without re-encrypting, say — is logged and returned as-is rather than
  raising. Refusing to answer would turn a key mistake into an outage, and the
  caller will fail anyway with a clearer error when the credential is rejected.

Only keys in :data:`SECRET_SETTING_KEYS` are encrypted. Encrypting everything
would make the settings table unreadable to an operator for no gain.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, TypeVar

from .secrets_redaction import SECRET_SETTING_KEYS

#: `decrypt` devolve o mesmo tipo que recebe: um `str` continua `str` e um
#: `None` continua `None`, para não obrigar cada chamador a re-estreitar.
_T = TypeVar("_T", str, Optional[str])

logger = logging.getLogger(__name__)

PREFIX = "enc:v1:"
_ENV_KEY = "SETTINGS_ENCRYPTION_KEY"

_fernet = None
_resolved = False


def _cipher():
    """Resolve the Fernet instance once; ``None`` when no key is configured."""
    global _fernet, _resolved
    if _resolved:
        return _fernet
    _resolved = True
    raw = (os.getenv(_ENV_KEY) or "").strip()
    if not raw:
        return None
    try:
        from cryptography.fernet import Fernet

        _fernet = Fernet(raw.encode())
        logger.info("[settings] Cifra de segredos activa (%s configurada).", _ENV_KEY)
    except Exception as exc:
        # Uma chave malformada não pode derrubar o arranque: o efeito é o
        # mesmo de não ter chave nenhuma, que é o estado anterior.
        logger.error("[settings] %s inválida, a cifra fica desligada: %s", _ENV_KEY, exc)
        _fernet = None
    return _fernet


def reset_cipher_cache() -> None:
    """Re-read the key (tests, and a key rotation without a restart)."""
    global _fernet, _resolved
    _fernet, _resolved = None, False


def is_secret(key: str) -> bool:
    """Whether this setting holds a credential.

    Reuses the same list the admin redaction uses, so a key can never be
    masked in one place and stored in the clear in the other.
    """
    upper = key.upper()
    return upper in SECRET_SETTING_KEYS or any(
        token in upper for token in ("PASSWORD", "SECRET", "TOKEN", "API_KEY")
    )


def encrypt(key: str, value: str) -> str:
    """Ciphertext for a secret setting; the value unchanged otherwise."""
    if not value or not is_secret(key):
        return value
    cipher = _cipher()
    if cipher is None:
        return value
    if str(value).startswith(PREFIX):
        return value  # já cifrado: não cifrar duas vezes
    try:
        return PREFIX + cipher.encrypt(str(value).encode()).decode()
    except Exception as exc:
        logger.error("[settings] Falha ao cifrar '%s', a gravar em claro: %s", key, exc)
        return value


def decrypt(value: _T) -> _T:
    """Plaintext for a stored value, whether or not it was encrypted."""
    if not isinstance(value, str) or not value.startswith(PREFIX):
        return value
    cipher = _cipher()
    if cipher is None:
        logger.error("[settings] Valor cifrado encontrado sem %s configurada.", _ENV_KEY)
        return value
    try:
        return cipher.decrypt(value[len(PREFIX) :].encode()).decode()
    except Exception as exc:
        logger.error("[settings] Falha ao decifrar um segredo (chave rodada?): %s", exc)
        return value
