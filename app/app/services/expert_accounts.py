# Objective: Expert account credentials and admin-managed registration.
"""Expert account storage, password hashing, and authentication."""

from __future__ import annotations

import hashlib
import re
import secrets
from typing import Any, Dict, List, Optional

from app.roadmap_features import (
    create_expert_account as _db_create_expert_account,
)
from app.roadmap_features import (
    get_expert_account_by_email,
    get_expert_account_by_id,
    upsert_expert_profile,
)
from app.roadmap_features import (
    list_expert_accounts as _db_list_expert_accounts,
)
from app.roadmap_features import (
    update_expert_account as _db_update_expert_account,
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PBKDF2_ITERATIONS = 120_000


def normalize_email(email: str) -> str:
    """Normalize email for storage and login lookup."""
    return str(email or "").strip().lower()


def validate_email(email: str) -> bool:
    """Basic email format validation."""
    normalized = normalize_email(email)
    return bool(normalized) and bool(_EMAIL_RE.match(normalized))


def hash_password(password: str) -> str:
    """Hash password with PBKDF2-SHA256 and random salt."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify plaintext password against stored PBKDF2 hash."""
    try:
        algo, iterations, salt, digest_hex = str(stored_hash).split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        computed = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            int(iterations),
        ).hex()
        return secrets.compare_digest(computed, digest_hex)
    except Exception:
        return False


def register_expert_account(
    *,
    display_name: str,
    email: str,
    phone: Optional[str],
    password: str,
) -> Dict[str, Any]:
    """Create expert account and linked review profile."""
    normalized = normalize_email(email)
    if not validate_email(normalized):
        raise ValueError("E-mail inválido.")
    if len(password) < 8:
        raise ValueError("Senha deve ter pelo menos 8 caracteres.")
    if get_expert_account_by_email(normalized):
        raise ValueError("E-mail já cadastrado.")

    account_id = _db_create_expert_account(
        email=normalized,
        password_hash=hash_password(password),
        display_name=display_name.strip(),
        phone=(phone or "").strip() or None,
    )
    upsert_expert_profile(normalized, display_name=display_name.strip(), theme_ids=[])
    account = get_expert_account_by_id(account_id)
    return _public_account_view(account) if account else {"id": account_id, "email": normalized}


def authenticate_expert_account(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Return account dict when email/password are valid and account is enabled."""
    normalized = normalize_email(email)
    account = get_expert_account_by_email(normalized)
    if not account:
        return None
    if not account.get("enabled"):
        return None
    if not verify_password(password, str(account.get("password_hash") or "")):
        return None
    return account


def list_expert_accounts_public() -> List[Dict[str, Any]]:
    """List accounts without password hashes."""
    return [_public_account_view(row) for row in _db_list_expert_accounts()]


def update_expert_account_admin(
    account_id: int,
    *,
    display_name: Optional[str] = None,
    phone: Optional[str] = None,
    password: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Update expert account fields from admin UI."""
    account = get_expert_account_by_id(account_id)
    if not account:
        return None

    password_hash = None
    if password is not None:
        if len(password) < 8:
            raise ValueError("Senha deve ter pelo menos 8 caracteres.")
        password_hash = hash_password(password)

    _db_update_expert_account(
        account_id,
        display_name=display_name.strip() if display_name is not None else None,
        phone=phone.strip() if phone is not None else None,
        password_hash=password_hash,
        enabled=enabled,
    )

    if display_name is not None:
        upsert_expert_profile(str(account["email"]), display_name=display_name.strip())

    updated = get_expert_account_by_id(account_id)
    return _public_account_view(updated) if updated else None


def _public_account_view(account: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Strip secrets and enrich with profile theme ids."""
    if not account:
        return {}
    from app.roadmap_features import get_expert_profile

    email = str(account.get("email") or "")
    profile = get_expert_profile(email) or {}
    return {
        "id": account.get("id"),
        "email": email,
        "display_name": account.get("display_name"),
        "phone": account.get("phone"),
        "enabled": bool(account.get("enabled")),
        "theme_ids": profile.get("theme_ids") or [],
        "created_at": account.get("created_at"),
        "updated_at": account.get("updated_at"),
    }
