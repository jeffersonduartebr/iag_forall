# -*- coding: utf-8 -*-
# Objective: Application runtime code for guardrails.
"""Basic content guardrails (MVP).

Provides lightweight checks for:
- prompt injection markers
- obvious unsafe requests
- simple PII masking in outputs
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

_INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"system\s+prompt",
    r"developer\s+message",
    r"bypass\s+safety",
]

_UNSAFE_PATTERNS = [
    r"como\s+fazer\s+bomba",
    r"how\s+to\s+build\s+a\s+bomb",
    r"malware\s+code",
]

_EMAIL_RE = re.compile(r"([\w.\-+]+@[\w\-]+\.[\w.\-]+)", re.IGNORECASE)
_PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{7,}\d)")


@dataclass
class GuardrailDecision:
    """Represent `GuardrailDecision` within this module.

The class groups the state and behavior required for GuardrailDecision."""
    allowed: bool
    reasons: List[str]


def check_input_guardrails(prompt: str) -> GuardrailDecision:
    """Evaluate input prompt against basic guardrail patterns."""
    text = (prompt or "").lower()
    reasons: List[str] = []

    for pat in _INJECTION_PATTERNS:
        if re.search(pat, text):
            reasons.append("prompt_injection_signal")
            break

    for pat in _UNSAFE_PATTERNS:
        if re.search(pat, text):
            reasons.append("unsafe_content_signal")
            break

    return GuardrailDecision(allowed=len(reasons) == 0, reasons=reasons)


def sanitize_output_guardrails(text: str) -> Tuple[str, List[str]]:
    """Mask obvious PII in model output."""
    out = text or ""
    tags: List[str] = []

    if _EMAIL_RE.search(out):
        out = _EMAIL_RE.sub("[REDACTED_EMAIL]", out)
        tags.append("masked_email")

    if _PHONE_RE.search(out):
        out = _PHONE_RE.sub("[REDACTED_PHONE]", out)
        tags.append("masked_phone")

    return out, tags
