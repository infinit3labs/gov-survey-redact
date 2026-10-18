"""Masking strategies for direct identifiers, organisations, and quasi-identifiers."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from collections.abc import Callable

from .types import MaskStyle

_DEFAULT_MASK_CHAR = "*"


def full_mask(value: str, char: str = _DEFAULT_MASK_CHAR) -> str:
    """Replace every non-whitespace character with ``char``."""
    return "".join(char if not c.isspace() else c for c in value)


def partial_mask(
    value: str,
    char: str = _DEFAULT_MASK_CHAR,
    keep_prefix: int = 0,
    keep_suffix: int = 4,
) -> str:
    """Keep a few leading or trailing characters and mask the middle.

    Negative keep counts are clamped to zero so malformed policy values can
    never produce garbled replacements.
    """
    keep_prefix = max(0, keep_prefix)
    keep_suffix = max(0, keep_suffix)
    if len(value) <= keep_prefix + keep_suffix:
        return full_mask(value, char)
    hidden = len(value) - keep_prefix - keep_suffix
    return value[:keep_prefix] + char * hidden + (value[-keep_suffix:] if keep_suffix else "")


def stable_hash(value: str, salt: bytes | None = None, length: int = 12) -> str:
    """Deterministic salted digest ``[hex...]``."""
    if salt:
        digest = hmac.new(salt, value.encode("utf-8"), hashlib.sha256).hexdigest()
    else:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"[{digest[:length]}]"


def consistent_token(category: str, value: str, salt: bytes | None = None) -> str:
    """Category-scoped HMAC-SHA256 pseudonym such as ``email_ab12cd34``."""
    salt = salt or _process_salt()
    digest = hmac.new(salt, f"{category}:{value}".encode(), hashlib.sha256).hexdigest()
    return f"{category}_{digest[:8]}"


TOKEN_PATTERN = r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*_[0-9a-f]{8}"
LABEL_PATTERN = r"\[[A-Z][A-Z ]*\]"
PLACEHOLDER_PATTERN = rf"(?:{LABEL_PATTERN}|{TOKEN_PATTERN})"


def is_placeholder(text: str) -> bool:
    """True when ``text`` is an already-redacted placeholder."""
    return re.fullmatch(PLACEHOLDER_PATTERN, text) is not None


_PROCESS_SALT: bytes | None = None


def set_process_salt(salt: bytes | None) -> None:
    """Configure the module-level salt used by :func:`consistent_token`."""
    global _PROCESS_SALT
    _PROCESS_SALT = salt


def _process_salt() -> bytes:
    global _PROCESS_SALT
    if _PROCESS_SALT is None:
        _PROCESS_SALT = secrets.token_bytes(32)
    return _PROCESS_SALT


Masker = Callable[[str], str]


def category_label(category: str) -> str:
    """Category marker such as ``[NAME]`` or ``[ORGANISATION NAME]``."""
    return f"[{category.upper().replace('_', ' ')}]"


def build_masker(
    style: MaskStyle,
    category: str,
    salt: bytes | None = None,
    mask_char: str = _DEFAULT_MASK_CHAR,
    keep_prefix: int = 0,
    keep_suffix: int = 4,
) -> Masker:
    """Build a replacement function for ``style`` and ``category``."""
    if style is MaskStyle.REMOVE:
        return lambda value: ""
    if style is MaskStyle.LABEL:
        return lambda value: category_label(category)
    if style is MaskStyle.HASH:
        return lambda value: stable_hash(value, salt)
    if style is MaskStyle.TOKEN:
        return lambda value: consistent_token(category, value, salt)
    if style is MaskStyle.PARTIAL:
        return lambda value: partial_mask(value, mask_char, keep_prefix, keep_suffix)
    return lambda value: full_mask(value, mask_char)
