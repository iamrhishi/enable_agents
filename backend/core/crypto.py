"""
Shared symmetric encryption for sensitive values at rest (API keys, OAuth
tokens, etc.) using Fernet. Extracted from core/settings.py so any module
storing encrypted values (UserSettingModel, ProjectSettingModel, ...) uses
the exact same key handling instead of each keeping its own private copy.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

# Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
SETTINGS_ENCRYPTION_KEY = os.environ.get("SETTINGS_ENCRYPTION_KEY")


def _get_fernet() -> Optional[Fernet]:
    if not SETTINGS_ENCRYPTION_KEY:
        logger.warning("SETTINGS_ENCRYPTION_KEY not set - values will be stored unencrypted")
        return None
    try:
        return Fernet(SETTINGS_ENCRYPTION_KEY.encode())
    except Exception as e:
        logger.error(f"Invalid encryption key: {e}")
        return None


def encrypt(value: str) -> str:
    fernet = _get_fernet()
    if fernet:
        return fernet.encrypt(value.encode()).decode()
    return value  # Fallback: store unencrypted (dev only)


def decrypt(encrypted: str) -> str:
    fernet = _get_fernet()
    if fernet:
        try:
            return fernet.decrypt(encrypted.encode()).decode()
        except Exception:
            return encrypted  # Return as-is if decryption fails
    return encrypted


def mask_value(value: str, show_chars: int = 4) -> str:
    """Mask a sensitive value for display (e.g. in a settings list UI)."""
    if not value or len(value) <= show_chars:
        return "****"
    return value[:show_chars] + "*" * (len(value) - show_chars)
