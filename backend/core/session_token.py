"""
Signed browser/session tokens for API authentication (Bearer in Authorization).

Uses SECRET_KEY; separate from CONTEXT_API_SECRET (service keys use X-Context-Api-Key).
"""
from __future__ import annotations

import os
from typing import Optional

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

SALT_BROWSER_SESSION_V1 = "ea-browser-session-v1"


def _max_age_seconds() -> int:
    return int(os.environ.get("SESSION_TOKEN_MAX_AGE_SECONDS", str(86400 * 30)))


def issue_browser_session_token(secret: str, user_id: str) -> str:
    if not secret or not isinstance(secret, str):
        raise ValueError("SECRET_KEY must be configured to issue session tokens")
    serializer = URLSafeTimedSerializer(secret, salt=SALT_BROWSER_SESSION_V1)
    return serializer.dumps({"uid": (user_id or "").strip()})


def verify_browser_session_token(secret: str, token: str) -> Optional[str]:
    if not secret or not token or not isinstance(token, str):
        return None
    serializer = URLSafeTimedSerializer(secret, salt=SALT_BROWSER_SESSION_V1)
    try:
        data = serializer.loads(token, max_age=_max_age_seconds())
    except (BadSignature, SignatureExpired):
        return None
    uid = data.get("uid") if isinstance(data, dict) else None
    if isinstance(uid, str) and uid.strip():
        return uid.strip()
    return None
