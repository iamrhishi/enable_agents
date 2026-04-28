"""
ContextStore — shared context layer for all agents.

Architecture
────────────
Two-tier storage, abstracted behind a single API:

  Redis (ephemeral)            MySQL AgentContext table (persistent)
  ─────────────────            ────────────────────────────────────
  Keyed by user_id + key       Keyed by user_id + agent_id + key
  TTL: CONTEXT_TTL_SECONDS     Survives restarts / Redis flushes
  (default 7200 = 2 hours)     Queryable, auditable, exportable

Agents always go through ContextStore — never directly to Redis or the
AgentContext model — so the storage backend can be swapped without
touching agent code.

Usage in agent service.py
─────────────────────────
    from core.context import ContextStore

    ctx = ContextStore()

    # Write (always writes to both tiers)
    ctx.set(user_id="u1", agent_id="market_research", key="company_profile",
            value={"industry": "SaaS", "name": "Acme"})

    # Read (Redis first, falls back to MySQL)
    profile = ctx.get(user_id="u1", key="company_profile")

    # Read everything available for a user
    snapshot = ctx.snapshot(user_id="u1")

    # Delete a key
    ctx.delete(user_id="u1", agent_id="market_research", key="company_profile")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONTEXT_TTL_SECONDS = int(os.environ.get("CONTEXT_TTL_SECONDS", 7200))
REDIS_KEY_PREFIX = "agent_ctx"


def _redis_key(user_id: str, key: str) -> str:
    return f"{REDIS_KEY_PREFIX}:{user_id}:{key}"


def _get_redis():
    """Return a Redis client, or None if Redis is unavailable."""
    try:
        import redis as redis_lib
        url = os.environ.get("CELERY_BROKER_URL") or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        client = redis_lib.from_url(url, decode_responses=True, socket_connect_timeout=1)
        client.ping()
        return client
    except Exception as exc:
        logger.debug("Redis unavailable for ContextStore (falling back to DB only): %s", exc)
        return None


class ContextStore:
    """
    Thread-safe context store.  Create one instance per request or share
    across a module — all state lives in Redis / MySQL, not in the object.
    """

    def set(
        self,
        user_id: str,
        agent_id: str,
        key: str,
        value: Any,
        ttl: int = CONTEXT_TTL_SECONDS,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Write a key-value pair for a user.

        value can be any JSON-serialisable object (dict, list, str, etc.).
        ttl controls the Redis expiry (seconds); the MySQL copy has no expiry.
        """
        serialised = json.dumps(value)

        # ── Redis write ───────────────────────────────────────────────────────
        rc = _get_redis()
        if rc:
            try:
                rc.set(_redis_key(user_id, key), serialised, ex=ttl)
            except Exception as exc:
                logger.warning("Redis write failed for %s/%s: %s", user_id, key, exc)

        # ── MySQL write ───────────────────────────────────────────────────────
        try:
            from core.models import AgentContext
            from core.database import db

            row = AgentContext.query.filter_by(
                user_id=user_id, agent_id=agent_id, key=key
            ).first()
            if row:
                row.value = serialised
                row.updated_at = datetime.utcnow()
                if session_id:
                    row.session_id = session_id
            else:
                db.session.add(AgentContext(
                    user_id=user_id,
                    agent_id=agent_id,
                    key=key,
                    value=serialised,
                    session_id=session_id,
                ))
            db.session.commit()
        except Exception as exc:
            logger.error("MySQL write failed for %s/%s: %s", user_id, key, exc)

    def get(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        Read a value by user_id + key.

        Tries Redis first (fast); falls back to MySQL if Redis misses or is
        unavailable.  Returns default if the key doesn't exist anywhere.
        """
        # ── Redis read ────────────────────────────────────────────────────────
        rc = _get_redis()
        if rc:
            try:
                raw = rc.get(_redis_key(user_id, key))
                if raw is not None:
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("Redis read failed for %s/%s: %s", user_id, key, exc)

        # ── MySQL fallback ────────────────────────────────────────────────────
        try:
            from core.models import AgentContext

            row = (
                AgentContext.query
                .filter_by(user_id=user_id, key=key)
                .order_by(AgentContext.updated_at.desc())
                .first()
            )
            if row:
                value = json.loads(row.value)
                # Warm Redis cache back up
                if rc:
                    try:
                        rc.set(_redis_key(user_id, key), row.value, ex=CONTEXT_TTL_SECONDS)
                    except Exception:
                        pass
                return value
        except Exception as exc:
            logger.error("MySQL read failed for %s/%s: %s", user_id, key, exc)

        return default

    def snapshot(self, user_id: str) -> Dict[str, Any]:
        """Return all context keys currently stored for a user (from MySQL)."""
        try:
            from core.models import AgentContext

            rows = AgentContext.query.filter_by(user_id=user_id).all()
            result: Dict[str, Any] = {}
            for row in rows:
                if row.key not in result:  # keep latest (ordered by updated_at desc below)
                    result[row.key] = json.loads(row.value)
            return result
        except Exception as exc:
            logger.error("Snapshot failed for %s: %s", user_id, exc)
            return {}

    def delete(self, user_id: str, agent_id: str, key: str) -> None:
        """Delete a specific key for a user (both Redis and MySQL)."""
        rc = _get_redis()
        if rc:
            try:
                rc.delete(_redis_key(user_id, key))
            except Exception:
                pass

        try:
            from core.models import AgentContext
            from core.database import db

            AgentContext.query.filter_by(
                user_id=user_id, agent_id=agent_id, key=key
            ).delete()
            db.session.commit()
        except Exception as exc:
            logger.error("Delete failed for %s/%s: %s", user_id, key, exc)

    def clear(self, user_id: str) -> None:
        """Delete ALL context for a user (logout / reset)."""
        rc = _get_redis()
        if rc:
            try:
                pattern = _redis_key(user_id, "*")
                keys = rc.keys(pattern)
                if keys:
                    rc.delete(*keys)
            except Exception:
                pass

        try:
            from core.models import AgentContext
            from core.database import db

            AgentContext.query.filter_by(user_id=user_id).delete()
            db.session.commit()
        except Exception as exc:
            logger.error("Clear failed for %s: %s", user_id, exc)
