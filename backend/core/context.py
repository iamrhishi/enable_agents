"""
ContextStore — shared context layer for all agents.

Architecture
────────────
Two-tier storage, abstracted behind a single API:

  Redis (ephemeral)            PostgreSQL AgentContext table (persistent)
  ─────────────────            ────────────────────────────────────
  Keyed by user+agent+key      Keyed by user_id + agent_id + key
  TTL: CONTEXT_TTL_SECONDS     Survives restarts / Redis flushes
  (default 7200 = 2 hours)     Queryable, auditable, exportable

Agents always go through ContextStore — never directly to Redis or the
AgentContext model — so the storage backend can be swapped without
touching agent code.

Flexible "ask" interface (recommended)
──────────────────────────────────────
    from core.context import ContextStore

    ctx = ContextStore()

    # Ask for what you need — don't care which agent provides it
    docs = ctx.ask(user_id, "documents")              # List of available docs
    context = ctx.ask(user_id, "document_context", query="pricing")  # RAG context
    profile = ctx.ask(user_id, "company_profile")     # From any agent
    all_data = ctx.ask(user_id, "all")                # Full snapshot
    results = ctx.ask(user_id, "search", query="competitor")  # Search all

    # See what's available
    available = ctx.available(user_id)  # { agent_id: [keys...], ... }

Explicit API (when you know the source)
───────────────────────────────────────
    # Write (always writes to both tiers)
    ctx.set(user_id="u1", agent_id="market_research", key="company_profile",
            value={"industry": "SaaS", "name": "Acme"})

    # Read (Redis first, falls back to PostgreSQL)
    profile = ctx.get(user_id="u1", agent_id="market_research", key="company_profile")

    # Read everything available for a user (nested: agent_id -> key -> value)
    snapshot = ctx.snapshot(user_id="u1")

    # Delete a key
    ctx.delete(user_id="u1", agent_id="market_research", key="company_profile")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def _escape_sql_like(value: Optional[str]) -> str:
    text_value = value or ''
    return text_value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


def _validate_search_limit(limit: Any) -> int:
    try:
        n = int(limit)
    except (TypeError, ValueError):
        raise ValueError('Invalid limit; must be an integer')
    if n < 1:
        raise ValueError('Invalid limit; must be greater than 0')
    return min(n, 100)

CONTEXT_TTL_SECONDS = int(os.environ.get("CONTEXT_TTL_SECONDS", 7200))
REDIS_KEY_PREFIX = "agent_ctx"


def _redis_key(user_id: str, agent_id: str, key: str) -> str:
    """Redis cache key aligned with PostgreSQL unique (user_id, agent_id, key)."""
    return f"{REDIS_KEY_PREFIX}:{user_id}:{agent_id}:{key}"


def _redis_pattern_for_user(user_id: str) -> str:
    """Match all cached context rows for a user (any agent, any logical key)."""
    return f"{REDIS_KEY_PREFIX}:{user_id}:*"


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
    across a module — all state lives in Redis / PostgreSQL, not in the object.
    """

    @staticmethod
    def _mysql_upsert(
        user_id: str,
        agent_id: str,
        key: str,
        serialised: str,
        session_id: Optional[str],
    ) -> None:
        from core.models import AgentContext
        from core.database import db

        row = AgentContext.query.filter_by(
            user_id=user_id, agent_id=agent_id, key=key
        ).first()
        if row:
            row.value = serialised
            row.updated_at = datetime.utcnow()
            if session_id is not None:
                row.session_id = session_id
        else:
            db.session.add(
                AgentContext(
                    user_id=user_id,
                    agent_id=agent_id,
                    key=key,
                    value=serialised,
                    session_id=session_id,
                )
            )

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
        ttl controls the Redis expiry (seconds); the PostgreSQL copy has no expiry.
        """
        serialised = json.dumps(value, default=str)

        # ── Redis write ───────────────────────────────────────────────────────
        rc = _get_redis()
        if rc:
            try:
                rc.set(_redis_key(user_id, agent_id, key), serialised, ex=ttl)
            except Exception as exc:
                logger.warning("Redis write failed for %s/%s: %s", user_id, key, exc)

        # ── PostgreSQL write ───────────────────────────────────────────────────────
        try:
            from core.database import db

            self._mysql_upsert(user_id, agent_id, key, serialised, session_id)
            db.session.commit()
        except Exception as exc:
            from core.database import db

            db.session.rollback()
            logger.error("PostgreSQL write failed for %s/%s: %s", user_id, key, exc)

    def set_many(
        self,
        user_id: str,
        agent_id: str,
        entries: List[Union[Tuple[str, Any], Tuple[str, Any, Optional[str]]]],
        ttl: int = CONTEXT_TTL_SECONDS,
    ) -> None:
        """
        Batch write many keys for one user + agent. One PostgreSQL commit at the end.

        Each entry is (key, value) or (key, value, session_id).
        """
        if not entries:
            return

        from core.database import db

        rc = _get_redis()
        try:
            for raw in entries:
                if len(raw) == 2:
                    key, value = raw[0], raw[1]
                    session_id: Optional[str] = None
                else:
                    key, value, session_id = raw[0], raw[1], raw[2]
                serialised = json.dumps(value, default=str)
                if rc:
                    try:
                        rc.set(_redis_key(user_id, agent_id, key), serialised, ex=ttl)
                    except Exception as exc:
                        logger.warning(
                            "Redis write failed for %s/%s: %s", user_id, key, exc
                        )
                self._mysql_upsert(user_id, agent_id, key, serialised, session_id)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.error("PostgreSQL batch write failed for %s/%s: %s", user_id, agent_id, exc)
            raise

    def search(self, user_id: str, query: str = '', limit: int = 10):
        """
        Search persisted context by substring match on JSON value or key (PostgreSQL only).
        Returns AgentContext rows, newest first.
        """
        from core.models import AgentContext

        limit = _validate_search_limit(limit)
        q = AgentContext.query.filter(AgentContext.user_id == user_id)
        qstr = (query or '').strip()
        if qstr:
            like = f"%{_escape_sql_like(qstr)}%"
            q = q.filter(
                (AgentContext.value.ilike(like, escape='\\'))
                | (AgentContext.key.ilike(like, escape='\\'))
            )
        return q.order_by(AgentContext.updated_at.desc()).limit(limit).all()

    def get(self, user_id: str, agent_id: str, key: str, default: Any = None) -> Any:
        """
        Read a value by user_id + agent_id + key (matches persistent row identity).

        Tries Redis first (fast); falls back to PostgreSQL if Redis misses or is
        unavailable.  Returns default if the key doesn't exist anywhere.
        """
        # ── Redis read ────────────────────────────────────────────────────────
        rc = _get_redis()
        if rc:
            try:
                raw = rc.get(_redis_key(user_id, agent_id, key))
                if raw is not None:
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("Redis read failed for %s/%s/%s: %s", user_id, agent_id, key, exc)

        # ── PostgreSQL fallback ────────────────────────────────────────────────────
        try:
            from core.models import AgentContext

            row = AgentContext.query.filter_by(
                user_id=user_id, agent_id=agent_id, key=key
            ).first()
            if row:
                value = json.loads(row.value)
                # Warm Redis cache back up
                if rc:
                    try:
                        rc.set(
                            _redis_key(user_id, agent_id, key),
                            row.value,
                            ex=CONTEXT_TTL_SECONDS,
                        )
                    except Exception:
                        pass
                return value
        except Exception as exc:
            logger.error(
                "PostgreSQL read failed for %s/%s/%s: %s", user_id, agent_id, key, exc
            )

        return default

    def snapshot(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Return all context for a user from PostgreSQL.

        Structure: ``{ agent_id: { logical_key: value, ... }, ... }``.
        When the same agent+key appears more than once (should not happen),
        the newest ``updated_at`` wins.
        """
        try:
            from core.models import AgentContext

            rows = (
                AgentContext.query.filter_by(user_id=user_id)
                .order_by(AgentContext.updated_at.desc())
                .all()
            )
            result: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                agent_bucket = result.setdefault(row.agent_id, {})
                if row.key in agent_bucket:
                    continue
                agent_bucket[row.key] = json.loads(row.value)
            return result
        except Exception as exc:
            logger.error("Snapshot failed for %s: %s", user_id, exc)
            return {}

    def delete(self, user_id: str, agent_id: str, key: str) -> None:
        """Delete a specific key for a user (both Redis and PostgreSQL)."""
        rc = _get_redis()
        if rc:
            try:
                rc.delete(_redis_key(user_id, agent_id, key))
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
                pattern = _redis_pattern_for_user(user_id)
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

    # ─────────────────────────────────────────────────────────────────────────
    # Flexible "ask" interface — agent doesn't need to know where data comes from
    # ─────────────────────────────────────────────────────────────────────────

    def ask(
        self,
        user_id: str,
        what: str,
        query: Optional[str] = None,
        limit: int = 10,
        **kwargs,
    ) -> Any:
        """
        Ask for data by logical name — don't need to know which agent provides it.

        Args:
            user_id: User ID
            what: What you're asking for (see examples below)
            query: Optional search query for semantic/text search
            limit: Max results for list-type requests
            **kwargs: Additional parameters for specific data types

        Examples:
            ctx.ask(user_id, "documents")           # List of available documents
            ctx.ask(user_id, "document_context", query="pricing")  # RAG context
            ctx.ask(user_id, "company_profile")     # From any agent that has it
            ctx.ask(user_id, "all")                 # Full snapshot
            ctx.ask(user_id, "search", query="competitor")  # Search all context

        Returns:
            Requested data if available, None otherwise
        """
        what = what.lower().strip()

        # Documents — from document_intelligence
        if what in ("documents", "docs", "document_list"):
            return self.get(user_id, "document_intelligence", "documents:index", default=[])

        # Document context for RAG — semantic search across documents
        if what in ("document_context", "doc_context", "rag_context"):
            if not query:
                return None
            try:
                from agents.document_intelligence.service import get_document_context_for_prompt
                return get_document_context_for_prompt(user_id, query, max_tokens=kwargs.get("max_tokens", 2000))
            except Exception as e:
                logger.warning("Document context fetch failed: %s", e)
                return None

        # Document search — raw vector search
        if what in ("document_search", "doc_search", "search_documents"):
            if not query:
                return []
            try:
                from agents.document_intelligence.service import search_documents
                return search_documents(user_id, query, top_k=limit)
            except Exception as e:
                logger.warning("Document search failed: %s", e)
                return []

        # Full snapshot — everything available for user
        if what in ("all", "snapshot", "everything"):
            return self.snapshot(user_id)

        # Search across all context
        if what == "search":
            if not query:
                return []
            return self.search(user_id, query, limit=limit)

        # External sources (web search, crawl, API results)
        if what in ("web_search", "web_results", "search_results"):
            return self.get(user_id, "external", "web_search:index", default=[])

        if what in ("web_crawl", "crawl_results", "scraped"):
            return self.get(user_id, "external", "web_crawl:index", default=[])

        if what == "external":
            # All external data indices
            return {
                "web_search": self.get(user_id, "external", "web_search:index", default=[]),
                "web_crawl": self.get(user_id, "external", "web_crawl:index", default=[]),
                "api": self.get(user_id, "external", "api:index", default=[]),
            }

        # Generic key lookup — search across all agents for this key
        return self._find_key_any_agent(user_id, what)

    def _find_key_any_agent(self, user_id: str, key: str) -> Any:
        """
        Find a key from any agent that provides it.

        Checks all agents for the requested key and returns first match.
        """
        try:
            from core.models import AgentContext

            # Look for exact key match across all agents
            row = (
                AgentContext.query
                .filter_by(user_id=user_id, key=key)
                .order_by(AgentContext.updated_at.desc())
                .first()
            )
            if row:
                return json.loads(row.value)

            # Try partial key match (e.g., "company" matches "company_profile")
            rows = (
                AgentContext.query
                .filter(AgentContext.user_id == user_id)
                .filter(AgentContext.key.ilike(f"%{key}%"))
                .order_by(AgentContext.updated_at.desc())
                .limit(1)
                .all()
            )
            if rows:
                return json.loads(rows[0].value)

        except Exception as exc:
            logger.warning("Key lookup failed for %s/%s: %s", user_id, key, exc)

        return None

    def available(self, user_id: str) -> Dict[str, List[str]]:
        """
        List all available context keys for a user, grouped by agent.

        Returns:
            { agent_id: [key1, key2, ...], ... }
        """
        try:
            from core.models import AgentContext

            rows = AgentContext.query.filter_by(user_id=user_id).all()
            result: Dict[str, List[str]] = {}
            for row in rows:
                if row.agent_id not in result:
                    result[row.agent_id] = []
                if row.key not in result[row.agent_id]:
                    result[row.agent_id].append(row.key)
            return result
        except Exception as exc:
            logger.error("Available check failed for %s: %s", user_id, exc)
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # Standardized store patterns — agents use these instead of raw set()
    # ─────────────────────────────────────────────────────────────────────────

    def store_result(
        self,
        user_id: str,
        agent_id: str,
        result_type: str,
        result_id: str,
        data: Dict[str, Any],
        index_key: Optional[str] = None,
    ) -> None:
        """
        Standard pattern for storing agent results with automatic indexing.

        Args:
            user_id: User ID
            agent_id: Agent storing the result
            result_type: Type of result (e.g., "document", "research", "content")
            result_id: Unique ID for this result
            data: The result data to store
            index_key: Optional custom index key (defaults to "{result_type}s:index")

        This method:
        1. Stores the result under "{result_type}:{result_id}"
        2. Updates the index list under "{result_type}s:index"

        Example:
            ctx.store_result(
                user_id, "document_intelligence",
                result_type="document",
                result_id=doc_id,
                data={"file_name": "report.pdf", "chunks": 10, ...}
            )
            # Stores: document:{doc_id} = data
            # Updates: documents:index = [{doc_id, ...}, ...]
        """
        # Store the result
        result_key = f"{result_type}:{result_id}"
        self.set(user_id, agent_id, result_key, data)

        # Update the index
        idx_key = index_key or f"{result_type}s:index"
        existing_index = self.get(user_id, agent_id, idx_key, default=[])

        # Create index entry (subset of data for quick listing)
        index_entry = {
            "id": result_id,
            "type": result_type,
            "updated_at": datetime.utcnow().isoformat(),
        }
        # Include common fields if present
        for field in ("name", "file_name", "title", "status"):
            if field in data:
                index_entry[field] = data[field]

        # Remove old entry, add new at front
        existing_index = [e for e in existing_index if e.get("id") != result_id]
        existing_index.insert(0, index_entry)
        existing_index = existing_index[:100]  # Keep last 100

        self.set(user_id, agent_id, idx_key, existing_index)

    def store_from_source(
        self,
        user_id: str,
        source: str,
        data: Any,
        key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store data from an external source with automatic provenance tracking.

        Args:
            user_id: User ID
            source: Source identifier ("web_search", "web_crawl", "api", "upload")
            data: The data to store
            key: Optional specific key (auto-generated if not provided)
            metadata: Optional metadata (url, query, timestamp, etc.)

        Returns:
            The key where data was stored

        Example:
            ctx.store_from_source(
                user_id,
                source="web_search",
                data={"results": [...]},
                metadata={"query": "competitor analysis", "engine": "google"}
            )
        """
        from uuid import uuid4

        # Generate key if not provided
        if not key:
            key = f"{source}:{uuid4().hex[:8]}"

        # Wrap data with provenance
        wrapped = {
            "source": source,
            "data": data,
            "stored_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        # Store under "external" agent namespace
        self.set(user_id, agent_id="external", key=key, value=wrapped)

        # Update source index
        idx_key = f"{source}:index"
        existing = self.get(user_id, "external", idx_key, default=[])
        existing.insert(0, {
            "key": key,
            "stored_at": wrapped["stored_at"],
            **(metadata or {}),
        })
        existing = existing[:50]
        self.set(user_id, "external", idx_key, existing)

        logger.info(f"Stored external data: {source} -> {key}")
        return key
