"""
Core SQLAlchemy models — User and OAuth token.

These are the foundational models used by the auth blueprint and
referenced by multiple agents.
"""
from datetime import datetime
from core.database import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password = db.Column(db.String(512), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    first_name = db.Column(db.String(255))
    last_name = db.Column(db.String(255))
    company = db.Column(db.String(255))
    linkedin = db.Column(db.String(512))
    short_intro = db.Column(db.Text)
    company_intro = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class AgentContext(db.Model):
    """
    Persistent cross-agent context store (the "context lake").

    Each row is one key-value pair scoped to a user (and optionally a session).
    The agent_id column records which agent wrote the value so the registry can
    enforce manifest-declared 'provides' / 'consumes' contracts.

    Use ContextStore (core/context.py) to read and write — do not query this
    table directly from agent code.
    """
    __tablename__ = "agent_context"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(255), nullable=False, index=True)
    session_id = db.Column(db.String(36), index=True)          # optional — group related keys
    agent_id = db.Column(db.String(100), nullable=False)        # which agent wrote this
    key = db.Column(db.String(255), nullable=False)             # e.g. "company_profile"
    value = db.Column(db.Text, nullable=False)                  # JSON-serialised value
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("user_id", "agent_id", "key", name="uq_agent_context"),
        db.Index("ix_agent_context_user_key", "user_id", "key"),
    )


class GoogleOAuthToken(db.Model):
    __tablename__ = "google_oauth_tokens"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_email = db.Column(db.String(255), nullable=False, index=True)
    access_token = db.Column(db.Text)
    refresh_token = db.Column(db.Text)
    token_expiry = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
