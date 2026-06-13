"""
Core SQLAlchemy models.

Only models that are unique to the agent registry / context system live here.
The primary application models (User, GoogleOAuthToken, EmailCampaign, etc.)
are defined in app.py alongside the routes that own them, using the same db
instance created there.
"""
from datetime import datetime
from core.database import db


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
    session_id = db.Column(db.String(36), index=True)
    agent_id = db.Column(db.String(100), nullable=False)
    key = db.Column(db.String(255), nullable=False)
    value = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("user_id", "agent_id", "key", name="uq_agent_context"),
        db.Index("ix_agent_context_user_key", "user_id", "key"),
    )


class UserSettingModel(db.Model):
    """
    Encrypted per-user settings storage.

    Stores API keys, OAuth tokens, connector configurations, etc.
    All sensitive values are encrypted at rest.

    Use UserSettings (core/settings.py) to read/write — provides
    encryption/decryption automatically.
    """
    __tablename__ = "user_settings"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(255), nullable=False, index=True)
    category = db.Column(db.String(50), nullable=False)  # ai, connectors, scraping, oauth
    key = db.Column(db.String(100), nullable=False)
    value_encrypted = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("user_id", "category", "key", name="uq_user_setting"),
        db.Index("ix_user_settings_user_category", "user_id", "category"),
    )
