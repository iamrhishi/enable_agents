"""
Shared SQLAlchemy db instance and Flask-Migrate setup.

Usage (in app factory):
    from core.database import db, migrate
    db.init_app(app)
    migrate.init_app(app, db)
"""
import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()


def init_db(app):
    """Attach db and migrate to the Flask app, set pool options."""
    db_uri = os.environ.get("DATABASE_URI") or os.environ.get("DATABASE_URL")
    if not db_uri:
        raise ValueError("DATABASE_URI or DATABASE_URL environment variable is required. PostgreSQL connection string expected.")
    if db_uri.startswith("sqlite"):
        raise ValueError("SQLite is not supported. PostgreSQL connection string expected in DATABASE_URI.")

    app.config.setdefault("SQLALCHEMY_DATABASE_URI", db_uri)
    app.config.setdefault("SQLALCHEMY_TRACK_MODIFICATIONS", False)
    app.config.setdefault("SQLALCHEMY_ENGINE_OPTIONS", {
        "pool_size": 10,
        "pool_recycle": 300,
        "pool_pre_ping": True,
    })

    db.init_app(app)
    migrate.init_app(app, db)

    # Enable pgvector extension on first connection (PostgreSQL only)
    if "postgresql" in db_uri:
        with app.app_context():
            try:
                from sqlalchemy import text
                db.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                db.session.commit()
            except Exception:
                db.session.rollback()
