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
    db_uri = (
        os.environ.get("DATABASE_URI")
        or os.environ.get("DATABASE_URL")
        or "sqlite:///instance/enable_agents.db"
    )
    app.config.setdefault("SQLALCHEMY_DATABASE_URI", db_uri)
    app.config.setdefault("SQLALCHEMY_TRACK_MODIFICATIONS", False)
    # Pool options only apply to server-based databases (not SQLite)
    if not db_uri.startswith("sqlite"):
        app.config.setdefault("SQLALCHEMY_ENGINE_OPTIONS", {
            "pool_size": 10,
            "pool_recycle": 300,
            "pool_pre_ping": True,
        })

    db.init_app(app)
    migrate.init_app(app, db)
