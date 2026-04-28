"""
Shared pytest fixtures for integration tests.

Builds a MINIMAL Flask app from just the blueprints and agent packages
under test — intentionally does NOT import the giant app.py so that
tests run without the full dependency stack (faiss, selenium, etc.).

For full end-to-end tests that also exercise the legacy routes, run
inside Docker: docker compose exec backend-dev pytest tests/integration/
"""
import os
import sys

import pytest

# ── Env stubs (must be set before any module is imported) ────────────────────
os.environ.setdefault("DATABASE_URI", "sqlite:///:memory:")
os.environ.setdefault("PUBLIC_URL", "http://localhost:5000")
os.environ.setdefault("GOOGLE_CLIENT_ID", "test-client-id")
os.environ.setdefault("GOOGLE_CLIENT_SECRET", "test-client-secret")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")

# Add backend/ to the Python path
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


def _build_test_app():
    """Create a minimal Flask app for blueprint/agent testing."""
    from flask import Flask
    from flask_cors import CORS

    from core.database import db, init_db
    from core.logging_config import configure_logging

    app = Flask("test_app")
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    CORS(app)
    configure_logging(app)
    init_db(app)

    # Core blueprints
    from blueprints.auth_bp import auth_bp
    from blueprints.health_bp import health_bp
    from blueprints.prompts_bp import prompts_bp
    from blueprints.favorites_bp import favorites_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(prompts_bp)
    app.register_blueprint(favorites_bp)

    # Agent blueprints
    from agents.registry import registry_bp, _load_manifests, _registry, agent_dir_for
    import importlib

    _load_manifests()
    for agent_id, manifest in _registry.items():
        if not manifest.get("enabled"):
            continue
        dir_name = agent_dir_for(agent_id)
        try:
            module = importlib.import_module(f"agents.{dir_name}.routes")
            from flask import Blueprint as _BP
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, _BP) and obj.name != "agent_registry":
                    app.register_blueprint(obj)
                    break
        except Exception as exc:
            print(f"[test] Skipping agent {agent_id}: {exc}")

    app.register_blueprint(registry_bp)

    with app.app_context():
        # Ensure all models are imported so db.create_all() sees their tables
        import core.models  # noqa: F401 — registers User, GoogleOAuthToken
        db.create_all()

    return app


@pytest.fixture(scope="session")
def flask_app():
    app = _build_test_app()
    yield app
    with app.app_context():
        from core.database import db
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()
