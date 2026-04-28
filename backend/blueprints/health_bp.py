"""
Health Blueprint — lightweight liveness/readiness endpoints.
"""
from datetime import datetime

from flask import Blueprint, jsonify

from core.database import db

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health_check():
    db_ok = True
    try:
        db.session.execute(db.text("SELECT 1"))
    except Exception:
        db_ok = False

    status = "healthy" if db_ok else "degraded"
    code = 200 if db_ok else 503
    return jsonify({
        "status": status,
        "service": "enable-agents-api",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {"db": "ok" if db_ok else "error"},
    }), code
