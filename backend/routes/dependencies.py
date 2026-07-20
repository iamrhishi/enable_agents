"""
Dependencies API Routes.

Exposes dependency validation status for agents.
"""

from flask import Blueprint, request, jsonify
from core.dependency_validator import get_dependency_status, validator

dependencies_bp = Blueprint("dependencies", __name__, url_prefix="/api/dependencies")


@dependencies_bp.route("/status/<agent_id>", methods=["GET"])
def check_agent_dependencies(agent_id):
    """Get dependency status for a specific agent."""
    user_id = request.headers.get("X-User-Id", "")
    if not user_id:
        return jsonify({"error": "X-User-Id header required"}), 400

    status = get_dependency_status(agent_id, user_id)
    return jsonify(status)


@dependencies_bp.route("/config", methods=["GET"])
def get_dependency_config():
    """Get full dependency configuration."""
    return jsonify(validator.config)


@dependencies_bp.route("/reload", methods=["POST"])
def reload_config():
    """Reload dependency configuration from disk."""
    validator.reload_config()
    return jsonify({"success": True, "message": "Configuration reloaded"})
