"""
Shared request authentication for all routes (app.py and every agent blueprint).

The only trusted identity signal is a valid `Authorization: Bearer <session_token>`
issued at /login, /register, or the Google OAuth callback (see core/session_token.py).
Client-supplied identity (a `user_id`/`X-User-Id` in the body/headers/query string)
is never trusted on its own - it is spoofable by any caller.
"""
from __future__ import annotations

from functools import wraps
from typing import Optional

from flask import current_app, g, jsonify, request

from core.session_token import verify_browser_session_token


def get_authenticated_user_id() -> Optional[str]:
    """Verify the Authorization: Bearer <session_token> header. Returns the
    user id it was issued to, or None if missing/invalid/expired."""
    auth = (request.headers.get("Authorization") or "").strip()
    if not auth.lower().startswith("bearer "):
        return None
    raw = auth[7:].strip()
    if not raw:
        return None
    secret = current_app.config.get("SECRET_KEY") or ""
    return verify_browser_session_token(secret, raw)


def require_auth(view_func):
    """Reject the request with 401 unless a valid session Bearer token is
    present. On success, sets g.user_id for the view to use."""

    @wraps(view_func)
    def wrapped(*args, **kwargs):
        user_id = get_authenticated_user_id()
        if not user_id:
            return jsonify({
                "success": False,
                "error": "Missing or invalid session. Sign in and send Authorization: Bearer <session_token>.",
            }), 401
        g.user_id = user_id
        return view_func(*args, **kwargs)

    return wrapped


def user_can_access_project(user_id: str, project_id: str) -> bool:
    """True if user_id owns project_id directly, or is a member of the team
    that owns it. Unknown project_id -> False (caller should 404, not 403,
    to avoid confirming existence of other users' projects)."""
    if not user_id or not project_id:
        return False
    from core.models import Project, TeamMember

    project = Project.query.filter_by(project_id=project_id).first()
    if not project:
        return False
    if project.owner_id == user_id:
        return True
    return (
        TeamMember.query.filter_by(team_id=project.team_id, user_id=user_id).first()
        is not None
    )


def user_can_manage_project_settings(user_id: str, project_id: str) -> bool:
    """True if user_id can change project-level settings (API keys, etc.) -
    narrower than user_can_access_project: the project owner, or a team
    member with role owner/admin. Plain members/viewers can use a
    project's configured key but not change it."""
    if not user_id or not project_id:
        return False
    from core.models import Project, TeamMember

    project = Project.query.filter_by(project_id=project_id).first()
    if not project:
        return False
    if project.owner_id == user_id:
        return True
    member = TeamMember.query.filter_by(team_id=project.team_id, user_id=user_id).first()
    return member is not None and member.role in ("owner", "admin")


def require_project_access(get_project_id):
    """Decorator factory: like require_auth, but additionally requires that
    the authenticated user owns (or is a team member of) the project
    identified by calling get_project_id(*args, **kwargs) - e.g. a lambda
    pulling project_id from a URL kwarg, query string, or JSON body.

    Must be applied inside (below) @require_auth so g.user_id is set first.
    """

    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            project_id = get_project_id(*args, **kwargs)
            if not project_id or not user_can_access_project(g.user_id, project_id):
                return jsonify({"success": False, "error": "Project not found"}), 404
            return view_func(*args, **kwargs)

        return wrapped

    return decorator
