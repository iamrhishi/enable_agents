"""
Favorites Blueprint — user-saved profile favorites.
"""
import json
import os
from datetime import datetime

from flask import Blueprint, jsonify, request
from flask_cors import cross_origin

favorites_bp = Blueprint("favorites", __name__)

_USER_DATA_DIR = os.environ.get(
    "ENABLE_AGENTS_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data"),
)


def _favorites_path(user_id: str) -> str:
    return os.path.join(_USER_DATA_DIR, "user_data", user_id, "favorites.json")


def _load(path: str) -> list:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _dump(path: str, data: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@favorites_bp.post("/save_user_favorite")
@cross_origin()
def save_user_favorite():
    data = request.get_json() or {}
    user_id = data.get("user_id", "default_user")
    profile_data = data.get("profile_data")

    if not profile_data:
        return jsonify({"success": False, "error": "No profile_data provided"}), 400

    path = _favorites_path(user_id)
    favorites = _load(path)

    full_name = f"{profile_data.get('name', '')} {profile_data.get('lastname', '')}".strip()
    if any(
        f.get("full_name") == full_name and f.get("company") == profile_data.get("company")
        for f in favorites
    ):
        return jsonify({"success": False, "error": "Profile already in favorites"})

    favorites.append({
        **profile_data,
        "saved_at": datetime.utcnow().isoformat(),
        "favorite_id": len(favorites) + 1,
    })
    _dump(path, favorites)
    return jsonify({"success": True, "message": "Saved", "favorites_count": len(favorites)})


@favorites_bp.post("/get_user_favorites")
@cross_origin()
def get_user_favorites():
    data = request.get_json() or {}
    user_id = data.get("user_id", "default_user")
    path = _favorites_path(user_id)
    favorites = _load(path)
    return jsonify({"success": True, "favorites": favorites, "count": len(favorites)})


@favorites_bp.post("/remove_user_favorite")
@cross_origin()
def remove_user_favorite():
    data = request.get_json() or {}
    user_id = data.get("user_id", "default_user")
    favorite_id = data.get("favorite_id")

    if not favorite_id:
        return jsonify({"success": False, "error": "No favorite_id provided"}), 400

    path = _favorites_path(user_id)
    if not os.path.exists(path):
        return jsonify({"success": False, "error": "No favorites found"}), 404

    favorites = _load(path)
    original_count = len(favorites)
    favorites = [f for f in favorites if f.get("favorite_id") != favorite_id]

    if len(favorites) == original_count:
        return jsonify({"success": False, "error": "Favorite not found"}), 404

    _dump(path, favorites)
    return jsonify({"success": True, "message": "Removed", "favorites_count": len(favorites)})
