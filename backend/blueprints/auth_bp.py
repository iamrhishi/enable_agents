"""
Auth Blueprint — /register, /login, /auth/google/start, /auth/google/callback.

Owns: user registration, password login, Google OAuth flow.
Delegates Google token exchange to the legacy helpers still in app.py
during the incremental migration.
"""
import os
from urllib.parse import urlencode

from flask import Blueprint, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

from core.database import db
from core.models import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.post("/register")
def register():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 400

    try:
        user = User(
            username=email,
            password=generate_password_hash(password, method="pbkdf2:sha256"),
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            email=email,
            company=data.get("company"),
            linkedin=data.get("linkedin"),
            short_intro=data.get("short_intro"),
            company_intro=data.get("company_intro"),
        )
        db.session.add(user)
        db.session.commit()
        return jsonify({"message": "User registered successfully"}), 201
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": str(exc)}), 500


@auth_bp.post("/login")
def login():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password, password):
        return jsonify({"message": "Login successful", "username": user.username, "email": user.email}), 200
    return jsonify({"error": "Invalid email or password"}), 401


@auth_bp.get("/auth/google/start")
def google_auth_start():
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    if not client_id:
        return jsonify({"error": "Google OAuth is not configured on this server."}), 503

    # GOOGLE_REDIRECT_URI must match exactly what is registered in Google Cloud Console.
    redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

    scopes = [
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/gmail.send",
    ]
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "access_type": "offline",
        "prompt": "consent",
        "state": "user_login_flow",
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    return jsonify({"auth_url": auth_url, "state": "user_login_flow"})
