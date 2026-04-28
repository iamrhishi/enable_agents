"""Integration tests — Auth blueprint."""
import pytest


class TestRegister:
    def test_register_success(self, client):
        res = client.post("/register", json={
            "email": "test@example.com",
            "password": "secret123",
            "first_name": "Test",
            "last_name": "User",
        })
        assert res.status_code == 201
        assert res.get_json()["message"] == "User registered successfully"

    def test_register_duplicate_email(self, client):
        payload = {"email": "dup@example.com", "password": "secret"}
        client.post("/register", json=payload)
        res = client.post("/register", json=payload)
        assert res.status_code == 400
        assert "already registered" in res.get_json()["error"]

    def test_register_missing_fields(self, client):
        res = client.post("/register", json={"email": "no-pass@example.com"})
        assert res.status_code == 400


class TestLogin:
    def test_login_success(self, client):
        client.post("/register", json={"email": "login@example.com", "password": "pass123"})
        res = client.post("/login", json={"email": "login@example.com", "password": "pass123"})
        assert res.status_code == 200
        data = res.get_json()
        assert data["message"] == "Login successful"
        assert data["email"] == "login@example.com"

    def test_login_wrong_password(self, client):
        client.post("/register", json={"email": "wp@example.com", "password": "right"})
        res = client.post("/login", json={"email": "wp@example.com", "password": "wrong"})
        assert res.status_code == 401

    def test_login_unknown_user(self, client):
        res = client.post("/login", json={"email": "nobody@example.com", "password": "x"})
        assert res.status_code == 401
