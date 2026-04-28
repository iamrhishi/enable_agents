"""Integration tests — Content Marketing agent."""
import pytest


@pytest.fixture
def project_id(client):
    res = client.post("/api/content-marketing/projects", json={
        "user_id": "user_1",
        "project_name": "Test Project",
        "industry": "Technology",
    })
    assert res.status_code == 201
    return res.get_json()["project_id"]


def test_create_project_success(client):
    res = client.post("/api/content-marketing/projects", json={
        "user_id": "user_1",
        "project_name": "My Project",
    })
    assert res.status_code == 201
    assert "project_id" in res.get_json()


def test_create_project_missing_required(client):
    res = client.post("/api/content-marketing/projects", json={})
    assert res.status_code == 400


def test_get_project(client, project_id):
    res = client.get(f"/api/content-marketing/projects/{project_id}")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["project"]["project_id"] == project_id


def test_get_project_not_found(client):
    res = client.get("/api/content-marketing/projects/nonexistent-id")
    assert res.status_code == 404


def test_list_documents_empty(client, project_id):
    res = client.get(f"/api/content-marketing/documents/{project_id}")
    assert res.status_code == 200
    assert res.get_json()["documents"] == []
