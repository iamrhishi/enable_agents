"""Integration tests — Market Research agent."""
import pytest


@pytest.fixture
def project_id(client):
    res = client.post("/api/market-research/projects", json={
        "user_id": "user_1",
        "company_name": "Acme Corp",
        "industry": "Technology",
    })
    assert res.status_code == 201
    return res.get_json()["project_id"]


def test_create_project_success(client):
    res = client.post("/api/market-research/projects", json={
        "user_id": "user_1",
        "company_name": "Test Co",
        "industry": "Finance",
        "research_goals": ["competitor analysis"],
    })
    assert res.status_code == 201
    assert "project_id" in res.get_json()


def test_create_project_missing_fields(client):
    res = client.post("/api/market-research/projects", json={"user_id": "u1"})
    assert res.status_code == 400


def test_get_project(client, project_id):
    res = client.get(f"/api/market-research/projects/{project_id}")
    assert res.status_code == 200
    data = res.get_json()
    assert data["project"]["project_id"] == project_id


def test_get_project_not_found(client):
    res = client.get("/api/market-research/projects/nonexistent")
    assert res.status_code == 404


def test_get_results_empty(client, project_id):
    res = client.get(f"/api/market-research/projects/{project_id}/results")
    assert res.status_code == 200
    assert res.get_json()["results"] == []
