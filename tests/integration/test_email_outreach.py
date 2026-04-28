"""Integration tests — Email Outreach agent."""
import pytest


@pytest.fixture
def campaign_id(client):
    res = client.post("/api/email/campaigns", json={
        "name": "Test Campaign",
        "subject": "Hello!",
        "body_template": "Hi {{name}}, ...",
        "recipients": [{"email": "a@b.com", "name": "A"}],
    })
    assert res.status_code == 201
    return res.get_json()["campaign_id"]


def test_create_campaign_success(client):
    res = client.post("/api/email/campaigns", json={
        "name": "Campaign 1",
        "subject": "Subject",
        "body_template": "Body",
    })
    assert res.status_code == 201
    assert "campaign_id" in res.get_json()


def test_create_campaign_missing_fields(client):
    res = client.post("/api/email/campaigns", json={"name": "incomplete"})
    assert res.status_code == 400


def test_list_campaigns(client, campaign_id):
    res = client.get("/api/email/campaigns")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    ids = [c["campaign_id"] for c in data["campaigns"]]
    assert campaign_id in ids


def test_campaign_stats(client, campaign_id):
    res = client.get(f"/api/email/campaigns/{campaign_id}/stats")
    assert res.status_code == 200
    data = res.get_json()
    assert data["campaign_id"] == campaign_id
    assert "total" in data


def test_campaign_recipients(client, campaign_id):
    res = client.get(f"/api/email/campaigns/{campaign_id}/recipients")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["recipients"]) == 1
    assert data["recipients"][0]["email"] == "a@b.com"
