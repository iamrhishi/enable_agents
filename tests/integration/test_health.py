"""Integration tests — Health blueprint."""


def test_health_returns_200(client):
    res = client.get("/health")
    assert res.status_code in (200, 503)
    data = res.get_json()
    assert "status" in data
    assert "timestamp" in data


def test_health_has_db_check(client):
    res = client.get("/health")
    data = res.get_json()
    assert "checks" in data
    assert "db" in data["checks"]
