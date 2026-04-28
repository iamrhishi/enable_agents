"""Integration tests — Agent registry API."""


def test_list_agents_returns_list(client):
    res = client.get("/api/v1/agents/")
    assert res.status_code == 200
    data = res.get_json()
    assert isinstance(data, list)


def test_all_agents_have_required_fields(client):
    res = client.get("/api/v1/agents/")
    for agent in res.get_json():
        for field in ("id", "name", "description", "enabled"):
            assert field in agent, f"Agent missing field '{field}': {agent}"


def test_agent_health_known(client):
    res = client.get("/api/v1/agents/")
    agents = res.get_json()
    if not agents:
        return  # no agents registered — skip
    agent_id = agents[0]["id"]
    res2 = client.get(f"/api/v1/agents/{agent_id}/health")
    assert res2.status_code == 200
    assert res2.get_json()["id"] == agent_id


def test_toggle_agent(client):
    res = client.get("/api/v1/agents/")
    agents = res.get_json()
    if not agents:
        return
    agent_id = agents[0]["id"]
    original = agents[0]["enabled"]

    res2 = client.patch(f"/api/v1/agents/{agent_id}", json={"enabled": not original})
    assert res2.status_code == 200
    assert res2.get_json()["enabled"] == (not original)

    # restore
    client.patch(f"/api/v1/agents/{agent_id}", json={"enabled": original})


def test_unknown_agent_health_404(client):
    res = client.get("/api/v1/agents/does-not-exist/health")
    assert res.status_code == 404
