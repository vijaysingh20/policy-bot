def preflight(client, origin):
    return client.options(
        "/questions",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )


def test_allowed_origin_passes_preflight(client):
    response = preflight(client, "http://localhost:3000")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_unknown_origin_is_refused(client):
    response = preflight(client, "https://evil.example.com")
    assert "access-control-allow-origin" not in response.headers