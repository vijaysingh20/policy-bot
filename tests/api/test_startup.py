from uuid import uuid4


def test_app_starts_without_a_default_index(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert client.app.state.default_resources is None


def test_question_without_document_explains_what_to_do(client):
    response = client.post("/questions", json={"question": "How many leave days do I get?"})
    assert response.status_code == 409
    assert "Upload a PDF" in response.json()["detail"]


def test_question_about_unknown_document_is_404(client):
    response = client.post(
        "/questions", json={"question": "How many leave days?", "document_id": str(uuid4())}
    )
    assert response.status_code == 404