import pytest
from fastapi.testclient import TestClient

import backend.api as api


class FakeEmbeddingModel:
    """Placeholder: startup only stores the model, it never encodes anything here."""


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The real FastAPI app, with fake models and every path inside tmp_path."""
    monkeypatch.setattr(api, "load_model_for_ingestion",
                        lambda name: (FakeEmbeddingModel(), "test-revision"))
    monkeypatch.setattr(api, "load_reranker", lambda: object())
    monkeypatch.setattr(api, "INDEX_DIR", tmp_path / "storage" / "handbook-v1")  # does not exist
    monkeypatch.setattr(api, "DATABASE_DIR", tmp_path / "db")
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")

    with TestClient(api.app) as test_client:  # `with` runs the lifespan startup
        yield test_client