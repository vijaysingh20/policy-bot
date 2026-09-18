import pytest
from fastapi.testclient import TestClient

import backend.api as api


class FakeEmbeddingModel:
    """Placeholder: startup only stores the model, it never encodes anything here."""


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    """Tests must never send traces to LangSmith or call OpenAI."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")


@pytest.fixture
def make_context():
    from backend.schemas import ChunkRecord, ContextSource, PageMetaData, RetrievalContext

    def _make(n: int = 3) -> RetrievalContext:
        sources = [
            ContextSource(
                source_id=f"S{i}",
                chunk=ChunkRecord(
                    text=f"Policy text {i}",
                    metadata=PageMetaData(source="handbook.pdf", page_number=i),
                    chunk_number=i
                ),
            )
            for i in range(1, n + 1)
        ]
        return RetrievalContext(context_text="...", sources=sources)
    return _make


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The real FastAPI app, with fake models and every path inside tmp_path."""
    monkeypatch.setattr(api, "load_model_for_ingestion",
                        lambda name: (FakeEmbeddingModel(), "test-revision"))
    monkeypatch.setattr(api, "load_reranker", lambda: object())
    monkeypatch.setattr(api, "INDEX_DIR", tmp_path / "storage" / "handbook-v1")
    monkeypatch.setattr(api, "DATABASE_DIR", tmp_path / "db")
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")

    with TestClient(api.app) as test_client:  # `with` runs the lifespan startup
        yield test_client