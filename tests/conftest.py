import pytest


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    """Tests must never send traces to LangSmith or call OpenAI."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")


@pytest.fixture
def make_context():
    """Factory: a RetrievalContext with sources S1..Sn, chunk i on page i."""
    from backend.schemas import ChunkRecord, ContextSource, PageMetaData, RetrievalContext

    def _make(n: int = 3) -> RetrievalContext:
        sources = [
            ContextSource(
                source_id=f"S{i}",
                chunk=ChunkRecord(
                    text=f"Policy text {i}",
                    metadata=PageMetaData(source="handbook.pdf", page_number=i),
                    chunk_number=i,
                ),
            )
            for i in range(1, n + 1)
        ]
        return RetrievalContext(context_text="...", sources=sources)

    return _make