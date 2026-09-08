"""Rebuild the handbook index from the source PDF.

Run with: python -m backend.scripts.build_index
"""

from backend.config import FINAL_TOP_K
from backend.ingestion import build_index
from backend.retrieval.vector_index import search_faiss
from backend.schemas import SearchRequest


def main() -> None:
    index, embedded_chunks, manifest, model = build_index()

    print("\nManifest:")
    print(manifest.model_dump_json(indent=2))

    request = SearchRequest(
        query="How can an employee review their personnel record?",
        top_k=FINAL_TOP_K,
    )

    matches = search_faiss(
        request=request,
        index=index,
        chunks=[item.chunk for item in embedded_chunks],
        model=model,
    )

    for rank, match in enumerate(matches, start=1):
        print(
            "\nRank:", rank,
            "| Score:", round(match.score, 4),
            "| PDF page:", match.chunk.metadata.page_number,
            "| Chunk:", match.chunk.chunk_number,
        )
        print(match.chunk.text)


if __name__ == "__main__":
    main()
