from backend.schemas import ContextSource, RetrievalContext, SearchResult

def build_context(
    matches: list[SearchResult]
) -> RetrievalContext:
    blocks: list[str] = []
    sources: list[ContextSource] = []

    for number, match in enumerate(matches, start=1):
        source_id = f"S{number}"
        source = ContextSource(
            source_id=source_id,
            chunk=match.chunk
        )

        sources.append(source)
        chunk = match.chunk

        block = (
            f"[{source_id}]\n"
            f"Source: {chunk.metadata.source}\n"
            f"PDF page: {chunk.metadata.page_number}\n"
            f"Chunk: {chunk.chunk_number}\n"
            f"Text:\n{chunk.text}"
        )

        blocks.append(block)

    return RetrievalContext(
        context_text="\n\n".join(blocks),
        sources=sources
    )