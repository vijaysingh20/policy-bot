import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.schemas import ChunkRecord, PageRecord


def normalize_page_text(text: str) -> str:
    cleaned_lines = []

    for line in text.splitlines():
        cleaned_lines.append(line.rstrip())
    joined_lines = "\n".join(cleaned_lines)

    normalized_lines = re.sub(r"\n{3,}", "\n\n", joined_lines)
    return normalized_lines.strip()


def split_page_recursive(
    page: PageRecord,
    splitter: RecursiveCharacterTextSplitter
) -> list[ChunkRecord]:
    pieces = splitter.split_text(page.text)

    chunks = []

    for chunk_number, piece in enumerate(pieces, start=1):
        chunk = ChunkRecord(
            text = piece,
            metadata = page.metadata,
            chunk_number = chunk_number
        )

        chunks.append(chunk)
    return chunks

def build_document_chunks(
    pages: list[PageRecord],
    splitter: RecursiveCharacterTextSplitter
) -> list[ChunkRecord]:
    doc_chunks = []
    for page in pages:
        normalized_text = normalize_page_text(page.text)
        page_record = PageRecord(
            text=normalized_text,
            metadata=page.metadata
        )

        page_chunks = split_page_recursive(
            page=page_record,
            splitter= splitter
        )
        doc_chunks.extend(page_chunks)
    return doc_chunks
