from pathlib import Path
from pypdf import PdfReader

from backend.schemas import PageMetaData, PageRecord


def count_content_characters(text: str) -> int:
    return sum(1 for character in text if not character.isspace())

def load_pdf_pages(pdf_path):
    path = Path(pdf_path)
    pdf_reader = PdfReader(path)

    page_records = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        record = PageRecord(
            text = text,
            metadata = PageMetaData(
                source = path.name,
                page_number = page_number
            )
        )

        page_records.append(record)
    return page_records
