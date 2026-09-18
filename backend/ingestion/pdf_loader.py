from pathlib import Path
from pypdf import PdfReader

from backend.schemas import PageMetaData, PageRecord


def load_pdf_pages(
    pdf_path: str | Path,
    *,
    source_name: str | None = None
) -> list[PageRecord]:
    path = Path(pdf_path)

    display_name = path.name if source_name is None else source_name.strip()

    if not display_name:
        raise ValueError("Source name must not be blank")

    pdf_reader = PdfReader(path)

    page_records: list[PageRecord] = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        record = PageRecord(
            text = text,
            metadata = PageMetaData(
                source = display_name,
                page_number = page_number
            )
        )

        page_records.append(record)
    return page_records
