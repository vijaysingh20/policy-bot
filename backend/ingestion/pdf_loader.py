import logging
import re
from pathlib import Path

from pypdf import PdfReader

from backend.schemas import PageMetaData, PageRecord

logger = logging.getLogger(__name__)

# "3.7 BEREAVEMENT LEAVE ........ 26"  or  "VISION...………7"
TOC_ENTRY = re.compile(r"[.…]{4,}\s*\d+\s*$")


def is_table_of_contents(text: str, min_entries: int = 5, min_ratio: float = 0.5) -> bool:
    """A page is a TOC when most of its lines are 'title ....... page-number' entries.

    TOC pages list every section title but contain no answers, which makes them
    look highly relevant to a cross-encoder while being useless as context.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    entries = sum(bool(TOC_ENTRY.search(line)) for line in lines)
    return entries >= min_entries and entries / len(lines) >= min_ratio


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

    for page_number, page in enumerate(pdf_reader.pages, start=1):  # numbered before skipping
        text = (page.extract_text() or "").strip()

        if is_table_of_contents(text):
            logger.info("Skipping page %d of %s: table of contents", page_number, display_name)
            continue

        page_records.append(
            PageRecord(
                text=text,
                metadata=PageMetaData(source=display_name, page_number=page_number),
            )
        )
    return page_records