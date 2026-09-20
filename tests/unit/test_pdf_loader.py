import backend.ingestion.pdf_loader as loader_module
from backend.ingestion.pdf_loader import is_table_of_contents, load_pdf_pages

TOC_PAGE = "\n".join([
    "TABLE OF CONTENTS",
    "3.5 SICK LEAVE BENEFITS ........................ 24",
    "3.6 SICK LEAVE BANK ............................ 25",
    "3.7 BEREAVEMENT LEAVE .......................... 26",
    "3.8 EDUCATION ASSISTANCE ....................... 26",
    "VISION...…………………………7",
])
POLICY_PAGE = "\n".join([
    "3.7 BEREAVEMENT LEAVE",
    "Paid bereavement leave will be provided to all employees as follows:",
    "a) Five (5) working days in the passing of an employee's spouse, child, father, mother.",
    "b) Three (3) working days in the passing of an employee's grandparents.",
])


def test_toc_page_is_detected():
    assert is_table_of_contents(TOC_PAGE)


def test_policy_page_with_numbers_is_not_a_toc():
    assert not is_table_of_contents(POLICY_PAGE)


def test_a_few_dotted_lines_inside_prose_are_not_a_toc():
    page = POLICY_PAGE + "\nSee section 3.2 ...... 23\nSee section 3.5 ...... 24"
    assert not is_table_of_contents(page)


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


def test_loader_skips_toc_pages_but_keeps_real_page_numbers(monkeypatch, tmp_path):
    fake_pages = [FakePage("Cover"), FakePage(TOC_PAGE), FakePage(POLICY_PAGE)]
    monkeypatch.setattr(loader_module, "PdfReader", lambda path: type("R", (), {"pages": fake_pages}))

    records = load_pdf_pages(tmp_path / "handbook.pdf")

    assert [r.metadata.page_number for r in records] == [1, 3]      # page 2 skipped, not renumbered
    assert "BEREAVEMENT" in records[1].text