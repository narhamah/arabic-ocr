"""Generate test fixture PDFs and images with Arabic text."""

import fitz  # PyMuPDF
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def create_single_page_pdf() -> Path:
    """Create a 1-page PDF with Arabic text."""
    path = FIXTURES_DIR / "single_page.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    # Insert Arabic text
    text = "بسم الله الرحمن الرحيم"
    page.insert_text((100, 100), text, fontsize=24, fontname="helv")
    doc.save(str(path))
    doc.close()
    return path


def create_multi_page_pdf() -> Path:
    """Create a 3-page PDF."""
    path = FIXTURES_DIR / "multi_page.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=595, height=842)
        page.insert_text((100, 100), f"Page {i + 1}", fontsize=24, fontname="helv")
    doc.save(str(path))
    doc.close()
    return path


def create_not_a_pdf() -> Path:
    """Create a non-PDF file for error testing."""
    path = FIXTURES_DIR / "not_a_pdf.txt"
    path.write_text("This is not a PDF file.")
    return path


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    create_single_page_pdf()
    create_multi_page_pdf()
    create_not_a_pdf()
    print(f"Fixtures created in {FIXTURES_DIR}")
