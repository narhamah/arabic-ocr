"""Stage 0: PDF to images renderer using PyMuPDF."""

from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


def render_pdf(
    pdf_path: str | Path,
    dpi: int = 400,
    pages: list[int] | None = None,
) -> list[Image.Image]:
    """Render PDF pages to PIL Images.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering (default 400).
        pages: Optional list of 0-based page indices to render.
               If None, renders all pages.

    Returns:
        List of PIL Images, one per rendered page.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        RuntimeError: If the file is not a valid PDF.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))

    if not doc.is_pdf:
        doc.close()
        raise RuntimeError(f"Not a valid PDF file: {pdf_path}")

    page_indices = pages if pages is not None else range(len(doc))
    images: list[Image.Image] = []

    for idx in page_indices:
        page = doc.load_page(idx)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)

    doc.close()
    return images
