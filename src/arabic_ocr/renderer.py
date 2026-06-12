"""PDF inspection and rendering helpers."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from arabic_ocr.models import PageAsset, PageImages
from arabic_ocr.native_pdf import extract_best_native_page_text


def render_pdf(
    pdf_path: str | Path,
    dpi: int = 400,
    pages: list[int] | None = None,
) -> list[Image.Image]:
    """Render PDF pages to PIL Images."""
    assets = inspect_pdf(pdf_path, dpi=dpi, pages=pages, render_images=True)
    return [asset.images.raw for asset in assets if asset.images.raw is not None]


def inspect_pdf(
    pdf_path: str | Path,
    *,
    dpi: int = 400,
    pages: list[int] | None = None,
    render_images: bool = True,
) -> list[PageAsset]:
    """Inspect a PDF and optionally render page imagery."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    if not doc.is_pdf:
        doc.close()
        raise RuntimeError(f"Not a valid PDF file: {pdf_path}")

    page_indices = pages if pages is not None else list(range(len(doc)))
    assets: list[PageAsset] = []
    cmap_cache: dict[int, dict[int, str]] = {}

    try:
        for page_index in page_indices:
            page = doc.load_page(page_index)
            rect = page.rect
            embedded_text, metadata = extract_best_native_page_text(
                pdf_path,
                doc,
                page,
                cmap_cache=cmap_cache,
            )
            images = PageImages()
            if render_images:
                pix = page.get_pixmap(dpi=dpi)
                images.raw = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            assets.append(PageAsset(
                page_number=page_index,
                width=int(rect.width),
                height=int(rect.height),
                embedded_text=embedded_text,
                images=images,
                metadata=metadata,
            ))
    finally:
        doc.close()

    return assets
