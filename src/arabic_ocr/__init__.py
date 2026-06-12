"""Arabic OCR pipeline for scanned legal documents."""

from arabic_ocr.pipeline import process_pdf, process_pdf_document

__all__ = ["process_pdf", "process_pdf_document", "__version__"]

__version__ = "0.2.0"
