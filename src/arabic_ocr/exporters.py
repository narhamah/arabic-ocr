"""Export helpers for OCR documents."""

from __future__ import annotations

import json

from arabic_ocr.models import OCRDocument
from arabic_ocr.normalizer import normalize_faithful, normalize_llm


def document_to_dict(document: OCRDocument) -> dict:
    """Serialize the canonical OCR document."""
    return document.to_dict()


def document_to_json(document: OCRDocument) -> str:
    """Serialize the document as pretty-printed JSON."""
    return json.dumps(document_to_dict(document), ensure_ascii=False, indent=2)


def document_from_dict(data: dict) -> OCRDocument:
    """Deserialize the canonical OCR document."""
    return OCRDocument.from_dict(data)


def document_from_json(text: str) -> OCRDocument:
    """Deserialize a canonical OCR document JSON payload."""
    return document_from_dict(json.loads(text))


def render_text(document: OCRDocument, *, profile: str = "faithful") -> str:
    """Render a document into text with page markers."""
    pages = []
    for page in document.pages:
        body = _normalize_for_profile(page.text, profile=profile)
        header = f"=== Page {page.page_number + 1} ==="
        pages.append(f"{header}\n{body}".strip())
    return "\n\n".join(pages).strip()


def render_markdown(document: OCRDocument, *, profile: str = "llm") -> str:
    """Render a document into Markdown with explicit page sections."""
    sections = []
    for page in document.pages:
        body = _normalize_for_profile(page.text, profile=profile)
        sections.append(f"## Page {page.page_number + 1}\n\n{body}".strip())
    return "\n\n".join(sections).strip()


def legacy_page_results(document: OCRDocument, *, profile: str = "faithful") -> list[dict]:
    """Produce the legacy list-of-dicts result shape."""
    results = []
    for page in document.pages:
        results.append({
            "page": page.page_number,
            "text": _normalize_for_profile(page.text, profile=profile),
            "confidence": page.confidence,
            "regions": len(page.blocks),
            "method": page.source_kind,
        })
    return results


def _normalize_for_profile(text: str, *, profile: str) -> str:
    if profile == "llm":
        return normalize_llm(text)
    return normalize_faithful(text)
