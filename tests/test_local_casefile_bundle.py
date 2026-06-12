"""Tests for local-only casefile cleanup and bundling."""

from __future__ import annotations

from arabic_ocr.local_casefile_bundle import (
    build_manual_case_index,
    clean_page_text,
    render_grouped_context,
)
from arabic_ocr.models import BoundingBox, OCRBlock, OCRDocument, OCRLine, OCRPage, OCRSpan


def _sample_document(page_total: int = 82) -> OCRDocument:
    pages = []
    for page_number in range(page_total):
        pages.append(
            OCRPage(
                page_number=page_number,
                width=800,
                height=1000,
                source_kind="scan",
                blocks=[
                    OCRBlock(
                        block_type="text",
                        bbox=BoundingBox(0, 0, 800, 1000),
                        reading_order=0,
                        lines=[OCRLine(spans=[OCRSpan(text=f"page {page_number + 1}")])],
                    )
                ],
            )
        )
    return OCRDocument(source_pdf="ملف قضية الشيك الكامل.pdf", dpi=400, pages=pages)


def test_clean_page_text_removes_artifact_variants():
    cleaned, removed = clean_page_text(
        "\n".join(
            [
                "There is no visible or legible text in the provided image.",
                "Scanned with CS CamScannerTM",
                "[The image contains no visible text.]",
                "جوهر الصفحة",
            ]
        )
    )

    assert cleaned == "جوهر الصفحة"
    assert removed == [
        "no_visible_or_legible_text",
        "camscanner",
        "image_contains_no_visible_text",
    ]


def test_render_grouped_context_uses_manual_case_index():
    document = _sample_document()
    case_index = build_manual_case_index(document)

    rendered = render_grouped_context(document, case_index, [f"text {i}" for i in range(1, 83)])

    assert "# Grouped Case File Bundle" in rendered
    assert "## Document 01 - Court session cover sheet" in rendered
    assert "### Page 1" in rendered
    assert "### Page 82" in rendered
