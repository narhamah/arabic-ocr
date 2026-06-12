"""Tests for document models and exporters."""

from pathlib import Path

from arabic_ocr.exporters import (
    document_from_dict,
    document_from_json,
    document_to_dict,
    document_to_json,
    legacy_page_results,
    render_markdown,
    render_text,
)
from arabic_ocr.models import BoundingBox, OCRBlock, OCRDocument, OCRLine, OCRPage, OCRSpan


def _sample_document() -> OCRDocument:
    return OCRDocument(
        source_pdf="sample.pdf",
        dpi=400,
        pages=[
            OCRPage(
                page_number=0,
                width=800,
                height=1000,
                source_kind="scan",
                blocks=[
                    OCRBlock(
                        block_type="text",
                        bbox=BoundingBox(0, 0, 800, 1000),
                        reading_order=0,
                        confidence=0.9,
                        engine_id="mock",
                        lines=[
                            OCRLine(
                                spans=[OCRSpan(text="أحمد\n123", confidence=0.9, engine_id="mock")],
                                confidence=0.9,
                                engine_id="mock",
                            )
                        ],
                    )
                ],
            )
        ],
    )


def test_document_to_dict_contains_pages():
    result = document_to_dict(_sample_document())
    assert result["pages"][0]["page"] == 0


def test_document_to_json_serializes():
    result = document_to_json(_sample_document())
    assert '"source_pdf": "sample.pdf"' in result


def test_document_roundtrip_from_dict_and_json():
    data = document_to_dict(_sample_document())
    from_dict_result = document_from_dict(data)
    from_json_result = document_from_json(document_to_json(_sample_document()))

    assert from_dict_result.pages[0].text == _sample_document().pages[0].text
    assert from_json_result.pages[0].page_number == 0


def test_render_text_uses_page_markers():
    result = render_text(_sample_document(), profile="llm")
    assert "=== Page 1 ===" in result
    assert "123" in result


def test_render_markdown_uses_headings():
    result = render_markdown(_sample_document(), profile="faithful")
    assert "## Page 1" in result


def test_legacy_page_results_shape():
    result = legacy_page_results(_sample_document(), profile="faithful")
    assert result[0]["page"] == 0
    assert "text" in result[0]
