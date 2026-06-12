"""Tests for glyph-aware native PDF extraction helpers."""

from unittest.mock import patch

from PIL import Image

from arabic_ocr.models import BoundingBox, OCRBlock, OCRLine, OCRPage, OCRSpan, PageAsset, PageImages, PipelineConfig
from arabic_ocr.native_pdf import choose_best_page_text, should_compare_native_ocr
from arabic_ocr.pipeline import _resolve_page_from_native_and_ocr


def test_choose_best_page_text_prefers_ocr_when_native_is_glyph_noise():
    native_text = "Æ‡ÆÆ‡ÆÆ‡ÆÆ‡Æ"
    ocr_text = "هذه مادة قانونية صحيحة"

    chosen_text, metadata = choose_best_page_text(native_text, ocr_text)

    assert chosen_text == ocr_text
    assert metadata["winner"] == "ocr"
    assert metadata["replaced"] is True


def test_should_compare_native_ocr_for_suspicious_arabic_native_page():
    metadata = {
        "page_maybe_arabic": True,
        "quality_status": "warn",
        "quality_metrics": {
            "weird_glyph_ratio": 0.02,
            "suspicious_arabic_char_ratio": 0.0,
            "repeated_arabic_run_ratio": 0.0,
        },
    }

    assert should_compare_native_ocr("نص عربي", metadata) is True


def test_resolve_page_from_native_and_ocr_prefers_ocr_when_native_bad():
    asset = PageAsset(
        page_number=0,
        width=100,
        height=100,
        embedded_text="Æ‡ÆÆ‡ÆÆ‡ÆÆ‡Æ",
        images=PageImages(raw=Image.new("RGB", (100, 100), "white")),
        metadata={
            "source": "glyph",
            "page_maybe_arabic": True,
            "quality_status": "fail",
            "quality_metrics": {
                "weird_glyph_ratio": 0.3,
                "suspicious_arabic_char_ratio": 0.2,
                "repeated_arabic_run_ratio": 0.0,
            },
        },
    )
    config = PipelineConfig(compare_native_ocr=True)
    ocr_page = OCRPage(
        page_number=0,
        width=100,
        height=100,
        source_kind="scan",
        blocks=[
            OCRBlock(
                block_type="text",
                bbox=BoundingBox(0, 0, 100, 100),
                reading_order=0,
                lines=[
                    OCRLine(
                        spans=[OCRSpan(text="هذه مادة قانونية صحيحة", engine_id="openai")],
                        engine_id="openai",
                    )
                ],
                engine_id="openai",
            )
        ],
    )

    with patch("arabic_ocr.pipeline._process_page", return_value=ocr_page):
        resolved = _resolve_page_from_native_and_ocr(
            asset=asset,
            config=config,
            primary_engine=None,
            secondary_engine=None,
            page_index=1,
            page_total=1,
        )

    assert resolved.source_kind == "scan"
    assert resolved.text == "هذه مادة قانونية صحيحة"
    assert resolved.metadata["native_ocr_comparison"]["winner"] == "ocr"
