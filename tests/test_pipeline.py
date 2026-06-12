"""Tests for pipeline orchestrator with mocked external APIs."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image
from arabic_ocr.models import BoundingBox, OCRDocument
from arabic_ocr.pipeline import (
    _process_page,
    _filter_regions,
    _merge_outputs,
    _sanitize_ocr_text,
    _stitch_chunked_text,
    _text_to_lines,
    process_pdf,
    process_pdf_document,
)
from arabic_ocr.engines.base import EngineOutput

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_ocr():
    with patch("arabic_ocr.ocr_engines._call_gemini") as mock_gemini, \
         patch("arabic_ocr.verifier._call_claude_tiebreaker") as mock_claude, \
         patch("arabic_ocr.verifier._call_claude_page_resolver") as mock_claude_page, \
         patch("arabic_ocr.verifier._call_openai_page_resolver") as mock_openai_page:
        mock_gemini.return_value = {
            "text": "بسم الله الرحمن الرحيم",
            "success": True,
            "error": None,
        }
        mock_claude.return_value = None
        mock_claude_page.return_value = None
        mock_openai_page.return_value = None
        yield {"gemini": mock_gemini, "claude": mock_claude}


class TestFullPipeline:

    def test_pipeline_returns_list(self, mock_ocr):
        result = process_pdf(str(FIXTURES_DIR / "single_page.pdf"))
        assert isinstance(result, list)

    def test_single_page_returns_one_result(self, mock_ocr):
        result = process_pdf(str(FIXTURES_DIR / "single_page.pdf"))
        assert len(result) == 1

    def test_multi_page_returns_multiple_results(self, mock_ocr):
        result = process_pdf(str(FIXTURES_DIR / "multi_page.pdf"))
        assert len(result) == 3

    def test_result_has_required_keys(self, mock_ocr):
        result = process_pdf(str(FIXTURES_DIR / "single_page.pdf"))
        page_result = result[0]
        assert "page" in page_result
        assert "text" in page_result
        assert "confidence" in page_result
        assert "regions" in page_result
        assert "method" in page_result

    def test_result_has_page_number(self, mock_ocr):
        result = process_pdf(str(FIXTURES_DIR / "multi_page.pdf"))
        assert result[0]["page"] == 0
        assert result[1]["page"] == 1
        assert result[2]["page"] == 2

    def test_result_has_text(self, mock_ocr):
        result = process_pdf(str(FIXTURES_DIR / "single_page.pdf"))
        assert isinstance(result[0]["text"], str)
        assert len(result[0]["text"]) > 0

    def test_document_pipeline_returns_document(self, mock_ocr):
        result = process_pdf_document(str(FIXTURES_DIR / "single_page.pdf"))
        assert isinstance(result, OCRDocument)


class TestPipelineConfig:

    def test_custom_dpi(self, mock_ocr):
        result = process_pdf(
            str(FIXTURES_DIR / "single_page.pdf"),
            config={"dpi": 200},
        )
        assert len(result) == 1

    def test_specific_pages(self, mock_ocr):
        result = process_pdf(
            str(FIXTURES_DIR / "multi_page.pdf"),
            config={"pages": [0, 2]},
        )
        assert len(result) == 2

    def test_keep_diacritics(self, mock_ocr):
        result = process_pdf(
            str(FIXTURES_DIR / "single_page.pdf"),
            config={"strip_diacritics": False},
        )
        assert len(result) == 1

    def test_progress_callback_receives_events(self, mock_ocr):
        events = []

        process_pdf(
            str(FIXTURES_DIR / "single_page.pdf"),
            config={"progress_callback": lambda event, payload: events.append(event)},
        )

        assert "document_start" in events
        assert "page_complete" in events
        assert "document_complete" in events


class TestErrorRecovery:

    def test_invalid_pdf_raises(self):
        with pytest.raises(Exception):
            process_pdf(str(FIXTURES_DIR / "not_a_pdf.txt"))

    def test_nonexistent_file_raises(self):
        with pytest.raises(Exception):
            process_pdf("/nonexistent/path.pdf")

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_ocr_failure_doesnt_crash(self, mock_gemini):
        mock_gemini.return_value = {"text": "", "success": False, "error": "API error"}
        result = process_pdf(str(FIXTURES_DIR / "single_page.pdf"))
        assert isinstance(result, list)
        assert len(result) == 1


def test_filter_regions_drops_tiny_heuristic_fragments():
    img = Image.new("RGB", (10, 10), "white")
    regions = [
        {"label": "heuristic_block", "bbox": (2400, 0, 2814, 102), "crop": img, "confidence": 0.6},
        {"label": "heuristic_block", "bbox": (0, 97, 3307, 4600), "crop": img, "confidence": 0.6},
    ]

    filtered = _filter_regions(regions, page_size=(3307, 4678))

    assert len(filtered) == 1
    assert filtered[0]["bbox"] == (0, 97, 3307, 4600)


@patch("arabic_ocr.pipeline.reflow_block_text", return_value="line one\nline two")
def test_merge_outputs_reflows_long_openai_blocks(mock_reflow):
    primary = EngineOutput(
        text="line one line two " * 50,
        success=True,
        confidence=0.8,
        metadata={"engine_id": "openai"},
    )

    merged_text, confidence, metadata = _merge_outputs(
        primary_output=primary,
        secondary_output=None,
        image=Image.new("RGB", (100, 100), "white"),
        strip_diacritics=True,
        verify_disagreements=False,
    )

    assert merged_text == "line one\nline two"
    assert metadata["final_engine_id"] == "openai"
    mock_reflow.assert_called_once()


def test_merge_outputs_prefers_higher_quality_text_over_shorter_length():
    primary = EngineOutput(
        text="القضية رقم 2234 لسنة 2025 حصر",
        success=True,
        confidence=0.8,
        metadata={"engine_id": "openai"},
    )
    secondary = EngineOutput(
        text="2234 2025",
        success=True,
        confidence=0.8,
        metadata={"engine_id": "gemini_pro"},
    )

    merged_text, confidence, metadata = _merge_outputs(
        primary_output=primary,
        secondary_output=secondary,
        image=Image.new("RGB", (100, 100), "white"),
        strip_diacritics=True,
        verify_disagreements=True,
    )

    assert merged_text == "القضية رقم 2234 لسنة 2025 حصر"
    assert metadata["final_engine_id"] == "openai"


def test_merge_outputs_strips_hallucinated_no_text_response():
    primary = EngineOutput(
        text="There is no visible text in the provided image.",
        success=True,
        confidence=0.8,
        metadata={"engine_id": "openai"},
    )

    merged_text, confidence, metadata = _merge_outputs(
        primary_output=primary,
        secondary_output=None,
        image=Image.new("RGB", (100, 100), "white"),
        strip_diacritics=True,
        verify_disagreements=False,
    )

    assert merged_text == ""
    assert confidence == 0.0


def test_merge_outputs_strips_image_provided_variant():
    primary = EngineOutput(
        text="There is no visible text in the image provided.",
        success=True,
        confidence=0.8,
        metadata={"engine_id": "openai"},
    )

    merged_text, confidence, metadata = _merge_outputs(
        primary_output=primary,
        secondary_output=None,
        image=Image.new("RGB", (100, 100), "white"),
        strip_diacritics=True,
        verify_disagreements=False,
    )

    assert merged_text == ""
    assert confidence == 0.0


def test_sanitize_ocr_text_removes_inline_boilerplate_lines():
    cleaned = _sanitize_ocr_text(
        "\n".join([
            "Sorry, I can't extract any visible text from this image. The image appears to be blank or too faint to read.",
            "لا يوجد نص مرئي في هذه الصورة.",
            "في : 2023/9/24",
            "There is no visible text in the image provided.",
            "(No text is visible in the provided image.)",
            "[No text visible]",
            "Scanned with",
            "Scanned with CS CamScannerTM",
        ])
    )

    assert cleaned == "في : 2023/9/24"


def test_merge_outputs_strips_parenthesized_hallucinated_no_text_response():
    primary = EngineOutput(
        text="(No visible text detected in the provided image.)",
        success=True,
        confidence=0.8,
        metadata={"engine_id": "openai"},
    )

    merged_text, confidence, metadata = _merge_outputs(
        primary_output=primary,
        secondary_output=None,
        image=Image.new("RGB", (100, 100), "white"),
        strip_diacritics=True,
        verify_disagreements=False,
    )

    assert merged_text == ""
    assert confidence == 0.0


def test_merge_outputs_strips_bracketed_short_hallucinated_no_text_response():
    primary = EngineOutput(
        text="[No visible text detected.]",
        success=True,
        confidence=0.8,
        metadata={"engine_id": "openai"},
    )

    merged_text, confidence, metadata = _merge_outputs(
        primary_output=primary,
        secondary_output=None,
        image=Image.new("RGB", (100, 100), "white"),
        strip_diacritics=True,
        verify_disagreements=False,
    )

    assert merged_text == ""
    assert confidence == 0.0


def test_merge_outputs_strips_no_text_to_extract_response():
    primary = EngineOutput(
        text="(No visible text to extract)",
        success=True,
        confidence=0.8,
        metadata={"engine_id": "openai"},
    )

    merged_text, confidence, metadata = _merge_outputs(
        primary_output=primary,
        secondary_output=None,
        image=Image.new("RGB", (100, 100), "white"),
        strip_diacritics=True,
        verify_disagreements=False,
    )

    assert merged_text == ""
    assert confidence == 0.0


def test_merge_outputs_strips_printed_or_handwritten_no_text_response():
    primary = EngineOutput(
        text="(No visible printed or handwritten text to extract from the provided image.)",
        success=True,
        confidence=0.8,
        metadata={"engine_id": "openai"},
    )

    merged_text, confidence, metadata = _merge_outputs(
        primary_output=primary,
        secondary_output=None,
        image=Image.new("RGB", (100, 100), "white"),
        strip_diacritics=True,
        verify_disagreements=False,
    )

    assert merged_text == ""
    assert confidence == 0.0


def test_text_to_lines_uses_image_based_line_boxes(sample_arabic_image):
    lines = _text_to_lines(
        "line one\nline two\nline three",
        0.5,
        "openai",
        block_bbox=BoundingBox(0, 0, 800, 600),
        region_image=sample_arabic_image,
    )

    assert len(lines) == 3
    assert lines[0].bbox is not None
    assert lines[0].bbox.y1 < lines[1].bbox.y1 < lines[2].bbox.y1


def test_process_region_details_rescues_low_quality_text_with_better_variant():
    from arabic_ocr.pipeline import _process_region_details

    base_image = Image.new("RGB", (100, 100), "white")
    rescue_image = Image.new("RGB", (200, 200), "white")

    class DummyEngine:
        id = "openai"

        def recognize(self, image):
            if image is base_image:
                return EngineOutput(
                    text="1 2 3",
                    success=True,
                    confidence=0.2,
                    metadata={"engine_id": self.id},
                )
            if image is rescue_image:
                return EngineOutput(
                    text="القضية رقم 2234 لسنة 2025",
                    success=True,
                    confidence=0.95,
                    metadata={"engine_id": self.id},
                )
            return EngineOutput(
                text="",
                success=False,
                confidence=0.0,
                metadata={"engine_id": self.id},
            )

    with patch("arabic_ocr.pipeline._build_rescue_variants", return_value=[("enhanced_upscaled", rescue_image)]):
        text, confidence, metadata = _process_region_details(
            original_crop=base_image,
            strip_diacritics=True,
            verify_disagreements=False,
            primary_engine=DummyEngine(),
            secondary_engine=None,
        )

    assert text == "القضية رقم 2234 لسنة 2025"
    assert confidence == 0.95
    assert metadata["final_engine_id"] == "openai"
    assert metadata["ocr_rescue"]["variant"] == "enhanced_upscaled"


def test_stitch_chunked_text_deduplicates_overlap_lines():
    stitched = _stitch_chunked_text([
        "السطر الاول\nالسطر الثاني\nالسطر الثالث",
        "السطر الثالث\nالسطر الرابع",
    ])

    assert stitched == "السطر الاول\nالسطر الثاني\nالسطر الثالث\nالسطر الرابع"


def test_process_region_details_rescues_sparse_large_crop_with_chunked_ocr():
    from arabic_ocr.pipeline import _process_region_details

    base_image = Image.new("RGB", (3300, 2600), "white")
    chunk_a = Image.new("RGB", (3300, 1600), "white")
    chunk_b = Image.new("RGB", (3300, 1200), "white")

    class DummyEngine:
        id = "openai"

        def recognize(self, image):
            if image is base_image:
                return EngineOutput(
                    text="2234 2025",
                    success=True,
                    confidence=0.2,
                    metadata={"engine_id": self.id},
                )
            if image is chunk_a:
                return EngineOutput(
                    text="القضية رقم 2234 لسنة 2025\nالسطر الاول",
                    success=True,
                    confidence=0.9,
                    metadata={"engine_id": self.id},
                )
            if image is chunk_b:
                return EngineOutput(
                    text="السطر الاول\nالسطر الثاني",
                    success=True,
                    confidence=0.85,
                    metadata={"engine_id": self.id},
                )
            return EngineOutput(
                text="",
                success=False,
                confidence=0.0,
                metadata={"engine_id": self.id},
            )

    with patch("arabic_ocr.pipeline._build_chunked_ocr_crops", return_value=[("chunk_1", chunk_a), ("chunk_2", chunk_b)]), \
         patch("arabic_ocr.pipeline._build_rescue_variants", return_value=[]):
        text, confidence, metadata = _process_region_details(
            original_crop=base_image,
            strip_diacritics=True,
            verify_disagreements=False,
            primary_engine=DummyEngine(),
            secondary_engine=None,
        )

    assert text == "القضية رقم 2234 لسنة 2025\nالسطر الاول\nالسطر الثاني"
    assert confidence == pytest.approx(0.875)
    assert metadata["final_engine_id"] == "openai"
    assert metadata["ocr_rescue"]["variant"] == "chunked_horizontal"
    assert metadata["ocr_rescue"]["chunk_count"] == 2
    assert metadata["ocr_rescue"]["successful_chunk_count"] == 2


@patch("arabic_ocr.pipeline._region_to_block")
@patch("arabic_ocr.pipeline.detect_regions")
@patch("arabic_ocr.pipeline.preprocess")
def test_process_page_drops_empty_blocks(mock_preprocess, mock_detect_regions, mock_region_to_block):
    from arabic_ocr.models import OCRBlock, PageAsset, PageImages, PipelineConfig

    image = Image.new("RGB", (100, 100), "white")
    mock_preprocess.return_value = image
    mock_detect_regions.return_value = [
        {"label": "heuristic_block", "bbox": (0, 0, 100, 100), "crop": image, "confidence": 0.5}
    ]
    mock_region_to_block.return_value = OCRBlock(
        block_type="heuristic_block",
        bbox=BoundingBox(0, 0, 100, 100),
        reading_order=0,
        lines=[],
        confidence=0.0,
        engine_id="openai",
    )

    page = _process_page(
        asset=PageAsset(page_number=0, width=100, height=100, images=PageImages(raw=image)),
        config=PipelineConfig(),
        primary_engine=MagicMock(),
        secondary_engine=None,
        page_index=1,
        page_total=1,
    )

    assert page.blocks == []
