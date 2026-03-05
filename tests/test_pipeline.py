"""Tests for pipeline orchestrator — integration tests with mocked APIs."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image
from arabic_ocr.pipeline import process_pdf

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_ocr():
    """Mock all external API calls."""
    with patch("arabic_ocr.ocr_engines._call_gemini") as mock_gemini, \
         patch("arabic_ocr.verifier._call_claude_tiebreaker") as mock_claude:
        mock_gemini.return_value = {
            "text": "بسم الله الرحمن الرحيم",
            "success": True,
            "error": None,
        }
        mock_claude.return_value = None  # No tiebreaker needed for identical outputs
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

    def test_result_has_page_number(self, mock_ocr):
        result = process_pdf(str(FIXTURES_DIR / "multi_page.pdf"))
        assert result[0]["page"] == 0
        assert result[1]["page"] == 1
        assert result[2]["page"] == 2

    def test_result_has_text(self, mock_ocr):
        result = process_pdf(str(FIXTURES_DIR / "single_page.pdf"))
        assert isinstance(result[0]["text"], str)
        assert len(result[0]["text"]) > 0


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


class TestErrorRecovery:

    def test_invalid_pdf_raises(self):
        with pytest.raises(Exception):
            process_pdf(str(FIXTURES_DIR / "not_a_pdf.txt"))

    def test_nonexistent_file_raises(self):
        with pytest.raises(Exception):
            process_pdf("/nonexistent/path.pdf")

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_ocr_failure_doesnt_crash(self, mock_gemini):
        """If OCR fails on a region, pipeline should continue."""
        mock_gemini.return_value = {"text": "", "success": False, "error": "API error"}
        result = process_pdf(str(FIXTURES_DIR / "single_page.pdf"))
        assert isinstance(result, list)
        assert len(result) == 1
