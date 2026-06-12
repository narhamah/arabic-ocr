"""Tests targeting specific uncovered lines for 100% coverage.

These tests cover edge cases and error paths that the main test suites
didn't exercise, without needing live API keys.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image
from click.testing import CliRunner

from arabic_ocr.models import BoundingBox, OCRBlock, OCRDocument, OCRLine, OCRPage, OCRSpan

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _sample_document() -> OCRDocument:
    page = OCRPage(
        page_number=0,
        width=800,
        height=1000,
        source_kind="scan",
        blocks=[
            OCRBlock(
                block_type="text",
                bbox=BoundingBox(0, 0, 800, 1000),
                reading_order=0,
                confidence=1.0,
                engine_id="mock",
                lines=[
                    OCRLine(
                        spans=[OCRSpan(text="Output", confidence=1.0, engine_id="mock")],
                        confidence=1.0,
                        engine_id="mock",
                    )
                ],
            )
        ],
    )
    return OCRDocument(source_pdf=str(FIXTURES_DIR / "single_page.pdf"), dpi=400, pages=[page])


# ---------------------------------------------------------------------------
# cli.py — lines 40, 48-50, 60
# ---------------------------------------------------------------------------


class TestCLICoverageGaps:

    @patch("arabic_ocr.pipeline.process_pdf_document")
    def test_cli_with_pages_option(self, mock_process):
        """Cover line 40: config['pages'] = _parse_pages(pages)"""
        from arabic_ocr.cli import main

        mock_process.return_value = _sample_document()
        runner = CliRunner()
        result = runner.invoke(main, [
            str(FIXTURES_DIR / "multi_page.pdf"), "-p", "1-2"
        ])
        assert result.exit_code == 0
        call_config = mock_process.call_args[1].get("config") or mock_process.call_args[0][1]
        assert "pages" in call_config

    @patch("arabic_ocr.pipeline.process_pdf_document")
    def test_cli_process_raises_exception(self, mock_process):
        """Cover lines 48-50: except Exception -> sys.exit(1)"""
        from arabic_ocr.cli import main

        mock_process.side_effect = RuntimeError("PDF corrupt")
        runner = CliRunner()
        result = runner.invoke(main, [str(FIXTURES_DIR / "single_page.pdf")])
        assert result.exit_code != 0
        assert "PDF corrupt" in result.output or "PDF corrupt" in (result.stderr_bytes or b"").decode("utf-8", errors="replace")

    @patch("arabic_ocr.pipeline.process_pdf_document")
    def test_cli_verbose_with_output_file(self, mock_process, repo_tmp_path):
        """Cover line 60: verbose echo of output file path."""
        from arabic_ocr.cli import main

        mock_process.return_value = _sample_document()
        out_file = repo_tmp_path / "out.txt"
        runner = CliRunner()
        result = runner.invoke(main, [
            str(FIXTURES_DIR / "single_page.pdf"),
            "-o", str(out_file),
            "-v",
        ])
        assert result.exit_code == 0
        assert "Output" in out_file.read_text()


# ---------------------------------------------------------------------------
# layout.py — lines 43-69 (YOLO path), 93-129 (RTL multi-region sort)
# ---------------------------------------------------------------------------


class TestLayoutYOLOPath:

    def test_yolo_detection_with_mock_yolo(self):
        """Cover lines 43-69 by mocking doclayout_yolo and huggingface_hub."""
        from arabic_ocr.layout import _load_yolo_model, _try_yolo_detection

        _load_yolo_model.cache_clear()

        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [10, 20, 300, 400]
        mock_box.cls = [MagicMock()]
        mock_box.cls[0] = 0
        mock_box.conf = [MagicMock()]
        mock_box.conf[0] = 0.95

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "text"}

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        mock_yolo_module = MagicMock()
        mock_yolo_module.YOLOv10.return_value = mock_model

        mock_hf_module = MagicMock()
        mock_hf_module.hf_hub_download.return_value = "/fake/model.pt"

        import sys
        with patch.dict(sys.modules, {
            "doclayout_yolo": mock_yolo_module,
            "huggingface_hub": mock_hf_module,
        }):
            img = Image.new("RGB", (800, 600), "white")
            regions = _try_yolo_detection(img)

        assert len(regions) == 1
        assert regions[0]["label"] == "text"
        assert regions[0]["confidence"] == 0.95
        assert regions[0]["bbox"] == (10, 20, 300, 400)
        assert isinstance(regions[0]["crop"], Image.Image)

    def test_yolo_detection_handles_exception(self):
        """Cover line 68-69: YOLO exception returns empty list."""
        from arabic_ocr.layout import _load_yolo_model, _try_yolo_detection

        _load_yolo_model.cache_clear()

        mock_yolo_module = MagicMock()
        mock_hf_module = MagicMock()
        mock_hf_module.hf_hub_download.side_effect = Exception("Download failed")

        import sys
        with patch.dict(sys.modules, {
            "doclayout_yolo": mock_yolo_module,
            "huggingface_hub": mock_hf_module,
        }):
            img = Image.new("RGB", (800, 600), "white")
            regions = _try_yolo_detection(img)

        assert regions == []

    def test_rtl_sort_multiple_regions(self):
        """Cover lines 93-129: RTL sorting with multiple disjoint regions."""
        from arabic_ocr.layout import _sort_rtl_reading_order

        img = Image.new("RGB", (10, 10), "white")
        regions = [
            {"label": "a", "bbox": (10, 10, 100, 50), "crop": img, "confidence": 1.0},
            {"label": "b", "bbox": (500, 10, 600, 50), "crop": img, "confidence": 1.0},
            {"label": "c", "bbox": (500, 100, 600, 150), "crop": img, "confidence": 1.0},
            {"label": "d", "bbox": (10, 100, 100, 150), "crop": img, "confidence": 1.0},
        ]
        sorted_regions = _sort_rtl_reading_order(regions)
        labels = [r["label"] for r in sorted_regions]
        # Right column (b, c) first, then left column (a, d)
        # Within columns: top to bottom
        assert labels == ["b", "c", "a", "d"]

    def test_rtl_sort_single_region(self):
        """Single region returns unchanged."""
        from arabic_ocr.layout import _sort_rtl_reading_order

        img = Image.new("RGB", (10, 10), "white")
        regions = [{"label": "a", "bbox": (0, 0, 100, 100), "crop": img, "confidence": 1.0}]
        assert _sort_rtl_reading_order(regions) == regions


# ---------------------------------------------------------------------------
# pipeline.py — lines 100-102, 145-148, 152, 154
# ---------------------------------------------------------------------------


class TestPipelineCoverageGaps:

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_region_exception_logs_and_continues(self, mock_gemini):
        """Cover lines 100-102: exception in _process_region caught and skipped."""
        from arabic_ocr.pipeline import process_pdf

        # First call succeeds, second call will cause exception via side effect
        call_count = [0]
        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:  # First region: 2 calls (pro + flash)
                return {"text": "good text", "success": True, "error": None}
            raise RuntimeError("API crashed")

        mock_gemini.side_effect = side_effect

        # We need 2 regions to test this. Mock layout to return 2 regions.
        with patch("arabic_ocr.pipeline.detect_regions") as mock_layout:
            img = Image.new("RGB", (100, 100), "white")
            mock_layout.return_value = [
                {"label": "a", "bbox": (0, 0, 50, 100), "crop": img, "confidence": 1.0},
                {"label": "b", "bbox": (50, 0, 100, 100), "crop": img, "confidence": 1.0},
            ]
            result = process_pdf(str(FIXTURES_DIR / "single_page.pdf"))

        assert len(result) == 1
        # Should have processed at least 1 region successfully

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_hallucination_guard_primary_shorter(self, mock_gemini):
        """Cover lines 145-147: primary is shorter, use primary."""
        from arabic_ocr.pipeline import _process_region

        # Primary = short, secondary = much longer (>30% ratio)
        calls = [0]
        def gemini_side_effect(**kwargs):
            calls[0] += 1
            if calls[0] % 2 == 1:  # Pro (odd calls)
                return {"text": "short text", "success": True, "error": None}
            else:  # Flash (even calls)
                return {"text": "this is a much much much much much longer text output", "success": True, "error": None}
        mock_gemini.side_effect = gemini_side_effect

        img = Image.new("RGB", (100, 100), "white")
        text, confidence = _process_region(original_crop=img, strip_diacritics=True)
        assert text == "short text"
        assert confidence == 0.5

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_hallucination_guard_secondary_shorter(self, mock_gemini):
        """Cover line 148: secondary is shorter, use secondary."""
        from arabic_ocr.pipeline import _process_region

        calls = [0]
        def gemini_side_effect(**kwargs):
            calls[0] += 1
            if calls[0] % 2 == 1:  # Pro
                return {"text": "this is a much much much much much longer text output here", "success": True, "error": None}
            else:  # Flash
                return {"text": "short text", "success": True, "error": None}
        mock_gemini.side_effect = gemini_side_effect

        img = Image.new("RGB", (100, 100), "white")
        text, confidence = _process_region(original_crop=img, strip_diacritics=True)
        assert text == "short text"
        assert confidence == 0.5

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_only_secondary_has_text(self, mock_gemini):
        """Cover line 152: primary empty, secondary has text."""
        from arabic_ocr.pipeline import _process_region

        calls = [0]
        def gemini_side_effect(**kwargs):
            calls[0] += 1
            if calls[0] % 2 == 1:
                return {"text": "", "success": True, "error": None}
            else:
                return {"text": "secondary only", "success": True, "error": None}
        mock_gemini.side_effect = gemini_side_effect

        img = Image.new("RGB", (100, 100), "white")
        text, confidence = _process_region(original_crop=img, strip_diacritics=True)
        assert text == "secondary only"
        assert confidence == 0.5

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_only_primary_has_text(self, mock_gemini):
        """Cover line 154: primary has text, secondary empty."""
        from arabic_ocr.pipeline import _process_region

        calls = [0]
        def gemini_side_effect(**kwargs):
            calls[0] += 1
            if calls[0] % 2 == 1:
                return {"text": "primary only", "success": True, "error": None}
            else:
                return {"text": "", "success": True, "error": None}
        mock_gemini.side_effect = gemini_side_effect

        img = Image.new("RGB", (100, 100), "white")
        text, confidence = _process_region(original_crop=img, strip_diacritics=True)
        assert text == "primary only"
        assert confidence == 0.5

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_both_empty(self, mock_gemini):
        """Cover line 155-156: both engines return empty text."""
        from arabic_ocr.pipeline import _process_region

        mock_gemini.return_value = {"text": "", "success": True, "error": None}

        img = Image.new("RGB", (100, 100), "white")
        text, confidence = _process_region(original_crop=img, strip_diacritics=True)
        assert text == ""
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# preprocessor.py — lines 74, 81 (deskew edge cases)
# ---------------------------------------------------------------------------


class TestPreprocessorCoverageGaps:

    def test_deskew_no_lines_detected(self):
        """Cover line 74: no lines found, return unchanged."""
        from arabic_ocr.preprocessor import _deskew
        import numpy as np

        # Completely blank image — no lines to detect
        gray = np.full((200, 200), 200, dtype=np.uint8)
        result = _deskew(gray)
        assert result.shape == gray.shape

    def test_deskew_only_vertical_lines(self):
        """Cover line 81: lines found but all vertical (no near-horizontal)."""
        from arabic_ocr.preprocessor import _deskew
        import numpy as np

        # Image with only vertical lines
        gray = np.full((200, 200), 255, dtype=np.uint8)
        gray[:, 50] = 0
        gray[:, 100] = 0
        gray[:, 150] = 0
        result = _deskew(gray)
        assert result.shape == gray.shape


# ---------------------------------------------------------------------------
# utils.py — line 11 (load_dotenv when .env exists)
# ---------------------------------------------------------------------------


class TestUtilsCoverageGaps:

    def test_load_env_when_file_exists(self, repo_tmp_path, monkeypatch):
        """Cover line 11: .env file exists and gets loaded."""
        env_file = repo_tmp_path / ".env"
        env_file.write_text("TEST_ARABIC_OCR_VAR=hello_world\n")
        monkeypatch.chdir(repo_tmp_path)
        from arabic_ocr.utils import load_env
        load_env()
        assert os.getenv("TEST_ARABIC_OCR_VAR") == "hello_world"

    def test_load_env_when_file_missing(self, repo_tmp_path, monkeypatch):
        """Ensure load_env doesn't crash when .env is absent."""
        monkeypatch.chdir(repo_tmp_path)
        from arabic_ocr.utils import load_env
        load_env()  # Should not raise


# ---------------------------------------------------------------------------
# verifier.py — line 97 (unreachable tail return for non-disagreement path)
# ---------------------------------------------------------------------------


class TestVerifierCoverageGaps:

    def test_verify_words_differ_only_whitespace(self):
        """Cover line 97: words lists differ but SequenceMatcher finds no disagreements."""
        from arabic_ocr.verifier import verify_and_merge

        img = Image.new("RGB", (100, 100), "white")
        # These are word-equal after split, triggering the early return at line 42
        result = verify_and_merge("a  b  c", "a b c", img)
        assert result["text"] == "a  b  c"
        assert result["confidence"] == 1.0

    def test_call_claude_missing_key_returns_none(self):
        """Cover lines 125-127 of verifier.py."""
        from arabic_ocr.verifier import _call_claude_tiebreaker

        img = Image.new("RGB", (100, 100), "white")
        with patch.dict(os.environ, {}, clear=True):
            result = _call_claude_tiebreaker(
                primary="a",
                secondary="b",
                disagreements=[{"primary": "a", "secondary": "b", "position": 0}],
                image=img,
            )
        assert result is None

    def test_verify_and_merge_defensive_tail_return(self):
        """Cover line 97: defensive return when SequenceMatcher finds no disagreements
        despite word lists differing (edge case via mocked matcher)."""
        from arabic_ocr.verifier import verify_and_merge
        import difflib

        img = Image.new("RGB", (100, 100), "white")
        # Mock SequenceMatcher to return only "equal" opcodes even though words differ
        mock_matcher = MagicMock()
        mock_matcher.get_opcodes.return_value = [("equal", 0, 2, 0, 2)]
        with patch.object(difflib, "SequenceMatcher", return_value=mock_matcher):
            result = verify_and_merge("word1 word2", "word1 word3", img)
        assert result["confidence"] == 1.0
        assert result["disagreements"] == 0

    def test_call_claude_tiebreaker_full_body(self, monkeypatch):
        """Cover lines 130-175: full _call_claude_tiebreaker body with mocked anthropic."""
        from arabic_ocr.verifier import _call_claude_tiebreaker

        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake_key")

        mock_content_block = MagicMock()
        mock_content_block.text = "resolved text output"

        mock_message = MagicMock()
        mock_message.content = [mock_content_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        import sys
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            img = Image.new("RGB", (100, 100), "white")
            result = _call_claude_tiebreaker(
                primary="الوزير أكد أن العدد كبير",
                secondary="الوزير اكد ان العدد كبير",
                disagreements=[
                    {"primary": "أكد", "secondary": "اكد", "position": 1},
                    {"primary": "أن", "secondary": "ان", "position": 2},
                ],
                image=img,
            )

        assert result == "resolved text output"
        # Verify the client was constructed with the fake key
        mock_anthropic.Anthropic.assert_called_once_with(api_key="fake_key")
        # Verify messages.create was called with correct model
        create_call = mock_client.messages.create.call_args
        assert create_call[1]["model"] == "claude-sonnet-4-20250514"
        assert create_call[1]["max_tokens"] == 4096

    def test_call_claude_tiebreaker_empty_content(self, monkeypatch):
        """Cover line 175: response.content is empty list -> returns None."""
        from arabic_ocr.verifier import _call_claude_tiebreaker

        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake_key")

        mock_message = MagicMock()
        mock_message.content = []

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        import sys
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            img = Image.new("RGB", (100, 100), "white")
            result = _call_claude_tiebreaker(
                primary="a",
                secondary="b",
                disagreements=[{"primary": "a", "secondary": "b", "position": 0}],
                image=img,
            )

        assert result is None


# ---------------------------------------------------------------------------
# ocr_engines.py — line 84-85 (missing key path)
# ---------------------------------------------------------------------------


class TestOCREnginesCoverageGaps:

    def test_call_gemini_no_key_returns_error(self, monkeypatch):
        """GEMINI_API_KEY missing returns an error payload."""
        from arabic_ocr.ocr_engines import _call_gemini

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        img = Image.new("RGB", (100, 100), "white")
        result = _call_gemini(image=img, model="test", prompt="test")
        assert result["success"] is False
        assert "GEMINI_API_KEY" in result["error"]

    def test_call_gemini_exception_returns_error(self, monkeypatch):
        """Exceptions from google.genai are surfaced as error payloads."""
        from arabic_ocr.ocr_engines import _call_gemini

        monkeypatch.setenv("GEMINI_API_KEY", "fake_key")

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("Network error")
        mock_genai_module = MagicMock()
        mock_genai_module.Client.return_value = mock_client
        mock_types_module = MagicMock()
        mock_types_module.Part.from_bytes.return_value = "image-part"
        mock_genai_module.types = mock_types_module
        mock_google_package = MagicMock()
        mock_google_package.genai = mock_genai_module

        import sys
        with patch.dict(sys.modules, {
            "google": mock_google_package,
            "google.genai": mock_genai_module,
            "google.genai.types": mock_types_module,
        }):
            img = Image.new("RGB", (100, 100), "white")
            result = _call_gemini(image=img, model="test", prompt="test")
        assert result["success"] is False
        assert "Network error" in result["error"]

    def test_call_gemini_response_no_text(self, monkeypatch):
        """Empty response text still returns a success payload."""
        from arabic_ocr.ocr_engines import _call_gemini

        monkeypatch.setenv("GEMINI_API_KEY", "fake_key")

        mock_response = MagicMock()
        mock_response.text = None
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_module = MagicMock()
        mock_genai_module.Client.return_value = mock_client
        mock_types_module = MagicMock()
        mock_types_module.Part.from_bytes.return_value = "image-part"
        mock_genai_module.types = mock_types_module
        mock_google_package = MagicMock()
        mock_google_package.genai = mock_genai_module

        import sys
        with patch.dict(sys.modules, {
            "google": mock_google_package,
            "google.genai": mock_genai_module,
            "google.genai.types": mock_types_module,
        }):
            img = Image.new("RGB", (100, 100), "white")
            result = _call_gemini(image=img, model="test", prompt="test")
        assert result["success"] is True
        assert result["text"] == ""

    def test_call_openai_missing_key_returns_error(self, monkeypatch):
        from arabic_ocr.ocr_engines import _call_openai

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        img = Image.new("RGB", (100, 100), "white")
        result = _call_openai(image=img, model="gpt-4.1-mini", prompt="test")
        assert result["success"] is False
        assert "OPENAI_API_KEY" in result["error"]
