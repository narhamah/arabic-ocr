"""Live API integration tests for Gemini and Claude.

These tests call real APIs and are SKIPPED unless the corresponding
API keys are set in the environment (via .env file or env vars).

Run with: pytest tests/test_live_api.py -v
"""

import os
import pytest
from pathlib import Path
from PIL import Image, ImageDraw

from arabic_ocr.utils import load_env

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Load .env at import time so keys are available for skipif checks
load_env()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
FIXTURES_DIR = Path(__file__).parent / "fixtures"

skip_no_gemini = pytest.mark.skipif(
    not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_key_here",
    reason="GEMINI_API_KEY not configured",
)
skip_no_anthropic = pytest.mark.skipif(
    not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_anthropic_key_here",
    reason="ANTHROPIC_API_KEY not configured",
)
skip_no_both = pytest.mark.skipif(
    not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_key_here"
    or not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_anthropic_key_here",
    reason="GEMINI_API_KEY and/or ANTHROPIC_API_KEY not configured",
)


@pytest.fixture
def arabic_text_image() -> Image.Image:
    """Create a simple image with clearly readable Arabic text using drawing."""
    img = Image.new("RGB", (400, 100), "white")
    draw = ImageDraw.Draw(img)
    # Draw Arabic-like text patterns (horizontal lines to simulate text)
    draw.text((10, 30), "123 ABC", fill="black")
    return img


# ---------------------------------------------------------------------------
# Gemini OCR Engine Tests (live)
# ---------------------------------------------------------------------------


@skip_no_gemini
class TestLiveGeminiPro:
    """Live tests for Gemini 2.5 Pro OCR."""

    def test_gemini_pro_returns_success(self, arabic_text_image):
        from arabic_ocr.ocr_engines import ocr_gemini_pro

        result = ocr_gemini_pro(arabic_text_image)
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["error"] is None
        assert isinstance(result["text"], str)

    def test_gemini_pro_extracts_text(self, arabic_text_image):
        from arabic_ocr.ocr_engines import ocr_gemini_pro

        result = ocr_gemini_pro(arabic_text_image)
        assert result["success"] is True
        assert len(result["text"]) > 0


@skip_no_gemini
class TestLiveGeminiFlash:
    """Live tests for Gemini 2.0 Flash OCR."""

    def test_gemini_flash_returns_success(self, arabic_text_image):
        from arabic_ocr.ocr_engines import ocr_gemini_flash

        result = ocr_gemini_flash(arabic_text_image)
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["error"] is None
        assert isinstance(result["text"], str)

    def test_gemini_flash_extracts_text(self, arabic_text_image):
        from arabic_ocr.ocr_engines import ocr_gemini_flash

        result = ocr_gemini_flash(arabic_text_image)
        assert result["success"] is True
        assert len(result["text"]) > 0


@skip_no_gemini
class TestLiveDualOCR:
    """Live tests for dual OCR engine."""

    def test_dual_ocr_returns_both(self, arabic_text_image):
        from arabic_ocr.ocr_engines import run_dual_ocr

        result = run_dual_ocr(arabic_text_image)
        assert "primary" in result
        assert "secondary" in result
        assert result["primary"]["success"] is True
        assert result["secondary"]["success"] is True


@skip_no_gemini
class TestLiveCallGeminiCoverage:
    """Tests that exercise the full _call_gemini code path."""

    def test_call_gemini_configures_and_calls(self, arabic_text_image):
        """Exercises lines 80-96 of ocr_engines.py."""
        from arabic_ocr.ocr_engines import _call_gemini

        result = _call_gemini(
            image=arabic_text_image,
            model="gemini-2.0-flash",
            prompt="What text do you see? Reply with just the text.",
        )
        assert result["success"] is True
        assert result["error"] is None
        assert isinstance(result["text"], str)

    def test_call_gemini_missing_key(self, arabic_text_image, monkeypatch):
        """_call_gemini returns error when key is missing."""
        from arabic_ocr.ocr_engines import _call_gemini

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = _call_gemini(
            image=arabic_text_image,
            model="gemini-2.0-flash",
            prompt="test",
        )
        assert result["success"] is False
        assert "GEMINI_API_KEY" in result["error"]

    def test_call_gemini_invalid_model(self, arabic_text_image):
        """_call_gemini handles exception from invalid model gracefully."""
        from arabic_ocr.ocr_engines import _call_gemini

        result = _call_gemini(
            image=arabic_text_image,
            model="nonexistent-model-xyz",
            prompt="test",
        )
        # Should not crash, should return error
        assert result["success"] is False
        assert result["error"] is not None


# ---------------------------------------------------------------------------
# Claude Verifier Tests (live)
# ---------------------------------------------------------------------------


@skip_no_anthropic
class TestLiveClaudeTiebreaker:
    """Live tests for Claude tiebreaker verification."""

    def test_claude_tiebreaker_resolves(self, arabic_text_image):
        """Exercises lines 123-175 of verifier.py."""
        from arabic_ocr.verifier import _call_claude_tiebreaker

        result = _call_claude_tiebreaker(
            primary="بسم الله الرحمن الرحيم",
            secondary="بسم الله الرحمان الرحيم",
            disagreements=[{
                "primary": "الرحمن",
                "secondary": "الرحمان",
                "position": 3,
            }],
            image=arabic_text_image,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_claude_tiebreaker_missing_key(self, arabic_text_image, monkeypatch):
        """Returns None when ANTHROPIC_API_KEY not set."""
        from arabic_ocr.verifier import _call_claude_tiebreaker

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = _call_claude_tiebreaker(
            primary="test a",
            secondary="test b",
            disagreements=[{"primary": "a", "secondary": "b", "position": 1}],
            image=arabic_text_image,
        )
        assert result is None


@skip_no_anthropic
class TestLiveVerifyAndMerge:
    """Live tests for full verify_and_merge with Claude."""

    def test_verify_merge_with_disagreement(self, arabic_text_image):
        from arabic_ocr.verifier import verify_and_merge

        result = verify_and_merge(
            "الوزير أكد أن العدد كبير",
            "الوزير اكد ان العدد كبير",
            arabic_text_image,
        )
        assert "text" in result
        assert "confidence" in result
        assert result["disagreements"] > 0
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0


# ---------------------------------------------------------------------------
# Full Pipeline Integration (live, needs both keys)
# ---------------------------------------------------------------------------


@skip_no_both
class TestLiveFullPipeline:
    """End-to-end live pipeline test."""

    def test_process_pdf_live(self):
        from arabic_ocr.pipeline import process_pdf

        results = process_pdf(
            str(FIXTURES_DIR / "single_page.pdf"),
            config={"dpi": 150},  # Low DPI to save API cost
        )
        assert isinstance(results, list)
        assert len(results) == 1
        page = results[0]
        assert page["page"] == 0
        assert isinstance(page["text"], str)
        assert isinstance(page["confidence"], float)

    def test_process_pdf_specific_pages(self):
        from arabic_ocr.pipeline import process_pdf

        results = process_pdf(
            str(FIXTURES_DIR / "multi_page.pdf"),
            config={"dpi": 150, "pages": [0]},
        )
        assert len(results) == 1
