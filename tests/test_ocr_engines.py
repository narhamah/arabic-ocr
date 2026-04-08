"""Tests for OCR engines. All API calls must be mocked."""

from unittest.mock import patch

import pytest
from PIL import Image

from arabic_ocr.ocr_engines import ocr_gemini_flash, ocr_gemini_pro, run_dual_ocr


@pytest.fixture
def small_image():
    return Image.new("RGB", (100, 100), "white")


class TestGeminiPro:

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_returns_dict_with_text(self, mock_call, small_image):
        mock_call.return_value = {"text": "بسم الله", "success": True, "error": None}
        result = ocr_gemini_pro(small_image)
        assert isinstance(result, dict)
        assert result["text"] == "بسم الله"
        assert result["success"] is True

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_sends_correct_model(self, mock_call, small_image):
        mock_call.return_value = {"text": "test", "success": True, "error": None}
        ocr_gemini_pro(small_image)
        call_args = mock_call.call_args
        assert "gemini-2.5-pro" in call_args[1]["model"]

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_handles_api_error(self, mock_call, small_image):
        mock_call.return_value = {"text": "", "success": False, "error": "API error"}
        result = ocr_gemini_pro(small_image)
        assert result["success"] is False
        assert result["error"] is not None


class TestGeminiFlash:

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_returns_dict_with_text(self, mock_call, small_image):
        mock_call.return_value = {"text": "نص عربي", "success": True, "error": None}
        result = ocr_gemini_flash(small_image)
        assert isinstance(result, dict)
        assert result["text"] == "نص عربي"

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_sends_correct_model(self, mock_call, small_image):
        mock_call.return_value = {"text": "test", "success": True, "error": None}
        ocr_gemini_flash(small_image)
        call_args = mock_call.call_args
        assert "gemini-2.0-flash" in call_args[1]["model"]


class TestDualOCR:

    @patch("arabic_ocr.ocr_engines.ocr_openai")
    @patch("arabic_ocr.ocr_engines.ocr_gemini_flash")
    @patch("arabic_ocr.ocr_engines.ocr_gemini_pro")
    def test_returns_both_results(self, mock_pro, mock_flash, mock_openai, small_image):
        mock_pro.return_value = {"text": "primary text", "success": True, "error": None}
        mock_flash.return_value = {"text": "secondary text", "success": True, "error": None}
        mock_openai.return_value = {"text": "fallback", "success": True, "error": None}
        result = run_dual_ocr(small_image)
        assert result["primary"]["text"] == "primary text"
        assert result["secondary"]["text"] == "secondary text"

    @patch("arabic_ocr.ocr_engines.ocr_openai")
    @patch("arabic_ocr.ocr_engines.ocr_gemini_flash")
    @patch("arabic_ocr.ocr_engines.ocr_gemini_pro")
    def test_both_engines_called(self, mock_pro, mock_flash, mock_openai, small_image):
        mock_pro.return_value = {"text": "", "success": True, "error": None}
        mock_flash.return_value = {"text": "", "success": True, "error": None}
        mock_openai.return_value = {"text": "unused", "success": True, "error": None}
        run_dual_ocr(small_image)
        mock_pro.assert_called_once_with(small_image)
        mock_flash.assert_called_once_with(small_image)

    @patch("arabic_ocr.ocr_engines.ocr_openai")
    @patch("arabic_ocr.ocr_engines.ocr_gemini_flash")
    @patch("arabic_ocr.ocr_engines.ocr_gemini_pro")
    def test_uses_openai_fallback_when_one_engine_fails(self, mock_pro, mock_flash, mock_openai, small_image):
        mock_pro.return_value = {"text": "good text", "success": True, "error": None}
        mock_flash.return_value = {"text": "", "success": False, "error": "failed"}
        mock_openai.return_value = {"text": "fallback text", "success": True, "error": None}
        result = run_dual_ocr(small_image)
        assert result["primary"]["success"] is True
        assert result["secondary"]["text"] == "fallback text"


class TestPromptContent:

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_prompt_contains_arabic_instructions(self, mock_call, small_image):
        mock_call.return_value = {"text": "", "success": True, "error": None}
        ocr_gemini_pro(small_image)
        prompt_arg = mock_call.call_args[1]["prompt"]
        assert "Arabic" in prompt_arg or "arabic" in prompt_arg

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_prompt_instructs_exact_extraction(self, mock_call, small_image):
        mock_call.return_value = {"text": "", "success": True, "error": None}
        ocr_gemini_pro(small_image)
        prompt_arg = mock_call.call_args[1]["prompt"]
        assert "fidelity" in prompt_arg.lower() or "exact" in prompt_arg.lower()
