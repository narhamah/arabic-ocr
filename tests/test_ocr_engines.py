"""Tests for OCR engine helpers with all API calls mocked."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from PIL import Image

from arabic_ocr.ocr_engines import ocr_gemini_flash, ocr_gemini_pro, ocr_openai, run_dual_ocr


@pytest.fixture
def small_image():
    return Image.new("RGB", (100, 100), "white")


class TestGeminiPro:

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_returns_dict_with_text(self, mock_call, small_image):
        mock_call.return_value = {"text": "بسم الله", "success": True, "error": None}
        result = ocr_gemini_pro(small_image)
        assert result["text"] == "بسم الله"
        assert result["success"] is True

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_sends_correct_model(self, mock_call, small_image):
        mock_call.return_value = {"text": "test", "success": True, "error": None}
        ocr_gemini_pro(small_image)
        assert mock_call.call_args.kwargs["model"] == "gemini-2.5-pro"

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_handles_api_error(self, mock_call, small_image):
        mock_call.return_value = {"text": "", "success": False, "error": "API error"}
        result = ocr_gemini_pro(small_image)
        assert result["success"] is False
        assert result["error"] == "API error"


class TestGeminiFlash:

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_returns_dict_with_text(self, mock_call, small_image):
        mock_call.return_value = {"text": "نص عربي", "success": True, "error": None}
        result = ocr_gemini_flash(small_image)
        assert result["text"] == "نص عربي"

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_sends_correct_model(self, mock_call, small_image):
        mock_call.return_value = {"text": "test", "success": True, "error": None}
        ocr_gemini_flash(small_image)
        assert mock_call.call_args.kwargs["model"] == "gemini-2.5-flash"

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_handles_api_error(self, mock_call, small_image):
        mock_call.return_value = {"text": "", "success": False, "error": "timeout"}
        result = ocr_gemini_flash(small_image)
        assert result["success"] is False


class TestOpenAI:

    @patch("arabic_ocr.ocr_engines._call_openai")
    def test_returns_dict_with_text(self, mock_call, small_image):
        mock_call.return_value = {"text": "نص عربي", "success": True, "error": None}
        result = ocr_openai(small_image)
        assert result["success"] is True
        assert result["text"] == "نص عربي"

    @patch("arabic_ocr.ocr_engines._call_openai")
    def test_uses_configured_default_model(self, mock_call, small_image, monkeypatch):
        monkeypatch.setenv("OPENAI_OCR_MODEL", "gpt-4.1-mini")
        mock_call.return_value = {"text": "ok", "success": True, "error": None}
        ocr_openai(small_image)
        assert mock_call.call_args.kwargs["model"] == "gpt-4.1-mini"


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


class TestPromptContent:

    @patch("arabic_ocr.ocr_engines._call_gemini")
    def test_prompt_contains_arabic_instructions(self, mock_call, small_image):
        mock_call.return_value = {"text": "", "success": True, "error": None}
        ocr_gemini_pro(small_image)
        prompt_arg = mock_call.call_args.kwargs["prompt"]
        assert "Arabic" in prompt_arg or "arabic" in prompt_arg

    @patch("arabic_ocr.ocr_engines._call_openai")
    def test_openai_prompt_instructs_exact_extraction(self, mock_call, small_image):
        mock_call.return_value = {"text": "", "success": True, "error": None}
        ocr_openai(small_image)
        prompt_arg = mock_call.call_args.kwargs["prompt"]
        assert "fidelity" in prompt_arg.lower() or "exact" in prompt_arg.lower()
