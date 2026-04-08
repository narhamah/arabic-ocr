"""Tests for OCR verification and native-vs-OCR adjudication."""

from unittest.mock import patch

import pytest
from PIL import Image

from arabic_ocr.verifier import verify_and_merge, verify_native_vs_ocr


@pytest.fixture
def small_image():
    return Image.new("RGB", (100, 100), "white")


class TestIdenticalOutputs:

    def test_identical_outputs_accepted(self, small_image):
        text = "بسم الله الرحمن الرحيم"
        result = verify_and_merge(text, text, small_image)
        assert result["text"] == text
        assert result["confidence"] == 1.0
        assert result["disagreements"] == 0

    def test_returns_required_keys(self, small_image):
        result = verify_and_merge("test", "test", small_image)
        assert "text" in result
        assert "confidence" in result
        assert "disagreements" in result
        assert "total_words" in result


class TestDisagreementDetection:

    def test_minor_differences_detected(self, small_image):
        primary = "الوزير أكد أن العدد كبير"
        secondary = "الوزير اكد ان العدد كبير"
        with patch("arabic_ocr.verifier._call_claude_tiebreaker") as mock_claude:
            mock_claude.return_value = primary
            result = verify_and_merge(primary, secondary, small_image)
        assert result["disagreements"] > 0

    def test_major_differences_flagged(self, small_image):
        primary = "نص مختلف تماما عن النص الاخر"
        secondary = "كلام آخر مغاير بشكل كامل للأول"
        with patch("arabic_ocr.verifier._call_claude_tiebreaker") as mock_claude:
            mock_claude.return_value = primary
            result = verify_and_merge(primary, secondary, small_image)
        assert result["disagreements"] > 0


class TestTiebreaker:

    @patch("arabic_ocr.verifier._call_claude_tiebreaker")
    def test_tiebreaker_called_on_disagreement(self, mock_claude, small_image):
        mock_claude.return_value = "الوزير أكد"
        verify_and_merge("الوزير أكد", "الوزير اكد", small_image)
        mock_claude.assert_called_once()

    @patch("arabic_ocr.verifier._call_claude_tiebreaker")
    def test_tiebreaker_not_called_when_identical(self, mock_claude, small_image):
        verify_and_merge("same text", "same text", small_image)
        mock_claude.assert_not_called()


class TestFallback:

    @patch("arabic_ocr.verifier._call_claude_tiebreaker")
    def test_handles_claude_api_error(self, mock_claude, small_image):
        mock_claude.side_effect = Exception("API error")
        result = verify_and_merge("primary text", "secondary text", small_image)
        assert result["text"] == "primary text"

    @patch("arabic_ocr.verifier._call_claude_tiebreaker")
    def test_fallback_returns_primary_on_none(self, mock_claude, small_image):
        mock_claude.return_value = None
        result = verify_and_merge("primary text", "secondary text", small_image)
        assert result["text"] == "primary text"


class TestNativeVsOcr:

    def test_accepts_native_when_it_matches_ocr(self, small_image):
        result = verify_native_vs_ocr(
            native_text="كشف حساب 12345",
            ocr_text="كشف حساب 12345",
            image=small_image,
        )
        assert result["source"] == "native_confirmed_by_ocr"
        assert result["digit_agreement"] is True

    @patch("arabic_ocr.verifier._call_claude_page_resolver")
    @patch("arabic_ocr.verifier._call_openai_page_resolver")
    def test_uses_resolver_on_meaningful_difference(self, mock_openai, mock_claude, small_image):
        mock_claude.return_value = "كشف حساب 12345"
        mock_openai.return_value = "كشف حساب 12345"
        result = verify_native_vs_ocr(
            native_text="كشف حساب 1234S",
            ocr_text="كشف حساب 12345",
            image=small_image,
        )
        assert result["text"] == "كشف حساب 12345"
        assert result["source"] in {"claude", "openai"}
