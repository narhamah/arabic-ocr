"""Tests for cross-model verifier — TDD RED phase."""

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from arabic_ocr.verifier import verify_and_merge


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
            mock_claude.return_value = "الوزير أكد أن العدد كبير"
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

    @patch("arabic_ocr.verifier._call_claude_tiebreaker")
    def test_tiebreaker_receives_image(self, mock_claude, small_image):
        mock_claude.return_value = "resolved"
        verify_and_merge("word1 word2", "word1 word3", small_image)
        call_args = mock_claude.call_args
        # Image should be passed
        assert any(isinstance(arg, Image.Image) for arg in call_args[0]) or \
               any(isinstance(v, Image.Image) for v in call_args[1].values())

    @patch("arabic_ocr.verifier._call_claude_tiebreaker")
    def test_tiebreaker_receives_both_versions(self, mock_claude, small_image):
        mock_claude.return_value = "text a"
        verify_and_merge("text a", "text b", small_image)
        call_args = mock_claude.call_args
        # Both texts should be in args somehow
        all_str_args = " ".join(str(a) for a in call_args[0]) + " ".join(str(v) for v in call_args[1].values())
        assert "text a" in all_str_args or "text b" in all_str_args


class TestFallback:

    @patch("arabic_ocr.verifier._call_claude_tiebreaker")
    def test_handles_claude_api_error(self, mock_claude, small_image):
        """If Claude fails, falls back to primary (Gemini Pro) output."""
        mock_claude.side_effect = Exception("API error")
        result = verify_and_merge("primary text", "secondary text", small_image)
        # Should not raise, should return primary
        assert result["text"] == "primary text"

    @patch("arabic_ocr.verifier._call_claude_tiebreaker")
    def test_fallback_returns_primary_on_none(self, mock_claude, small_image):
        mock_claude.return_value = None
        result = verify_and_merge("primary text", "secondary text", small_image)
        assert result["text"] == "primary text"


class TestConfidence:

    def test_confidence_score_returned(self, small_image):
        result = verify_and_merge("same text", "same text", small_image)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_identical_gives_full_confidence(self, small_image):
        result = verify_and_merge("identical", "identical", small_image)
        assert result["confidence"] == 1.0

    @patch("arabic_ocr.verifier._call_claude_tiebreaker")
    def test_disagreements_lower_confidence(self, mock_claude, small_image):
        mock_claude.return_value = "a b c"
        result = verify_and_merge("a b c", "a x c", small_image)
        assert result["confidence"] < 1.0


class TestEdgeCases:

    def test_empty_strings(self, small_image):
        result = verify_and_merge("", "", small_image)
        assert result["text"] == ""
        assert result["confidence"] == 1.0

    def test_one_empty_string(self, small_image):
        with patch("arabic_ocr.verifier._call_claude_tiebreaker") as mock_claude:
            mock_claude.return_value = "some text"
            result = verify_and_merge("some text", "", small_image)
        assert isinstance(result["text"], str)
