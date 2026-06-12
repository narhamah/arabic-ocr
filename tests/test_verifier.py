"""Tests for cross-model verification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
        assert set(result) == {"text", "confidence", "disagreements", "total_words"}


class TestDisagreementDetection:

    @patch("arabic_ocr.verifier._call_tiebreaker")
    def test_minor_differences_detected(self, mock_tiebreaker, small_image):
        mock_tiebreaker.return_value = "الوزير أكد أن العدد كبير"
        result = verify_and_merge("الوزير أكد أن العدد كبير", "الوزير اكد ان العدد كبير", small_image)
        assert result["disagreements"] > 0

    @patch("arabic_ocr.verifier._call_tiebreaker")
    def test_major_differences_flagged(self, mock_tiebreaker, small_image):
        mock_tiebreaker.return_value = "نص مختلف تماما عن النص الاخر"
        result = verify_and_merge("نص مختلف تماما عن النص الاخر", "كلام آخر مغاير بشكل كامل للأول", small_image)
        assert result["disagreements"] > 0


class TestTiebreaker:

    @patch("arabic_ocr.verifier._call_tiebreaker")
    def test_tiebreaker_called_on_disagreement(self, mock_tiebreaker, small_image):
        mock_tiebreaker.return_value = "الوزير أكد"
        verify_and_merge("الوزير أكد", "الوزير اكد", small_image)
        mock_tiebreaker.assert_called_once()

    @patch("arabic_ocr.verifier._call_tiebreaker")
    def test_tiebreaker_not_called_when_identical(self, mock_tiebreaker, small_image):
        verify_and_merge("same text", "same text", small_image)
        mock_tiebreaker.assert_not_called()

    @patch("arabic_ocr.verifier._call_tiebreaker")
    def test_tiebreaker_receives_image(self, mock_tiebreaker, small_image):
        mock_tiebreaker.return_value = "resolved"
        verify_and_merge("word1 word2", "word1 word3", small_image)
        call_args = mock_tiebreaker.call_args
        assert any(isinstance(arg, Image.Image) for arg in call_args[0]) or any(
            isinstance(value, Image.Image) for value in call_args[1].values()
        )

    @patch("arabic_ocr.verifier._call_tiebreaker")
    def test_tiebreaker_receives_both_versions(self, mock_tiebreaker, small_image):
        mock_tiebreaker.return_value = "text a"
        verify_and_merge("text a", "text b", small_image)
        all_str_args = " ".join(str(arg) for arg in mock_tiebreaker.call_args[0]) + " ".join(
            str(value) for value in mock_tiebreaker.call_args[1].values()
        )
        assert "text a" in all_str_args or "text b" in all_str_args


class TestFallback:

    @patch("arabic_ocr.verifier._call_tiebreaker")
    def test_handles_tiebreaker_error(self, mock_tiebreaker, small_image):
        mock_tiebreaker.side_effect = Exception("API error")
        result = verify_and_merge("primary text", "secondary text", small_image)
        assert result["text"] == "primary text"

    @patch("arabic_ocr.verifier._call_tiebreaker")
    def test_fallback_returns_primary_on_none(self, mock_tiebreaker, small_image):
        mock_tiebreaker.return_value = None
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

    def test_identical_gives_full_confidence(self, small_image):
        result = verify_and_merge("identical", "identical", small_image)
        assert result["confidence"] == 1.0

    @patch("arabic_ocr.verifier._call_tiebreaker")
    def test_disagreements_lower_confidence(self, mock_tiebreaker, small_image):
        mock_tiebreaker.return_value = "a b c"
        result = verify_and_merge("a b c", "a x c", small_image)
        assert result["confidence"] < 1.0


class TestEdgeCases:

    def test_empty_strings(self, small_image):
        result = verify_and_merge("", "", small_image)
        assert result["text"] == ""
        assert result["confidence"] == 1.0

    @patch("arabic_ocr.verifier._call_tiebreaker")
    def test_one_empty_string(self, mock_tiebreaker, small_image):
        mock_tiebreaker.return_value = "some text"
        result = verify_and_merge("some text", "", small_image)
        assert isinstance(result["text"], str)


def test_openai_tiebreaker_returns_none_without_key(monkeypatch, small_image):
    from arabic_ocr.verifier import _call_openai_tiebreaker

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _call_openai_tiebreaker(primary="a", secondary="b", disagreements=[], image=small_image) is None


def test_claude_tiebreaker_returns_none_without_key(monkeypatch, small_image):
    from arabic_ocr.verifier import _call_claude_tiebreaker

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _call_claude_tiebreaker(primary="a", secondary="b", disagreements=[], image=small_image) is None


def test_openai_is_preferred_when_available(monkeypatch, small_image):
    from arabic_ocr.verifier import _call_tiebreaker

    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    with patch("arabic_ocr.verifier.openai_ready", return_value=True), patch(
        "arabic_ocr.verifier._call_openai_tiebreaker",
        return_value="resolved",
    ) as mock_openai, patch("arabic_ocr.verifier._call_claude_tiebreaker") as mock_claude:
        result = _call_tiebreaker(primary="a", secondary="b", disagreements=[], image=small_image)

    assert result == "resolved"
    mock_openai.assert_called_once()
    mock_claude.assert_not_called()


def test_claude_full_body(monkeypatch, small_image):
    from arabic_ocr.verifier import _call_claude_tiebreaker

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake_key")
    mock_content_block = MagicMock(text="resolved text output")
    mock_message = MagicMock(content=[mock_content_block])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client

    import sys

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        result = _call_claude_tiebreaker(
            primary="الوزير أكد أن العدد كبير",
            secondary="الوزير اكد ان العدد كبير",
            disagreements=[{"primary": "أكد", "secondary": "اكد", "position": 1}],
            image=small_image,
        )

    assert result == "resolved text output"
