"""Tests for engine registry and optional adapters."""

from __future__ import annotations

from unittest.mock import patch

from PIL import Image

from arabic_ocr.engines.paddle_ppstructure import PaddlePPStructureEngine
from arabic_ocr.engines.paddleocr_vl import PaddleOCRVLEngine
from arabic_ocr.engines.registry import available_engine_ids, get_primary_secondary


def test_gemini_mode_returns_two_engines():
    primary, secondary = get_primary_secondary("gemini")
    assert primary.id == "gemini_pro"
    assert secondary is not None and secondary.id == "gemini_flash"


def test_openai_mode_prefers_openai(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    primary, secondary = get_primary_secondary("openai")
    assert primary.id == "openai"
    assert secondary is None


def test_auto_prefers_cloud_vlms_over_paddle(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini")
    with patch("arabic_ocr.engines.openai_vision.OpenAIVisionEngine.is_available", return_value=True), patch(
        "arabic_ocr.engines.gemini.GeminiProEngine.is_available",
        return_value=True,
    ), patch(
        "arabic_ocr.engines.gemini.GeminiFlashEngine.is_available",
        return_value=True,
    ), patch(
        "arabic_ocr.engines.paddleocr_vl.PaddleOCRVLEngine.is_available",
        return_value=True,
    ), patch(
        "arabic_ocr.engines.paddle_ppstructure.PaddlePPStructureEngine.is_available",
        return_value=True,
    ):
        primary, secondary = get_primary_secondary("auto")

    assert primary.id == "openai"
    assert secondary is not None and secondary.id == "gemini_pro"


def test_available_engines_include_openai():
    engines = available_engine_ids()
    assert "openai" in engines


@patch("importlib.util.find_spec", return_value=None)
def test_gemini_engine_unavailable_without_google_genai(mock_find_spec):
    from arabic_ocr.engines.gemini import GeminiProEngine

    assert GeminiProEngine().is_available() is False


@patch("importlib.util.find_spec", return_value=None)
def test_paddle_ppstructure_unavailable(mock_find_spec):
    engine = PaddlePPStructureEngine()
    result = engine.recognize(Image.new("RGB", (10, 10), "white"))
    assert result.success is False
    assert "paddleocr" in result.error


@patch("importlib.util.find_spec", return_value=None)
def test_paddle_vl_unavailable(mock_find_spec):
    engine = PaddleOCRVLEngine()
    result = engine.recognize(Image.new("RGB", (10, 10), "white"))
    assert result.success is False
    assert "paddleocr" in result.error
