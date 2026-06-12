"""Tests for faithful and LLM normalization profiles."""

from arabic_ocr.normalizer import normalize_faithful, normalize_llm


def test_normalize_faithful_preserves_newlines():
    result = normalize_faithful("أحمد\n\n١٢٣")
    assert "\n\n" in result
    assert "١٢٣" in result


def test_normalize_llm_preserves_newlines_and_converts_digits():
    result = normalize_llm("أحمد\n\n١٢٣")
    assert "\n\n" in result
    assert "123" in result
