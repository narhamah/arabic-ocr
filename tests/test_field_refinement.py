"""Tests for targeted field refinement."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from PIL import Image

from arabic_ocr.field_refinement import refine_high_value_lines
from arabic_ocr.models import BoundingBox, OCRBlock, OCRLine, OCRSpan
from arabic_ocr.quality import classify_field_candidate


def _sample_block(texts: list[str]) -> OCRBlock:
    return OCRBlock(
        block_type="heuristic_block",
        bbox=BoundingBox(0, 0, 1000, 800),
        reading_order=0,
        confidence=0.5,
        engine_id="openai",
        lines=[
            OCRLine(
                spans=[OCRSpan(text=text, confidence=0.5, engine_id="openai")],
                confidence=0.5,
                bbox=BoundingBox(0, index * 40, 1000, (index + 1) * 40),
                engine_id="openai",
            )
            for index, text in enumerate(texts)
        ],
    )


def test_refinement_skips_non_openai_blocks():
    block = _sample_block(["في القضية رقم 123 لسنة 2025"])
    block.engine_id = "gemini_pro"
    result = refine_high_value_lines(block, region_crop=Image.new("RGB", (1000, 800), "white"))
    assert "field_refinements" not in result.metadata


@patch("arabic_ocr.field_refinement.openai_ready", return_value=True)
@patch("arabic_ocr.field_refinement.create_client")
def test_refinement_accepts_better_case_line(mock_create_client, mock_ready, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    mock_response = MagicMock(output_text="في القضية رقم 2234 لسنة 2025 حصر")
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mock_create_client.return_value = mock_client

    block = _sample_block(["في القضية رقم 324 لسنة 2025 حصر"])
    result = refine_high_value_lines(block, region_crop=Image.new("RGB", (1000, 800), "white"))

    assert result.lines[0].text == "في القضية رقم 2234 لسنة 2025 حصر"
    assert result.lines[0].metadata["original_text"] == "في القضية رقم 324 لسنة 2025 حصر"
    assert result.metadata["field_refinements"][0]["accepted"] is True


@patch("arabic_ocr.field_refinement.openai_ready", return_value=True)
@patch("arabic_ocr.field_refinement.create_client")
def test_refinement_rejects_digit_loss(mock_create_client, mock_ready, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    mock_response = MagicMock(output_text="في القضية رقم لسنة")
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mock_create_client.return_value = mock_client

    block = _sample_block(["في القضية رقم 324 لسنة 2025 حصر"])
    result = refine_high_value_lines(block, region_crop=Image.new("RGB", (1000, 800), "white"))

    assert result.lines[0].text == "في القضية رقم 324 لسنة 2025 حصر"
    assert result.metadata["field_refinements"][0]["accepted"] is False
    assert result.metadata["field_refinements"][0]["reason"] == "lost_digits"


@patch("arabic_ocr.field_refinement.openai_ready", return_value=True)
@patch("arabic_ocr.field_refinement.create_client")
def test_refinement_emits_progress(mock_create_client, mock_ready, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    mock_response = MagicMock(output_text="PUBLIC PROSECUTION")
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mock_create_client.return_value = mock_client
    events = []

    block = _sample_block(["PUBLIC PROSECUTION"])
    refine_high_value_lines(
        block,
        region_crop=Image.new("RGB", (1000, 800), "white"),
        progress_callback=lambda event, payload: events.append((event, payload["field_label"])),
        page_number=0,
        page_index=1,
        page_total=3,
        block_index=0,
    )

    assert events == [("field_refinement", "header")]


@patch("arabic_ocr.field_refinement.openai_ready", return_value=True)
@patch("arabic_ocr.field_refinement.create_client")
def test_refinement_rejects_cross_script_header(mock_create_client, mock_ready, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    mock_response = MagicMock(output_text="في القضية رقم ٢٣٢٤ لسنة ٢٠٢٥")
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mock_create_client.return_value = mock_client

    block = _sample_block(["Deputy Attorney General Office"])
    result = refine_high_value_lines(block, region_crop=Image.new("RGB", (1000, 800), "white"))

    assert result.lines[0].text == "Deputy Attorney General Office"
    assert result.metadata["field_refinements"][0]["accepted"] is False
    assert result.metadata["field_refinements"][0]["reason"] == "script_shift"


@patch("arabic_ocr.field_refinement.openai_ready", return_value=True)
@patch("arabic_ocr.field_refinement.create_client")
def test_refinement_rejects_same_script_unrelated_header(mock_create_client, mock_ready, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    mock_response = MagicMock(output_text="MINISTRY OF JUSTICE")
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mock_create_client.return_value = mock_client

    block = _sample_block(["PUBLIC PROSECUTION"])
    result = refine_high_value_lines(block, region_crop=Image.new("RGB", (1000, 800), "white"))

    assert result.lines[0].text == "PUBLIC PROSECUTION"
    assert result.metadata["field_refinements"][0]["accepted"] is False
    assert result.metadata["field_refinements"][0]["reason"] == "low_token_overlap"


def test_name_line_classification_requires_slashed_party_label():
    assert classify_field_candidate(
        "واقر الشاكي بالمحادثات الشخصية التي تمت بينه وبين ديسوس",
        line_index=4,
        total_lines=12,
    ) is None

    assert classify_field_candidate(
        "الشاكي / نواف احمد سالم",
        line_index=4,
        total_lines=12,
    ) == "name_line"


def test_date_line_classification_skips_mixed_header_date_rows():
    assert classify_field_candidate(
        "مكتب المحامي العام الكويت في / / الموافق / /",
        line_index=1,
        total_lines=12,
    ) is None
