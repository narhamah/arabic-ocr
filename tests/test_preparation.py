"""Tests for context preparation helpers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from arabic_ocr.models import BoundingBox, OCRBlock, OCRDocument, OCRLine, OCRPage, OCRSpan
from arabic_ocr.preparation import prepare_context_bundle, prepare_context_markdown, reflow_block_text


def _sample_document() -> OCRDocument:
    page = OCRPage(
        page_number=0,
        width=800,
        height=1000,
        source_kind="scan",
        blocks=[
            OCRBlock(
                block_type="text",
                bbox=BoundingBox(0, 0, 100, 100),
                reading_order=0,
                lines=[OCRLine(spans=[OCRSpan(text="سطر أول"), OCRSpan(text="\nسطر ثان")])],
            )
        ],
    )
    return OCRDocument(source_pdf="sample.pdf", dpi=400, pages=[page])


def test_rule_strategy_returns_baseline_markdown():
    text = prepare_context_markdown(_sample_document(), strategy="rule")
    assert "## Page 1" in text
    assert "سطر" in text


def test_auto_strategy_falls_back_without_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    text = prepare_context_markdown(_sample_document(), strategy="auto")
    assert "## Page 1" in text


@patch("arabic_ocr.preparation.openai_ready", return_value=True)
@patch("arabic_ocr.preparation.create_client")
def test_openai_strategy_uses_client(mock_create_client, mock_ready, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    mock_response = MagicMock(output_text="Formatted markdown")
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mock_create_client.return_value = mock_client

    text = prepare_context_markdown(_sample_document(), strategy="openai")

    assert "Formatted markdown" in text
    mock_client.responses.create.assert_called_once()


@patch("arabic_ocr.preparation.openai_ready", return_value=True)
@patch("arabic_ocr.preparation.create_client")
def test_preparation_emits_progress(mock_create_client, mock_ready, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    mock_response = MagicMock(output_text="Formatted markdown")
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mock_create_client.return_value = mock_client
    events = []

    prepare_context_markdown(
        _sample_document(),
        strategy="openai",
        progress_callback=lambda event, payload: events.append((event, payload["page_number"])),
    )

    assert events == [("context_preparation_page", 0)]


@patch("arabic_ocr.preparation.openai_ready", return_value=True)
@patch("arabic_ocr.preparation.create_client")
def test_openai_casefile_returns_bundle_outputs(mock_create_client, mock_ready, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    payload = {
        "bundle_title": "Case bundle",
        "bundle_description": "Mixed prosecution file.",
        "documents": [
            {
                "id": "doc-001",
                "label": "Opening memorandum",
                "document_type": "prosecution memo",
                "page_start": 1,
                "page_end": 1,
                "confidence": "medium",
                "language": "ar",
                "summary": "Opening memo summary.",
                "defense_relevance": "Check whether the allegations are framed consistently.",
                "key_dates": ["22 / 7 / 2025"],
                "date_events": [{"date": "22 / 7 / 2025", "event": "Meeting referenced in complaint."}],
                "people": [{"name": "نواف", "role": "complainant"}],
                "organizations": ["النيابة العامة"],
                "topics": ["threats", "checks"],
                "evidence_types": ["complaint memo"],
                "ocr_watchlist": ["Verify case number digits."],
            }
        ],
        "bundle_notes": ["Grouping inferred from OCR headers."],
        "global_ocr_watchlist": ["Repeated header noise."],
    }
    mock_response = MagicMock(output_text=json.dumps(payload, ensure_ascii=False))
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mock_create_client.return_value = mock_client

    bundle = prepare_context_bundle(_sample_document(), strategy="openai_casefile")

    assert bundle.case_index is not None
    assert bundle.case_packet_markdown is not None
    assert "# Case File Bundle" in bundle.context_markdown
    assert "Opening memorandum" in bundle.context_markdown
    assert "# Defense Workbench" in bundle.case_packet_markdown
    assert "22 / 7 / 2025" in bundle.case_packet_markdown
    mock_client.responses.create.assert_called_once()


@patch("arabic_ocr.preparation.openai_ready", return_value=False)
def test_reflow_returns_original_without_openai(mock_ready):
    text = "سطر اول سطر ثان"
    assert reflow_block_text(text) == text


@patch("arabic_ocr.preparation.openai_ready", return_value=True)
@patch("arabic_ocr.preparation.create_client")
def test_reflow_uses_openai_when_available(mock_create_client, mock_ready, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    mock_response = MagicMock(output_text="سطر اول\nسطر ثان")
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mock_create_client.return_value = mock_client

    result = reflow_block_text("سطر اول سطر ثان")

    assert result == "سطر اول\nسطر ثان"
