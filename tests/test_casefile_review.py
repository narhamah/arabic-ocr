"""Tests for case-file review helpers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from arabic_ocr.casefile_review import review_casefile_bundle
from arabic_ocr.models import BoundingBox, OCRBlock, OCRDocument, OCRLine, OCRPage, OCRSpan


def _sample_document() -> OCRDocument:
    pages = [
        OCRPage(
            page_number=0,
            width=800,
            height=1000,
            source_kind="scan",
            blocks=[
                OCRBlock(
                    block_type="text",
                    bbox=BoundingBox(0, 0, 800, 1000),
                    reading_order=0,
                    lines=[OCRLine(spans=[OCRSpan(text="سطر اول\nسطر ثان")])],
                )
            ],
        ),
        OCRPage(
            page_number=1,
            width=800,
            height=1000,
            source_kind="scan",
            blocks=[
                OCRBlock(
                    block_type="text",
                    bbox=BoundingBox(0, 0, 800, 1000),
                    reading_order=0,
                    lines=[OCRLine(spans=[OCRSpan(text="صفحة ثانية")])],
                )
            ],
        ),
    ]
    return OCRDocument(source_pdf="sample.pdf", dpi=400, pages=pages)


@patch("arabic_ocr.casefile_review.openai_ready", return_value=True)
@patch("arabic_ocr.casefile_review.create_client")
def test_review_casefile_bundle_returns_outputs(mock_create_client, mock_ready, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    responses = [
        json.dumps({
            "cleaned_text": "### Page 1\n\nسطر أول\nسطر ثان",
            "uncertainties": ["راجع اسم الشخص في هذا الموضع."],
        }, ensure_ascii=False),
        json.dumps({
            "cleaned_text": "### Page 2\n\nصفحة ثانية",
            "uncertainties": [],
        }, ensure_ascii=False),
    ]
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = [MagicMock(output_text=item) for item in responses]
    mock_create_client.return_value = mock_client

    case_index = {
        "page_total": 2,
        "documents": [
            {
                "id": "doc-001",
                "label": "مذكرة",
                "document_type": "prosecution memo",
                "page_start": 1,
                "page_end": 1,
                "summary": "ملخص 1",
                "ocr_watchlist": [],
            },
            {
                "id": "doc-002",
                "label": "مرفق",
                "document_type": "attachment",
                "page_start": 2,
                "page_end": 2,
                "summary": "ملخص 2",
                "ocr_watchlist": [],
            },
        ],
        "global_ocr_watchlist": ["تحقق من التاريخ."],
    }

    reviewed = review_casefile_bundle(_sample_document(), case_index)

    assert "# Reviewed Case File Bundle" in reviewed.reviewed_context_markdown
    assert "### Page 1" in reviewed.reviewed_context_markdown
    assert "# Review Notes" in reviewed.review_notes_markdown
    assert "راجع اسم الشخص" in reviewed.review_notes_markdown
    assert mock_client.responses.create.call_count == 2


@patch("arabic_ocr.casefile_review.openai_ready", return_value=True)
@patch("arabic_ocr.casefile_review.create_client")
def test_review_casefile_bundle_passes_reasoning_effort(mock_create_client, mock_ready, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    mock_client = MagicMock()
    mock_client.responses.create.return_value = MagicMock(output_text=json.dumps({
        "cleaned_text": "### Page 1\n\ncleaned",
        "uncertainties": [],
    }))
    mock_create_client.return_value = mock_client

    case_index = {
        "page_total": 1,
        "documents": [
            {
                "id": "doc-001",
                "label": "memo",
                "document_type": "prosecution memo",
                "page_start": 1,
                "page_end": 1,
                "summary": "",
                "ocr_watchlist": [],
            },
        ],
    }

    review_casefile_bundle(
        _sample_document(),
        case_index,
        model="gpt-5.4",
        reasoning_effort="xhigh",
    )

    assert mock_client.responses.create.call_count == 1
    assert mock_client.responses.create.call_args.kwargs["model"] == "gpt-5.4"
    assert mock_client.responses.create.call_args.kwargs["reasoning"] == {"effort": "xhigh"}


@patch("arabic_ocr.casefile_review.openai_ready", return_value=True)
@patch("arabic_ocr.casefile_review.create_client")
def test_review_casefile_bundle_splits_large_documents_into_segments(mock_create_client, mock_ready, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    responses = [
        json.dumps({
            "cleaned_text": "### Page 1\n\nsegment one",
            "uncertainties": ["verify first page name"],
        }),
        json.dumps({
            "cleaned_text": "### Page 2\n\nsegment two",
            "uncertainties": [],
        }),
    ]
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = [MagicMock(output_text=item) for item in responses]
    mock_create_client.return_value = mock_client

    case_index = {
        "page_total": 2,
        "documents": [
            {
                "id": "doc-001",
                "label": "memo",
                "document_type": "prosecution memo",
                "page_start": 1,
                "page_end": 2,
                "summary": "",
                "ocr_watchlist": [],
            },
        ],
    }

    reviewed = review_casefile_bundle(
        _sample_document(),
        case_index,
        max_chunk_pages=1,
    )

    assert mock_client.responses.create.call_count == 2
    assert "segment one" in reviewed.reviewed_context_markdown
    assert "segment two" in reviewed.reviewed_context_markdown
    assert "Page 1: verify first page name" in reviewed.review_notes_markdown


@patch("arabic_ocr.casefile_review.openai_ready", return_value=True)
@patch("arabic_ocr.casefile_review.create_client")
def test_review_casefile_bundle_resumes_from_existing_reviews(mock_create_client, mock_ready, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    mock_client = MagicMock()
    mock_client.responses.create.return_value = MagicMock(output_text=json.dumps({
        "cleaned_text": "### Page 2\n\nfresh review",
        "uncertainties": [],
    }))
    mock_create_client.return_value = mock_client

    case_index = {
        "page_total": 2,
        "documents": [
            {
                "id": "doc-001",
                "label": "memo one",
                "document_type": "prosecution memo",
                "page_start": 1,
                "page_end": 1,
                "summary": "",
                "ocr_watchlist": [],
            },
            {
                "id": "doc-002",
                "label": "memo two",
                "document_type": "attachment",
                "page_start": 2,
                "page_end": 2,
                "summary": "",
                "ocr_watchlist": [],
            },
        ],
    }
    existing_reviews = [{
        "entry": case_index["documents"][0],
        "cleaned_text": "### Page 1\n\ncached review",
        "uncertainties": [],
    }]

    reviewed = review_casefile_bundle(
        _sample_document(),
        case_index,
        existing_reviews=existing_reviews,
    )

    assert mock_client.responses.create.call_count == 1
    assert "cached review" in reviewed.reviewed_context_markdown
    assert "fresh review" in reviewed.reviewed_context_markdown
