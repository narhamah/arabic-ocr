"""OpenAI-assisted cleanup for OCR-derived case-file bundles."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from arabic_ocr.models import OCRDocument
from arabic_ocr.normalizer import normalize_llm
from arabic_ocr.openai_support import create_client, extract_output_text, openai_ready

_REVIEW_PROMPT = """You are correcting OCR-extracted Arabic legal case-file text for counsel review.

Rules:
- Preserve all substantive facts, allegations, denials, dates, numbers, amounts, names, roles, citations, and document references unless the OCR error is obvious from context
- Correct obvious OCR errors, spacing, punctuation, broken line wraps, duplicated fragments, and spelling mistakes
- Keep the document in Arabic when the source is Arabic, while preserving existing English text
- Do not summarize, omit, translate, soften, or rewrite the substance
- Preserve page markers exactly if they appear in the input, for example: ### Page 17
- If a token is too ambiguous to correct confidently, keep the closest supported reading in the text and list the issue in `uncertainties`
- Return valid JSON only with this shape:
  {
    "cleaned_text": "full cleaned text",
    "uncertainties": ["issue 1", "issue 2"]
  }"""

_DEFAULT_MAX_REVIEW_CHUNK_CHARS = 3000
_DEFAULT_MAX_REVIEW_CHUNK_PAGES = 2
_DEFAULT_REVIEW_MAX_ATTEMPTS = 3
_DEFAULT_REVIEW_REQUEST_TIMEOUT = 900.0


@dataclass(slots=True)
class ReviewedCasefileBundle:
    """Reviewed case-file outputs."""

    reviewed_context_markdown: str
    review_notes_markdown: str


def review_casefile_bundle(
    document: OCRDocument,
    case_index: dict[str, Any],
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_chunk_chars: int | None = None,
    max_chunk_pages: int | None = None,
    existing_reviews: list[dict[str, Any]] | None = None,
    checkpoint_callback=None,
    progress_callback=None,
) -> ReviewedCasefileBundle:
    """Clean and normalize each logical document chunk in a case-file bundle."""
    if not openai_ready():
        raise RuntimeError("OpenAI credentials are required for case-file review")

    client = create_client()
    selected_model = model or os.getenv("OPENAI_CASEFILE_REVIEW_MODEL") or os.getenv("OPENAI_CASEFILE_MODEL") or "gpt-5"
    selected_reasoning_effort = reasoning_effort or os.getenv("OPENAI_CASEFILE_REVIEW_REASONING_EFFORT")
    selected_max_chunk_chars = max_chunk_chars or _env_int(
        "OPENAI_CASEFILE_REVIEW_MAX_CHARS",
        _DEFAULT_MAX_REVIEW_CHUNK_CHARS,
    )
    selected_max_chunk_pages = max_chunk_pages or _env_int(
        "OPENAI_CASEFILE_REVIEW_MAX_PAGES",
        _DEFAULT_MAX_REVIEW_CHUNK_PAGES,
    )
    selected_request_timeout = _env_float(
        "OPENAI_CASEFILE_REVIEW_REQUEST_TIMEOUT",
        _DEFAULT_REVIEW_REQUEST_TIMEOUT,
    )
    existing_by_key = {
        _entry_review_key(item["entry"]): item
        for item in existing_reviews or []
        if item.get("entry")
    }
    reviewed_documents: list[dict[str, Any]] = []

    for index, entry in enumerate(case_index.get("documents") or [], start=1):
        entry_key = _entry_review_key(entry)
        if entry_key in existing_by_key:
            reviewed_documents.append(existing_by_key[entry_key])
            if progress_callback is not None:
                progress_callback("casefile_review_resume", {
                    "document_index": index,
                    "document_total": len(case_index.get("documents") or []),
                    "page_start": entry.get("page_start"),
                    "page_end": entry.get("page_end"),
                    "label": entry.get("label"),
                })
            continue
        segments = _build_review_segments(
            document,
            entry,
            max_chunk_chars=selected_max_chunk_chars,
            max_chunk_pages=selected_max_chunk_pages,
        )
        cleaned_parts: list[str] = []
        uncertainties: list[str] = []
        for segment_index, segment in enumerate(segments, start=1):
            if progress_callback is not None:
                progress_callback("casefile_review_document", {
                    "document_index": index,
                    "document_total": len(case_index.get("documents") or []),
                    "segment_index": segment_index,
                    "segment_total": len(segments),
                    "page_start": segment["page_start"],
                    "page_end": segment["page_end"],
                    "label": entry.get("label"),
                })
            cleaned_text, segment_uncertainties = _review_chunk(
                client,
                entry=entry,
                raw_text=segment["raw_text"],
                model=selected_model,
                reasoning_effort=selected_reasoning_effort,
                request_timeout=selected_request_timeout,
            )
            cleaned_parts.append(cleaned_text)
            uncertainties.extend(
                _qualify_uncertainties(
                    segment_uncertainties,
                    page_start=segment["page_start"],
                    page_end=segment["page_end"],
                )
            )
        reviewed_item = {
            "entry": entry,
            "cleaned_text": "\n\n".join(part.strip() for part in cleaned_parts if part and part.strip()).strip(),
            "uncertainties": uncertainties,
        }
        reviewed_documents.append(reviewed_item)
        if checkpoint_callback is not None:
            checkpoint_callback(reviewed_documents)

    return ReviewedCasefileBundle(
        reviewed_context_markdown=_render_reviewed_context(document, case_index, reviewed_documents),
        review_notes_markdown=_render_review_notes(case_index, reviewed_documents),
    )


def _chunk_text(document: OCRDocument, entry: dict[str, Any]) -> str:
    start = int(entry["page_start"])
    end = int(entry["page_end"])
    parts: list[str] = []
    for page_number in range(start, end + 1):
        page = document.pages[page_number - 1]
        parts.append(f"### Page {page_number}\n\n{normalize_llm(page.text)}".strip())
    return "\n\n".join(parts).strip()


def _build_review_segments(
    document: OCRDocument,
    entry: dict[str, Any],
    *,
    max_chunk_chars: int,
    max_chunk_pages: int,
) -> list[dict[str, Any]]:
    start = int(entry["page_start"])
    end = int(entry["page_end"])
    segments: list[dict[str, Any]] = []
    current_pages: list[int] = []
    current_parts: list[str] = []
    current_chars = 0

    for page_number in range(start, end + 1):
        page = document.pages[page_number - 1]
        page_text = f"### Page {page_number}\n\n{normalize_llm(page.text)}".strip()
        projected_chars = current_chars + len(page_text) + (2 if current_parts else 0)
        exceeds_page_limit = bool(current_parts) and len(current_pages) >= max_chunk_pages
        exceeds_char_limit = bool(current_parts) and projected_chars > max_chunk_chars
        if exceeds_page_limit or exceeds_char_limit:
            segments.append({
                "page_start": current_pages[0],
                "page_end": current_pages[-1],
                "raw_text": "\n\n".join(current_parts).strip(),
            })
            current_pages = []
            current_parts = []
            current_chars = 0

        current_pages.append(page_number)
        current_parts.append(page_text)
        current_chars += len(page_text) + (2 if len(current_parts) > 1 else 0)

    if current_parts:
        segments.append({
            "page_start": current_pages[0],
            "page_end": current_pages[-1],
            "raw_text": "\n\n".join(current_parts).strip(),
        })

    return segments


def _review_chunk(
    client,
    *,
    entry: dict[str, Any],
    raw_text: str,
    model: str,
    reasoning_effort: str | None,
    request_timeout: float,
) -> tuple[str, list[str]]:
    request: dict[str, Any] = {
        "model": model,
        "input": [{
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": (
                    f"{_REVIEW_PROMPT}\n\n"
                    f"Document label: {entry.get('label', 'Unknown')}\n"
                    f"Document type: {entry.get('document_type', 'unknown')}\n"
                    f"Pages: {entry.get('page_start')}-{entry.get('page_end')}\n"
                    f"Summary hint: {entry.get('summary', '')}\n"
                    f"Known OCR risks: {'; '.join(entry.get('ocr_watchlist') or [])}\n\n"
                    "Raw OCR text follows:\n"
                    f"{raw_text}"
                ),
            }],
        }],
    }
    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort}
    request["timeout"] = request_timeout
    response = None
    for attempt in range(1, _DEFAULT_REVIEW_MAX_ATTEMPTS + 1):
        try:
            response = client.responses.create(**request)
            break
        except Exception:
            if attempt >= _DEFAULT_REVIEW_MAX_ATTEMPTS:
                raise
            time.sleep(min(5, attempt * 2))
    assert response is not None
    payload_text = extract_output_text(response)
    cleaned_text, uncertainties = _parse_review_payload(payload_text)
    if not cleaned_text:
        cleaned_text = raw_text
        uncertainties.append("Review model returned empty cleaned text; raw OCR chunk was preserved.")
    return cleaned_text, uncertainties


def _parse_review_payload(text: str) -> tuple[str, list[str]]:
    payload = _parse_json_object(text)
    cleaned_text = str(payload.get("cleaned_text") or "").strip()
    uncertainties = [
        str(item).strip()
        for item in payload.get("uncertainties") or []
        if str(item).strip()
    ]
    return cleaned_text, uncertainties


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Review response was not a JSON object")
    return payload


def _render_reviewed_context(
    document: OCRDocument,
    case_index: dict[str, Any],
    reviewed_documents: list[dict[str, Any]],
) -> str:
    lines = [
        "# Reviewed Case File Bundle",
        "",
        f"Source PDF: {document.source_pdf}",
        f"Total pages: {case_index.get('page_total', len(document.pages))}",
        f"Logical documents: {len(reviewed_documents)}",
        "",
        "This file is a cleaned LLM-reviewed edition of the OCR bundle. It is intended for analysis and drafting, but any disputed wording should still be checked against the scan.",
    ]

    for index, item in enumerate(reviewed_documents, start=1):
        entry = item["entry"]
        lines.extend([
            "",
            f"## Document {index:02d} - {entry.get('label', 'Unlabeled document')}",
            "",
            f"Pages: {entry.get('page_start')}-{entry.get('page_end')}",
            f"Type: {entry.get('document_type', 'unknown')}",
        ])
        if entry.get("summary"):
            lines.append(f"Summary: {entry['summary']}")
        lines.extend([
            "",
            item["cleaned_text"].strip(),
        ])

    return "\n".join(lines).strip()


def _render_review_notes(case_index: dict[str, Any], reviewed_documents: list[dict[str, Any]]) -> str:
    lines = [
        "# Review Notes",
        "",
        "These are the remaining ambiguities flagged during the LLM cleanup pass. They are the places to verify directly against the scan before quoting or relying on them.",
    ]
    global_watchlist = case_index.get("global_ocr_watchlist") or []
    if global_watchlist:
        lines.extend(["", "## Bundle-Level OCR Risks", ""])
        lines.extend(f"- {item}" for item in global_watchlist)

    for index, item in enumerate(reviewed_documents, start=1):
        lines.extend([
            "",
            f"## Document {index:02d} - {item['entry'].get('label', 'Unlabeled document')}",
            "",
        ])
        uncertainties = item.get("uncertainties") or []
        if uncertainties:
            lines.extend(f"- {issue}" for issue in uncertainties)
        else:
            lines.append("- No remaining high-confidence ambiguities were flagged by the review model.")

    return "\n".join(lines).strip()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _qualify_uncertainties(items: list[str], *, page_start: int, page_end: int) -> list[str]:
    if not items:
        return []
    if page_start == page_end:
        prefix = f"Page {page_start}: "
    else:
        prefix = f"Pages {page_start}-{page_end}: "
    return [f"{prefix}{item}" for item in items]


def _entry_review_key(entry: dict[str, Any]) -> str:
    entry_id = str(entry.get("id") or "").strip()
    if entry_id:
        return entry_id
    return (
        f"{entry.get('page_start', '')}:"
        f"{entry.get('page_end', '')}:"
        f"{entry.get('label', '')}:"
        f"{entry.get('document_type', '')}"
    )
