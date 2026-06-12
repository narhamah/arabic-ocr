"""Helpers for preparing OCR text for downstream LLM use."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from arabic_ocr.exporters import render_markdown
from arabic_ocr.models import OCRDocument, OCRPage
from arabic_ocr.normalizer import normalize_llm
from arabic_ocr.openai_support import create_client, extract_output_text, openai_ready

_PREPARATION_PROMPT = """You are preparing OCR-extracted Arabic legal text for downstream question answering.

Rules:
- Preserve the wording, numbers, names, dates, citations, and ordering exactly
- Do not invent or complete missing text
- Do not summarize
- Improve only structure and readability
- Keep headings and paragraph boundaries
- Render obvious tables or lists using simple Markdown when safe
- If structure is uncertain, keep the original line breaks
- Return Markdown only for the page body, with no explanation"""

_REFLOW_PROMPT = """You are cleaning OCR text from an Arabic legal document.

Rules:
- Keep exactly the same words, numbers, punctuation, and ordering
- Do not add, remove, summarize, translate, or correct wording
- Your only job is to insert line breaks and paragraph breaks where they are obviously missing
- Keep bilingual headers on separate lines when appropriate
- Keep one printed line per output line when the structure is clear
- Return plain text only, with no commentary"""

_CASEFILE_INDEX_PROMPT = """You are organizing OCR-extracted Arabic legal case-bundle text for counsel review.

The input is a single scanned bundle that may contain multiple distinct documents.

Rules:
- Use only the OCR text that is provided
- Do not invent facts, dates, names, titles, or page ranges
- Group pages into contiguous logical documents in reading order
- Cover every page exactly once with non-overlapping page ranges
- If uncertain, prefer broader ranges and lower confidence
- Keep summaries neutral, factual, and document-focused
- "defense_relevance" should identify facts or issues worth defense review, not legal conclusions
- Return valid JSON only

Return an object with this shape:
{
  "bundle_title": "short title",
  "bundle_description": "1-3 sentence neutral description",
  "documents": [
    {
      "id": "doc-001",
      "label": "short document label",
      "document_type": "complaint|prosecution memo|witness statement|investigation summary|correspondence|attachment|evidence appendix|chat transcript|financial record|court order|unknown",
      "page_start": 1,
      "page_end": 3,
      "confidence": "high|medium|low",
      "language": "ar|ar-en|mixed|unknown",
      "summary": "2-4 sentence neutral summary",
      "defense_relevance": "why this document may matter for defense review",
      "key_dates": ["date strings exactly as seen or conservatively normalized"],
      "date_events": [{"date": "date string", "event": "event description"}],
      "people": [{"name": "person name", "role": "role in this document"}],
      "organizations": ["organization names"],
      "topics": ["short topic labels"],
      "evidence_types": ["evidence or material type"],
      "ocr_watchlist": ["uncertain names, numbers, or OCR issues that need manual verification"]
    }
  ],
  "bundle_notes": ["bundle-level notes"],
  "global_ocr_watchlist": ["bundle-level OCR uncertainties"]
}"""


@dataclass(slots=True)
class PreparedContextBundle:
    """Prepared LLM bundle outputs."""

    context_markdown: str
    case_packet_markdown: str | None = None
    case_index: dict[str, Any] | None = None

    def case_index_json(self) -> str | None:
        if self.case_index is None:
            return None
        return json.dumps(self.case_index, ensure_ascii=False, indent=2)


def prepare_context_markdown(
    document: OCRDocument,
    *,
    strategy: str = "rule",
    progress_callback=None,
) -> str:
    """Render LLM-ready Markdown, optionally refined with OpenAI."""
    return prepare_context_bundle(
        document,
        strategy=strategy,
        progress_callback=progress_callback,
    ).context_markdown


def prepare_context_bundle(
    document: OCRDocument,
    *,
    strategy: str = "rule",
    progress_callback=None,
) -> PreparedContextBundle:
    """Render an LLM-ready bundle, optionally adding case-file artifacts."""
    baseline = render_markdown(document, profile="llm")
    if strategy == "rule":
        return PreparedContextBundle(context_markdown=baseline)
    if strategy == "auto" and not openai_ready():
        return PreparedContextBundle(context_markdown=baseline)
    if strategy == "openai_casefile":
        if not openai_ready():
            return PreparedContextBundle(context_markdown=baseline)
        return _prepare_casefile_bundle(
            document,
            baseline_markdown=baseline,
            progress_callback=progress_callback,
        )
    if strategy not in {"auto", "openai"}:
        raise ValueError(f"Unsupported context preparation strategy: {strategy}")
    if not openai_ready():
        return PreparedContextBundle(context_markdown=baseline)

    sections: list[str] = []
    for index, page in enumerate(document.pages, start=1):
        if progress_callback is not None:
            progress_callback("context_preparation_page", {
                "page_number": page.page_number,
                "page_index": index,
                "page_total": len(document.pages),
            })
        try:
            sections.append(_prepare_page_markdown(page))
        except Exception:
            sections.append(f"## Page {page.page_number + 1}\n\n{normalize_llm(page.text)}".strip())
    return PreparedContextBundle(
        context_markdown="\n\n".join(section for section in sections if section).strip(),
    )


def _prepare_page_markdown(page: OCRPage) -> str:
    normalized = normalize_llm(page.text)
    if not normalized:
        return f"## Page {page.page_number + 1}"

    client = create_client()
    model = os.getenv("OPENAI_PREP_MODEL", "gpt-4.1-mini")
    response = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": (
                    f"{_PREPARATION_PROMPT}\n\n"
                    f"Page number: {page.page_number + 1}\n\n"
                    "OCR text follows:\n"
                    f"{normalized}"
                ),
            }],
        }],
    )
    body = extract_output_text(response) or normalized
    return f"## Page {page.page_number + 1}\n\n{body}".strip()


def reflow_block_text(text: str) -> str:
    """Use OpenAI to restore conservative line breaks in long OCR blocks."""
    normalized = normalize_llm(text)
    if not normalized or not openai_ready():
        return normalized

    client = create_client()
    model = os.getenv("OPENAI_PREP_MODEL", "gpt-4.1-mini")
    response = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": f"{_REFLOW_PROMPT}\n\nOCR text follows:\n{normalized}",
            }],
        }],
    )
    return extract_output_text(response) or normalized


def _prepare_casefile_bundle(
    document: OCRDocument,
    *,
    baseline_markdown: str,
    progress_callback=None,
) -> PreparedContextBundle:
    if progress_callback is not None:
        progress_callback("casefile_preparation_start", {
            "page_total": len(document.pages),
        })
    case_index = _classify_casefile(document, baseline_markdown=baseline_markdown)
    if progress_callback is not None:
        progress_callback("casefile_preparation_complete", {
            "page_total": len(document.pages),
            "document_total": len(case_index["documents"]),
        })
    return PreparedContextBundle(
        context_markdown=_render_casefile_context(document, case_index),
        case_packet_markdown=_render_case_packet(document, case_index),
        case_index=case_index,
    )


def _classify_casefile(document: OCRDocument, *, baseline_markdown: str) -> dict[str, Any]:
    client = create_client()
    model = os.getenv("OPENAI_CASEFILE_MODEL") or os.getenv("OPENAI_PREP_MODEL", "gpt-4.1")
    response = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": (
                    f"{_CASEFILE_INDEX_PROMPT}\n\n"
                    f"Total pages: {len(document.pages)}\n\n"
                    "Full OCR bundle markdown follows:\n"
                    f"{baseline_markdown}"
                ),
            }],
        }],
    )
    payload = _parse_json_object(extract_output_text(response))
    return _normalize_case_index(payload, document=document)


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
        raise ValueError("Case-file classification did not return a JSON object")
    return payload


def _normalize_case_index(payload: dict[str, Any], *, document: OCRDocument) -> dict[str, Any]:
    total_pages = len(document.pages)
    bundle_title = _clean_scalar(payload.get("bundle_title")) or "Case file bundle"
    bundle_description = _clean_scalar(payload.get("bundle_description"))
    bundle_notes = _clean_string_list(payload.get("bundle_notes"))
    global_ocr_watchlist = _clean_string_list(payload.get("global_ocr_watchlist"))

    normalized_candidates: list[dict[str, Any]] = []
    for raw_doc in payload.get("documents") or []:
        normalized_doc = _normalize_case_document_candidate(raw_doc)
        if normalized_doc is not None:
            normalized_candidates.append(normalized_doc)
    normalized_candidates.sort(key=lambda item: (item["page_start"], item["page_end"]))

    documents: list[dict[str, Any]] = []
    next_page = 1
    for candidate in normalized_candidates:
        start = max(1, min(total_pages, candidate["page_start"]))
        end = max(1, min(total_pages, candidate["page_end"]))
        if end < next_page:
            continue
        start = max(start, next_page)
        if start > next_page:
            documents.append(_unknown_case_document(next_page, start - 1, len(documents) + 1))
        documents.append(_finalize_case_document(candidate, start, end, len(documents) + 1))
        next_page = end + 1

    if not documents:
        documents.append(_unknown_case_document(1, total_pages, 1))
    elif next_page <= total_pages:
        documents.append(_unknown_case_document(next_page, total_pages, len(documents) + 1))

    return {
        "bundle_title": bundle_title,
        "bundle_description": bundle_description,
        "source_pdf": document.source_pdf,
        "page_total": total_pages,
        "documents": documents,
        "bundle_notes": bundle_notes,
        "global_ocr_watchlist": global_ocr_watchlist,
    }


def _normalize_case_document_candidate(raw_doc: Any) -> dict[str, Any] | None:
    if not isinstance(raw_doc, dict):
        return None
    try:
        page_start = int(raw_doc.get("page_start"))
        page_end = int(raw_doc.get("page_end"))
    except (TypeError, ValueError):
        return None
    if page_start < 1 or page_end < 1:
        return None
    if page_end < page_start:
        page_start, page_end = page_end, page_start
    return {
        "label": _clean_scalar(raw_doc.get("label")) or "Unlabeled document",
        "document_type": _clean_scalar(raw_doc.get("document_type")) or "unknown",
        "page_start": page_start,
        "page_end": page_end,
        "confidence": _clean_scalar(raw_doc.get("confidence")) or "low",
        "language": _clean_scalar(raw_doc.get("language")) or "unknown",
        "summary": _clean_scalar(raw_doc.get("summary")),
        "defense_relevance": _clean_scalar(raw_doc.get("defense_relevance")),
        "key_dates": _clean_string_list(raw_doc.get("key_dates")),
        "date_events": _clean_date_events(raw_doc.get("date_events")),
        "people": _clean_people(raw_doc.get("people")),
        "organizations": _clean_string_list(raw_doc.get("organizations")),
        "topics": _clean_string_list(raw_doc.get("topics")),
        "evidence_types": _clean_string_list(raw_doc.get("evidence_types")),
        "ocr_watchlist": _clean_string_list(raw_doc.get("ocr_watchlist")),
    }


def _finalize_case_document(candidate: dict[str, Any], page_start: int, page_end: int, index: int) -> dict[str, Any]:
    finalized = dict(candidate)
    finalized["id"] = f"doc-{index:03d}"
    finalized["page_start"] = page_start
    finalized["page_end"] = page_end
    return finalized


def _unknown_case_document(page_start: int, page_end: int, index: int) -> dict[str, Any]:
    return {
        "id": f"doc-{index:03d}",
        "label": "Unclassified bundle segment",
        "document_type": "unknown",
        "page_start": page_start,
        "page_end": page_end,
        "confidence": "low",
        "language": "unknown",
        "summary": "The OCR text in this page range should be reviewed manually to confirm the document type and relevance.",
        "defense_relevance": "Manual review recommended because the bundle classifier could not confidently label this segment.",
        "key_dates": [],
        "date_events": [],
        "people": [],
        "organizations": [],
        "topics": [],
        "evidence_types": [],
        "ocr_watchlist": ["Document boundary classification was uncertain for this page range."],
    }


def _render_casefile_context(document: OCRDocument, case_index: dict[str, Any]) -> str:
    lines = [
        "# Case File Bundle",
        "",
        f"Source PDF: {case_index['source_pdf']}",
        f"Total pages: {case_index['page_total']}",
        f"Logical documents: {len(case_index['documents'])}",
    ]
    if case_index.get("bundle_description"):
        lines.extend(["", case_index["bundle_description"]])
    if case_index.get("bundle_notes"):
        lines.extend(["", "## Bundle Notes", ""])
        lines.extend(f"- {note}" for note in case_index["bundle_notes"])

    lines.extend(["", "## Document Inventory", ""])
    for index, entry in enumerate(case_index["documents"], start=1):
        summary = entry.get("summary") or "No summary available."
        lines.append(
            f"{index}. {entry['label']} (pages {entry['page_start']}-{entry['page_end']}; "
            f"type: {entry['document_type']}; confidence: {entry['confidence']})"
        )
        lines.append(f"   {summary}")

    for index, entry in enumerate(case_index["documents"], start=1):
        lines.extend([
            "",
            f"## Document {index:02d} - {entry['label']}",
            "",
            f"Pages: {entry['page_start']}-{entry['page_end']}",
            f"Type: {entry['document_type']}",
            f"Confidence: {entry['confidence']}",
            f"Language: {entry['language']}",
        ])
        if entry.get("summary"):
            lines.append(f"Summary: {entry['summary']}")
        if entry.get("defense_relevance"):
            lines.append(f"Defense relevance: {entry['defense_relevance']}")
        if entry.get("topics"):
            lines.append(f"Topics: {', '.join(entry['topics'])}")
        if entry.get("evidence_types"):
            lines.append(f"Evidence types: {', '.join(entry['evidence_types'])}")
        if entry.get("key_dates"):
            lines.append(f"Key dates: {', '.join(entry['key_dates'])}")
        if entry.get("organizations"):
            lines.append(f"Organizations: {', '.join(entry['organizations'])}")
        if entry.get("people"):
            lines.append(
                "People: "
                + ", ".join(
                    f"{person['name']} ({person['role']})" if person.get("role") else person["name"]
                    for person in entry["people"]
                )
            )
        if entry.get("ocr_watchlist"):
            lines.append("OCR watchlist: " + "; ".join(entry["ocr_watchlist"]))

        for page_number in range(entry["page_start"], entry["page_end"] + 1):
            page = document.pages[page_number - 1]
            body = normalize_llm(page.text)
            lines.extend([
                "",
                f"### Page {page_number}",
                "",
                body or "_No text extracted._",
            ])

    return "\n".join(lines).strip()


def _render_case_packet(document: OCRDocument, case_index: dict[str, Any]) -> str:
    people = _aggregate_people(case_index["documents"])
    chronology = _aggregate_date_events(case_index["documents"])
    evidence_types = _aggregate_unique(case_index["documents"], "evidence_types")
    topics = _aggregate_unique(case_index["documents"], "topics")
    global_watchlist = list(case_index.get("global_ocr_watchlist") or [])
    for entry in case_index["documents"]:
        global_watchlist.extend(entry.get("ocr_watchlist") or [])
    global_watchlist = _dedupe_preserve_order(global_watchlist)

    lines = [
        "# Defense Workbench",
        "",
        "This packet is an OCR-based organizational aid for counsel review. All names, dates, and quotations should be verified against the source pages before relying on them.",
        "",
        "## Bundle Snapshot",
        "",
        f"- Source PDF: {document.source_pdf}",
        f"- Total pages: {case_index['page_total']}",
        f"- Logical documents inferred: {len(case_index['documents'])}",
    ]
    if case_index.get("bundle_description"):
        lines.append(f"- Bundle description: {case_index['bundle_description']}")

    lines.extend(["", "## Document Map", ""])
    for entry in case_index["documents"]:
        lines.append(
            f"- Pages {entry['page_start']}-{entry['page_end']}: {entry['label']} "
            f"[{entry['document_type']}]"
        )
        if entry.get("summary"):
            lines.append(f"  Summary: {entry['summary']}")
        if entry.get("defense_relevance"):
            lines.append(f"  Defense relevance: {entry['defense_relevance']}")

    if people:
        lines.extend(["", "## People and Roles", ""])
        for item in people:
            if item["roles"]:
                lines.append(f"- {item['name']}: {', '.join(item['roles'])}")
            else:
                lines.append(f"- {item['name']}")

    if chronology:
        lines.extend(["", "## Chronology", ""])
        for item in chronology:
            date_label = item["date"] or "Undated"
            lines.append(
                f"- {date_label}: {item['event']} "
                f"(Document {item['document_id']}, pages {item['page_start']}-{item['page_end']})"
            )

    if topics:
        lines.extend(["", "## Topics", ""])
        lines.extend(f"- {topic}" for topic in topics)

    if evidence_types:
        lines.extend(["", "## Evidence and Materials", ""])
        lines.extend(f"- {item}" for item in evidence_types)

    lines.extend(["", "## Defense-Relevant Review Points", ""])
    for entry in case_index["documents"]:
        if entry.get("defense_relevance"):
            lines.append(
                f"- Pages {entry['page_start']}-{entry['page_end']} ({entry['label']}): "
                f"{entry['defense_relevance']}"
            )

    if global_watchlist:
        lines.extend(["", "## OCR / Verification Watchlist", ""])
        lines.extend(f"- {item}" for item in global_watchlist)

    return "\n".join(lines).strip()


def _aggregate_people(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    people: dict[str, list[str]] = {}
    for document in documents:
        for person in document.get("people") or []:
            name = person.get("name") or ""
            role = person.get("role") or ""
            if not name:
                continue
            bucket = people.setdefault(name, [])
            if role and role not in bucket:
                bucket.append(role)
    return [{"name": name, "roles": roles} for name, roles in people.items()]


def _aggregate_date_events(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for document in documents:
        for event in document.get("date_events") or []:
            if not event.get("date") and not event.get("event"):
                continue
            events.append({
                "document_id": document["id"],
                "page_start": document["page_start"],
                "page_end": document["page_end"],
                "date": event.get("date", ""),
                "event": event.get("event", ""),
            })
    return events


def _aggregate_unique(documents: list[dict[str, Any]], field_name: str) -> list[str]:
    values: list[str] = []
    for document in documents:
        values.extend(document.get(field_name) or [])
    return _dedupe_preserve_order(values)


def _clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _dedupe_preserve_order(
        [str(item).strip() for item in value if str(item).strip()]
    )


def _clean_people(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    people: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            name = _clean_scalar(item.get("name"))
            role = _clean_scalar(item.get("role"))
        else:
            name = _clean_scalar(item)
            role = ""
        if name:
            people.append({"name": name, "role": role})
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in people:
        marker = (item["name"], item["role"])
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(item)
    return deduped


def _clean_date_events(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    events: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        date = _clean_scalar(item.get("date"))
        event = _clean_scalar(item.get("event"))
        if date or event:
            events.append({"date": date, "event": event})
    return events


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
