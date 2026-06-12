"""Local-only cleanup and manual bundling for OCR case files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from arabic_ocr.models import OCRDocument
from arabic_ocr.normalizer import normalize_faithful, normalize_llm

_LINE_ARTIFACT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("no_text_visible", re.compile(r"^\(?No text is visible in the provided image\.\)?$", re.IGNORECASE)),
    ("no_visible_text_to_extract", re.compile(r"^\(?No visible text to extract\)?$", re.IGNORECASE)),
    (
        "no_visible_printed_or_handwritten_text",
        re.compile(
            r"^\(?No visible printed or handwritten text to extract from the provided image\.\)?$",
            re.IGNORECASE,
        ),
    ),
    (
        "no_visible_or_legible_text",
        re.compile(r"^There is no visible or legible text in the provided image\.$", re.IGNORECASE),
    ),
    (
        "image_contains_no_visible_text",
        re.compile(r"^\[The image contains no visible text\.\]$", re.IGNORECASE),
    ),
    (
        "empty_image_no_text_visible",
        re.compile(r"^\(?empty image, no text visible\)?$", re.IGNORECASE),
    ),
    ("camscanner", re.compile(r".*CamScanner.*", re.IGNORECASE)),
]
_BLANK_LINE_RE = re.compile(r"\n{3,}")
_CHECK_CASE01_SOURCE_MARKERS = (
    "ملف قضية الشيك الكامل",
    "out_check_case01",
)
_CHECK_CASE01_DOCUMENTS: list[dict[str, Any]] = [
    {
        "id": "doc-001",
        "label": "Court session cover sheet",
        "document_type": "court session record",
        "page_start": 1,
        "page_end": 1,
        "confidence": "low",
        "language": "ar",
        "summary": "Opening court-style cover sheet with mixed Kuwaiti and Iraqi-looking headings and session metadata.",
        "defense_relevance": "Low direct relevance; use mainly as a provenance warning that the PDF is a compiled dossier rather than one continuous pleading.",
        "ocr_watchlist": [
            "Header appears mixed between Kuwaiti formatting and a Basra criminal-court reference.",
            "Names and court titles should be verified directly against the scan.",
        ],
    },
    {
        "id": "doc-002",
        "label": "Commercial-prosecution charge sheets",
        "document_type": "charge sheet",
        "page_start": 2,
        "page_end": 5,
        "confidence": "low",
        "language": "ar",
        "summary": "Multiple صحيفة اتهام pages from cheque matters with different case numbers and repeated OCR drift in names and docket references.",
        "defense_relevance": "Important to show the bundle contains several charge sheets and not one single continuous case narrative.",
        "ocr_watchlist": [
            "Case numbers and registry numbers drift heavily across these pages.",
            "Accused name is unstable and should not be quoted from OCR alone.",
        ],
    },
    {
        "id": "doc-003",
        "label": "Commercial cheque investigation minutes",
        "document_type": "investigation minutes",
        "page_start": 6,
        "page_end": 18,
        "confidence": "medium",
        "language": "ar",
        "summary": "Prosecution investigation sessions in a cheque-without-funds matter, including presentation of the accused and questions about cheque issuance, debt, and surrounding dispute facts.",
        "defense_relevance": "Core file for the commercial cheque allegations and the accused's explanations; useful for building the factual timeline and identifying admissions versus denials.",
        "ocr_watchlist": [
            "Numeric fields such as case number, civil ID, and dates vary across pages.",
            "Narrative answers contain notable lexical corruption in Arabic prose.",
        ],
    },
    {
        "id": "doc-004",
        "label": "Additional investigation appendices",
        "document_type": "investigation appendix",
        "page_start": 19,
        "page_end": 20,
        "confidence": "low",
        "language": "ar",
        "summary": "Short evidentiary appendix pages with forensic/police styling and a separate investigator or prosecutor name.",
        "defense_relevance": "Potentially relevant as supporting investigation material, but the OCR is too weak to rely on the exact wording without scan review.",
        "ocr_watchlist": [
            "Page 20 is effectively a stub that points to decisions on the next page.",
            "Page origin and relationship to the main cheque file need confirmation.",
        ],
    },
    {
        "id": "doc-005",
        "label": "Investigation continuation and charging order re cheque 120",
        "document_type": "investigation and order",
        "page_start": 21,
        "page_end": 24,
        "confidence": "medium",
        "language": "ar",
        "summary": "Investigation continuation, accused questioning, and a dated prosecutorial order that appears to charge Nawaf over cheque number 120 and a large amount.",
        "defense_relevance": "Central for tracing the commercial cheque accusation tied to cheque 120 and the prosecution's framing of bad-faith issuance.",
        "ocr_watchlist": [
            "A hallucinated English no-text line was removed from page 24.",
            "Amount, statutory article, and date fields are high-stakes and should be checked against the scan.",
        ],
    },
    {
        "id": "doc-006",
        "label": "Prosecution manager follow-up note",
        "document_type": "administrative note",
        "page_start": 25,
        "page_end": 25,
        "confidence": "medium",
        "language": "ar",
        "summary": "Short manager-level note directing follow-up on case 2264/2025 and asking for completion of inquiry before return with final view.",
        "defense_relevance": "Useful for linking the cheque file to the separate coercion-complaint track and showing parallel prosecutorial handling.",
        "ocr_watchlist": [
            "Page is compressed and partially distorted by scan artifacts.",
        ],
    },
    {
        "id": "doc-007",
        "label": "Separate commercial-prosecution investigation and order",
        "document_type": "investigation and order",
        "page_start": 26,
        "page_end": 28,
        "confidence": "low",
        "language": "ar",
        "summary": "Another short investigation sequence in the commercial prosecution ending with a charging/order page tied to cheque 120 and a multi-million-dinar amount.",
        "defense_relevance": "Useful for mapping parallel proceedings and comparing how cheque 120 was described across documents.",
        "ocr_watchlist": [
            "Page 28 contains severe OCR errors in statutory citation and company name.",
            "Dates appear inconsistent and should be verified from the scan.",
        ],
    },
    {
        "id": "doc-008",
        "label": "Criminal status certificate",
        "document_type": "certificate",
        "page_start": 29,
        "page_end": 29,
        "confidence": "medium",
        "language": "mixed",
        "summary": "Standalone criminal status certificate page from the forensic/criminal-evidence directorate.",
        "defense_relevance": "Background-only unless tied to admissibility or character issues in the defense memo.",
        "ocr_watchlist": [
            "Certificate details are not fully legible in OCR output.",
        ],
    },
    {
        "id": "doc-009",
        "label": "Deputy Attorney General memo on alleged coercion",
        "document_type": "prosecution memo",
        "page_start": 30,
        "page_end": 35,
        "confidence": "low",
        "language": "ar",
        "summary": "Legal memo discussing the coercion allegation tied to signing cheques and citing article 229, with reasoning that appears to undermine the coercion theory.",
        "defense_relevance": "Potentially the strongest exculpatory document in the bundle if authenticated, because it frames the coercion allegation through the prosecutor's legal analysis.",
        "ocr_watchlist": [
            "Arabic prose on pages 30-35 is heavily corrupted and should never be quoted without scan verification.",
            "Case number and date on page 30 are likely wrong as OCR readings.",
        ],
    },
    {
        "id": "doc-010",
        "label": "Execution, travel-ban, detention, and remand records",
        "document_type": "custody and enforcement records",
        "page_start": 36,
        "page_end": 42,
        "confidence": "medium",
        "language": "mixed",
        "summary": "Criminal implementation records, travel-ban request, remand renewal paperwork, detention scheduling, and prison-custody routing pages.",
        "defense_relevance": "Important for custody timeline, procedural posture, and showing how the cheque matter escalated into detention and travel-ban measures.",
        "ocr_watchlist": [
            "Case numbers across these pages may refer to more than one proceeding.",
            "Civil ID values should be verified directly before use.",
        ],
    },
    {
        "id": "doc-011",
        "label": "Cheque image and supporting communications",
        "document_type": "evidence appendix",
        "page_start": 43,
        "page_end": 49,
        "confidence": "high",
        "language": "mixed",
        "summary": "Cheque image, emails requesting a copy of the cheque, chat screenshots, and banking/deposit support pages tied to cheque 120 and related communications.",
        "defense_relevance": "One of the most useful evidentiary sections for reconstructing cheque handling, the voluntary-language chats, and subsequent bank communication.",
        "ocr_watchlist": [
            "Hallucinated no-text lines were removed from pages 43 and 47.",
            "Numeric cheque fields and English email headers should still be checked against the scans.",
        ],
    },
    {
        "id": "doc-012",
        "label": "Police referral and prosecution correspondence",
        "document_type": "administrative correspondence",
        "page_start": 50,
        "page_end": 51,
        "confidence": "medium",
        "language": "ar",
        "summary": "Police referral style page followed by prosecution correspondence addressed to the capital networks office.",
        "defense_relevance": "Helps establish the referral chain and administrative movement of the cheque case between police and prosecution.",
        "ocr_watchlist": [
            "Page 50 appears to contain an older or separate reference number alongside the live file.",
        ],
    },
    {
        "id": "doc-013",
        "label": "Gmail correspondence and settlement WhatsApp extracts",
        "document_type": "correspondence",
        "page_start": 52,
        "page_end": 57,
        "confidence": "high",
        "language": "mixed",
        "summary": "Email chain about cheque 118 and supporting documents, followed by WhatsApp messages discussing issuance of the 350,000 cheque and a blank cheque as part of a dispute resolution narrative.",
        "defense_relevance": "High-value defense material because the chats use language consistent with consensual settlement rather than immediate coercion.",
        "ocr_watchlist": [
            "Screenshots include UI fragments and mixed Arabic/English text.",
            "Speaker attribution in chat screenshots should be confirmed visually.",
        ],
    },
    {
        "id": "doc-014",
        "label": "Police, prosecution, and registry tracking records",
        "document_type": "registry and service records",
        "page_start": 58,
        "page_end": 64,
        "confidence": "medium",
        "language": "ar",
        "summary": "Police escort or custody forms, prosecution letters, justice IT registry printouts, and execution/status notifications.",
        "defense_relevance": "Useful for procedural chronology, service proof, and identifying the exact registry trail of the cheque proceedings.",
        "ocr_watchlist": [
            "Hallucinated no-text line was removed from page 61.",
            "These pages contain several near-duplicate case numbers that require scan confirmation.",
        ],
    },
    {
        "id": "doc-015",
        "label": "Boubyan Petrochemical corporate disclosure records",
        "document_type": "corporate records",
        "page_start": 65,
        "page_end": 66,
        "confidence": "high",
        "language": "mixed",
        "summary": "Corporate disclosure records about Boubyan Petrochemical, including a title-change disclosure mentioning Nawaf and a board-level signatory from the Dabbous side.",
        "defense_relevance": "Useful for background on corporate roles, governance structure, and relationship between the main actors.",
        "ocr_watchlist": [
            "Corporate name and officer titles drift slightly in OCR.",
        ],
    },
    {
        "id": "doc-016",
        "label": "Statement and affidavit in another cheque complaint",
        "document_type": "affidavit",
        "page_start": 67,
        "page_end": 69,
        "confidence": "low",
        "language": "ar",
        "summary": "Affidavit-style statement in a different cheque complaint with a separate complainant or agent, different cheque number, and a different factual setting.",
        "defense_relevance": "Likely collateral rather than central, but it may matter if the prosecution tries to use it to show a broader pattern.",
        "ocr_watchlist": [
            "Complaint number, parties, and bank name appear to belong to another matter.",
            "Do not merge this statement into the main facts without scan confirmation.",
        ],
    },
    {
        "id": "doc-017",
        "label": "Boubyan Bank insufficient-funds letters",
        "document_type": "bank correspondence",
        "page_start": 70,
        "page_end": 71,
        "confidence": "high",
        "language": "mixed",
        "summary": "Boubyan Bank letters to the prosecutor explaining that cheque 120 was not paid because the account lacked sufficient available funds.",
        "defense_relevance": "Strong banking evidence for the return reason and the state of the account on the presentation date.",
        "ocr_watchlist": [
            "Date is rendered as 2023 in OCR and should be verified against the scan and surrounding chronology.",
        ],
    },
    {
        "id": "doc-018",
        "label": "Complaint-intake forms and prosecution checklist exhibits",
        "document_type": "forms and guidance",
        "page_start": 72,
        "page_end": 75,
        "confidence": "low",
        "language": "ar",
        "summary": "Cheque-complaint intake forms and prosecution checklist or guidance material; some forms appear to reference Yemen and Saudi locations or currencies rather than the main Kuwait cheque file.",
        "defense_relevance": "Primarily procedural; also important as a warning that the PDF includes mixed exemplars or unrelated forms.",
        "ocr_watchlist": [
            "Pages 72-74 appear cross-jurisdictional and may not belong to the core Kuwait dispute.",
            "These pages should be treated as ancillary until the scans are reviewed.",
        ],
    },
    {
        "id": "doc-019",
        "label": "Returned-cheque letters and ABK advice for cheque 120",
        "document_type": "banking evidence",
        "page_start": 76,
        "page_end": 81,
        "confidence": "high",
        "language": "mixed",
        "summary": "Boubyan and ABK letters, returned-cheque advice, cheque image, and return-reason records for cheque 120 in the amount of 5,920,000 KWD.",
        "defense_relevance": "This is the strongest banking packet in the file for amount, date, return reason, and named signer.",
        "ocr_watchlist": [
            "Hallucinated no-text line was removed from page 76.",
            "Amount, cheque date, and signer name are critical and should be checked directly before citation.",
        ],
    },
    {
        "id": "doc-020",
        "label": "Detention-renewal tracking sheet",
        "document_type": "administrative tracking sheet",
        "page_start": 82,
        "page_end": 82,
        "confidence": "low",
        "language": "ar",
        "summary": "Single-page detention-renewal tracking table with session placeholders and dates.",
        "defense_relevance": "Collateral scheduling material; useful only if detention chronology becomes contested.",
        "ocr_watchlist": [
            "Table OCR is weak and some date fields appear inconsistent.",
        ],
    },
]
_CHECK_CASE01_BUNDLE_NOTES = [
    "This PDF is a compiled dossier of separate documents, exhibits, and procedural records. It should not be treated as a single continuous document.",
    "The local cleanup removed scanner watermarks and hallucinated no-text filler lines only. It did not perform any LLM correction of body wording.",
    "Several sections appear to belong to adjacent or collateral cheque matters; document relevance should be confirmed against the scanned pages before drafting.",
]
_CHECK_CASE01_GLOBAL_WATCHLIST = [
    "Names drift materially in OCR, especially Nawaf Arhamah / ارحمه variants and Dabbous / دبوس variants.",
    "Case numbers and registry numbers differ across pages and may represent separate proceedings rather than a single docket.",
    "Pages 30-35 contain severe Arabic prose corruption; verify from the scan before quoting the legal analysis.",
    "Pages 72-74 appear to contain cross-jurisdictional cheque forms and may not belong to the core Kuwait dispute.",
    "Pages 79-81 contain high-stakes banking fields such as amount, return reason, and signer name; do not rely on OCR alone for those fields.",
]


@dataclass(slots=True)
class CleanupResult:
    """Result of local page cleanup."""

    cleaned_llm_pages: list[str]
    cleaned_verbatim_pages: list[str]
    removed_counts: dict[str, int]
    removed_pages: dict[str, list[int]]
    empty_pages: list[int]


def clean_page_text(text: str) -> tuple[str, list[str]]:
    """Remove obvious scan artifacts and OCR hallucination lines from one page."""
    if not text:
        return "", []

    removed_labels: list[str] = []
    kept_lines: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        matched_label = next(
            (label for label, pattern in _LINE_ARTIFACT_PATTERNS if pattern.search(stripped)),
            None,
        )
        if matched_label:
            removed_labels.append(matched_label)
            continue
        kept_lines.append(line.rstrip())

    cleaned = "\n".join(kept_lines)
    cleaned = _BLANK_LINE_RE.sub("\n\n", cleaned).strip()
    return cleaned, removed_labels


def clean_document(document: OCRDocument) -> CleanupResult:
    """Clean all pages of an OCR document using local deterministic rules."""
    removed_counts = {label: 0 for label, _ in _LINE_ARTIFACT_PATTERNS}
    removed_pages: dict[str, set[int]] = {label: set() for label, _ in _LINE_ARTIFACT_PATTERNS}
    cleaned_llm_pages: list[str] = []
    cleaned_verbatim_pages: list[str] = []
    empty_pages: list[int] = []

    for page_number, page in enumerate(document.pages, start=1):
        cleaned_raw, removed_labels = clean_page_text(page.text)
        for label in removed_labels:
            removed_counts[label] += 1
            removed_pages[label].add(page_number)
        cleaned_llm = normalize_llm(cleaned_raw)
        cleaned_verbatim = normalize_faithful(cleaned_raw)
        cleaned_llm_pages.append(cleaned_llm)
        cleaned_verbatim_pages.append(cleaned_verbatim)
        if not cleaned_llm.strip():
            empty_pages.append(page_number)

    return CleanupResult(
        cleaned_llm_pages=cleaned_llm_pages,
        cleaned_verbatim_pages=cleaned_verbatim_pages,
        removed_counts=removed_counts,
        removed_pages={label: sorted(pages) for label, pages in removed_pages.items() if pages},
        empty_pages=empty_pages,
    )


def build_manual_case_index(document: OCRDocument, *, preset: str = "check_case01") -> dict[str, Any]:
    """Build a manual case index for a known OCR bundle."""
    if preset != "check_case01":
        raise ValueError(f"Unsupported preset: {preset}")

    marker_text = f"{document.source_pdf} {' '.join(document.metadata.keys())}"
    if not any(marker in marker_text for marker in _CHECK_CASE01_SOURCE_MARKERS) and len(document.pages) != 82:
        raise ValueError("check_case01 preset expects the 82-page cheque case bundle")

    return {
        "bundle_title": "Manual local bundle - cheque case file",
        "bundle_description": (
            "OCR-only rerun of the 82-page cheque-case dossier, cleaned locally and grouped "
            "manually without any post-OCR API review."
        ),
        "source_pdf": document.source_pdf,
        "page_total": len(document.pages),
        "documents": [dict(item) for item in _CHECK_CASE01_DOCUMENTS],
        "bundle_notes": list(_CHECK_CASE01_BUNDLE_NOTES),
        "global_ocr_watchlist": list(_CHECK_CASE01_GLOBAL_WATCHLIST),
    }


def render_cleaned_markdown(
    document: OCRDocument,
    cleaned_pages: list[str],
    *,
    title: str,
    intro: str,
) -> str:
    """Render cleaned pages with explicit page markers."""
    lines = [
        f"# {title}",
        "",
        f"Source PDF: {document.source_pdf}",
        f"Total pages: {len(cleaned_pages)}",
        "",
        intro,
    ]
    for page_number, text in enumerate(cleaned_pages, start=1):
        lines.extend([
            "",
            f"## Page {page_number}",
            "",
            text if text else "(No retained text after local cleanup)",
        ])
    return "\n".join(lines).strip()


def render_manual_case_index_markdown(case_index: dict[str, Any]) -> str:
    """Render a human-readable manual case index."""
    documents = case_index.get("documents") or []
    lines = [
        "# Manual Case Index",
        "",
        "This index was prepared locally from the OCR rerun. No API or model review was used after OCR.",
        "",
        "## Bundle Snapshot",
        "",
        f"- Source PDF: {case_index.get('source_pdf', '')}",
        f"- Total pages: {case_index.get('page_total', 0)}",
        f"- Logical documents: {len(documents)}",
        f"- Bundle description: {case_index.get('bundle_description', '')}",
        "",
        "## Document Map",
    ]
    for entry in documents:
        lines.extend([
            "",
            f"- Pages {entry['page_start']}-{entry['page_end']}: {entry['label']} [{entry['document_type']}]",
            f"  Summary: {entry.get('summary', '')}",
            f"  Defense relevance: {entry.get('defense_relevance', '')}",
        ])
        watchlist = entry.get("ocr_watchlist") or []
        if watchlist:
            lines.append("  OCR watchlist:")
            lines.extend(f"  - {item}" for item in watchlist)

    bundle_notes = case_index.get("bundle_notes") or []
    if bundle_notes:
        lines.extend(["", "## Bundle Notes", ""])
        lines.extend(f"- {item}" for item in bundle_notes)

    global_watchlist = case_index.get("global_ocr_watchlist") or []
    if global_watchlist:
        lines.extend(["", "## Global OCR Watchlist", ""])
        lines.extend(f"- {item}" for item in global_watchlist)

    return "\n".join(lines).strip()


def render_grouped_context(
    document: OCRDocument,
    case_index: dict[str, Any],
    cleaned_pages: list[str],
) -> str:
    """Render cleaned pages grouped into manually classified logical documents."""
    lines = [
        "# Grouped Case File Bundle",
        "",
        "This file was cleaned locally and grouped manually. No API or model review was used after OCR.",
        "",
        f"Source PDF: {document.source_pdf}",
        f"Total pages: {len(cleaned_pages)}",
        f"Logical documents: {len(case_index.get('documents') or [])}",
    ]
    for index, entry in enumerate(case_index.get("documents") or [], start=1):
        lines.extend([
            "",
            f"## Document {index:02d} - {entry['label']}",
            "",
            f"Pages: {entry['page_start']}-{entry['page_end']}",
            f"Type: {entry.get('document_type', 'unknown')}",
            f"Confidence: {entry.get('confidence', 'unknown')}",
            f"Summary: {entry.get('summary', '')}",
            f"Defense relevance: {entry.get('defense_relevance', '')}",
        ])
        watchlist = entry.get("ocr_watchlist") or []
        if watchlist:
            lines.extend(["OCR watchlist:"] + [f"- {item}" for item in watchlist])
        for page_number in range(int(entry["page_start"]), int(entry["page_end"]) + 1):
            lines.extend([
                "",
                f"### Page {page_number}",
                "",
                cleaned_pages[page_number - 1] if cleaned_pages[page_number - 1] else "(No retained text after local cleanup)",
            ])
    return "\n".join(lines).strip()


def render_cleanup_report(document: OCRDocument, cleanup: CleanupResult, *, source_dir: str) -> str:
    """Render a markdown report of local cleanup actions."""
    lines = [
        "# Local Cleanup Report",
        "",
        f"Source OCR directory: {source_dir}",
        f"Source PDF: {document.source_pdf}",
        f"Total pages: {len(document.pages)}",
        f"Pages with no retained text after cleanup: {len(cleanup.empty_pages)}",
    ]
    if cleanup.empty_pages:
        lines.append(f"Empty pages after cleanup: {', '.join(str(page) for page in cleanup.empty_pages)}")

    lines.extend(["", "## Removed Artifact Counts", ""])
    for label, count in cleanup.removed_counts.items():
        pages = cleanup.removed_pages.get(label) or []
        page_text = f" (pages: {', '.join(str(page) for page in pages)})" if pages else ""
        lines.append(f"- `{label}`: {count}{page_text}")

    return "\n".join(lines).strip()


def render_ocr_risk_report(case_index: dict[str, Any], cleanup: CleanupResult) -> str:
    """Render the bundle-level OCR risk report."""
    lines = [
        "# OCR Risk Report",
        "",
        "This report was prepared locally after the OCR rerun. It identifies where the cleaned text is still risky for citation or legal analysis.",
        "",
        "## Cleanup Alerts",
        "",
    ]
    removed_pages = cleanup.removed_pages
    for label in (
        "no_text_visible",
        "no_visible_text_to_extract",
        "no_visible_printed_or_handwritten_text",
        "no_visible_or_legible_text",
        "image_contains_no_visible_text",
        "empty_image_no_text_visible",
    ):
        pages = removed_pages.get(label) or []
        if pages:
            lines.append(f"- Removed `{label}` hallucination lines from pages: {', '.join(str(page) for page in pages)}.")
    camscanner_pages = removed_pages.get("camscanner") or []
    lines.append(f"- Removed scanner watermark lines from {len(camscanner_pages)} pages.")

    global_watchlist = case_index.get("global_ocr_watchlist") or []
    if global_watchlist:
        lines.extend(["", "## Bundle-Level Risks", ""])
        lines.extend(f"- {item}" for item in global_watchlist)

    lines.extend([
        "",
        "## Page-Specific High-Risk Areas",
        "",
        "- Pages 1-5: opening sheets contain unstable court names, charge-sheet headers, and case numbers.",
        "- Pages 24, 43, 47, 61, and 76: OCR inserted hallucinated no-text filler lines that were removed during cleanup.",
        "- Pages 30-35: the coercion-analysis memo is present, but the Arabic prose is heavily degraded and must be checked against the scan before quotation.",
        "- Pages 72-74: complaint-form exhibits contain Yemen and Saudi references that may not belong to the core Kuwait file.",
        "- Pages 79-81: bank return advice and reply letters contain the amount, date, return reason, and signer fields; treat OCR text there as reference only until visually verified.",
        "",
        "## Document Watchlist",
        "",
    ])
    for entry in case_index.get("documents") or []:
        watchlist = entry.get("ocr_watchlist") or []
        if not watchlist:
            continue
        lines.append(f"- Pages {entry['page_start']}-{entry['page_end']} ({entry['label']}): {'; '.join(watchlist)}")

    return "\n".join(lines).strip()
