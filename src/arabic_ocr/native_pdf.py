"""Glyph-aware native Arabic PDF extraction and OCR comparison helpers."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import fitz

from arabic_ocr.arabic_text import (
    assess_arabic_extraction_quality,
    clean_arabic_text,
    contains_arabic,
    detect_repeated_arabic_char_run_ratio,
    detect_rtl_ratio,
    detect_weird_glyph_char_ratio,
)
from arabic_ocr.glyph_extractor import extract_page_text_glyph_level


def _is_dot_fragment_span(span: dict, all_spans: list[dict]) -> bool:
    """Detect detached dot fragments emitted as separate spans by some fonts."""
    text = span["text"]
    if len(text) > 3:
        return False
    if not text.strip() or len(text.strip()) <= 1:
        bbox = span["bbox"]
        span_x0, span_x1 = bbox[0], bbox[2]
        for other in all_spans:
            if other is span:
                continue
            other_x0, other_x1 = other["bbox"][0], other["bbox"][2]
            if span_x0 >= other_x0 - 2 and span_x1 <= other_x1 + 2:
                return True
    return False


def _merge_line_spans(spans: list[dict]) -> str:
    """Merge line spans while suppressing detached Arabic dot artifacts."""
    if not spans:
        return ""

    parts: list[str] = []
    prev_was_dot_fragment = False
    for span in spans:
        text = span["text"]
        if _is_dot_fragment_span(span, spans):
            stripped = text.lstrip(" ")
            if stripped and parts:
                if parts[-1].endswith(" "):
                    parts[-1] = parts[-1][:-1]
                parts.append(stripped)
            prev_was_dot_fragment = True
            continue

        if prev_was_dot_fragment and text.startswith(" "):
            text = text[1:]
        parts.append(text)
        prev_was_dot_fragment = False

    return "".join(parts)


def _extract_pymupdf_text(page: fitz.Page) -> str:
    """Extract page text using dict mode with Arabic span merging."""
    blocks = page.get_text("dict")["blocks"]
    text_blocks: list[str] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        lines: list[str] = []
        for line in block.get("lines", []):
            line_text = _merge_line_spans(line.get("spans", []))
            if line_text.strip():
                lines.append(line_text)
        block_text = "\n".join(lines).strip()
        if block_text:
            text_blocks.append(block_text)
    if text_blocks:
        return "\n\n".join(text_blocks).strip()
    return (page.get_text("text") or "").strip()


def _has_pdftotext() -> bool:
    """Return whether Poppler's pdftotext is available."""
    return shutil.which("pdftotext") is not None


def _extract_page_pdftotext(pdf_path: Path, page_number: int) -> str | None:
    """Extract a single page through pdftotext when available."""
    try:
        result = subprocess.run(
            [
                "pdftotext",
                str(pdf_path),
                "-",
                "-f", str(page_number + 1),
                "-l", str(page_number + 1),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None
    return re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", result.stdout).strip()


def _single_letter_ratio(text: str) -> float:
    """Ratio of single-character Arabic words to total Arabic words."""
    arabic_words = [word for word in text.split() if re.search(r"[\u0600-\u06FF]", word)]
    if not arabic_words:
        return 0.0
    single_letters = sum(1 for word in arabic_words if len(word) == 1)
    return single_letters / len(arabic_words)


def _score_arabic_text_candidate(text: str) -> float:
    """Score page-level extraction candidates; higher is better."""
    if not text or not text.strip():
        return float("-inf")

    has_arabic = contains_arabic(text)
    rtl_ratio = detect_rtl_ratio(text)
    weird_ratio = detect_weird_glyph_char_ratio(text)
    repeated_run_ratio = detect_repeated_arabic_char_run_ratio(text)
    single_letter_ratio = _single_letter_ratio(text) if has_arabic else 1.0
    length_bonus = min(len(text.strip()), 2000) / 2000.0

    score = length_bonus
    if has_arabic:
        score += 100.0
    if rtl_ratio >= 0.5:
        score += 10.0
    score += rtl_ratio * 10.0
    score -= weird_ratio * 120.0
    score -= repeated_run_ratio * 250.0
    score -= single_letter_ratio * 15.0
    return score


def _score_document_quality(text: str) -> float:
    """Score Arabic document quality for native-vs-OCR comparisons."""
    quality = assess_arabic_extraction_quality(text)
    status_score = {
        "fail": 0.0,
        "not_arabic": 1.0,
        "warn": 2.0,
        "pass": 3.0,
    }.get(quality.status, 0.0)
    metrics = quality.metrics
    return (
        status_score * 1000.0
        + float(metrics.get("rtl_ratio", 0.0)) * 50.0
        + min(float(metrics.get("arabic_char_count", 0.0)), 5000.0) / 100.0
        - float(metrics.get("weird_glyph_ratio", 0.0)) * 1500.0
        - float(metrics.get("suspicious_arabic_char_ratio", 0.0)) * 1200.0
        - float(metrics.get("repeated_arabic_run_ratio", 0.0)) * 1500.0
    )


def _clean_extracted_page_text(page_text: str) -> str:
    """Normalize a page candidate before native-vs-OCR comparison."""
    if contains_arabic(page_text):
        return clean_arabic_text(page_text)
    return page_text.strip()


def _page_looks_like_arabic_font(page: fitz.Page) -> bool:
    """Heuristic for broken text layers backed by Arabic fonts."""
    for font in page.get_fonts(full=True):
        base_font = (font[3] or "").lower()
        encoding = font[5] or ""
        if "simplifiedarabic" in base_font or "traditionalarabic" in base_font:
            return True
        if encoding == "Identity-H" and "arab" in base_font:
            return True
    return False


def _glyph_output_is_better(glyph_text: str, pymupdf_text: str) -> bool:
    """Reject glyph output when it is materially worse than PyMuPDF."""
    glyph_ratio = detect_rtl_ratio(glyph_text)
    pymupdf_ratio = detect_rtl_ratio(pymupdf_text)
    if pymupdf_ratio > 0.1 and glyph_ratio < pymupdf_ratio * 0.5:
        return False

    glyph_avg_word = _average_arabic_word_length(glyph_text)
    pymupdf_avg_word = _average_arabic_word_length(pymupdf_text)
    if pymupdf_avg_word > 2.0 and glyph_avg_word < pymupdf_avg_word * 0.5:
        return False

    glyph_nonstandard = _nonstandard_char_ratio(glyph_text)
    pymupdf_nonstandard = _nonstandard_char_ratio(pymupdf_text)
    if glyph_nonstandard > 0.01 and glyph_nonstandard > pymupdf_nonstandard + 0.005:
        return False

    return True


def _average_arabic_word_length(text: str) -> float:
    """Average length of Arabic-containing words."""
    arabic_words = [word for word in text.split() if re.search(r"[\u0600-\u06FF]", word)]
    if not arabic_words:
        return 0.0
    return sum(len(word) for word in arabic_words) / len(arabic_words)


def _nonstandard_char_ratio(text: str) -> float:
    """Ratio of unusual codepoints likely caused by bad PDF CMaps."""
    if not text:
        return 0.0

    total = 0
    nonstandard = 0
    for char in text:
        if char.isspace():
            continue
        total += 1
        codepoint = ord(char)
        if not (
            codepoint <= 0x024F
            or 0x0600 <= codepoint <= 0x06FF
            or 0x0750 <= codepoint <= 0x077F
            or 0x08A0 <= codepoint <= 0x08FF
            or 0xFB50 <= codepoint <= 0xFDFF
            or 0xFE70 <= codepoint <= 0xFEFF
            or 0x2000 <= codepoint <= 0x206F
        ):
            nonstandard += 1
    return nonstandard / total if total else 0.0


def _quality_summary(text: str) -> dict[str, object]:
    """Compact quality information for diagnostics."""
    quality = assess_arabic_extraction_quality(text)
    return {
        "status": quality.status,
        "issues": quality.issues,
        "metrics": quality.metrics,
        "score": round(_score_document_quality(text), 3),
    }


def extract_best_native_page_text(
    pdf_path: str | Path,
    doc: fitz.Document,
    page: fitz.Page,
    *,
    cmap_cache: dict[int, dict[int, str]] | None = None,
) -> tuple[str, dict[str, object]]:
    """Extract the best native text layer for a page."""
    pdf_path = Path(pdf_path)
    pymupdf_text = _extract_pymupdf_text(page)
    page_maybe_arabic = (
        contains_arabic(pymupdf_text)
        or detect_weird_glyph_char_ratio(pymupdf_text) >= 0.05
        or _page_looks_like_arabic_font(page)
    )

    best_source = "pymupdf"
    best_text = pymupdf_text
    poppler_text = None
    glyph_text = None

    if page_maybe_arabic:
        glyph_candidate = extract_page_text_glyph_level(doc, page, cmap_cache or {})
        if glyph_candidate and (
            (contains_arabic(glyph_candidate) and not contains_arabic(pymupdf_text))
            or detect_weird_glyph_char_ratio(pymupdf_text) >= 0.05
            or _glyph_output_is_better(glyph_candidate, pymupdf_text)
        ):
            glyph_text = glyph_candidate

        if _has_pdftotext():
            poppler_text = _extract_page_pdftotext(pdf_path, page.number)

        candidates = [("pymupdf", pymupdf_text)]
        if poppler_text and poppler_text.strip():
            candidates.append(("pdftotext", poppler_text))
        if glyph_text and glyph_text.strip():
            candidates.append(("glyph", glyph_text))

        best_source, best_text = max(
            candidates,
            key=lambda item: _score_arabic_text_candidate(item[1]),
        )

    cleaned_text = _clean_extracted_page_text(best_text)
    quality = assess_arabic_extraction_quality(cleaned_text)
    metadata = {
        "source": best_source,
        "page_maybe_arabic": page_maybe_arabic,
        "quality_status": quality.status,
        "quality_issues": quality.issues,
        "quality_metrics": quality.metrics,
        "candidate_sources": {
            "pymupdf": bool(pymupdf_text.strip()),
            "pdftotext": bool(poppler_text and poppler_text.strip()),
            "glyph": bool(glyph_text and glyph_text.strip()),
        },
    }
    return cleaned_text, metadata


def should_compare_native_ocr(native_text: str, metadata: dict[str, object]) -> bool:
    """Decide whether a native text page is suspicious enough to OCR-check."""
    if not native_text.strip():
        return False
    if not metadata.get("page_maybe_arabic"):
        return False

    # Hybrid mode is intentionally accuracy-first for Arabic PDFs:
    # always compare the native text layer against OCR, then keep the winner.
    return True


def choose_best_page_text(native_text: str, ocr_text: str) -> tuple[str, dict[str, object]]:
    """Choose the better page text between native extraction and OCR."""
    cleaned_native = _clean_extracted_page_text(native_text)
    cleaned_ocr = _clean_extracted_page_text(ocr_text)

    if not cleaned_ocr:
        return cleaned_native, {
            "winner": "native",
            "native": _quality_summary(cleaned_native),
            "ocr": _quality_summary(""),
            "replaced": False,
            "skipped_reason": "ocr_empty",
        }

    native_score = _score_document_quality(cleaned_native) if contains_arabic(cleaned_native) else _score_arabic_text_candidate(cleaned_native)
    ocr_score = _score_document_quality(cleaned_ocr) if contains_arabic(cleaned_ocr) else _score_arabic_text_candidate(cleaned_ocr)

    if ocr_score > native_score:
        winner = "ocr"
        chosen_text = cleaned_ocr
    else:
        winner = "native"
        chosen_text = cleaned_native

    return chosen_text, {
        "winner": winner,
        "native": _quality_summary(cleaned_native),
        "ocr": _quality_summary(cleaned_ocr),
        "replaced": winner == "ocr",
    }
