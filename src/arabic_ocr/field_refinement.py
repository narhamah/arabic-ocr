"""Targeted high-value line refinement using OpenAI."""

from __future__ import annotations

import os

import re

from PIL import Image

from arabic_ocr.models import BoundingBox, OCRBlock, OCRLine, OCRSpan
from arabic_ocr.openai_support import create_client, extract_output_text, image_to_data_url, openai_ready
from arabic_ocr.quality import (
    classify_field_candidate,
    looks_like_hallucinated_ocr_text,
    token_overlap_ratio,
)

_FIELD_REFINEMENT_PROMPT = """You are refining OCR for a single line from an Arabic legal document.

Rules:
- Use the image as the source of truth
- Return only the exact text for this one line
- Preserve wording, names, digits, punctuation, and order exactly
- Do not summarize, explain, translate, or normalize
- Do not add text that is not visible
- If uncertain, stay as close as possible to the original OCR guess
- Output a single line only"""

_LATIN_RE = re.compile(r"[A-Za-z]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_CASE_KEYWORDS = ("القضية", "رقم", "لسنة", "حصر", "نيابة")


def refine_high_value_lines(
    block: OCRBlock,
    *,
    region_crop: Image.Image,
    progress_callback=None,
    page_number: int | None = None,
    block_index: int | None = None,
    page_index: int | None = None,
    page_total: int | None = None,
) -> OCRBlock:
    """Refine likely header/field lines with a stricter OpenAI image pass."""
    if block.engine_id != "openai" or not openai_ready() or not block.lines:
        return block

    refinements: list[dict] = []
    max_candidates = 8
    candidates_seen = 0
    for line_index, line in enumerate(block.lines):
        label = classify_field_candidate(line.text, line_index=line_index, total_lines=len(block.lines))
        if label is None:
            continue
        candidates_seen += 1
        if candidates_seen > max_candidates:
            break

        if progress_callback is not None:
            progress_callback("field_refinement", {
                "page_number": page_number,
                "page_index": page_index,
                "page_total": page_total,
                "block_index": block_index,
                "line_index": line_index,
                "field_label": label,
            })

        candidate_crop = _line_crop(region_crop, line=line, block_bbox=block.bbox)
        refined_text = _refine_line(line.text, candidate_crop, label=label)
        accepted, reason = _should_accept_refinement(line.text, refined_text, label=label)

        refinements.append({
            "line_index": line_index,
            "label": label,
            "original_text": line.text,
            "refined_text": refined_text,
            "accepted": accepted,
            "reason": reason,
        })
        if accepted:
            _apply_refinement(line, refined_text, label=label)

    if refinements:
        accepted_count = sum(1 for item in refinements if item["accepted"])
        block.metadata["field_refinements"] = refinements
        block.metadata["field_refinement_count"] = accepted_count
    return block


def _refine_line(original_text: str, image: Image.Image, *, label: str) -> str:
    client = create_client()
    model = os.getenv("OPENAI_FIELD_MODEL", "gpt-4.1")
    response = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        f"{_FIELD_REFINEMENT_PROMPT}\n\n"
                        f"Line type: {label}\n"
                        f"Original OCR guess: {original_text}"
                    ),
                },
                {"type": "input_image", "image_url": image_to_data_url(image)},
            ],
        }],
    )
    return extract_output_text(response).strip()


def _line_crop(image: Image.Image, *, line: OCRLine, block_bbox: BoundingBox) -> Image.Image:
    width, height = image.size
    if line.bbox is None:
        return image

    pad_y = max(8, height // 80)
    local_y1 = max(0, line.bbox.y1 - block_bbox.y1 - pad_y)
    local_y2 = min(height, line.bbox.y2 - block_bbox.y1 + pad_y)
    return image.crop((0, local_y1, width, max(local_y2, local_y1 + 1)))


def _should_accept_refinement(original_text: str, refined_text: str, *, label: str) -> tuple[bool, str]:
    if not refined_text:
        return False, "empty"
    if looks_like_hallucinated_ocr_text(refined_text):
        return False, "hallucinated"

    original_clean = " ".join(original_text.split())
    refined_clean = " ".join(refined_text.split())
    if not refined_clean:
        return False, "blank_after_cleanup"
    if refined_clean == original_clean:
        return False, "unchanged"

    original_len = max(len(original_clean), 1)
    refined_len = len(refined_clean)
    ratio = refined_len / original_len
    if ratio < 0.45 or ratio > 2.2:
        return False, "length_ratio"

    original_tokens = original_clean.split()
    refined_tokens = refined_clean.split()
    if len(original_tokens) <= 5 and len(refined_tokens) > len(original_tokens) + 3:
        return False, "too_many_new_tokens"

    overlap_ratio = token_overlap_ratio(original_clean, refined_clean)

    if any(char.isdigit() for char in original_clean) and not any(char.isdigit() for char in refined_clean):
        return False, "lost_digits"

    if label == "case_line" and not any(char.isdigit() for char in refined_clean):
        return False, "case_line_without_digits"

    if label in {"header", "title_line"}:
        original_has_latin = bool(_LATIN_RE.search(original_clean))
        refined_has_latin = bool(_LATIN_RE.search(refined_clean))
        original_has_arabic = bool(_ARABIC_RE.search(original_clean))
        refined_has_arabic = bool(_ARABIC_RE.search(refined_clean))

        if original_has_latin != refined_has_latin:
            return False, "script_shift"
        if original_has_arabic != refined_has_arabic:
            return False, "script_shift"
        if not any(char.isdigit() for char in original_clean) and any(char.isdigit() for char in refined_clean):
            return False, "unexpected_digits"
        if overlap_ratio < 0.6:
            return False, "low_token_overlap"
        if "الموافق" in original_clean and "الموافق" not in refined_clean:
            return False, "missing_date_keyword"

    if label == "case_line":
        if any(keyword in original_clean for keyword in _CASE_KEYWORDS) and not any(keyword in refined_clean for keyword in _CASE_KEYWORDS):
            return False, "missing_case_keywords"
        if overlap_ratio < 0.4:
            return False, "low_token_overlap"

    if label == "date_line":
        if "الموافق" in original_clean and "الموافق" not in refined_clean:
            return False, "missing_date_keyword"
        if overlap_ratio < 0.6:
            return False, "low_token_overlap"

    if label == "name_line":
        if "/" in original_clean and "/" not in refined_clean:
            return False, "missing_name_separator"
        if overlap_ratio < 0.5:
            return False, "low_token_overlap"

    return True, "accepted"


def _apply_refinement(line: OCRLine, refined_text: str, *, label: str) -> None:
    line.metadata["original_text"] = line.text
    line.metadata["refined_text"] = refined_text
    line.metadata["field_label"] = label
    line.metadata["refined"] = True
    line.spans = [
        OCRSpan(
            text=refined_text,
            confidence=line.confidence,
            bbox=line.bbox,
            engine_id=line.engine_id,
            adjudicated=True,
            metadata={
                "original_text": line.text,
                "field_label": label,
                "refined": True,
            },
        )
    ]
