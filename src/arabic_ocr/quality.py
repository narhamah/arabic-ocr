"""Quality heuristics and simple validators for OCR routing."""

from __future__ import annotations

import re

CASE_NUMBER_RE = re.compile(r"\b\d[\d/\-]{2,}\b")
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
ARTICLE_RE = re.compile(r"(المادة|مادة)\s+\d+")
_HALLUCINATED_OCR_RESPONSES = (
    "there is no visible text in the provided image.",
    "there is no text in the provided image.",
    "there is no visible text in the image provided.",
    "no text is visible in the provided image.",
    "no text visible",
    "no visible text in the provided image.",
    "no visible text found.",
    "no visible text detected in the provided image.",
    "no visible text detected.",
    "no visible text detected",
    "no visible text to extract.",
    "no visible text to extract",
    "no visible text to extract from the provided image.",
    "no visible printed or handwritten text to extract from the provided image.",
    "no visible printed or handwritten text to extract.",
    "sorry, i can't extract any visible text from this image.",
    "sorry, i cannot extract any visible text from this image.",
    "the image appears to be blank or too faint to read.",
    "the image appears blank or too faint to read.",
    "لا يوجد نص مرئي في هذه الصورة.",
    "لا يوجد نص مرئي في الصورة المقدمة.",
    "لا يوجد نص ظاهر في هذه الصورة.",
)
_HEADER_HINT_RE = re.compile(
    r"(النيابة|وزارة|العدل|المحامي العام|prosecution|attorney general|justice|deputy attorney)",
    re.IGNORECASE,
)
_CASE_HINT_RE = re.compile(r"(القضية|رقم|لسنة|حصر|نيابة)", re.IGNORECASE)
_DATE_HINT_RE = re.compile(r"(بتاريخ|الموافق|تاريخ)", re.IGNORECASE)
_NAME_HINT_RE = re.compile(r"(الشاكي|المشكو|المتهم|ضد|المدعي|المدعى|المجني)", re.IGNORECASE)
_SLASHED_NAME_HINT_RE = re.compile(
    r"(الشاكي|المشكو(?:\s+في\s+حقه)?|المتهم|المدعي|المدعى|المجني)\s*/",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[0-9A-Za-z\u0600-\u06FF]+")
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_DIGIT_CHAR_RE = re.compile(r"\d")
_LONG_REPEAT_RE = re.compile(r"(.)\1{4,}")


def token_stats(text: str) -> dict[str, float | int]:
    """Compute lightweight OCR quality heuristics."""
    tokens = text.split()
    if not tokens:
        return {
            "token_count": 0,
            "single_char_ratio": 0.0,
            "digit_fragment_ratio": 0.0,
        }

    single_chars = sum(1 for token in tokens if len(token) == 1)
    digit_fragments = sum(
        1
        for token in tokens
        if any(char.isdigit() for char in token) and len(token) <= 2
    )
    token_count = len(tokens)
    return {
        "token_count": token_count,
        "single_char_ratio": single_chars / token_count,
        "digit_fragment_ratio": digit_fragments / token_count,
    }


def extract_high_value_fields(text: str) -> dict[str, list[str]]:
    """Extract high-value legal fields for downstream validation."""
    return {
        "case_numbers": CASE_NUMBER_RE.findall(text),
        "dates": DATE_RE.findall(text),
        "articles": ARTICLE_RE.findall(text),
    }


def is_low_quality_text(text: str) -> bool:
    """Flag OCR output that likely needs a second opinion."""
    stats = token_stats(text)
    if stats["token_count"] == 0:
        return True
    if stats["single_char_ratio"] > 0.3:
        return True
    if stats["digit_fragment_ratio"] > 0.2:
        return True
    return False


def looks_like_hallucinated_ocr_text(text: str) -> bool:
    """Flag common model-side non-OCR boilerplate responses."""
    normalized = " ".join(text.strip().lower().split())
    normalized = normalized.strip("()[]{} .")
    if not normalized:
        return False
    if normalized.startswith("scanned with"):
        return True
    if "camscanner" in normalized:
        return True
    if "image" in normalized and "text" in normalized:
        if any(token in normalized for token in ("visible", "extract", "extractable", "faint", "blank", "legible", "read")):
            return True
    if "الصورة" in normalized and "نص" in normalized:
        if any(token in normalized for token in ("مرئي", "واضح", "مقروء", "ظاهر")):
            return True
    return any(normalized.startswith(pattern.strip("()[]{} .")) for pattern in _HALLUCINATED_OCR_RESPONSES)


def strip_ocr_boilerplate_lines(text: str) -> str:
    """Remove hallucinated boilerplate and scanner watermark lines from OCR output."""
    if not text:
        return ""

    kept_lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if looks_like_hallucinated_ocr_text(line):
            continue
        kept_lines.append(line.rstrip())

    cleaned = "\n".join(kept_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def normalized_tokens(text: str) -> list[str]:
    """Normalize OCR text into stable tokens for overlap checks."""
    normalized = text.translate(_ARABIC_INDIC_DIGITS).lower().replace("ـ", " ")
    return _TOKEN_RE.findall(normalized)


def token_overlap_ratio(left: str, right: str) -> float:
    """Return the fraction of original tokens preserved in a refinement."""
    left_tokens = normalized_tokens(left)
    right_tokens = set(normalized_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0

    shared = sum(1 for token in left_tokens if token in right_tokens)
    return shared / len(left_tokens)


def classify_field_candidate(text: str, *, line_index: int, total_lines: int) -> str | None:
    """Classify OCR lines that are worth a higher-precision second pass."""
    stripped = text.strip()
    if not stripped:
        return None
    token_count = len(stripped.split())
    top_band = line_index <= min(total_lines // 2, 10)
    has_header_hint = bool(_HEADER_HINT_RE.search(stripped))
    has_case_hint = bool(_CASE_HINT_RE.search(stripped) or CASE_NUMBER_RE.search(stripped))
    has_date_hint = bool(_DATE_HINT_RE.search(stripped) or DATE_RE.search(stripped))
    if top_band and has_header_hint and has_date_hint and token_count > 4:
        return None

    if token_count <= 8 and has_header_hint and top_band:
        return "header"
    if token_count <= 14 and top_band and has_case_hint:
        return "case_line"
    if token_count <= 10 and top_band and has_date_hint:
        # Mixed header/date lines are too unstable for single-line correction.
        if has_header_hint and token_count > 4:
            return None
        return "date_line"
    if token_count <= 10 and _SLASHED_NAME_HINT_RE.search(stripped) and _NAME_HINT_RE.search(stripped):
        return "name_line"
    if line_index <= 2 and token_stats(stripped)["token_count"] <= 8:
        return "title_line"
    return None


def ocr_quality_score(text: str) -> float:
    """Score OCR text quality for merge and retry decisions."""
    stripped = text.strip()
    if not stripped or looks_like_hallucinated_ocr_text(stripped):
        return -1.0

    stats = token_stats(stripped)
    token_count = int(stats["token_count"])
    single_char_ratio = float(stats["single_char_ratio"])
    digit_fragment_ratio = float(stats["digit_fragment_ratio"])
    line_count = sum(1 for line in stripped.splitlines() if line.strip())

    arabic_chars = len(_ARABIC_CHAR_RE.findall(stripped))
    latin_chars = len(_LATIN_CHAR_RE.findall(stripped))
    digit_chars = len(_DIGIT_CHAR_RE.findall(stripped))
    script_ratio = (arabic_chars + latin_chars + digit_chars) / max(len(stripped), 1)

    extracted_fields = extract_high_value_fields(stripped)
    field_bonus = sum(1 for values in extracted_fields.values() if values)
    long_repeat_penalty = len(_LONG_REPEAT_RE.findall(stripped))

    score = 0.0
    score += min(token_count, 120) / 120 * 1.2
    score += min(line_count, 12) / 12 * 0.25
    score += min(script_ratio, 1.0) * 0.45
    score += min(field_bonus, 3) * 0.12
    if arabic_chars:
        score += 0.2

    score -= single_char_ratio * 1.6
    score -= digit_fragment_ratio * 1.3
    score -= min(long_repeat_penalty, 3) * 0.1

    if token_count >= 20 and line_count <= 1:
        score -= 0.2

    return score
