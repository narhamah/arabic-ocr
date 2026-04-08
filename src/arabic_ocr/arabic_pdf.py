"""Native Arabic PDF extraction with glyph-level recovery for broken CMaps."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz

from arabic_ocr.cmap_fixer import parse_tounicode_cmap

_ARABIC_CHAR_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_BIDI_MARKS_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_ARABIC_WORD_RE = re.compile(r"[\u0600-\u06FF]+")


@dataclass
class NativePageResult:
    page: int
    text: str
    source: str
    has_text_layer: bool


@dataclass
class _TextSpan:
    x: float
    y: float
    glyph_units: list[str]
    is_rtl: bool

    @property
    def text(self) -> str:
        if self.is_rtl:
            return "".join(reversed(self.glyph_units))
        return "".join(self.glyph_units)


def extract_native_pdf(
    pdf_path: str | Path,
    pages: list[int] | None = None,
) -> list[NativePageResult]:
    """Extract text directly from the PDF when a text layer exists."""
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    pypdf_reader = None
    try:
        from pypdf import PdfReader
        pypdf_reader = PdfReader(str(pdf_path))
    except Exception:
        pypdf_reader = None
    page_numbers = pages if pages is not None else list(range(len(doc)))
    cmap_cache: dict[int, dict[int, str]] = {}
    results: list[NativePageResult] = []

    try:
        for page_num in page_numbers:
            page = doc.load_page(page_num)
            pypdf_text = ""
            if pypdf_reader is not None:
                try:
                    pypdf_text = pypdf_reader.pages[page_num].extract_text() or ""
                except Exception:
                    pypdf_text = ""
            text = _extract_page_native(doc, page, cmap_cache, pypdf_text)
            results.append(
                NativePageResult(
                    page=page_num,
                    text=text,
                    source="native" if text else "none",
                    has_text_layer=bool(text.strip()),
                )
            )
    finally:
        doc.close()

    return results


def _extract_page_native(
    doc: fitz.Document,
    page: fitz.Page,
    cmap_cache: dict[int, dict[int, str]],
    pypdf_text: str,
) -> str:
    blocks = _extract_blocks_with_fonts(page)
    page_text = "\n".join(text for text, _font_sizes in blocks if text.strip()).strip()
    if not page_text:
        return clean_arabic_pdf_text(pypdf_text) if pypdf_text.strip() else ""

    glyph_text = extract_page_text_glyph_level(doc, page, cmap_cache)
    best_text = _pick_best_arabic_text(page_text, glyph_text, pypdf_text)
    return clean_arabic_pdf_text(best_text)


def clean_arabic_pdf_text(text: str) -> str:
    """Normalize common Arabic PDF extraction artifacts without flattening content."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = _BIDI_MARKS_RE.sub("", text)
    text = text.replace("\u0640", "")
    text = text.replace("\u060C", ",").replace("\u061B", ";").replace("\u061F", "?")
    text = text.replace("\u066B", ".").replace("\u066C", ",")
    text = _fix_lam_ligatures(text)
    text = _collapse_arabic_letter_spaces(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def contains_arabic(text: str) -> bool:
    return bool(_ARABIC_CHAR_RE.search(text))


def detect_rtl_ratio(text: str) -> float:
    if not text:
        return 0.0
    rtl = 0
    total = 0
    for char in text:
        bidi = unicodedata.bidirectional(char)
        if bidi in ("R", "AL", "AN"):
            rtl += 1
            total += 1
        elif bidi == "L":
            total += 1
    return rtl / total if total else 0.0


def _fix_lam_ligatures(text: str) -> str:
    text = text.replace("\u0627\u0625\u0644", "\u0627\u0644\u0625")
    text = text.replace("\u0627\u0623\u0644", "\u0627\u0644\u0623")
    text = text.replace("\u0627\u0622\u0644", "\u0627\u0644\u0622")
    text = text.replace("\u0627\u0627\u0644", "\u0627\u0644\u0627")
    text = re.sub(r"(?<![\u0600-\u06FF])\u0627\u0645\u0644", "\u0627\u0644\u0645", text)
    return text


def _collapse_arabic_letter_spaces(text: str) -> str:
    pattern = re.compile(
        r"([\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF])"
        r"(?:\s([\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF])){2,}"
    )

    def _collapse(match: re.Match) -> str:
        return re.sub(
            r"\s+(?=[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF])",
            "",
            match.group(0),
        )

    return pattern.sub(_collapse, text)


def _extract_blocks_with_fonts(page: fitz.Page) -> list[tuple[str, list[float]]]:
    result: list[tuple[str, list[float]]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        text_parts: list[str] = []
        all_font_sizes: list[float] = []
        for line in block.get("lines", []):
            line_text, font_sizes = _merge_line_spans(line.get("spans", []))
            if line_text:
                text_parts.append(line_text)
                all_font_sizes.extend(font_sizes)
        text = "\n".join(text_parts).strip()
        if text:
            result.append((text, all_font_sizes))
    return result


def _merge_line_spans(spans: list[dict]) -> tuple[str, list[float]]:
    if not spans:
        return "", []

    font_sizes: list[float] = []
    parts: list[str] = []
    prev_dot_fragment = False

    for span in spans:
        text = span["text"]
        if _is_dot_fragment_span(span, spans):
            stripped = text.lstrip(" ")
            if stripped and parts:
                if parts[-1].endswith(" "):
                    parts[-1] = parts[-1][:-1]
                parts.append(stripped)
            prev_dot_fragment = True
            continue

        if prev_dot_fragment and text.startswith(" "):
            text = text[1:]
        parts.append(text)
        if text.strip():
            font_sizes.append(span["size"])
        prev_dot_fragment = False

    return "".join(parts), font_sizes


def _is_dot_fragment_span(span: dict, all_spans: list[dict]) -> bool:
    if len(span["text"]) > 3:
        return False
    if span["text"].strip():
        return False

    x0, _, x1, _ = span["bbox"]
    for other in all_spans:
        if other is span:
            continue
        other_x0, _, other_x1, _ = other["bbox"]
        if x0 >= other_x0 - 2 and x1 <= other_x1 + 2:
            return True
    return False


def _pick_best_arabic_text(
    pymupdf_text: str,
    glyph_text: str | None,
    pypdf_text: str | None,
) -> str:
    candidates = [("pymupdf", pymupdf_text)]
    if glyph_text and glyph_text.strip() and _glyph_output_is_better(glyph_text, pymupdf_text):
        candidates.append(("glyph", glyph_text))
    if pypdf_text and pypdf_text.strip():
        candidates.append(("pypdf", pypdf_text))

    best_text = pymupdf_text
    best_ratio = _single_letter_ratio(pymupdf_text)
    for _name, text in candidates[1:]:
        ratio = _single_letter_ratio(text)
        if ratio < best_ratio - 0.01:
            best_ratio = ratio
            best_text = text
        elif abs(ratio - best_ratio) <= 0.01 and len(text.strip()) > len(best_text.strip()):
            best_text = text
    return best_text


def _glyph_output_is_better(glyph_text: str, pymupdf_text: str) -> bool:
    if detect_rtl_ratio(pymupdf_text) > 0.1 and detect_rtl_ratio(glyph_text) < detect_rtl_ratio(pymupdf_text) * 0.5:
        return False
    if _avg_arabic_word_length(pymupdf_text) > 2.0 and _avg_arabic_word_length(glyph_text) < _avg_arabic_word_length(pymupdf_text) * 0.5:
        return False
    if _nonstandard_char_ratio(glyph_text) > 0.01 and _nonstandard_char_ratio(glyph_text) > _nonstandard_char_ratio(pymupdf_text) + 0.005:
        return False
    return True


def _single_letter_ratio(text: str) -> float:
    words = [word for word in text.split() if re.search(r"[\u0600-\u06FF]", word)]
    if not words:
        return 0.0
    single = sum(1 for word in words if len(word) == 1)
    return single / len(words)


def _avg_arabic_word_length(text: str) -> float:
    words = _ARABIC_WORD_RE.findall(text)
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)


def _nonstandard_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = 0
    bad = 0
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
            bad += 1
    return bad / total if total else 0.0


def extract_page_text_glyph_level(
    doc: fitz.Document,
    page: fitz.Page,
    cmap_cache: dict[int, dict[int, str]] | None = None,
) -> str | None:
    if cmap_cache is None:
        cmap_cache = {}

    font_cmaps = _load_page_font_cmaps(doc, page, cmap_cache)
    if not _has_multichar_cmaps(font_cmaps):
        return None

    spans = _extract_page_glyph_level(doc, page, font_cmaps)
    if not spans:
        return None

    lines = _group_spans_into_lines(spans)
    text = _lines_to_text(lines)
    return text if text.strip() else None


def _load_page_font_cmaps(
    doc: fitz.Document,
    page: fitz.Page,
    cmap_cache: dict[int, dict[int, str]],
) -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = {}
    for font in page.get_fonts(full=True):
        xref = font[0]
        refname = font[4]
        if xref in cmap_cache:
            result[refname] = cmap_cache[xref]
            continue

        try:
            font_obj = doc.xref_object(xref)
            match = re.search(r"/ToUnicode\s+(\d+)\s+\d+\s+R", font_obj)
            if match:
                raw = doc.xref_stream(int(match.group(1)))
                cmap = parse_tounicode_cmap(raw.decode("utf-8", errors="replace"))
            else:
                cmap = {}
        except Exception:
            cmap = {}

        cmap_cache[xref] = cmap
        result[refname] = cmap
    return result


def _has_multichar_cmaps(font_cmaps: dict[str, dict[int, str]]) -> bool:
    for cmap in font_cmaps.values():
        for chars in cmap.values():
            if len(chars) >= 2 and contains_arabic(chars):
                return True
    return False


def _extract_page_glyph_level(
    doc: fitz.Document,
    page: fitz.Page,
    font_cmaps: dict[str, dict[int, str]],
) -> list[_TextSpan]:
    contents = page.get_contents()
    if not contents:
        return []

    stream_data = b""
    for content_xref in contents:
        try:
            stream_data += doc.xref_stream(content_xref)
        except Exception:
            continue

    if not stream_data:
        return []

    content_text = stream_data.decode("latin-1", errors="replace")
    current_cmap: dict[int, str] = {}
    text_x = 0.0
    text_y = 0.0
    text_scale = 1.0
    spans: list[_TextSpan] = []

    pattern = re.compile(
        r"/(\w+)\s+([\d.]+)\s+Tf"
        r"|(\S+(?:\s+\S+){5})\s+Tm"
        r"|([-\d.]+)\s+([-\d.]+)\s+Td"
        r"|([-\d.]+)\s+([-\d.]+)\s+TD"
        r"|\[([^\]]*)\]\s*TJ"
        r"|<([0-9A-Fa-f]+)>\s*Tj"
    )

    for match in pattern.finditer(content_text):
        if match.group(1):
            current_cmap = font_cmaps.get(match.group(1), {})
        elif match.group(3):
            parsed = _parse_text_matrix(match.group(3))
            if parsed:
                text_x, text_y, text_scale = parsed
        elif match.group(4) is not None:
            text_x += float(match.group(4)) * text_scale
            text_y += float(match.group(5)) * text_scale
        elif match.group(6) is not None:
            text_x += float(match.group(6)) * text_scale
            text_y += float(match.group(7)) * text_scale
        elif match.group(8) is not None:
            units: list[str] = []
            for hex_match in re.finditer(r"<([0-9A-Fa-f]+)>", match.group(8)):
                units.extend(_decode_hex_glyphs(hex_match.group(1), current_cmap))
            if units:
                spans.append(
                    _TextSpan(
                        x=text_x,
                        y=text_y,
                        glyph_units=units,
                        is_rtl=_is_rtl_glyph_run(units),
                    )
                )
        elif match.group(9) is not None:
            units = _decode_hex_glyphs(match.group(9), current_cmap)
            if units:
                spans.append(
                    _TextSpan(
                        x=text_x,
                        y=text_y,
                        glyph_units=units,
                        is_rtl=_is_rtl_glyph_run(units),
                    )
                )

    return spans


def _decode_hex_glyphs(hex_string: str, cmap: dict[int, str]) -> list[str]:
    units: list[str] = []
    for index in range(0, len(hex_string), 4):
        if index + 4 > len(hex_string):
            break
        glyph_id = int(hex_string[index:index + 4], 16)
        chars = cmap.get(glyph_id, "")
        if chars:
            units.append(chars)
    return units


def _is_rtl_glyph_run(units: list[str]) -> bool:
    rtl = 0
    ltr = 0
    for unit in units:
        for char in unit:
            bidi = unicodedata.bidirectional(char)
            if bidi in ("R", "AL", "AN"):
                rtl += 1
            elif bidi == "L":
                ltr += 1
    return rtl > ltr


def _parse_text_matrix(args: str) -> tuple[float, float, float] | None:
    numbers = re.findall(r"[-+]?\d*\.?\d+", args)
    if len(numbers) < 6:
        return None
    a = float(numbers[0])
    c = float(numbers[2])
    e = float(numbers[4])
    f = float(numbers[5])
    return e, f, (a * a + c * c) ** 0.5


def _group_spans_into_lines(
    spans: list[_TextSpan],
    line_tolerance: float = 2.0,
) -> list[list[_TextSpan]]:
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda span: (-span.y, span.x))
    lines: list[list[_TextSpan]] = [[sorted_spans[0]]]
    line_ys = [sorted_spans[0].y]

    for span in sorted_spans[1:]:
        if abs(span.y - line_ys[-1]) <= line_tolerance:
            lines[-1].append(span)
        else:
            lines.append([span])
            line_ys.append(span.y)

    return lines


def _lines_to_text(lines: list[list[_TextSpan]]) -> str:
    if not lines:
        return ""

    line_texts: list[str] = []
    for line in lines:
        rtl_count = sum(1 for span in line if span.is_rtl)
        reverse = rtl_count > len(line) / 2
        ordered = sorted(line, key=lambda span: span.x, reverse=reverse)
        parts: list[str] = []
        for span in ordered:
            text = span.text.strip()
            if not text:
                continue
            if parts and not parts[-1].endswith(" ") and not text.startswith(" "):
                parts.append(" ")
            parts.append(text)
        line_texts.append("".join(parts).strip())

    return "\n".join(text for text in line_texts if text)
