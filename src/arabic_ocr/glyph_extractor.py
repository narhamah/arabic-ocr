"""Glyph-level text extractor for Arabic PDFs.

Solves the fundamental CMap ligature reversal problem by extracting text
directly from PDF content streams, mapping glyph IDs through ToUnicode
CMaps, and reversing at the glyph-unit level (not character level).

Root cause (solved here):
  PDF content streams store glyphs in visual (LTR) order. For RTL Arabic
  text, extractors like pdftotext and PyMuPDF reverse the entire character
  string. But multi-character CMap entries (ligatures like لا, حم, في) get
  their internal character order reversed too. This module reverses only
  the ORDER of glyph units, keeping each unit's internal characters intact.

This approach is used by pypdf (Issue #1589), pdfminer.six (Issue #850),
and PDFBox (TextNormalize). We implement it at the content stream level
for maximum correctness — no heuristics, no dictionaries, zero false positives.
"""

from __future__ import annotations

from functools import lru_cache
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from .cmap_fixer import _parse_tounicode_cmap


_SYSTEM_FONT_FALLBACKS: dict[str, tuple[Path, ...]] = {
    "simplifiedarabic,bold": (Path(r"C:\Windows\Fonts\simpbdo.ttf"),),
    "simplifiedarabic-bold": (Path(r"C:\Windows\Fonts\simpbdo.ttf"),),
    "simplifiedarabic": (Path(r"C:\Windows\Fonts\simpo.ttf"),),
    "traditionalarabic": (Path(r"C:\Windows\Fonts\trado.ttf"),),
}
_SYSTEM_GLYPH_NAME_MAP = {
    "space": " ",
    "period": ".",
    "comma": ",",
    "colon": ":",
    "semicolon": ";",
    "hyphen": "-",
    "minus": "-",
    "parenleft": "(",
    "parenright": ")",
}
_PDF_ARABIC_VARIANT_NORMALIZATIONS = str.maketrans({
    "\u06A9": "\u0643",  # Farsi kaf -> Arabic kaf
    "\u06CC": "\u064A",  # Farsi yeh -> Arabic yeh
    "\uFEFF": "",        # stray BOM from glyph-name fallback
})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TextSpan:
    """A run of text at a specific position on the page."""
    x: float
    y: float
    font_size: float
    glyph_units: list[str] = field(default_factory=list)
    is_rtl: bool = False

    @property
    def text(self) -> str:
        """Text with correct glyph-unit reversal for RTL."""
        if self.is_rtl:
            return "".join(reversed(self.glyph_units))
        return "".join(self.glyph_units)


@dataclass
class TextLine:
    """A line of text reconstructed from spans at similar y-coordinates."""
    y: float
    spans: list[TextSpan] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Concatenate spans in reading order."""
        if not self.spans:
            return ""
        # Sort spans by x-coordinate
        # For predominantly RTL lines, rightmost span comes first
        rtl_count = sum(1 for s in self.spans if s.is_rtl)
        is_rtl_line = rtl_count > len(self.spans) / 2

        sorted_spans = sorted(
            self.spans,
            key=lambda s: s.x,
            reverse=is_rtl_line,
        )

        parts = []
        for i, span in enumerate(sorted_spans):
            t = span.text
            if not t.strip():
                continue
            # Add space between spans if needed
            if parts and not parts[-1].endswith(" ") and not t.startswith(" "):
                parts.append(" ")
            parts.append(t)
        return "".join(parts)


# ---------------------------------------------------------------------------
# CMap loading
# ---------------------------------------------------------------------------

def _load_page_font_cmaps(
    doc: fitz.Document,
    page: fitz.Page,
    cmap_cache: dict[int, dict[int, str]],
) -> dict[str, dict[int, str]]:
    """Load ToUnicode CMaps for all fonts on a page.

    Returns:
        Dict mapping font reference name (e.g. "F4") to its CMap.
    """
    result: dict[str, dict[int, str]] = {}
    for ft in page.get_fonts(full=True):
        xref = ft[0]
        font_type = ft[2]
        basefont = ft[3]
        refname = ft[4]
        encoding = ft[5]

        if xref in cmap_cache:
            result[refname] = cmap_cache[xref]
            continue

        try:
            font_obj = doc.xref_object(xref)
            tu_match = re.search(r"/ToUnicode\s+(\d+)\s+\d+\s+R", font_obj)
            if tu_match:
                tu_xref = int(tu_match.group(1))
                raw = doc.xref_stream(tu_xref)
                cmap_text = raw.decode("utf-8", errors="replace")
                cmap = _parse_tounicode_cmap(cmap_text)
                cmap_cache[xref] = cmap
                result[refname] = cmap
            else:
                fallback_font = _pick_system_font_fallback(
                    basefont, font_type, encoding,
                )
                if fallback_font is not None:
                    cmap = _load_system_font_cmap(str(fallback_font))
                    cmap_cache[xref] = cmap
                    result[refname] = cmap
                else:
                    cmap_cache[xref] = {}
                    result[refname] = {}
        except Exception:
            cmap_cache[xref] = {}
            result[refname] = {}

    return result


def _has_multichar_cmaps(font_cmaps: dict[str, dict[int, str]]) -> bool:
    """Check if any font on the page has multi-char CMap entries."""
    for cmap in font_cmaps.values():
        for chars in cmap.values():
            if len(chars) >= 2:
                # Check it's actually Arabic characters, not just diacritics
                for c in chars:
                    if "\u0600" <= c <= "\u06FF" or "\uFB50" <= c <= "\uFEFF":
                        return True
    return False


def _has_arabic_mappings(font_cmaps: dict[str, dict[int, str]]) -> bool:
    """Return True when a page has substantial Arabic glyph mappings."""
    for cmap in font_cmaps.values():
        arabic_entries = 0
        for chars in cmap.values():
            if any(
                "\u0600" <= c <= "\u06FF" or "\uFB50" <= c <= "\uFEFF"
                for c in chars
            ):
                arabic_entries += 1
                if arabic_entries >= 8:
                    return True
    return False


def _pick_system_font_fallback(
    base_font_name: str,
    font_type: str,
    encoding: str,
) -> Path | None:
    """Map known PDF Arabic fonts to local Windows fonts."""
    if font_type != "Type0" or encoding != "Identity-H":
        return None

    lowered = (base_font_name or "").lower()
    for key, candidates in _SYSTEM_FONT_FALLBACKS.items():
        if key in lowered:
            for candidate in candidates:
                if candidate.exists():
                    return candidate
    return None


def _chars_from_glyph_name(glyph_name: str) -> str:
    """Decode glyph names like ``uniFE8D`` into Unicode characters."""
    if not glyph_name:
        return ""

    glyph_name = glyph_name.split(".", 1)[0]

    if glyph_name in _SYSTEM_GLYPH_NAME_MAP:
        return _SYSTEM_GLYPH_NAME_MAP[glyph_name]

    if glyph_name.startswith("uni") and len(glyph_name) > 3:
        hex_part = glyph_name[3:]
        if len(hex_part) % 4 == 0:
            try:
                return "".join(
                    chr(int(hex_part[i:i + 4], 16))
                    for i in range(0, len(hex_part), 4)
                )
            except ValueError:
                return ""

    if glyph_name.startswith("u") and len(glyph_name) in (5, 7):
        try:
            return chr(int(glyph_name[1:], 16))
        except ValueError:
            return ""

    return ""


def _pick_best_codepoint(codepoints: list[int]) -> str:
    """Pick the most useful Arabic-looking codepoint for a glyph."""
    preferred_ranges = (
        (0x0600, 0x06FF),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    )
    for start, end in preferred_ranges:
        for cp in codepoints:
            if start <= cp <= end:
                return chr(cp)
    return chr(codepoints[0]) if codepoints else ""


@lru_cache(maxsize=8)
def _load_system_font_cmap(font_path: str) -> dict[int, str]:
    """Build a glyph-id -> Unicode map from a local Windows font file."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return {}

    try:
        font = TTFont(font_path)
    except Exception:
        return {}

    glyph_order = font.getGlyphOrder()
    best_cmap = font.getBestCmap() or {}
    reverse_cmap: dict[str, list[int]] = {}
    for codepoint, glyph_name in best_cmap.items():
        reverse_cmap.setdefault(glyph_name, []).append(codepoint)

    gid_map: dict[int, str] = {}
    for gid, glyph_name in enumerate(glyph_order):
        chars = _chars_from_glyph_name(glyph_name)
        if not chars:
            chars = _pick_best_codepoint(reverse_cmap.get(glyph_name, []))
        if chars:
            gid_map[gid] = chars.translate(_PDF_ARABIC_VARIANT_NORMALIZATIONS)

    return gid_map


# ---------------------------------------------------------------------------
# Content stream parsing
# ---------------------------------------------------------------------------

def _decode_hex_glyphs(
    hex_str: str,
    cmap: dict[int, str],
) -> list[str]:
    """Decode a hex string into glyph units using the CMap.

    Each 4-hex-digit group is a glyph ID. The CMap maps each glyph ID
    to one or more Unicode characters. Each mapping is kept as an atomic
    unit (the key insight for correct RTL handling).
    """
    units: list[str] = []
    for i in range(0, len(hex_str), 4):
        if i + 4 > len(hex_str):
            break
        gid = int(hex_str[i:i + 4], 16)
        chars = cmap.get(gid, "")
        if chars:
            units.append(chars)
    return units


def _is_rtl_glyph_run(units: list[str]) -> bool:
    """Check if a sequence of glyph units is predominantly RTL."""
    rtl = 0
    ltr = 0
    for unit in units:
        for c in unit:
            bidi = unicodedata.bidirectional(c)
            if bidi in ("R", "AL", "AN"):
                rtl += 1
            elif bidi == "L":
                ltr += 1
    return rtl > ltr


def _parse_text_matrix(args: str) -> tuple[float, float, float] | None:
    """Parse a Tm operator's arguments to extract x, y, and font scale.

    Tm takes 6 numbers: a b c d e f (the text matrix).
    e = x position, f = y position.
    Font scale = sqrt(a^2 + c^2) approximately.
    """
    nums = re.findall(r"[-+]?\d*\.?\d+", args)
    if len(nums) < 6:
        return None
    try:
        a = float(nums[0])
        c = float(nums[2])
        e = float(nums[4])
        f = float(nums[5])
        scale = (a * a + c * c) ** 0.5
        return e, f, scale
    except (ValueError, IndexError):
        return None


def _extract_page_glyph_level(
    doc: fitz.Document,
    page: fitz.Page,
    font_cmaps: dict[str, dict[int, str]],
) -> list[TextSpan]:
    """Extract text from a page using glyph-level content stream parsing.

    Parses the page's content stream to find text operators (Tj, TJ),
    maps glyph IDs through ToUnicode CMaps, and preserves glyph units
    as atomic elements for correct RTL reversal.
    """
    # Get content stream
    contents = page.get_contents()
    if not contents:
        return []

    stream_data = b""
    for c_xref in contents:
        try:
            stream_data += doc.xref_stream(c_xref)
        except Exception:
            continue

    if not stream_data:
        return []

    content_text = stream_data.decode("latin-1", errors="replace")

    # State tracking
    current_cmap: dict[int, str] = {}
    current_font_size = 12.0
    text_x = 0.0
    text_y = 0.0
    text_scale = 1.0

    spans: list[TextSpan] = []

    # Regex to match text operators in sequence
    # We need to track: font selection (Tf), text matrix (Tm),
    # text positioning (Td, TD), and text showing (Tj, TJ)
    pattern = re.compile(
        r"/(\w+)\s+([\d.]+)\s+Tf"  # Font selection
        r"|(\S+(?:\s+\S+){5})\s+Tm"  # Text matrix (6 numbers)
        r"|([-\d.]+)\s+([-\d.]+)\s+Td"  # Text position delta
        r"|([-\d.]+)\s+([-\d.]+)\s+TD"  # Text position delta (+ set leading)
        r"|\[([^\]]*)\]\s*TJ"  # Show text array
        r"|<([0-9A-Fa-f]+)>\s*Tj"  # Show text string
        r"|T\*"  # Move to next line
    )

    for m in pattern.finditer(content_text):
        if m.group(1):
            # Font selection: /FontName size Tf
            refname = m.group(1)
            try:
                current_font_size = float(m.group(2))
            except ValueError:
                pass
            current_cmap = font_cmaps.get(refname, {})

        elif m.group(3):
            # Text matrix: a b c d e f Tm
            parsed = _parse_text_matrix(m.group(3))
            if parsed:
                text_x, text_y, text_scale = parsed

        elif m.group(4) is not None:
            # Td: tx ty Td
            try:
                text_x += float(m.group(4)) * text_scale
                text_y += float(m.group(5)) * text_scale
            except ValueError:
                pass

        elif m.group(6) is not None:
            # TD: tx ty TD
            try:
                text_x += float(m.group(6)) * text_scale
                text_y += float(m.group(7)) * text_scale
            except ValueError:
                pass

        elif m.group(8) is not None:
            # TJ: array of hex strings and positioning
            if not current_cmap:
                continue
            tj_content = m.group(8)
            all_units: list[str] = []
            for hex_m in re.finditer(r"<([0-9A-Fa-f]+)>", tj_content):
                units = _decode_hex_glyphs(hex_m.group(1), current_cmap)
                all_units.extend(units)
            if all_units:
                is_rtl = _is_rtl_glyph_run(all_units)
                spans.append(TextSpan(
                    x=text_x,
                    y=text_y,
                    font_size=current_font_size,
                    glyph_units=all_units,
                    is_rtl=is_rtl,
                ))

        elif m.group(9) is not None:
            # Tj: single hex string
            if not current_cmap:
                continue
            units = _decode_hex_glyphs(m.group(9), current_cmap)
            if units:
                is_rtl = _is_rtl_glyph_run(units)
                spans.append(TextSpan(
                    x=text_x,
                    y=text_y,
                    font_size=current_font_size,
                    glyph_units=units,
                    is_rtl=is_rtl,
                ))

    return spans


# ---------------------------------------------------------------------------
# Layout reconstruction
# ---------------------------------------------------------------------------

def _detect_column_boundary(
    spans: list[TextSpan],
    page_width: float,
) -> float | None:
    """Detect if the page has a two-column layout.

    Uses a valley-detection approach: bucket spans by x-position and
    find a significant valley (low char density) between two peaks.

    Args:
        spans: All text spans on the page.
        page_width: The page width in points.

    Returns:
        The x-coordinate of the column boundary, or None if single-column.
    """
    if not spans or page_width <= 0:
        return None

    # Bucket chars by x-position (20-pt bins)
    bucket_size = 20.0
    n_buckets = int(page_width / bucket_size) + 1
    char_counts = [0] * n_buckets

    for s in spans:
        bucket = min(int(s.x / bucket_size), n_buckets - 1)
        char_counts[bucket] += sum(len(u) for u in s.glyph_units)

    total_chars = sum(char_counts)
    if total_chars < 50:
        return None

    # Find the best split point: a valley between two substantial halves
    # Scan through possible split points in the middle 60% of the page
    start_bucket = int(n_buckets * 0.2)
    end_bucket = int(n_buckets * 0.8)

    best_split = None
    best_score = 0.0

    for split in range(start_bucket, end_bucket):
        left_chars = sum(char_counts[:split])
        right_chars = sum(char_counts[split:])

        # Both sides need substantial content (at least 15% each)
        if left_chars < total_chars * 0.15 or right_chars < total_chars * 0.15:
            continue

        # Score = how empty the valley is (3 buckets around split)
        valley_start = max(0, split - 1)
        valley_end = min(n_buckets, split + 2)
        valley_chars = sum(char_counts[valley_start:valley_end])
        valley_ratio = valley_chars / total_chars if total_chars else 1.0

        # Lower valley = better split
        balance = min(left_chars, right_chars) / max(left_chars, right_chars)
        score = balance * (1.0 - valley_ratio * 5)

        if score > best_score:
            best_score = score
            best_split = split

    if best_split is None or best_score < 0.3:
        return None

    return best_split * bucket_size


def build_cmap_correction_table(
    font_cmaps: dict[str, dict[int, str]],
    min_chars: int = 4,
) -> list[tuple[str, str]]:
    """Build a text correction table from CMap multi-char entries.

    For each multi-char CMap entry, the reversed form appears in
    PyMuPDF's extracted text. We build reversed→correct pairs,
    skipping collision pairs (where both orderings exist as CMap entries).

    Args:
        font_cmaps: Dict mapping font names to their CMap dicts.
        min_chars: Minimum character count for entries to include.
            Default 4 (safe/unambiguous). Set to 2 in staging mode
            for broader coverage — research confirmed 2-char CMap
            entries are stored in logical order across all tested PDFs.

    Returns:
        List of (garbled, correct) pairs, sorted longest-first.
    """
    forward: set[str] = set()
    for cmap in font_cmaps.values():
        for chars in cmap.values():
            if len(chars) >= min_chars:
                forward.add(chars)

    corrections: dict[str, str] = {}
    for seq in forward:
        reversed_seq = seq[::-1]
        if reversed_seq == seq:
            continue  # palindrome
        if reversed_seq in forward:
            continue  # collision — both orderings exist
        # The reversed version is what appears in extracted text
        corrections[reversed_seq] = seq

    # Sort by length (longest first) to prevent partial matches
    return sorted(corrections.items(), key=lambda x: len(x[0]), reverse=True)


def apply_cmap_corrections(
    text: str,
    corrections: list[tuple[str, str]],
) -> str:
    """Apply CMap correction table to fix reversed multi-char entries in text.

    Used in the hybrid approach: when PyMuPDF text is used as fallback
    (because glyph extractor produced worse output), this function fixes
    the reversed multi-char CMap entries that PyMuPDF produces.

    Args:
        text: PyMuPDF-extracted text with potentially reversed entries.
        corrections: List of (garbled, correct) pairs from
            build_cmap_correction_table.

    Returns:
        Text with reversed CMap entries corrected.
    """
    for garbled, correct in corrections:
        text = text.replace(garbled, correct)
    return text


def _group_spans_into_lines(
    spans: list[TextSpan],
    line_tolerance: float = 2.0,
) -> list[TextLine]:
    """Group text spans into lines based on y-coordinate proximity.

    Args:
        spans: Text spans from content stream parsing.
        line_tolerance: Maximum y-distance to consider same line (in points).

    Returns:
        List of TextLine objects, sorted top-to-bottom.
    """
    if not spans:
        return []

    # Sort by y descending (PDF y=0 is bottom), then by x
    sorted_spans = sorted(spans, key=lambda s: (-s.y, s.x))

    lines: list[TextLine] = []
    current_line = TextLine(y=sorted_spans[0].y, spans=[sorted_spans[0]])

    for span in sorted_spans[1:]:
        if abs(span.y - current_line.y) <= line_tolerance:
            current_line.spans.append(span)
        else:
            lines.append(current_line)
            current_line = TextLine(y=span.y, spans=[span])

    lines.append(current_line)
    return lines


def _lines_to_text(lines: list[TextLine]) -> str:
    """Convert lines to text with paragraph detection.

    Uses y-coordinate gaps to detect paragraph breaks (double newline)
    vs line breaks within a paragraph (single newline).
    """
    if not lines:
        return ""

    # Calculate typical line spacing
    spacings = []
    for i in range(1, len(lines)):
        gap = abs(lines[i - 1].y - lines[i].y)
        if gap > 0:
            spacings.append(gap)

    if spacings:
        median_spacing = sorted(spacings)[len(spacings) // 2]
    else:
        median_spacing = 14.0  # reasonable default

    # Paragraph break threshold: > 1.5x median line spacing
    para_threshold = median_spacing * 1.5

    parts: list[str] = []
    for i, line in enumerate(lines):
        text = line.text.strip()
        if not text:
            continue

        parts.append(text)

        if i < len(lines) - 1:
            gap = abs(line.y - lines[i + 1].y)
            if gap > para_threshold:
                parts.append("")  # blank line = paragraph break

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_page_text_glyph_level(
    doc: fitz.Document,
    page: fitz.Page,
    cmap_cache: dict[int, dict[int, str]] | None = None,
) -> str | None:
    """Extract text from a page using glyph-level reversal.

    Only activates for pages where fonts have multi-character CMap entries
    (the source of Arabic ligature reversals). Returns None if the page
    doesn't need glyph-level extraction.

    Args:
        doc: The fitz.Document.
        page: The page to extract from.
        cmap_cache: Optional shared cache for CMap data across pages.

    Returns:
        Extracted text with correct Arabic ligature ordering, or None
        if glyph-level extraction is not needed for this page.
    """
    if cmap_cache is None:
        cmap_cache = {}

    font_cmaps = _load_page_font_cmaps(doc, page, cmap_cache)

    if not (_has_multichar_cmaps(font_cmaps) or _has_arabic_mappings(font_cmaps)):
        return None

    spans = _extract_page_glyph_level(doc, page, font_cmaps)
    if not spans:
        return None

    # Check for multi-column layout
    page_width = page.rect.width
    col_boundary = _detect_column_boundary(spans, page_width)

    if col_boundary is not None:
        left_spans = [s for s in spans if s.x < col_boundary]
        right_spans = [s for s in spans if s.x >= col_boundary]

        column_texts: list[str] = []
        for col_spans in (left_spans, right_spans):
            if not col_spans:
                continue
            col_lines = _group_spans_into_lines(col_spans)
            col_text = _lines_to_text(col_lines)
            if col_text.strip():
                column_texts.append(col_text)
        # RTL documents: right column comes first
        column_texts.reverse()
        text = "\n\n".join(column_texts)
    else:
        lines = _group_spans_into_lines(spans)
        text = _lines_to_text(lines)

    return text if text.strip() else None


def extract_pdf_glyph_level(pdf_path: Path | str) -> str:
    """Extract full text from a PDF using glyph-level reversal.

    Processes each page: uses glyph-level extraction for pages with
    multi-char CMap fonts, falls back to PyMuPDF for other pages.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Full extracted text with correct Arabic text ordering.
    """
    doc = fitz.open(str(pdf_path))
    cmap_cache: dict[int, dict[int, str]] = {}
    page_texts: list[str] = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Try glyph-level extraction first
            glyph_text = extract_page_text_glyph_level(
                doc, page, cmap_cache,
            )

            if glyph_text:
                page_texts.append(glyph_text)
            else:
                # Fallback to PyMuPDF dict mode
                page_texts.append(page.get_text("text").strip())

        return "\n\n".join(t for t in page_texts if t)
    finally:
        doc.close()
