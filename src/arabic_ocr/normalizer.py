"""Arabic text normalization for OCR post-processing.

Applies transformations in strict order:
1. NFKC Unicode normalization
2. Strip BiDi control characters
3. Strip tatweel/kashida
4. Optionally strip diacritics
5. Normalize Alef variants
6. Normalize Alef Maksura -> Yeh
7. Eastern -> Western numerals
8. Collapse whitespace
"""

import re
import unicodedata

# BiDi control characters to strip
_BIDI_CHARS = set(
    "\u200B\u200C\u200D\u200E\u200F"  # zero-width and directional marks
    "\u202A\u202B\u202C\u202D\u202E"  # bidi embedding/override
    "\u2066\u2067\u2068\u2069"  # bidi isolate
    "\uFEFF"  # BOM / zero-width no-break space
)

# Arabic diacritics (tashkeel/harakat) Unicode ranges
_DIACRITICS_RE = re.compile(
    "[\u064B-\u065F\u0610-\u061A\u0670\u06D6-\u06ED]"
)

# Alef variants -> bare Alef
_ALEF_VARIANTS = {
    "\u0623": "\u0627",  # أ Alef with Hamza above
    "\u0625": "\u0627",  # إ Alef with Hamza below
    "\u0622": "\u0627",  # آ Alef with Madda above
    "\u0671": "\u0627",  # ٱ Alef Wasla
}

# Eastern Arabic numerals -> Western
_EASTERN_NUMERALS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Whitespace collapse
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str, strip_diacritics: bool = True) -> str:
    """Normalize Arabic OCR output text.

    Args:
        text: Raw OCR output text.
        strip_diacritics: If True, remove tashkeel/harakat diacritical marks.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    # 1. NFKC normalization (decomposes presentation forms)
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip BiDi control characters
    text = "".join(ch for ch in text if ch not in _BIDI_CHARS)

    # 3. Strip tatweel/kashida U+0640
    text = text.replace("\u0640", "")

    # 4. Optionally strip diacritics
    if strip_diacritics:
        text = _DIACRITICS_RE.sub("", text)

    # 5. Normalize Alef variants
    text = "".join(_ALEF_VARIANTS.get(ch, ch) for ch in text)

    # 6. Normalize Alef Maksura -> Yeh
    text = text.replace("\u0649", "\u064A")

    # 7. Eastern -> Western numerals
    text = text.translate(_EASTERN_NUMERALS)

    # 8. Collapse whitespace and strip edges
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text
