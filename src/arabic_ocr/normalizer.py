"""Arabic text normalization helpers."""

from __future__ import annotations

import re
import unicodedata

_BIDI_CHARS = set(
    "\u200B\u200C\u200D\u200E\u200F"
    "\u202A\u202B\u202C\u202D\u202E"
    "\u2066\u2067\u2068\u2069"
    "\uFEFF"
)
_DIACRITICS_RE = re.compile("[\u064B-\u065F\u0610-\u061A\u0670\u06D6-\u06ED]")
_ALEF_VARIANTS = {
    "\u0623": "\u0627",
    "\u0625": "\u0627",
    "\u0622": "\u0627",
    "\u0671": "\u0627",
}
_EASTERN_NUMERALS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_SPACE_RE = re.compile(r"[^\S\r\n]+")
_BLANK_LINE_RE = re.compile(r"\n{3,}")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str, strip_diacritics: bool = True) -> str:
    """Backwards-compatible normalization used by the legacy pipeline."""
    return _normalize_text(
        text,
        strip_diacritics=strip_diacritics,
        normalize_alef=True,
        normalize_alef_maksura=True,
        convert_eastern_numerals=True,
        preserve_newlines=False,
    )


def normalize_faithful(text: str, strip_diacritics: bool = False) -> str:
    """Minimal normalization for archival text output."""
    return _normalize_text(
        text,
        strip_diacritics=strip_diacritics,
        normalize_alef=False,
        normalize_alef_maksura=False,
        convert_eastern_numerals=False,
        preserve_newlines=True,
    )


def normalize_llm(
    text: str,
    *,
    strip_diacritics: bool = True,
    normalize_alef: bool = True,
    convert_eastern_numerals: bool = True,
) -> str:
    """Safer normalization for LLM-ready context output."""
    return _normalize_text(
        text,
        strip_diacritics=strip_diacritics,
        normalize_alef=normalize_alef,
        normalize_alef_maksura=normalize_alef,
        convert_eastern_numerals=convert_eastern_numerals,
        preserve_newlines=True,
    )


def _normalize_text(
    text: str,
    *,
    strip_diacritics: bool,
    normalize_alef: bool,
    normalize_alef_maksura: bool,
    convert_eastern_numerals: bool,
    preserve_newlines: bool,
) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch not in _BIDI_CHARS)
    text = text.replace("\u0640", "")

    if strip_diacritics:
        text = _DIACRITICS_RE.sub("", text)
    if normalize_alef:
        text = "".join(_ALEF_VARIANTS.get(ch, ch) for ch in text)
    if normalize_alef_maksura:
        text = text.replace("\u0649", "\u064A")
    if convert_eastern_numerals:
        text = text.translate(_EASTERN_NUMERALS)

    if preserve_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _SPACE_RE.sub(" ", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = _BLANK_LINE_RE.sub("\n\n", text)
        return text.strip()

    return _WHITESPACE_RE.sub(" ", text).strip()
