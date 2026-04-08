"""ToUnicode CMap parser used by glyph-level PDF extraction."""

from __future__ import annotations

import re


def parse_tounicode_cmap(cmap_text: str) -> dict[int, str]:
    """Parse a ToUnicode CMap stream into glyph_id -> Unicode text."""
    mapping: dict[int, str] = {}

    def _hex_to_chars(value: str) -> str:
        return "".join(
            chr(int(value[i:i + 4], 16)) for i in range(0, len(value), 4)
        )

    for match in re.finditer(r"<([0-9A-Fa-f]{4})>\s+<([0-9A-Fa-f]+)>", cmap_text):
        mapping[int(match.group(1), 16)] = _hex_to_chars(match.group(2))

    for match in re.finditer(
        r"<([0-9A-Fa-f]{4})>\s+<([0-9A-Fa-f]{4})>\s+<([0-9A-Fa-f]+)>\s*$",
        cmap_text,
        re.MULTILINE,
    ):
        start = int(match.group(1), 16)
        end = int(match.group(2), 16)
        base = [
            int(match.group(3)[i:i + 4], 16)
            for i in range(0, len(match.group(3)), 4)
        ]
        for glyph_id in range(start, end + 1):
            chars = list(base)
            chars[-1] += glyph_id - start
            mapping[glyph_id] = "".join(chr(codepoint) for codepoint in chars)

    for match in re.finditer(
        r"<([0-9A-Fa-f]{4})>\s+<([0-9A-Fa-f]{4})>\s+\[([^\]]+)\]",
        cmap_text,
    ):
        start = int(match.group(1), 16)
        for offset, value in enumerate(re.findall(r"<([0-9A-Fa-f]+)>", match.group(3))):
            mapping[start + offset] = _hex_to_chars(value)

    return mapping
