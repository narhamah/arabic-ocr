"""ToUnicode CMap parser for PDF font glyph-to-Unicode mapping.

Parses PDF ToUnicode CMap streams to produce glyph_id -> unicode_chars
mappings.  Used by glyph_extractor.py for content-stream-level text
extraction with correct Arabic ligature handling.
"""

import re


def _parse_tounicode_cmap(cmap_text: str) -> dict[int, str]:
    """Parse a ToUnicode CMap stream into glyph_id -> unicode_chars mapping."""
    gm: dict[int, str] = {}

    def _hex_to_chars(uhex: str) -> str:
        return "".join(
            chr(int(uhex[i : i + 4], 16)) for i in range(0, len(uhex), 4)
        )

    # beginbfchar: <glyph> <unicode>
    for m in re.finditer(r"<([0-9A-Fa-f]{4})>\s+<([0-9A-Fa-f]+)>", cmap_text):
        gm[int(m.group(1), 16)] = _hex_to_chars(m.group(2))

    # beginbfrange (simple): <start> <end> <startUnicode>
    for m in re.finditer(
        r"<([0-9A-Fa-f]{4})>\s+<([0-9A-Fa-f]{4})>\s+<([0-9A-Fa-f]+)>\s*$",
        cmap_text,
        re.MULTILINE,
    ):
        start = int(m.group(1), 16)
        end = int(m.group(2), 16)
        uhex = m.group(3)
        base = [int(uhex[i : i + 4], 16) for i in range(0, len(uhex), 4)]
        for g in range(start, end + 1):
            cl = list(base)
            cl[-1] += g - start
            gm[g] = "".join(chr(c) for c in cl)

    # beginbfrange (array): <start> <end> [<v1> <v2> ...]
    for m in re.finditer(
        r"<([0-9A-Fa-f]{4})>\s+<([0-9A-Fa-f]{4})>\s+\[([^\]]+)\]", cmap_text
    ):
        start = int(m.group(1), 16)
        for i, v in enumerate(re.findall(r"<([0-9A-Fa-f]+)>", m.group(3))):
            gm[start + i] = _hex_to_chars(v)

    return gm
