from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import json
import re
import statistics
import unicodedata

import fitz
from fontTools.ttLib import TTFont


ROOT = Path(r"C:\Users\narha\arabic-ocr")
PDF_PATH = ROOT / "precedents" / "law104.pdf"
OUT_PATH = ROOT / "precedents" / "law104.clean_arabic.txt"
ARTIFACT_DIR = ROOT / "precedents" / "law104_local_glyph_artifacts"
PAGE_DIR = ARTIFACT_DIR / "pages"
SUMMARY_PATH = ARTIFACT_DIR / "law104_local_glyph_summary.json"

SIMPLIFIED_ARABIC = Path(r"C:\Windows\Fonts\simpo.ttf")
SIMPLIFIED_ARABIC_BOLD = Path(r"C:\Windows\Fonts\simpbdo.ttf")


ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_EXT_RE = re.compile(r"[\u00C0-\u02AF\u1D00-\u1EFF]")
REPLACEMENT = "\uFFFD"
MIRROR = str.maketrans({"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{", "<": ">", ">": "<"})


@dataclass
class FontInfo:
    refname: str
    basefont: str
    font_type: str
    encoding: str
    gid_map: dict[int, str] | None


@dataclass
class Span:
    x: float
    y: float
    size: float
    units: list[str]
    order: int


def decode_glyph_name(name: str) -> str:
    if not name:
        return ""
    name = name.split(".", 1)[0]
    named = {
        "space": " ",
        "period": ".",
        "comma": ",",
        "colon": ":",
        "semicolon": ";",
        "hyphen": "-",
        "minus": "-",
        "slash": "/",
        "plus": "+",
        "percent": "%",
        "parenleft": "(",
        "parenright": ")",
        "bracketleft": "[",
        "bracketright": "]",
        "braceleft": "{",
        "braceright": "}",
        "question": "؟",
        "exclam": "!",
        "quotesingle": "'",
        "quotedbl": '"',
    }
    if name in named:
        return named[name]
    if name.startswith("uni") and len(name) > 3:
        hex_part = name[3:]
        if len(hex_part) % 4 == 0:
            try:
                return "".join(chr(int(hex_part[i : i + 4], 16)) for i in range(0, len(hex_part), 4))
            except ValueError:
                return ""
    if name.startswith("u") and len(name) in (5, 7):
        try:
            return chr(int(name[1:], 16))
        except ValueError:
            return ""
    lotus = decode_lotus_glyph_name(name)
    if lotus:
        return lotus
    return ""


def decode_lotus_glyph_name(name: str) -> str:
    """Decode Lotus/Muna-style Arabic glyph names embedded in index pages."""
    base = name.split(".", 1)[0]
    base = re.sub(r"[IFMUA]$", "", base)
    mapping = {
        "space": " ",
        "return": "",
        "kashidashort": "",
        "kashidashorter": "",
        "ba": "ب",
        "ta": "ت",
        "tha": "ث",
        "jim": "ج",
        "ha": "ح",
        "kha": "خ",
        "sin": "س",
        "shin": "ش",
        "sad": "ص",
        "dad": "ض",
        "ta-": "ط",
        "_ta": "ط",
        "za": "ظ",
        "ayn": "ع",
        "ghayn": "غ",
        "fa": "ف",
        "qaf": "ق",
        "kaf": "ك",
        "lam": "ل",
        "mim": "م",
        "nun": "ن",
        "he": "ه",
        "ha": "ح",
        "waw": "و",
        "ya": "ي",
        "bari_ye": "ي",
        "alif": "ا",
        "alifhamA": "أ",
        "alifhamB": "إ",
        "alifmadda": "آ",
        "alifmaq": "ى",
        "alifmaqham": "ئ",
        "wawham": "ؤ",
        "hamza": "ء",
        "dal": "د",
        "_dal": "د",
        "dhal": "ذ",
        "ra": "ر",
        "_ra": "ر",
        "zay": "ز",
        "tamar": "ة",
        "farsigaf": "گ",
        "gaf": "گ",
        "pa": "پ",
        "pe": "پ",
        "hayc": "ہ",
        "she": "ش",
        "tjim": "چ",
        "nun_e_g": "ں",
        "lamali": "لا",
        "lamalihamA": "لأ",
        "lamalihamB": "لإ",
        "lamalimad": "لآ",
        "lamjim": "لج",
        "lamha": "لح",
        "lamkha": "لخ",
        "lammim": "لم",
        "lammimjim": "لمج",
        "lammimha": "لمح",
        "lammimkha": "لمخ",
        "nunjim": "نج",
        "nunmim": "نم",
        "nunnun": "نن",
        "banun": "بن",
        "tanun": "تن",
        "yanun": "ين",
        "bamim": "بم",
        "tamim": "تم",
        "yamim": "يم",
        "mimmim": "مم",
        "tajim": "تج",
        "taha": "طح",
        "lamalimad": "لآ",
        "zero": "٠",
        "one": "١",
        "two": "٢",
        "three": "٣",
        "four": "٤",
        "five": "٥",
        "six": "٦",
        "seven": "٧",
        "eight": "٨",
        "nine": "٩",
        "decimal": ".",
        "thousandsep": ",",
        "fullpoint": ".",
        "comma": "،",
        "colon": ":",
        "semicolon": "؛",
        "solidus": "/",
        "minus": "-",
        "plus": "+",
        "percent": "%",
        "question": "؟",
        "exclam": "!",
        "asterisk": "*",
        "openparen": "(",
        "closeparen": ")",
        "openbracket": "[",
        "closebracket": "]",
        "openparendec": "(",
        "closeparendec": ")",
        "ellipsis": ".",
        "openchevron": "",
        "closechevron": "",
        "openschevron": "",
        "closeschevron": "",
        "damma": "",
        "dammatan": "",
        "fatha": "",
        "fathatan": "",
        "kasraB": "",
        "kasratanB": "",
        "shadda": "",
        "shaddadam": "",
        "shaddadamtan": "",
        "shaddafat": "",
        "shaddakas": "",
        "shaddakastan": "",
        "sukun": "",
    }
    return mapping.get(base, "")


def load_gid_map(font_path: Path) -> dict[int, str]:
    font = TTFont(str(font_path))
    gid_map: dict[int, str] = {}
    for gid, glyph_name in enumerate(font.getGlyphOrder()):
        text = decode_glyph_name(glyph_name)
        if text:
            text = unicodedata.normalize("NFKC", text)
            text = text.translate(str.maketrans({"\u06A9": "\u0643", "\u06CC": "\u064A", "\uFEFF": ""}))
            gid_map[gid] = text
    return gid_map


REGULAR_GID_MAP = load_gid_map(SIMPLIFIED_ARABIC)
BOLD_GID_MAP = load_gid_map(SIMPLIFIED_ARABIC_BOLD)
EMBEDDED_GID_MAP_CACHE: dict[int, dict[int, str] | None] = {}


def embedded_gid_map(doc: fitz.Document, xref: int) -> dict[int, str] | None:
    if xref in EMBEDDED_GID_MAP_CACHE:
        return EMBEDDED_GID_MAP_CACHE[xref]
    try:
        name, ext, _font_type, content = doc.extract_font(xref)
        if not content or ext.lower() not in {"ttf", "otf"}:
            EMBEDDED_GID_MAP_CACHE[xref] = None
            return None
        font = TTFont(BytesIO(content))
        gid_map: dict[int, str] = {}
        for gid, glyph_name in enumerate(font.getGlyphOrder()):
            text = decode_glyph_name(glyph_name)
            if text:
                text = unicodedata.normalize("NFKC", text)
                text = text.translate(str.maketrans({"\u06A9": "\u0643", "\u06CC": "\u064A", "\uFEFF": ""}))
                gid_map[gid] = text
        EMBEDDED_GID_MAP_CACHE[xref] = gid_map
        return gid_map
    except Exception:
        EMBEDDED_GID_MAP_CACHE[xref] = None
        return None


def font_infos(doc: fitz.Document, page: fitz.Page) -> dict[str, FontInfo]:
    result: dict[str, FontInfo] = {}
    for ft in page.get_fonts(full=True):
        font_type = ft[2] or ""
        basefont = ft[3] or ""
        refname = ft[4] or ""
        encoding = ft[5] or ""
        gid_map = None
        if font_type == "Type0" and encoding == "Identity-H":
            if "simplifiedarabic" in basefont.lower():
                gid_map = BOLD_GID_MAP if "bold" in basefont.lower() else REGULAR_GID_MAP
            else:
                gid_map = embedded_gid_map(doc, ft[0])
        result[refname] = FontInfo(refname, basefont, font_type, encoding, gid_map)
    return result


def parse_text_matrix(args: str) -> tuple[float, float, float] | None:
    nums = re.findall(r"[-+]?\d*\.?\d+", args)
    if len(nums) < 6:
        return None
    a = float(nums[0])
    c = float(nums[2])
    e = float(nums[4])
    f = float(nums[5])
    return e, f, (a * a + c * c) ** 0.5


def pdf_literal_to_text(raw: str) -> str:
    out = bytearray()
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch != "\\":
            out.append(ord(ch) & 0xFF)
            i += 1
            continue
        i += 1
        if i >= len(raw):
            break
        esc = raw[i]
        i += 1
        escapes = {"n": 10, "r": 13, "t": 9, "b": 8, "f": 12, "(": 40, ")": 41, "\\": 92}
        if esc in escapes:
            out.append(escapes[esc])
        elif esc in "\r\n":
            if esc == "\r" and i < len(raw) and raw[i] == "\n":
                i += 1
        elif esc in "01234567":
            octal = esc
            for _ in range(2):
                if i < len(raw) and raw[i] in "01234567":
                    octal += raw[i]
                    i += 1
                else:
                    break
            out.append(int(octal, 8) & 0xFF)
        else:
            out.append(ord(esc) & 0xFF)
    return out.decode("latin-1", errors="replace")


def simple_font_units(text: str) -> list[str]:
    """Keep only structural punctuation from undecoded simple-font operators."""
    units: list[str] = []
    for ch in text:
        if ch == "\u00A1":
            units.append("،")
        elif ch.isspace():
            units.append(" ")
        elif ch in "/\\()[]{}-–—.,:;!?%+*،؛؟":
            units.append(ch)
        elif ch.isdigit() or "\u0660" <= ch <= "\u0669":
            units.append(ch)
    return units


def decode_hex_units(hex_str: str, current_font: FontInfo | None) -> list[str]:
    if not hex_str:
        return []
    if current_font and current_font.gid_map:
        units: list[str] = []
        for i in range(0, len(hex_str), 4):
            if i + 4 > len(hex_str):
                break
            gid = int(hex_str[i : i + 4], 16)
            text = current_font.gid_map.get(gid)
            if text:
                units.append(text)
        return units
    # Simple fonts in this file carry punctuation such as slashes and parentheses.
    data = bytes.fromhex(hex_str[: len(hex_str) - (len(hex_str) % 2)])
    return simple_font_units(data.decode("latin-1", errors="replace"))


def decode_tj_array(array_text: str, current_font: FontInfo | None) -> list[str]:
    units: list[str] = []
    token_re = re.compile(r"<([0-9A-Fa-f]+)>|\(((?:\\.|[^\\()])*)\)", re.S)
    for match in token_re.finditer(array_text):
        if match.group(1) is not None:
            units.extend(decode_hex_units(match.group(1), current_font))
        else:
            text = pdf_literal_to_text(match.group(2))
            if current_font and current_font.gid_map:
                units.extend(text)
            else:
                units.extend(simple_font_units(text))
    return units


CONTENT_RE = re.compile(
    r"/(?P<font>\w+)\s+(?P<size>[-+]?\d*\.?\d+)\s+Tf"
    r"|(?P<tm>(?:[-+]?\d*\.?\d+\s+){5}[-+]?\d*\.?\d+)\s+Tm"
    r"|(?P<tdx>[-+]?\d*\.?\d+)\s+(?P<tdy>[-+]?\d*\.?\d+)\s+Td"
    r"|(?P<TDx>[-+]?\d*\.?\d+)\s+(?P<TDy>[-+]?\d*\.?\d+)\s+TD"
    r"|\[(?P<TJ>[^\]]*)\]\s*TJ"
    r"|<(?P<hex>[0-9A-Fa-f]+)>\s*Tj"
    r"|\((?P<lit>(?:\\.|[^\\()])*)\)\s*Tj"
    r"|(?P<star>T\*)",
    re.S,
)


def extract_spans(doc: fitz.Document, page: fitz.Page) -> list[Span]:
    infos = font_infos(doc, page)
    current_font: FontInfo | None = None
    current_size = 12.0
    x = 0.0
    y = 0.0
    scale = 1.0
    spans: list[Span] = []
    order = 0
    stream_data = b""
    for xref in page.get_contents() or []:
        try:
            stream_data += doc.xref_stream(xref)
        except Exception:
            continue
    content = stream_data.decode("latin-1", errors="replace")
    for match in CONTENT_RE.finditer(content):
        if match.group("font"):
            current_font = infos.get(match.group("font"))
            current_size = float(match.group("size"))
        elif match.group("tm"):
            parsed = parse_text_matrix(match.group("tm"))
            if parsed:
                x, y, scale = parsed
        elif match.group("tdx"):
            x += float(match.group("tdx")) * scale
            y += float(match.group("tdy")) * scale
        elif match.group("TDx"):
            x += float(match.group("TDx")) * scale
            y += float(match.group("TDy")) * scale
        elif match.group("TJ") is not None:
            units = decode_tj_array(match.group("TJ"), current_font)
            if units:
                spans.append(Span(x, y, current_size, units, order))
                order += 1
        elif match.group("hex") is not None:
            units = decode_hex_units(match.group("hex"), current_font)
            if units:
                spans.append(Span(x, y, current_size, units, order))
                order += 1
        elif match.group("lit") is not None:
            text = pdf_literal_to_text(match.group("lit"))
            units = list(text) if current_font and current_font.gid_map else simple_font_units(text)
            if units:
                spans.append(Span(x, y, current_size, units, order))
                order += 1
        elif match.group("star"):
            y -= current_size * 1.2
    return spans


def contains_arabic_unit(units: list[str]) -> bool:
    return any(ARABIC_RE.search(unit) for unit in units)


def group_lines(spans: list[Span], tolerance: float = 2.25) -> list[list[Span]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (-s.y, s.x, s.order))
    lines: list[list[Span]] = []
    current = [ordered[0]]
    current_y = ordered[0].y
    for span in ordered[1:]:
        if abs(span.y - current_y) <= tolerance:
            current.append(span)
        else:
            lines.append(current)
            current = [span]
            current_y = span.y
    lines.append(current)
    return lines


def is_digit_unit(unit: str) -> bool:
    return bool(unit) and all(ch.isdigit() for ch in unit)


def should_insert_visual_space(left: Span, right: Span) -> bool:
    if not left.units or not right.units:
        return False
    if abs(right.x - left.x) < 1.0:
        return False
    last = left.units[-1]
    first = right.units[0]
    if last.isspace() or first.isspace():
        return False
    if last in "/-–—(" or first in "/-–—),.;:؟،":
        return False
    if is_digit_unit(last) or is_digit_unit(first):
        return True
    return abs(right.x - left.x) > max(left.size, right.size) * 0.75


def visual_units_for_line(line: list[Span]) -> list[str]:
    spans = sorted(line, key=lambda s: (s.x, s.order))
    units: list[str] = []
    prev: Span | None = None
    for span in spans:
        if prev is not None and should_insert_visual_space(prev, span):
            units.append(" ")
        units.extend(span.units)
        prev = span
    return units


def logical_rtl_line(line: list[Span]) -> str:
    units = visual_units_for_line(line)
    rev: list[str] = []
    for unit in reversed(units):
        if len(unit) == 1 and unit in "()[]{}<>":
            unit = unit.translate(MIRROR)
        rev.append(unit)

    # Visual Arabic PDFs often store digit runs in display order; restore each run.
    i = 0
    while i < len(rev):
        if is_digit_unit(rev[i]):
            j = i + 1
            while j < len(rev) and is_digit_unit(rev[j]):
                j += 1
            rev[i:j] = list(reversed(rev[i:j]))
            i = j
        else:
            i += 1
    return clean_line("".join(rev))


def logical_ltr_line(line: list[Span]) -> str:
    spans = sorted(line, key=lambda s: (s.x, s.order))
    return clean_line("".join("".join(s.units) for s in spans))


def clean_line(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"\u06A9": "\u0643", "\u06CC": "\u064A"}))
    text = text.replace("\u0640", "")
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
    text = text.replace("¡", "،")
    text = text.replace("Æ", ".")
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([،؛؟,.])", r"\1", text)
    text = re.sub(r"([([])\s+", r"\1", text)
    text = re.sub(r"\s+([])])", r"\1", text)
    text = re.sub(r"([٠-٩0-9])\s*/\s*([٠-٩0-9])", r"\1/\2", text)
    text = re.sub(r"^([٠-٩0-9]+)\s*-\s*", r"\1 - ", text)
    text = re.sub(r"\s+-\s+", " - ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


SPLIT_WORD_CORRECTIONS = {
    "ي وجب": "يوجب",
    "ت وجب": "توجب",
    "ي جب": "يجب",
    "ي عد": "يعد",
    "ت عد": "تعد",
    "ت حم ل": "تحمل",
    "ي لزم": "يلزم",
    "ي ختصم": "يختصم",
    "ي سمح": "يسمح",
    "ي سأل": "يسأل",
    "ي فترض": "يفترض",
    "ي طمأن": "يطمأن",
    "ي فاجىء": "يفاجىء",
    "ي عطي": "يعطي",
    "ي عادل": "يعادل",
    "ي غطيها": "يغطيها",
    "ت فرغ": "تفرغ",
    "ت شترى": "تشترى",
    "ت جرى": "تجرى",
    "ت سأل": "تسأل",
    "س ريانها": "سريانها",
    "س ريان": "سريان",
    "س اعات": "ساعات",
    "س نوات": "سنوات",
    "س بب": "سبب",
    "س قوطها": "سقوطها",
    "س تة": "ستة",
    "ش أن": "شأن",
    "ش أنها": "شأنها",
    "ش مول": "شمول",
    "ش ركات": "شركات",
    "ش ريك": "شريك",
    "ش خصان": "شخصان",
    "ش خصيتها": "شخصيتها",
    "ع دم": "عدم",
    "ه ذا": "هذا",
    "ق بل": "قبل",
    "م كان": "مكان",
    "م حددة": "محددة",
    "م لزم": "ملزم",
    "م ؤدى": "مؤدى",
    "م ؤداه": "مؤداه",
    "ح سن": "حسن",
    "ص دور": "صدور",
    "أ سقط": "أسقط",
    "ب أن": "بأن",
    "و جد": "وجد",
    "ث بت": "ثبت",
    "ط بق": "طبق",
    "ف صل": "فصل",
    "ن سب": "نسب",
    "س لطات": "سلطات",
    "الترامات": "التزامات",
    "الترامه": "التزامه",
    "بالتويض": "بالتعويض",
    "الترامه بالتويض": "التزامه بالتعويض",
    "ا لتعويض": "التعويض",
    "مطالبه جهة": "مطالبة جهة",
    "عمل أخر": "عمل آخر",
    "وجوبأن": "وجوب أن",
    "يجبأن": "يجب أن",
    "لايجوز": "لا يجوز",
    "يتسليم": "بتسليم",
    "اللجنة الثلاثة": "اللجنة الثلاثية",
    "فيعدم": "في عدم",
    "فيعداد": "في عداد",
    "فيعدة": "في عدة",
    "فيالنظام": "في النظام",
    "فيننا": "فيينا",
    "بدونسبب": "بدون سبب",
    "دونسبب": "دون سبب",
    "منسبعة": "من سبعة",
    "المش تراة": "المشتراة",
    "صاحل": "صاحب",
    "النطام": "النظام",
    "سبا للمطالبة": "سببا للمطالبة",
    "حس ابه": "حسابه",
    "المادةة": "المادة",
    "الإلتزلم": "الالتزام",
    "متجرأن": "متجر أن",
    "يرتبعليه": "يترتب عليه",
    "بتقديه": "بتقديره",
    "ا لإكراه": "الإكراه",
    "وسائلالإكراه": "وسائل الإكراه",
    "هذاالوفاء": "هذا الوفاء",
    "هوالإجراء": "هو الإجراء",
    "الإندار": "الإنذار",
    "الش ئون": "الشئون",
}

FINAL_ALIF_SPLIT_RE = re.compile(r"(?<![\u0621-\u064A])([\u0621-\u064A]{3,}) \u0627(?![\u0621-\u064A])")
FINAL_ALIF_EXCEPTIONS = {"على", "إلى", "حتى", "لدى", "متى", "سوى"}


def join_split_final_alif(match: re.Match[str]) -> str:
    word = match.group(1)
    if word in FINAL_ALIF_EXCEPTIONS:
        return match.group(0)
    return f"{word}ا"


def clean_page_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    cleaned: list[str] = []
    for line in lines:
        line = clean_line(line)
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if re.fullmatch(r"[\u064B-\u065F\u0670]+", line):
            continue
        if re.fullmatch(r"%\s*[0-9٠-٩]+", line):
            continue
        line = FINAL_ALIF_SPLIT_RE.sub(join_split_final_alif, line)
        for bad, good in SPLIT_WORD_CORRECTIONS.items():
            line = line.replace(bad, good)
        cleaned.append(line)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return "\n".join(cleaned).strip()


def page_pixel_nonwhite_ratio(page: fitz.Page) -> float:
    pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5), alpha=False, colorspace=fitz.csGRAY)
    data = bytes(pix.samples)
    if not data:
        return 0.0
    return sum(1 for b in data if b < 245) / len(data)


MANUAL_PAGES = {
    1: """وزارة العدل
محكمة التمييز
المكتب الفني

المبادئ القانونية التي قررتها محكمة التمييز خلال أربعين عاما

خلال الفترة من 1/10/1972
حتى 31/12/2011

المجلد التاسع
التجاري والعمالي

رجب 1437 هـ
أبريل 2016 م

(حقوق الطبع محفوظة لوزارة العدل)""",
    3: """بسم الله الرحمن الرحيم

وإذا حكمتم بين الناس
أن تحكموا بالعدل

صدق الله العظيم""",
}


def extract_page(doc: fitz.Document, page_number: int) -> tuple[str, dict[str, object]]:
    page = doc[page_number - 1]
    ratio = page_pixel_nonwhite_ratio(page)
    raw_text = (page.get_text("text") or "").strip()
    if page_number in MANUAL_PAGES:
        text = clean_page_text(MANUAL_PAGES[page_number])
        return text, {
            "page": page_number,
            "source": "manual_visible_page_transcription",
            "raw_chars": len(raw_text),
            "nonwhite_ratio": ratio,
            "chars": len(text),
        }
    if not raw_text and ratio < 0.001:
        return "[صفحة بيضاء في المصدر]", {
            "page": page_number,
            "source": "blank_page_pixel_check",
            "raw_chars": 0,
            "nonwhite_ratio": ratio,
            "chars": 0,
        }

    spans = extract_spans(doc, page)
    lines = group_lines(spans)
    rendered_lines: list[str] = []
    for line in lines:
        line_units: list[str] = []
        for span in line:
            line_units.extend(span.units)
        if contains_arabic_unit(line_units):
            rendered_lines.append(logical_rtl_line(line))
        else:
            rendered_lines.append(logical_ltr_line(line))

    text = clean_page_text("\n".join(line for line in rendered_lines if line.strip()))
    if not text and raw_text:
        text = clean_page_text(raw_text)

    return text or "[لم يتم العثور على نص مقروء في هذه الصفحة]", {
        "page": page_number,
        "source": "custom_local_glyph_decoder_with_simple_font_punctuation",
        "raw_chars": len(raw_text),
        "nonwhite_ratio": ratio,
        "chars": len(text),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF_PATH)
    page_count = doc.page_count
    page_records: list[dict[str, object]] = []
    page_texts: list[str] = []

    try:
        for page_number in range(1, page_count + 1):
            text, meta = extract_page(doc, page_number)
            page_records.append(meta)
            page_file = PAGE_DIR / f"page_{page_number:04d}.txt"
            page_file.write_text(text, encoding="utf-8")
            page_texts.append(f"===== Page {page_number} =====\n{text}".rstrip())
            if page_number % 100 == 0:
                print(f"processed {page_number}/{page_count}")
    finally:
        doc.close()

    body = "\n\n".join(page_texts).strip() + "\n"
    header = (
        f"Source PDF: {PDF_PATH.name}\n"
        f"SHA256: {sha256(PDF_PATH)}\n"
        f"Pages: {page_count}\n"
        "Extraction: local deterministic glyph-ID decoder with simple-font punctuation recovery. "
        "No OpenAI, Gemini, cloud OCR, external LLM, or external API calls.\n"
        "Notes: pages 1 and 3 are visible image/title pages transcribed locally; pages 2, 4, 684, and 780 are blank by pixel check.\n\n"
    )
    OUT_PATH.write_text(header + body, encoding="utf-8")

    final_text = OUT_PATH.read_text(encoding="utf-8")
    marker_count = len(re.findall(r"^===== Page \d+ =====$", final_text, flags=re.M))
    latin_ext = len(LATIN_EXT_RE.findall(final_text))
    replacement = final_text.count(REPLACEMENT)
    arabic_chars = len(ARABIC_RE.findall(final_text))
    page_lengths = [int(record["chars"]) for record in page_records]
    summary = {
        "pdf": str(PDF_PATH),
        "output": str(OUT_PATH),
        "artifacts": str(ARTIFACT_DIR),
        "pages": page_count,
        "page_markers": marker_count,
        "sha256": sha256(PDF_PATH),
        "final_chars": len(final_text),
        "arabic_chars": arabic_chars,
        "latin_extended_glyph_noise": latin_ext,
        "replacement_chars": replacement,
        "empty_text_pages": [r["page"] for r in page_records if int(r["chars"]) == 0],
        "manual_pages": sorted(MANUAL_PAGES),
        "blank_pages": [r["page"] for r in page_records if r["source"] == "blank_page_pixel_check"],
        "min_page_chars": min(page_lengths) if page_lengths else 0,
        "median_page_chars": statistics.median(page_lengths) if page_lengths else 0,
        "max_page_chars": max(page_lengths) if page_lengths else 0,
        "records": page_records,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("output", "pages", "page_markers", "final_chars", "arabic_chars", "latin_extended_glyph_noise", "replacement_chars", "blank_pages")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
