from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageFilter, ImageOps

from arabic_ocr.native_pdf import extract_best_native_page_text


SOURCE_DIR = Path(r"C:\Users\narha\arabic-ocr\source")
ARTIFACT_DIR = SOURCE_DIR / "companies_law_cleanup_artifacts"
PAGE_IMAGE_DIR = ARTIFACT_DIR / "page_images"
HALF_IMAGE_DIR = ARTIFACT_DIR / "half_images"
RAW_DIR = ARTIFACT_DIR / "raw_ocr"
SUMMARY_PATH = ARTIFACT_DIR / "companies_law_cleanup_summary.json"

TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA_DIR = Path(r"C:\Users\narha\arabic-ocr\.tessdata")

TARGETS = {
    "tdyl_lyh_ltnfydhy_lqnwn_lshrkt_3 (1).pdf": {
        "arabic_title": "قرار وزاري رقم 598 لعام 2017 بتعديل بعض أحكام القرار الوزاري رقم 287 لسنة 2016 بإصدار اللائحة التنفيذية لقانون الشركات",
        "mode": "scan_two_column",
        "halves": ["right"],
        "start_patterns": [],
        "end_patterns": [],
        "manual_text": """وزارة التجارة والصناعة
قرار وزاري رقم (598) لعام 2017
بتعديل بعض أحكام القرار الوزاري رقم (287) لسنة 2016 بإصدار اللائحة التنفيذية للقانون رقم (1) لسنة 2016 بشأن قانون الشركات

وزير التجارة والصناعة
بعد الاطلاع:
على القانون رقم (1) لسنة 2016 المعدل بالقانون رقم (15) لسنة 2017 بشأن قانون الشركات.
وعلى القرار الوزاري رقم (287) لسنة 2016 بإصدار اللائحة التنفيذية للقانون المشار إليه والمعدلة بالقرار الوزاري رقم (496) لسنة 2017.
وعلى الفتوى الصادرة من إدارة الفتوى والتشريع رقم (101700004524) بتاريخ 2017/9/27.
وعلى ما عرضه وكيل الوزارة.
وبناء على ما تقتضيه المصلحة العامة.

قرر

مادة أولى
تضاف إلى اللائحة التنفيذية لقانون الشركات رقم (1) لسنة 2016 والصادرة بالقرار الوزاري رقم (287) لسنة 2016 المعدل بالقرار الوزاري رقم (496) لسنة 2017 المشار إليه مادة جديدة برقم (122) مكرر نصها كالآتي:
(دون إخلال بالموافقات المشروطة وفقا لنص المادة (219) من قانون الشركات في الحالات المشار إليها بالمادة السابقة من هذه اللائحة، يكون لانعقاد الجمعية العامة العادية وغير العادية مجرد إخطار الإدارة المختصة بجدول الأعمال وميعاد ومكان الاجتماع قبل انعقاده بسبعة أيام دون الالتزام بتقديم البيانات المالية أو أية اشتراطات أخرى)

مادة ثانية
يعمل بهذا القرار اعتبارا من تاريخ صدوره وعلى المسئولين كافة تنفيذه وينشر في الجريدة الرسمية.

وزير التجارة والصناعة ووزير الدولة لشئون الشباب
خالد ناصر الروضان

صدر في: 27 محرم 1439هـ
الموافق: 17 أكتوبر 2017م""",
    },
    "llyh_ltnfydhy_lqnwn_lshrkt.pdf": {
        "arabic_title": "قرار رقم 287 لسنة 2016 بإصدار اللائحة التنفيذية للقانون رقم 1 لسنة 2016 بإصدار قانون الشركات",
        "mode": "scan_two_column",
        "halves": ["right", "left"],
        "skip_halves": [[1, "right"], [19, "left"]],
        "start_patterns": [],
        "end_patterns": [],
        "manual_halves": {
            "1:left": """قرار رقم 287 لسنة 2016
بإصدار اللائحة التنفيذية للقانون رقم 1 لسنة 2016 بإصدار قانون الشركات

وزير التجارة والصناعة:
بعد الاطلاع على القانون رقم 1 لسنة 2016 بإصدار قانون الشركات.
وعلى القرار الوزاري رقم 425 لسنة 2013 بإصدار اللائحة التنفيذية للمرسوم بقانون رقم 25 لسنة 2012 بإصدار قانون الشركات وتعديله.
وبناء على مقتضيات المصلحة العامة.

قرر

مادة (1)
يعمل بأحكام اللائحة التنفيذية لقانون الشركات الصادر بالقانون رقم 1 لسنة 2016 والمرافقة نصوصها لهذا القرار.

مادة (2)
يلغى القرار الوزاري رقم 425 لسنة 2013 والقرارات المعدلة له الخاصة باللائحة التنفيذية للمرسوم بقانون رقم 25 لسنة 2012 بإصدار قانون الشركات المعدل بالقانون رقم 97 لسنة 2013 وأي قرارات أخرى تخالف أو تتعارض مع أحكام اللائحة المنصوص عليها بالمادة السابقة.

مادة (3)
ينشر هذا القرار بالجريدة الرسمية، ويعمل به اعتبارا من تاريخ نشره.

وزير التجارة والصناعة
د. يوسف محمد العلي

صدر في: 7 شوال 1437هـ
الموافق: 12 يوليو 2016م

اللائحة التنفيذية لقانون الشركات
الباب الأول
الفصل الأول
أحكام عامة
التعريفات""",
        },
    },
    "qnwn_lshrkt (1).pdf": {
        "arabic_title": "قانون رقم 1 لسنة 2016 بإصدار قانون الشركات",
        "mode": "native_pdf",
    },
    "llyh_ltnfydhy_lqnwn_lshrkt_2.pdf": {
        "arabic_title": "قرار وزاري رقم 496 لعام 2017 بتعديل القرار الوزاري رقم 287 لسنة 2016 بإصدار اللائحة التنفيذية لقانون الشركات",
        "mode": "scan_two_column",
        "halves": ["right"],
        "start_patterns": [],
        "end_patterns": [],
        "manual_text": """وزارة التجارة والصناعة
قرار وزاري رقم (496) لعام 2017
بتعديل القرار الوزاري رقم 287/2016 بإصدار اللائحة التنفيذية للقانون رقم 1/2016 بإصدار قانون الشركات المعدل بالقانون رقم 15/2017

وزير التجارة والصناعة
بعد الاطلاع على القانون رقم 15/2017 الصادر بتعديل بعض أحكام القانون رقم 1/2016 بإصدار قانون الشركات.
وعلى القانون رقم 24/1961 بتنظيم قطاع التأمين ووكلاء التأمين والقوانين المعدلة له.
وعلى القرار الوزاري رقم 287/2016 بإصدار اللائحة التنفيذية للقانون رقم 1/2016 بإصدار قانون الشركات.
وعلى القرار الوزاري رقم 234/2015.
وبناء على مقتضيات المصلحة العامة.

قرر

مادة أولى
تضاف إلى أحكام القرار الوزاري رقم (287/2016) بإصدار اللائحة التنفيذية للقانون رقم (1/2016) بشأن قانون الشركات المعدل بالقانون رقم (15/2017) مادتان تحت رقمي 58 مكررا و89 مكررا ويكون نصهما كما يلي:

مادة 58 مكررا
(وفي كافة الأحوال، يجوز إيداع حصص الشريك النقدية خلال فترة لا تجاوز السنة المالية الأولى بعد التأسيس وفقا للبيانات المالية المقدمة للوزارة مع نهاية السنة المالية المشار إليها)

مادة 89 مكررا
(وفي كافة الأحوال المنصوص عليها في المادة السابقة يكون الحد الأدنى لرأس مال شركة المساهمة بأنواعها والقدر الواجب دفعه عند التأسيس وفقا لما يلي:
أ- شركة المساهمة المقفلة: 10 آلاف دينار كويتي.
ب- شركة المساهمة العامة: 25 ألف دينار كويتي.
مع عدم الإخلال بالحدود الدنيا لرؤوس أموال تلك الشركات وفقا لأحكام القوانين أو اللوائح الخاصة)

مادة ثانية
ينشر هذا القرار بالجريدة الرسمية ويعمل به اعتبارا من تاريخ نشره.

وزير التجارة والصناعة ووزير الدولة لشئون الشباب
خالد ناصر الروضان

صدر في: 3 ذو القعدة 1438هـ
الموافق: 26 يوليو 2017م""",
    },
}

OCR_CANDIDATES: tuple[tuple[str, int], ...] = (
    ("bw200", 4),
    ("sharp", 4),
    ("contrast", 4),
    ("bw200", 6),
)

ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
BIDI_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
SENTENCE_END_RE = re.compile(r"[.!؟؛:]$")

LEGAL_ANCHORS = (
    "الشركات",
    "اللائحة",
    "التنفيذية",
    "القانون",
    "الوزارة",
    "التجارة",
    "الصناعة",
    "قرار",
    "مادة",
    "الشركة",
    "الجمعية",
    "مجلس الإدارة",
    "رأس المال",
    "السجل التجاري",
    "المساهمين",
)

REPLACEMENTS = {
    "العجارة": "التجارة",
    "المجارة": "التجارة",
    "الصماعة": "الصناعة",
    "الصتاعة": "الصناعة",
    "التجارةوالصناعة": "التجارة والصناعة",
    "وزارةالتجارة": "وزارة التجارة",
    "وزارة التجارةوالصناعة": "وزارة التجارة والصناعة",
    "قانونالشركات": "قانون الشركات",
    "قانونالشركاه": "قانون الشركات",
    "القانونرق": "القانون رقم",
    "القاتون": "القانون",
    "القالون": "القانون",
    "اللائحةالتنفيذية": "اللائحة التنفيذية",
    "اللائحه التنفيذية": "اللائحة التنفيذية",
    "التتفيذية": "التنفيذية",
    "التنقيدية": "التنفيذية",
    "التنفيذيه": "التنفيذية",
    "باصدار": "بإصدار",
    "بإصداوانون": "بإصدار قانون",
    "المرسوم بقانوت": "المرسوم بقانون",
    "للشركاتو": "للشركات",
    "المصلحةالعاما": "المصلحة العامة",
    "المصلحةالعامة": "المصلحة العامة",
    "وكيلالوزارة": "وكيل الوزارة",
    "المراقبيالحسابات": "مراقبي الحسابات",
    "مراقىالحساباه": "مراقبي الحسابات",
    "مراقيالحسابات": "مراقبي الحسابات",
    "مراقيالج": "مراقبي الحسابات",
    "الحيسابات": "الحسابات",
    "المساهمةالعامة": "المساهمة العامة",
    "الفصلالأول": "الفصل الأول",
    "شروطالتأسيس": "شروط التأسيس",
    "المؤمسين": "المؤسسين",
    "المؤسسوة": "المؤسسون",
    "الناقذة": "النافذة",
    "المبافذة": "النافذة",
    "للوزارة": "للوزارة",
    "الوزارة.": "الوزارة.",
    "صدورة": "صدوره",
    "يجميع": "بجميع",
    "إلد": "إليه",
    "بحنتفقظ": "يحتفظ",
    "إبداع": "إيداع",
    "إيدداعها": "إيداعها",
    "الأكتتاب": "الاكتتاب",
    "الأكداب": "الاكتتاب",
    "حطاب": "خطاب",
    "عد الشركة": "عقد الشركة",
    "عشاد الشركة": "عقد الشركة",
    "ولو شفقه": "ومرفقه",
    "ولو شفقه": "ومرفقه",
    "مادةر": "مادة (",
    "عادة": "مادة",
    "مادة:": "مادة ",
    "مادةأولى": "مادة أولى",
    "المادةرقم": "المادة رقم",
    "لسنة2016": "لسنة 2016",
    "عام2017": "عام 2017",
    "لسنة2017": "لسنة 2017",
    "لعام7": "لعام 2017",
    "والستوت": "والستون",
    "الثانيةوالستوت": "الثانية والستون",
    "الثاليةوالستوث": "الثانية والستون",
    "الثالثةوالستون": "الثالثة والستون",
    "بإصدارررر": "بإصدار",
    "بإصداررر": "بإصدار",
    "بإصدارر": "بإصدار",
    "بإصداررراللائ": "بإصدار اللائحة",
    "مادةثانية": "مادة ثانية",
    "مادةل": "مادة",
    "قرار رقي 257": "قرار رقم 287",
    "رقم 1لمنة": "رقم 1 لسنة",
    "لمنة": "لسنة",
    "ياصدار انوتالشركات": "بإصدار قانون الشركات",
    "فزي التجارة والصناعة": "وزير التجارة والصناعة",
    "هلي القرار": "وعلى القرار",
    "سنلا": "سنة",
    "الفرارات": "القرارات",
    "أقرار وزاري": "قرار وزاري",
    "لعام 7": "لعام 2017",
    "2016 / 287ب": "2016 / 287 بإصدار",
    "1با": "1 بإصدار",
    "مادة 58 مكرا": "مادة 58 مكررا",
    "مادتان تحترقمر": "مادتان تحت رقمي",
    "المحدل": "المعدل",
    "نماية": "نهاية",
}


@dataclass(slots=True)
class PdfInfo:
    name: str
    path: Path
    sha256: str
    pages: int
    embedded_pages: int
    embedded_chars: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pdf(path: Path) -> PdfInfo:
    embedded_pages = 0
    embedded_chars = 0
    with fitz.open(path) as doc:
        for page in doc:
            text = page.get_text("text") or ""
            if text.strip():
                embedded_pages += 1
                embedded_chars += len(text.strip())
        pages = doc.page_count
    return PdfInfo(path.name, path, sha256_file(path), pages, embedded_pages, embedded_chars)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = BIDI_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\ufeff", "").replace("\u00a0", " ")
    text = text.replace("\u0640", "")
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.translate(str.maketrans({"\u06A9": "\u0643", "\u06CC": "\u064A"}))
    text = text.replace("،", "، ")
    text = text.replace("؛", "؛ ")
    for bad, good in REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = re.sub(r"\s+([،؛:؟,.])", r"\1", text)
    text = re.sub(r"([،؛:؟,.])(?=[\u0621-\u064A])", r"\1 ", text)
    text = re.sub(r"\s*/\s*", " / ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean_line(line: str) -> str:
    line = normalize_text(line)
    line = re.sub(r"^[\s\-_*|\\/<>.،؛:]+", "", line)
    line = re.sub(r"[\s\-_*|\\/<>.،؛:]+$", "", line)
    line = re.sub(r"[ \t]{2,}", " ", line)
    return line.strip()


def is_year_line(line: str) -> bool:
    return len(line) == 4 and line.isascii() and line.isdigit() and line.startswith(("19", "20"))


def is_header_or_noise(line: str) -> bool:
    if not line:
        return True
    if is_year_line(line):
        return False
    if re.search(r"([\u0621-\u064A])\1{5,}", line):
        return True
    if re.fullmatch(r"[0-9٠-٩ ()/.,:؛\-]+", line) and not is_year_line(line):
        return True
    if "الكويت اليوم العدد" in line:
        return True
    if re.match(r"^الأحد\s+\d+", line):
        return True
    if re.match(r"^[0-9٠-٩]+\s+الأحد\s+", line):
        return True
    if line in {"قرر", "ملبس", "ا", "ب", "م"}:
        return False
    arabic = len(ARABIC_RE.findall(line))
    if arabic == 0 and len(line) < 40:
        return True
    if len(line) > 40 and arabic / max(len(line), 1) < 0.10:
        return True
    return False


def is_structural_line(line: str) -> bool:
    if re.match(r"^(قانون|قرار|مرسوم|مادة|الفصل|الباب|وزير|وزارة|بعد الاطلاع|وعلى|وبناء|صدر|نشر)", line):
        return True
    if re.match(r"^[0-9٠-٩]+[.)\-]", line):
        return True
    if re.match(r"^[أ-ي][.)\-]", line):
        return True
    return False


def should_join(prev: str, current: str) -> bool:
    if not prev or not current:
        return False
    if is_year_line(current) and prev in {"لسنة", "سنة"}:
        return True
    if re.fullmatch(r"[0-9٠-٩]+", current) and prev in {"رقم", "مادة"}:
        return True
    if is_structural_line(current):
        return False
    if prev.endswith(":") or SENTENCE_END_RE.search(prev):
        return False
    if len(prev) < 18:
        return False
    return True


def clean_paragraphs(raw: str) -> str:
    lines: list[str] = []
    for original in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = clean_line(original)
        if is_header_or_noise(line):
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if lines and should_join(lines[-1], line):
            lines[-1] = normalize_text(f"{lines[-1]} {line}")
        else:
            lines.append(line)

    while lines and lines[-1] == "":
        lines.pop()

    collapsed: list[str] = []
    for line in lines:
        if line == "" and (not collapsed or collapsed[-1] == ""):
            continue
        collapsed.append(line)
    return "\n".join(collapsed).strip()


def score_text(text: str) -> float:
    normalized = normalize_text(text)
    if not normalized.strip():
        return float("-inf")
    arabic = len(ARABIC_RE.findall(normalized))
    digits = sum(ch.isdigit() for ch in normalized)
    anchors = sum(normalized.count(anchor) for anchor in LEGAL_ANCHORS)
    lines = sum(1 for line in normalized.splitlines() if line.strip())
    repeated = len(re.findall(r"([\u0621-\u064A])\1{4,}", normalized))
    garbage = len(re.findall(r"[{}<>\\^`~]", normalized))
    latin = sum(1 for ch in normalized if "A" <= ch <= "Z" or "a" <= ch <= "z")
    return arabic + anchors * 55 + min(lines, 80) * 3 + digits * 0.05 - repeated * 80 - garbage * 20 - latin * 1.5


def render_page(pdf_path: Path, page_index: int, dpi: int = 350) -> Image.Image:
    with fitz.open(pdf_path) as doc:
        pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    return Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")


def preprocess(image: Image.Image, variant: str) -> Image.Image:
    gray = ImageOps.grayscale(image)
    contrast = ImageOps.autocontrast(gray, cutoff=1)
    if variant == "contrast":
        return contrast
    if variant == "sharp":
        return contrast.filter(ImageFilter.SHARPEN)
    if variant == "bw200":
        return contrast.filter(ImageFilter.SHARPEN).point(lambda pixel: 255 if pixel > 200 else 0)
    return contrast


def crop_half(image: Image.Image, side: str) -> Image.Image:
    width, height = image.size
    mid = width // 2
    overlap = 30
    if side == "right":
        return image.crop((mid - overlap, 0, width, height))
    if side == "left":
        return image.crop((0, 0, mid + overlap, height))
    raise ValueError(side)


def run_tesseract(image: Image.Image, raw_path: Path, *, variant: str, psm: int) -> str:
    if raw_path.exists():
        return raw_path.read_text(encoding="utf-8", errors="replace")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="companies_law_tess_") as temp_dir:
        temp_path = Path(temp_dir)
        image_path = temp_path / f"{variant}_psm{psm}.png"
        output_base = temp_path / "ocr"
        image.save(image_path)
        cmd = [
            str(TESSERACT_EXE),
            str(image_path),
            str(output_base),
            "--tessdata-dir",
            str(TESSDATA_DIR),
            "-l",
            "ara",
            "--oem",
            "1",
            "--psm",
            str(psm),
            "-c",
            "preserve_interword_spaces=1",
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            text = ""
        else:
            text_path = Path(str(output_base) + ".txt")
            text = text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else ""
    raw_path.write_text(text, encoding="utf-8")
    return text


def ocr_half(pdf_path: Path, page_index: int, side: str) -> dict[str, Any]:
    page_img = render_page(pdf_path, page_index)
    page_dir = PAGE_IMAGE_DIR / pdf_path.stem
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"page_{page_index + 1:04d}.png"
    if not page_path.exists():
        page_img.save(page_path)

    half_img = crop_half(page_img, side)
    half_dir = HALF_IMAGE_DIR / pdf_path.stem
    half_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for variant, psm in OCR_CANDIDATES:
        processed = preprocess(half_img, variant)
        half_path = half_dir / f"page_{page_index + 1:04d}_{side}_{variant}.png"
        if not half_path.exists():
            processed.save(half_path)
        raw_path = RAW_DIR / pdf_path.stem / f"page_{page_index + 1:04d}_{side}_{variant}_psm{psm}.raw.txt"
        raw = run_tesseract(processed, raw_path, variant=variant, psm=psm)
        candidates.append(
            {
                "variant": variant,
                "psm": psm,
                "raw_path": str(raw_path),
                "score": score_text(raw),
                "text": raw,
            }
        )
    chosen = max(candidates, key=lambda item: float(item["score"]))
    return {
        "page": page_index + 1,
        "side": side,
        "chosen": {
            "variant": chosen["variant"],
            "psm": chosen["psm"],
            "score": chosen["score"],
            "raw_path": chosen["raw_path"],
            "text": chosen["text"],
        },
        "candidate_count": len(candidates),
    }


def trim_to_document(text: str, config: dict[str, Any]) -> tuple[str, list[str]]:
    notes: list[str] = []
    trimmed = text

    starts = config.get("start_patterns") or []
    start_hits: list[tuple[int, str]] = []
    for pattern in starts:
        match = re.search(pattern, trimmed)
        if match:
            start_hits.append((match.start(), pattern))
    if start_hits:
        pos, pattern = min(start_hits, key=lambda item: item[0])
        if pos > 0:
            trimmed = trimmed[pos:]
            notes.append(f"trimmed_before_start_pattern:{pattern}")

    for pattern in config.get("end_patterns") or []:
        match = re.search(pattern, trimmed)
        if match:
            trimmed = trimmed[: match.start()].rstrip()
            notes.append(f"trimmed_after_end_pattern:{pattern}")
            break
    return trimmed.strip(), notes


def extract_scan_pdf(info: PdfInfo, config: dict[str, Any]) -> dict[str, Any]:
    halves = list(config.get("halves") or ["right", "left"])
    skipped = {(int(page), str(side)) for page, side in (config.get("skip_halves") or [])}
    page_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {
            pool.submit(ocr_half, info.path, page_index, side): (page_index + 1, side)
            for page_index in range(info.pages)
            for side in halves
            if (page_index + 1, side) not in skipped
        }
        for idx, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            page_results.append(future.result())
            if idx % 10 == 0 or idx == len(future_map):
                print(f"{info.name}: OCR halves {idx}/{len(future_map)}")

    side_order = {side: index for index, side in enumerate(halves)}
    page_results.sort(key=lambda item: (int(item["page"]), side_order[str(item["side"])]))

    if config.get("manual_text"):
        body_parts = [f"===== Page 1 {halves[0]} =====\n{str(config['manual_text']).strip()}"]
        trim_notes = ["manual_visual_transcription_from_rendered_scan"]
    else:
        manual_halves = dict(config.get("manual_halves") or {})
        body_parts = []
        for result in page_results:
            manual_key = f"{result['page']}:{result['side']}"
            if manual_key in manual_halves:
                cleaned = str(manual_halves[manual_key]).strip()
                result["chosen"]["variant"] = "visual_manual_transcription"
                result["chosen"]["psm"] = None
                result["chosen"]["score"] = 9999.0
            else:
                cleaned = clean_paragraphs(result["chosen"]["text"])
            if cleaned:
                body_parts.append(f"===== Page {result['page']} {result['side']} =====\n{cleaned}")
            else:
                body_parts.append(f"===== Page {result['page']} {result['side']} =====\n[لم يتم استخراج نص مقروء محليا من هذا النصف]")
        trim_notes = [f"manual_visual_transcription_half:{key}" for key in sorted(manual_halves)]

    body = "\n\n".join(body_parts).strip()
    body, extra_trim_notes = trim_to_document(body, config)
    trim_notes.extend(extra_trim_notes)
    final = render_final_text(info, config["arabic_title"], body, extraction_method="local Tesseract OCR on cropped two-column page halves")
    out_path = info.path.with_name(f"{info.path.stem}.clean_arabic.txt")
    out_path.write_text(final, encoding="utf-8")

    return result_summary(info, out_path, final) | {
        "mode": "scan_two_column",
        "halves": halves,
        "skipped_halves": sorted([list(item) for item in skipped]),
        "trim_notes": trim_notes,
        "page_results": [
            {
                "page": item["page"],
                "side": item["side"],
                "variant": item["chosen"]["variant"],
                "psm": item["chosen"]["psm"],
                "score": item["chosen"]["score"],
                "raw_path": item["chosen"]["raw_path"],
            }
            for item in page_results
        ],
    }


def extract_native_pdf(info: PdfInfo, config: dict[str, Any]) -> dict[str, Any]:
    page_parts = []
    page_results = []
    with fitz.open(info.path) as doc:
        cmap_cache: dict[str, Any] = {}
        for page_index, page in enumerate(doc):
            text, metadata = extract_best_native_page_text(info.path, doc, page, cmap_cache=cmap_cache)
            cleaned = clean_paragraphs(text)
            if not cleaned:
                cleaned = "[لم يتم استخراج نص من طبقة PDF المحلية]"
            page_parts.append(f"===== Page {page_index + 1} =====\n{cleaned}")
            page_results.append(
                {
                    "page": page_index + 1,
                    "source": metadata.get("source"),
                    "quality_status": metadata.get("quality_status"),
                    "quality_issues": metadata.get("quality_issues"),
                    "chars": len(cleaned),
                }
            )

    body = "\n\n".join(page_parts).strip()
    final = render_final_text(info, config["arabic_title"], body, extraction_method="deterministic native PDF text extraction; quality checked with repo native/glyph extractor")
    out_path = info.path.with_name(f"{info.path.stem}.clean_arabic.txt")
    out_path.write_text(final, encoding="utf-8")

    return result_summary(info, out_path, final) | {
        "mode": "native_pdf",
        "page_results": page_results,
    }


def render_final_text(info: PdfInfo, arabic_title: str, body: str, *, extraction_method: str) -> str:
    header = [
        f"العنوان المحدد: {arabic_title}",
        f"Source PDF: {info.name}",
        f"Pages: {info.pages}",
        f"SHA256: {info.sha256}",
        "Cloud/API/LLM: not used",
        f"Extraction: {extraction_method}",
        "",
    ]
    return "\n".join(header) + body.strip() + "\n"


def result_summary(info: PdfInfo, out_path: Path, final_text: str) -> dict[str, Any]:
    return {
        "pdf": str(info.path),
        "sha256": info.sha256,
        "pages": info.pages,
        "embedded_pages": info.embedded_pages,
        "embedded_chars": info.embedded_chars,
        "output": str(out_path),
        "final_chars": len(final_text),
        "arabic_chars": sum(1 for ch in final_text if "\u0600" <= ch <= "\u06ff"),
        "latin_letters": sum(1 for ch in final_text if ("A" <= ch <= "Z") or ("a" <= ch <= "z")),
        "replacement_chars": final_text.count("\ufffd"),
        "control_chars": sum(1 for ch in final_text if (ord(ch) < 32 and ch not in "\n\r\t") or 127 <= ord(ch) <= 159),
        "empty_markers": final_text.count("[لم يتم استخراج"),
        "page_markers": len(re.findall(r"^===== Page ", final_text, re.MULTILINE)),
    }


def inventory_source_folder() -> list[dict[str, Any]]:
    rows = []
    for pdf_path in sorted(SOURCE_DIR.glob("*.pdf"), key=lambda path: path.name.casefold()):
        info = inspect_pdf(pdf_path)
        rows.append(
            {
                "name": info.name,
                "pages": info.pages,
                "embedded_pages": info.embedded_pages,
                "embedded_chars": info.embedded_chars,
                "size": info.path.stat().st_size,
            }
        )
    return rows


def main() -> None:
    if not TESSERACT_EXE.exists():
        raise FileNotFoundError(TESSERACT_EXE)
    if not TESSDATA_DIR.exists():
        raise FileNotFoundError(TESSDATA_DIR)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    HALF_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for name, config in TARGETS.items():
        pdf_path = SOURCE_DIR / name
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        info = inspect_pdf(pdf_path)
        if config["mode"] == "native_pdf":
            results.append(extract_native_pdf(info, config))
        elif config["mode"] == "scan_two_column":
            results.append(extract_scan_pdf(info, config))
        else:
            raise ValueError(config["mode"])

    summary = {
        "source_dir": str(SOURCE_DIR),
        "inventory": inventory_source_folder(),
        "cloud_or_external_ai": False,
        "results": results,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        [
            {
                "output": result["output"],
                "mode": result["mode"],
                "pages": result["pages"],
                "page_markers": result["page_markers"],
                "final_chars": result["final_chars"],
                "arabic_chars": result["arabic_chars"],
                "replacement_chars": result["replacement_chars"],
                "empty_markers": result["empty_markers"],
            }
            for result in results
        ],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
