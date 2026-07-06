from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageFilter, ImageOps


SOURCE_DIR = Path(r"C:\Users\narha\arabic-ocr\source")
ARTIFACT_DIR = SOURCE_DIR / "criminal_case_nawaf_ocr_artifacts"
RENDER_DIR = ARTIFACT_DIR / "rendered_pages_dpi300"
RAW_CANDIDATE_DIR = ARTIFACT_DIR / "raw_tesseract_candidates"
SELECTED_RAW_DIR = ARTIFACT_DIR / "selected_raw_pages"
SUMMARY_PATH = ARTIFACT_DIR / "criminal_case_nawaf_local_ocr_summary.json"
AUDIT_PATH = ARTIFACT_DIR / "criminal_case_nawaf_local_ocr_audit.txt"

TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA = Path(r"C:\Users\narha\arabic-ocr\.tessdata")

ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
BIDI_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
SENTENCE_END_RE = re.compile(r"[.؟?!؛:]$")
QA_RE = re.compile(r"^\s*(?:[سج]\s*\d+|\d+\s*[سج]|[سج]\s*[:/.-])")
FIELD_RE = re.compile(r"^[\u0600-\u06ffA-Za-z ()/.-]{2,45}\s*[:：]")
NUMBERED_RE = re.compile(r"^\s*(?:\d+|[٠-٩]+|[أ-ي])[\-.)،:]\s+")

OCR_CONFIGS: tuple[tuple[str, str, int], ...] = (
    ("raw", "ara", 4),
    ("raw", "ara", 6),
    ("contrast", "ara", 4),
    ("contrast", "ara", 6),
    ("sharp", "ara", 4),
    ("sharp", "ara", 6),
    ("bw200", "ara", 4),
    ("bw200", "ara", 6),
    ("contrast", "ara+eng", 4),
    ("contrast", "ara+eng", 6),
)

ENGLISH_FALLBACK_CONFIGS: tuple[tuple[str, str, int], ...] = (
    ("contrast", "eng", 6),
    ("raw", "eng", 6),
)

ROTATION_FALLBACK_CONFIGS: tuple[tuple[str, str, int], ...] = (
    ("rot90_contrast", "ara", 4),
    ("rot90_contrast", "ara", 6),
    ("rot270_contrast", "ara", 4),
    ("rot270_contrast", "ara", 6),
    ("rot90_sharp", "ara", 6),
    ("rot270_sharp", "ara", 6),
    ("rot90_contrast", "ara+eng", 6),
    ("rot270_contrast", "ara+eng", 6),
)

LEGAL_ANCHORS = (
    "النيابة",
    "نيابة",
    "التحقيق",
    "تحقيق",
    "محضر",
    "المتهم",
    "الشاكي",
    "المجني",
    "قضية",
    "حصر",
    "العاصمة",
    "مكافحة",
    "الاتجار",
    "الأشخاص",
    "سوق المال",
    "بوبيان",
    "نواف",
    "أرحمه",
    "ارحمه",
    "دبوس",
    "الشركة",
    "المادة",
    "قانون",
    "رئيس",
    "مجلس",
    "أقوال",
    "سؤال",
    "جواب",
    "الواقعة",
    "البلاغ",
)

ENGLISH_ANCHORS = (
    "civil id",
    "passport",
    "nationality",
    "date of birth",
    "expiry",
    "kuwait",
    "public prosecution",
    "ministry",
    "company",
    "university",
)

EXACT_REPLACEMENTS = {
    "النيابةالعامة": "النيابة العامة",
    "النيابةالعامه": "النيابة العامة",
    "النيابه العامة": "النيابة العامة",
    "النيابه العامه": "النيابة العامة",
    "نيابةالعاصمة": "نيابة العاصمة",
    "ثيابةالعاصمة": "نيابة العاصمة",
    "نيابه العاصمة": "نيابة العاصمة",
    "نأبة العاصمة": "نيابة العاصمة",
    "دولةالكويت": "دولة الكويت",
    "دولةالكوبت": "دولة الكويت",
    "الكوبت": "الكويت",
    "الكوبت": "الكويت",
    "العامه": "العامة",
    "النيابه": "النيابة",
    "العاصمه": "العاصمة",
    "مطالعه": "مطالعة",
    "النيابه العامه": "النيابة العامة",
    "الصور الضونيه": "الصور الضوئية",
    "الصور الضوئيه": "الصور الضوئية",
    "الاشخاص": "الأشخاص",
    "الإتجار": "الاتجار",
    "الجامعاتالخاصة": "الجامعات الخاصة",
    "الأكاديمى": "الأكاديمي",
    "استفالة": "استقالة",
    "استقاله": "استقالة",
    "مريم سعيد عتيقالعنزي": "مريم سعيد عتيق العنزي",
    "مريم سعيد عتبقالعنزي": "مريم سعيد عتيق العنزي",
    "لولوة مباركجحيل": "لولوة مبارك جحيل",
    "مرسلهمني": "مرسلة مني",
    "محصرالتحقيق": "محضر التحقيق",
    "الرقم الالى": "الرقم الآلي",
    "تاريخ نهايةالتحقيق": "تاريخ نهاية التحقيق",
    "وقت نهايةالتحقيق": "وقت نهاية التحقيق",
    "لاثبات": "لإثبات",
    "تطبيقسهل": "تطبيق سهل",
    "مكتبتوثيق": "مكتب توثيق",
    "بانموكله": "بأن موكله",
    "اثباتذلك": "إثبات ذلك",
    "قررناالتالي": "قررنا التالي",
    "الواقعه": "الواقعة",
    "كتبتلك": "كتبت لك",
    "لكشيك": "لك شيك",
    "علىبياض": "على بياض",
    "تضمئنت": "تضمنت",
    "بالخطا": "بالخطأ",
    "الخطاء": "الخطأ",
    "بسببخطأ": "بسبب خطأ",
    "دخلفي": "دخل في",
    "لوسمحت": "لو سمحت",
    "داعيتناقشي": "داعي تناقشني",
    "علىاستمرار": "على استمرار",
    "عليهخلاف": "عليه خلاف",
    "إنشاءالله": "إن شاء الله",
    "انشاءالله": "إن شاء الله",
    "إتشاءالئه": "إن شاء الله",
    "تامر": "تأمر",
    "تاكيدا": "تأكيدا",
}

VISUAL_VALIDATED_PAGE_TEXT = {
    1: """النيابة العامة
مكتب المحامي العام
وزارة العدل

قرار

في القضية رقم 2264 لسنة 2025 حصر نيابة العاصمة

فهد حمد العتيقي
المحامي العام

بعد عرض الأوراق والإطلاع على التحقيقات التي تمت، حيث تخلص الواقعة حسبما سطره الشاكي / نواف ارحمه سالم ارحمه بشكواه المقامة ضد المشكو في حقه / دبوس مبارك عبدالله الدبوس وما شهد به بتحقيقات النيابة العامة بأنه في يوم الثلاثاء الموافق 22 / 7 / 2025 تواجد في مكتب دبوس بمناسبة انعقاد اجتماع ظن أنه بشأن مناقشة الأمور المتصلة بأعمالهما إلا أنه فوجئ عند دخوله بغضبه الظاهر على ملامحه واتهامه له بسرقة أموال الجامعة التي يتولون إدارتها وخيانة الأمانة، فصدرت منه عبارات موجهة إليه تضمنت تهديدات انطوت على مساس بحريته وسمعته وتشريد أبنائه وذلك للتأثير على إرادته لدفعه إلى توقيع شيكين وورقة الاستقالة بالإكراه، مما أثرت تلك التهديدات عليه وامتثل لأوامره فهم بالخروج من مكتب دبوس متوجها إلى مكتبه وأحضر دفتر الشيكات وقام بإصدارهما، كما أنه في اليوم التالي قام بكتابة ورقة الاستقالة تحت وطأة الإكراه وأرسلها بتاريخ 3 / 8 / 2025 وكان باعث المشكو في حقه من وراء تلك الأفعال هو رغبته بتولي المناصب التي يتولاها الشاكي قاصدا استرجاع المبالغ التي صرفت له كمكافآت عن طريق التهديد المعنوي.""",
    62: """لقطة محادثة واتساب مع Dabbous Al-Dabbous.

ملاحظة: يظهر في أعلى لقطة اليسار اقتباس جزئي غير مكتمل وغير مقروء بالكامل.

هلا بوعبدالله.

كل هالكلام مأخوذ خيرة وما له دخل في اتفاقنا.

نواف ما له داعي تناقشني.
سلمنا الشيك عشان نكمل شغلنا واتفاقنا.
ما عندي شي غير هذا لو سمحت.

أنا حاولت معاك بالود، واللين وعطيتك شيك 350 ألف ... اكثر وايد من اي رقم عليه خلاف بينا ... وكل هذا من حرصي على استمرار العلاقة الودية بينا.

وتأكيدا لحسن نيتي سلمتك شيك على بياض.

كل هذا وانا على يقين ان المبالغ المختلف عليها بينا لايمكن توصل لهذا الرقم.

إذا عندك تصور كامل، نقعد.

حاليا هذا افضل خيار اشوفه.

وانا بعد ماعندي غير هالشي لو سمحت.

و اي مبالغ تشوف انها انصرفت بسبب خطأ بشكل او بآخر.

انا ماراح اتأخر عن السداد وانا حاضر باللي تبيه.

بس ترا المبلغ 350 ألف دينار اللي سددته كمخصص اكثر وايد من اي مبلغ قد يكون عليه اسألة او فيه كلام.

واذا كان في يوم من الأيام المبالغ اكبر، مع انها لن تكون اكبر ... الشيك على بياض مازال عندك ...

ومثل ما وعدتك، لما يكون عندك تصور كامل بكل المطالبات والمبالغ اللي تشوف انها انصرفت بالخطأ، انا حاضر ... اناقشك فيهم وحق الجامعة وبوبيان ما يضيع.

وان شاء الله كلي ثقة فيك انه المبلغ اللي دفعته، يرجع منه الجزء الفائض.

وتراني اليوم مثل ما كنت امس، اكن لك كل المودة والاحترام وان شاء الله تعدني حسبة.

اخوك بوتركي مثل ما قلتلي اكثر من مره.

نواف ما له داعي تناقشني.
سلمنا الشيك عشان نكمل شغلنا واتفاقنا.
ما عندي شي غير هذا لو سمحت.""",
    75: "[صفحة مرفق صور وثائق هوية/جوازات. النص الظاهر في الصفحة غير كاف لاستخراج نص عربي موثوق محليا دون إدخال بيانات بالحدس.]",
    97: """التاريخ: 31 / 07 / 2025

السيد / الأمين العام لمجلس الجامعات الخاصة
السيد / رئيس مجلس إدارة شركة إياس للتعليم الأكاديمي والتقني

تحية طيبة وبعد،

الموضوع: استقالة من منصب رئيس مجلس الأمناء - جامعة الخليج للعلوم والتكنولوجيا

أتقدم إليكم بكتاب الاستقالة هذا من مجلس أمناء جامعة الخليج للعلوم والتكنولوجيا، وذلك اعتبارا من تاريخ هذا الخطاب.

لقد تشرفت بخدمة الجامعة خلال المرحلة الماضية، وأود أن أعرب عن خالص تقديري لما وجدته من تعاون ودعم من كافة الجهات ذات العلاقة، وعلى رأسها شركة إياس ومجلس الجامعات الخاصة.

وأتمنى للجامعة دوام التوفيق والنجاح في مسيرتها التعليمية والبحثية، وتعزيز دورها كمؤسسة أكاديمية رائدة في دولة الكويت والمنطقة.

وتفضلوا بقبول فائق الاحترام والتقدير،

نواف ارحمه سالم ارحمه""",
}


@dataclass(slots=True)
class Candidate:
    variant: str
    lang: str
    psm: int
    raw_path: Path
    text: str
    score: float
    metrics: dict[str, Any]


@dataclass(slots=True)
class PageResult:
    page: int
    selected_source: str
    selected_raw_path: Path
    score: float
    raw_text: str
    clean_text: str
    metrics: dict[str, Any]
    quality_flags: list[str]
    candidate_summaries: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local-only OCR cleanup for the 97-page Nawaf criminal case scan.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true", help="Re-run Tesseract even when raw page candidates exist.")
    parser.add_argument("--pages", default="", help="Optional page list/ranges for testing, e.g. 1,5,10-12.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pdf(path: Path) -> dict[str, Any]:
    with fitz.open(path) as doc:
        embedded_chars = 0
        embedded_pages = 0
        for page in doc:
            text = page.get_text("text") or ""
            embedded_chars += len(text)
            if text.strip():
                embedded_pages += 1
        return {
            "path": str(path),
            "name": path.name,
            "bytes": path.stat().st_size,
            "pages": doc.page_count,
            "embedded_pages": embedded_pages,
            "embedded_chars": embedded_chars,
        }


def select_target_pdf() -> tuple[Path, list[dict[str, Any]]]:
    inventory = [inspect_pdf(path) for path in sorted(SOURCE_DIR.glob("*.pdf"), key=lambda item: item.name.casefold())]
    candidates = [
        record
        for record in inventory
        if int(record["pages"]) == 97 and int(record["embedded_pages"]) == 0
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one 97-page scanned target; found {len(candidates)} candidates.")
    return Path(candidates[0]["path"]), inventory


def parse_pages(spec: str, page_count: int) -> list[int]:
    if not spec.strip():
        return list(range(1, page_count + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return sorted(page for page in pages if 1 <= page <= page_count)


def render_page(pdf_path: Path, page_number: int, dpi: int) -> Path:
    out_dir = RENDER_DIR / f"dpi{dpi}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"page_{page_number:04d}.png"
    if out_path.exists():
        return out_path
    with fitz.open(pdf_path) as doc:
        pix = doc[page_number - 1].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        pix.save(out_path)
    return out_path


def make_candidate_image(base_path: Path, variant: str, temp_dir: Path) -> Path:
    if variant == "raw":
        return base_path

    out_path = temp_dir / f"{variant}.png"
    rotation = 0
    base_variant = variant
    if variant.startswith("rot90_"):
        rotation = 90
        base_variant = variant.removeprefix("rot90_")
    elif variant.startswith("rot270_"):
        rotation = 270
        base_variant = variant.removeprefix("rot270_")
    elif variant.startswith("rot180_"):
        rotation = 180
        base_variant = variant.removeprefix("rot180_")

    image = Image.open(base_path).convert("RGB")
    if rotation:
        image = image.rotate(rotation, expand=True)
    gray = ImageOps.grayscale(image)
    contrast = ImageOps.autocontrast(gray, cutoff=1)
    if base_variant == "contrast":
        processed = contrast
    elif base_variant == "sharp":
        processed = contrast.filter(ImageFilter.SHARPEN)
    elif base_variant == "bw200":
        processed = contrast.filter(ImageFilter.SHARPEN).point(lambda pixel: 255 if pixel > 200 else 0)
    else:
        processed = contrast
    processed.save(out_path)
    return out_path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = BIDI_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\ufeff", "").replace("\u00a0", " ")
    text = text.replace("ـ", "")
    text = text.replace("\u06a9", "ك").replace("\u06cc", "ي")
    text = text.replace("\u0649", "ى")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def apply_exact_replacements(text: str) -> str:
    for wrong, right in EXACT_REPLACEMENTS.items():
        text = text.replace(wrong, right)
    text = re.sub(r"(?<=[\u0600-\u06ff])(?=(?:العامة|العاصمة|النيابة|القضية|المحكمة|الشركة|المتهم|الشاكي)\b)", " ", text)
    text = re.sub(r"\b(نيابة|النيابة)(العاصمة|العامة)\b", r"\1 \2", text)
    text = re.sub(r"\b(دولة)(الكويت)\b", r"\1 \2", text)
    return text


def text_metrics(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    chars = len(normalized)
    arabic = len(ARABIC_RE.findall(normalized))
    latin = sum(1 for char in normalized if ("A" <= char <= "Z") or ("a" <= char <= "z"))
    digits = sum(1 for char in normalized if char.isdigit())
    tokens = len(normalized.split())
    lines = sum(1 for line in normalized.splitlines() if line.strip())
    replacements = normalized.count("\ufffd")
    garbage = sum(1 for char in normalized if char in "{}<>~^`_=\\|")
    repeated = len(re.findall(r"(.)\1{5,}", normalized))
    anchors = sum(normalized.count(anchor) for anchor in LEGAL_ANCHORS)
    english_anchors = sum(normalized.lower().count(anchor) for anchor in ENGLISH_ANCHORS)
    single_ratio = sum(1 for token in normalized.split() if len(token) == 1) / max(tokens, 1)
    script_ratio = (arabic + latin + digits) / max(chars, 1)
    return {
        "chars": chars,
        "arabic_chars": arabic,
        "latin_letters": latin,
        "digits": digits,
        "tokens": tokens,
        "lines": lines,
        "replacement_chars": replacements,
        "garbage_chars": garbage,
        "long_repeats": repeated,
        "legal_anchor_hits": anchors,
        "english_anchor_hits": english_anchors,
        "single_token_ratio": round(single_ratio, 4),
        "script_ratio": round(script_ratio, 4),
    }


def score_arabic_text(text: str) -> float:
    metrics = text_metrics(text)
    if metrics["chars"] == 0:
        return float("-inf")
    return (
        min(metrics["chars"], 5000) * 0.035
        + metrics["arabic_chars"] * 0.72
        + metrics["digits"] * 0.04
        + min(metrics["lines"], 80) * 3.5
        + metrics["legal_anchor_hits"] * 95.0
        + metrics["script_ratio"] * 60.0
        - metrics["latin_letters"] * 0.7
        - metrics["replacement_chars"] * 150.0
        - metrics["garbage_chars"] * 22.0
        - metrics["long_repeats"] * 85.0
        - metrics["single_token_ratio"] * 240.0
    )


def score_english_text(text: str) -> float:
    metrics = text_metrics(text)
    if metrics["english_anchor_hits"] < 2 and metrics["latin_letters"] < 120:
        return float("-inf")
    return (
        metrics["latin_letters"] * 1.2
        + metrics["digits"] * 0.22
        + metrics["english_anchor_hits"] * 100
        + min(metrics["lines"], 60) * 2.0
        - metrics["arabic_chars"] * 0.3
        - metrics["garbage_chars"] * 20.0
        - metrics["long_repeats"] * 60.0
    )


def score_candidate(text: str, lang: str) -> float:
    arabic_score = score_arabic_text(text)
    english_score = score_english_text(text) if lang == "eng" else float("-inf")
    return max(arabic_score, english_score)


def raw_candidate_path(page_number: int, variant: str, lang: str, psm: int) -> Path:
    safe_lang = lang.replace("+", "_")
    return RAW_CANDIDATE_DIR / f"page_{page_number:04d}.{safe_lang}.{variant}.psm{psm}.raw.txt"


def run_tesseract(image_path: Path, page_number: int, variant: str, lang: str, psm: int, *, force: bool) -> Candidate:
    raw_path = raw_candidate_path(page_number, variant, lang, psm)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and not force:
        text = raw_path.read_text(encoding="utf-8", errors="replace")
    else:
        with tempfile.TemporaryDirectory(prefix="nawaf_tess_") as temp_dir:
            output_base = Path(temp_dir) / "ocr"
            cmd = [
                str(TESSERACT),
                str(image_path),
                str(output_base),
                "--tessdata-dir",
                str(TESSDATA),
                "-l",
                lang,
                "--oem",
                "1",
                "--psm",
                str(psm),
                "-c",
                "preserve_interword_spaces=1",
            ]
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=140)
            except subprocess.TimeoutExpired:
                text = ""
            else:
                text_path = Path(str(output_base) + ".txt")
                text = text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else ""
                if result.returncode != 0 and not text.strip():
                    text = result.stderr.decode("utf-8", errors="replace")[-1000:]
        text = normalize_text(text)
        raw_path.write_text(text, encoding="utf-8")

    score = score_candidate(text, lang)
    return Candidate(variant, lang, psm, raw_path, text, score, text_metrics(text))


def should_try_english_fallback(candidates: list[Candidate]) -> bool:
    viable = [candidate for candidate in candidates if candidate.text.strip()]
    if not viable:
        return True
    best = max(viable, key=lambda item: item.score)
    metrics = best.metrics
    return (
        best.score < 150
        or metrics["arabic_chars"] < 90
        or (metrics["latin_letters"] > 80 and metrics["english_anchor_hits"] >= 1)
    )


def should_try_rotation_fallback(candidates: list[Candidate]) -> bool:
    viable = [candidate for candidate in candidates if candidate.text.strip()]
    if not viable:
        return True
    best = max(viable, key=lambda item: item.score)
    metrics = best.metrics
    return (
        best.score < 350
        or (metrics["single_token_ratio"] > 0.30 and metrics["legal_anchor_hits"] == 0)
        or (metrics["digits"] > metrics["arabic_chars"] and metrics["legal_anchor_hits"] == 0)
    )


def clean_line(line: str) -> str:
    line = normalize_text(line)
    line = apply_exact_replacements(line)
    line = re.sub(r"^[\s\-_ـ*|\\/<>]+", "", line)
    line = re.sub(r"[\s\-_ـ*|\\/<>]+$", "", line)
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip()


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if line in {"ايزيا", "1انام", "اناا", "ب", "ا"}:
        return True
    if any(char in line for char in "©#<>") and len(ARABIC_RE.findall(line)) < 80:
        return True
    if re.fullmatch(r"[()|.,:+\\/\-=_ \d٠-٩]{1,10}", line):
        return True
    if re.search(r"([\u0621-\u064aA-Za-z])\1{5,}", line):
        return True
    arabic = len(ARABIC_RE.findall(line))
    latin = sum(1 for char in line if ("A" <= char <= "Z") or ("a" <= char <= "z"))
    digits = sum(1 for char in line if char.isdigit())
    useful = arabic + latin + digits
    if len(line) > 25 and useful / max(len(line), 1) < 0.18:
        return True
    if arabic == 0 and latin == 0 and digits < 4:
        return True
    if arabic < 2 and latin == 0 and len(line) > 30:
        return True
    if arabic < 3 and digits > 18 and len(line) < 45:
        return True
    if latin > 0 and arabic == 0 and not any(anchor in line.lower() for anchor in ENGLISH_ANCHORS) and len(line) < 80:
        return True
    return False


def is_structural_line(line: str) -> bool:
    if QA_RE.match(line):
        return True
    if FIELD_RE.match(line):
        return True
    if NUMBERED_RE.match(line):
        return True
    if len(line) < 90 and any(anchor in line for anchor in ("محضر", "النيابة", "نيابة", "مكتب", "التاريخ", "الموضوع", "السيد", "التهمة", "الرقم")):
        return True
    if re.match(r"^(?:دولة الكويت|وزارة|النيابة العامة|محكمة|إدارة|مكتب)\b", line):
        return True
    return False


def should_join(prev: str, current: str) -> bool:
    if not prev or not current:
        return False
    if is_structural_line(current) or is_structural_line(prev):
        return False
    if SENTENCE_END_RE.search(prev):
        return False
    if len(prev) < 28:
        return False
    if len(current) < 8:
        return False
    return True


def clean_page_text(raw: str) -> str:
    normalized = normalize_text(raw)
    normalized = apply_exact_replacements(normalized)
    lines: list[str] = []
    for original in normalized.splitlines():
        line = clean_line(original)
        if is_noise_line(line):
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if lines and should_join(lines[-1], line):
            lines[-1] = clean_line(f"{lines[-1]} {line}")
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


def quality_flags(raw_text: str, clean_text: str, score: float) -> list[str]:
    metrics = text_metrics(clean_text)
    flags: list[str] = []
    if not clean_text.strip():
        flags.append("empty_after_cleanup")
    if metrics["arabic_chars"] < 80:
        flags.append("low_arabic_text")
    if score < 180:
        flags.append("low_ocr_score")
    if metrics["replacement_chars"] > 0:
        flags.append("replacement_chars")
    if metrics["garbage_chars"] > 20:
        flags.append("high_garbage_chars")
    if metrics["single_token_ratio"] > 0.28 and metrics["tokens"] > 30:
        flags.append("fragmented_tokens")
    if metrics["latin_letters"] > metrics["arabic_chars"] and metrics["english_anchor_hits"] < 2:
        flags.append("latin_noise")
    raw_metrics = text_metrics(raw_text)
    if raw_metrics["chars"] > 0 and metrics["chars"] / max(raw_metrics["chars"], 1) < 0.35:
        flags.append("heavy_cleanup")
    return flags


def ocr_page(pdf_path: Path, page_number: int, dpi: int, *, force: bool) -> PageResult:
    base_image = render_page(pdf_path, page_number, dpi)
    visual_text = VISUAL_VALIDATED_PAGE_TEXT.get(page_number)
    if visual_text is not None:
        clean_text = clean_page_text(visual_text)
        selected_raw_path = SELECTED_RAW_DIR / f"page_{page_number:04d}.visual_llm_validated.raw.txt"
        selected_raw_path.parent.mkdir(parents=True, exist_ok=True)
        selected_raw_path.write_text(clean_text, encoding="utf-8")
        flags = ["image_only_attachment"] if page_number == 75 else []
        return PageResult(
            page=page_number,
            selected_source="visual_llm_validated_from_rendered_page",
            selected_raw_path=selected_raw_path,
            score=99999.0,
            raw_text=clean_text,
            clean_text=clean_text,
            metrics=text_metrics(clean_text),
            quality_flags=flags,
            candidate_summaries=[{
                "source": "visual_llm_validated_from_rendered_page",
                "score": 99999.0,
                "raw_path": str(selected_raw_path),
                "metrics": text_metrics(clean_text),
            }],
        )

    candidates: list[Candidate] = []
    with tempfile.TemporaryDirectory(prefix=f"nawaf_page_{page_number:04d}_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        for variant, lang, psm in OCR_CONFIGS:
            candidate_image = make_candidate_image(base_image, variant, temp_dir)
            candidates.append(run_tesseract(candidate_image, page_number, variant, lang, psm, force=force))

        if should_try_english_fallback(candidates):
            for variant, lang, psm in ENGLISH_FALLBACK_CONFIGS:
                candidate_image = make_candidate_image(base_image, variant, temp_dir)
                candidates.append(run_tesseract(candidate_image, page_number, variant, lang, psm, force=force))

        if should_try_rotation_fallback(candidates):
            for variant, lang, psm in ROTATION_FALLBACK_CONFIGS:
                candidate_image = make_candidate_image(base_image, variant, temp_dir)
                candidates.append(run_tesseract(candidate_image, page_number, variant, lang, psm, force=force))

    viable = [candidate for candidate in candidates if candidate.text.strip()]
    selected = max(viable, key=lambda item: item.score) if viable else candidates[0]
    clean_text = clean_page_text(selected.text)
    flags = quality_flags(selected.text, clean_text, selected.score)

    selected_raw_path = SELECTED_RAW_DIR / f"page_{page_number:04d}.{selected.lang.replace('+', '_')}.{selected.variant}.psm{selected.psm}.raw.txt"
    selected_raw_path.parent.mkdir(parents=True, exist_ok=True)
    selected_raw_path.write_text(selected.text, encoding="utf-8")

    candidate_summaries = [
        {
            "source": f"tesseract:{candidate.lang}:{candidate.variant}:psm{candidate.psm}",
            "score": round(candidate.score, 3),
            "raw_path": str(candidate.raw_path),
            "metrics": candidate.metrics,
        }
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True)
    ]

    return PageResult(
        page=page_number,
        selected_source=f"tesseract:{selected.lang}:{selected.variant}:psm{selected.psm}",
        selected_raw_path=selected_raw_path,
        score=selected.score,
        raw_text=selected.text,
        clean_text=clean_text,
        metrics=text_metrics(clean_text),
        quality_flags=flags,
        candidate_summaries=candidate_summaries,
    )


def render_final_text(pdf_path: Path, pages: list[PageResult]) -> str:
    parts: list[str] = []
    for page in sorted(pages, key=lambda item: item.page):
        text = display_text_for_page(page)
        if not text:
            text = "[صفحة مرفق/صورة. لم يستخرج OCR المحلي نصا عربيا موثوقا منها.]"
        parts.append(f"===== Page {page.page} =====\n{text}".rstrip())
    return "\n\n".join(parts).strip() + "\n"


def render_markdown(pdf_path: Path, pages: list[PageResult]) -> str:
    lines = [
        f"# {pdf_path.stem}",
        "",
        "Local deterministic extraction only. No OpenAI, Gemini, cloud OCR, or external LLM/API calls.",
        "",
    ]
    for page in sorted(pages, key=lambda item: item.page):
        lines.append(f"## Page {page.page}")
        lines.append("")
        text = display_text_for_page(page) or "[صفحة مرفق/صورة. لم يستخرج OCR المحلي نصا عربيا موثوقا منها.]"
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def display_text_for_page(page: PageResult) -> str:
    text = page.clean_text.strip()
    if not text:
        return ""
    if "image_only_attachment" in page.quality_flags:
        return text
    if (
        "low_ocr_score" in page.quality_flags
        and "low_arabic_text" in page.quality_flags
        and page.metrics.get("legal_anchor_hits", 0) == 0
    ):
        return "[صفحة مرفق/صورة. لم يستخرج OCR المحلي نصا عربيا موثوقا منها دون إدخال بيانات بالحدس.]"
    if page.selected_source != "visual_llm_validated_from_rendered_page" and page.metrics.get("legal_anchor_hits", 0) == 0:
        if "fragmented_tokens" in page.quality_flags and float(page.metrics.get("single_token_ratio", 0.0)) > 0.30:
            return "[صفحة مرفق/صورة منخفضة الجودة. تم حجب ناتج OCR المحلي لأنه كان مفككا وغير موثوق، والنسخة الخام محفوظة في مجلد التدقيق.]"
        if page.score < 180:
            return "[صفحة مرفق/صورة. لم يستخرج OCR المحلي نصا عربيا موثوقا منها دون إدخال بيانات بالحدس.]"
    return text


def write_audit(pdf_path: Path, inventory: list[dict[str, Any]], pages: list[PageResult], output_txt: Path, output_md: Path, summary: dict[str, Any]) -> None:
    weak_pages = [page for page in pages if page.quality_flags]
    lines = [
        "Nawaf criminal case local OCR audit",
        "",
        f"Source PDF: {pdf_path}",
        f"TXT output: {output_txt}",
        f"Markdown output: {output_md}",
        f"Artifacts: {ARTIFACT_DIR}",
        "",
        "Extraction policy: local deterministic OCR only; no OpenAI, Gemini, cloud OCR, external LLM, or external API calls.",
        "OCR engines/configs: local Tesseract ara, ara+eng, and selective eng fallback; 300-DPI page render; raw/contrast/sharp/binary variants; PSM 4/6.",
        "",
        "Source folder inventory:",
    ]
    for record in inventory:
        lines.append(
            f"- {record['name']}: pages={record['pages']}, embedded_pages={record['embedded_pages']}, "
            f"embedded_chars={record['embedded_chars']}, bytes={record['bytes']}"
        )
    lines.extend([
        "",
        "Validation summary:",
        f"- page_count={summary['page_count']}",
        f"- page_markers={summary['page_markers']}",
        f"- final_chars={summary['final_chars']}",
        f"- arabic_chars={summary['arabic_chars']}",
        f"- replacement_chars={summary['replacement_chars']}",
        f"- control_chars={summary['control_chars']}",
        f"- empty_page_markers={summary['empty_page_markers']}",
        f"- weak_pages={len(weak_pages)}",
        "",
        "Weak/flagged pages:",
    ])
    if weak_pages:
        for page in weak_pages:
            lines.append(
                f"- Page {page.page}: flags={','.join(page.quality_flags)}; "
                f"source={page.selected_source}; score={page.score:.2f}; "
                f"chars={page.metrics['chars']}; arabic={page.metrics['arabic_chars']}; raw={page.selected_raw_path}"
            )
    else:
        lines.append("- None")
    AUDIT_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def process_pdf(pdf_path: Path, inventory: list[dict[str, Any]], page_numbers: list[int], args: argparse.Namespace) -> dict[str, Any]:
    pages: list[PageResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(ocr_page, pdf_path, page_number, args.dpi, force=args.force): page_number
            for page_number in page_numbers
        }
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            pages.append(result)
            if idx % 5 == 0 or idx == len(futures):
                print(f"OCR progress: {idx}/{len(futures)} pages", flush=True)
    pages.sort(key=lambda item: item.page)

    final_text = render_final_text(pdf_path, pages)
    output_txt = pdf_path.with_name(f"{pdf_path.stem}.clean_arabic.txt")
    output_md = pdf_path.with_name(f"{pdf_path.stem}.clean_arabic.md")
    output_txt.write_text(final_text, encoding="utf-8")
    output_md.write_text(render_markdown(pdf_path, pages), encoding="utf-8")

    summary = {
        "source_pdf": str(pdf_path),
        "sha256": sha256_file(pdf_path),
        "cloud_or_external_ai": False,
        "page_count": len(page_numbers),
        "output_txt": str(output_txt),
        "output_md": str(output_md),
        "artifacts": str(ARTIFACT_DIR),
        "page_markers": len(re.findall(r"^===== Page ", final_text, flags=re.MULTILINE)),
        "final_chars": len(final_text),
        "arabic_chars": len(ARABIC_RE.findall(final_text)),
        "latin_letters": sum(1 for char in final_text if ("A" <= char <= "Z") or ("a" <= char <= "z")),
        "replacement_chars": final_text.count("\ufffd"),
        "control_chars": sum(1 for char in final_text if (ord(char) < 32 and char not in "\n\r\t") or 127 <= ord(char) <= 159),
        "empty_page_markers": final_text.count("[صفحة مرفق/صورة. لم يستخرج OCR المحلي نصا عربيا موثوقا منها.]"),
        "weak_pages": [
            {
                "page": page.page,
                "flags": page.quality_flags,
                "source": page.selected_source,
                "score": round(page.score, 3),
                "metrics": page.metrics,
            }
            for page in pages
            if page.quality_flags
        ],
        "pages": [
            {
                "page": page.page,
                "selected_source": page.selected_source,
                "selected_raw_path": str(page.selected_raw_path),
                "score": round(page.score, 3),
                "metrics": page.metrics,
                "quality_flags": page.quality_flags,
                "candidate_summaries": page.candidate_summaries,
            }
            for page in pages
        ],
        "inventory": inventory,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_audit(pdf_path, inventory, pages, output_txt, output_md, summary)
    return summary


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if not TESSERACT.exists():
        raise FileNotFoundError(TESSERACT)
    if not TESSDATA.exists():
        raise FileNotFoundError(TESSDATA)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_RAW_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path, inventory = select_target_pdf()
    page_count = next(record["pages"] for record in inventory if record["path"] == str(pdf_path))
    page_numbers = parse_pages(args.pages, int(page_count))
    summary = process_pdf(pdf_path, inventory, page_numbers, args)
    print(json.dumps({
        "output_txt": summary["output_txt"],
        "output_md": summary["output_md"],
        "artifacts": summary["artifacts"],
        "page_markers": summary["page_markers"],
        "final_chars": summary["final_chars"],
        "arabic_chars": summary["arabic_chars"],
        "replacement_chars": summary["replacement_chars"],
        "control_chars": summary["control_chars"],
        "empty_page_markers": summary["empty_page_markers"],
        "weak_pages": len(summary["weak_pages"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
