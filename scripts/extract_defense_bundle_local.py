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
ARTIFACT_DIR = SOURCE_DIR / "defense_bundle_ocr_artifacts"
RENDER_DIR = ARTIFACT_DIR / "rendered_pages_dpi320"
RAW_CANDIDATE_DIR = ARTIFACT_DIR / "raw_tesseract_candidates"
SELECTED_RAW_DIR = ARTIFACT_DIR / "selected_raw_pages"
SUMMARY_PATH = ARTIFACT_DIR / "defense_bundle_local_ocr_summary.json"
AUDIT_PATH = ARTIFACT_DIR / "defense_bundle_local_ocr_audit.txt"

TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA = Path(r"C:\Users\narha\arabic-ocr\.tessdata")

ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
BIDI_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
SENTENCE_END_RE = re.compile(r"[.؟?!؛:]$")
NUMBERED_RE = re.compile(r"^\s*(?:\d+|[٠-٩]+|[أ-ي])[\-.)،:]\s+")
FIELD_RE = re.compile(r"^[\u0600-\u06ffA-Za-z ()/.-]{2,55}\s*[:：]")

OCR_CONFIGS: tuple[tuple[str, str, int], ...] = (
    ("raw", "ara", 4),
    ("raw", "ara", 6),
    ("contrast", "ara", 4),
    ("contrast", "ara", 6),
    ("sharp", "ara", 4),
    ("sharp", "ara", 6),
    ("bw190", "ara", 4),
    ("bw190", "ara", 6),
    ("bw210", "ara", 4),
    ("bw210", "ara", 6),
    ("contrast", "ara+eng", 4),
    ("contrast", "ara+eng", 6),
)

ROTATION_CONFIGS: tuple[tuple[str, str, int], ...] = (
    ("rot90_contrast", "ara", 4),
    ("rot90_contrast", "ara", 6),
    ("rot270_contrast", "ara", 4),
    ("rot270_contrast", "ara", 6),
    ("rot90_sharp", "ara", 6),
    ("rot270_sharp", "ara", 6),
)

LEGAL_ANCHORS = (
    "مذكرة",
    "دفاع",
    "حافظة",
    "مستندات",
    "محكمة",
    "القضية",
    "الدعوى",
    "النيابة",
    "الشاكي",
    "المتهم",
    "المدعي",
    "المدعى",
    "المادة",
    "القانون",
    "شركة",
    "إياس",
    "اياس",
    "بوبيان",
    "البتروكيماويات",
    "دبوس",
    "الدبوس",
    "نواف",
    "ارحمه",
    "أرحمه",
    "جامعة",
    "الخليج",
    "مجلس",
    "الأمناء",
    "الاستقالة",
    "الشيك",
    "شيك",
    "الإكراه",
    "التهديد",
    "شاهد",
    "المحامي",
    "وكيل",
    "دفوع",
    "طلبات",
)

EXACT_REPLACEMENTS = {
    "يوسفالحربش": "يوسف الحريش",
    "يوسفالحربيش": "يوسف الحريش",
    "يوسفالحريش": "يوسف الحريش",
    "يوسفالمريش": "يوسف الحريش",
    "يوسفالحيش": "يوسف الحريش",
    "الدكتوريوسف": "الدكتور يوسف",
    "الاكتوريوسف": "الدكتور يوسف",
    "للمحاماة والإستشاراتالقانونية": "للمحاماة والاستشارات القانونية",
    "للمحاماة والإستشارانالقانونية": "للمحاماة والاستشارات القانونية",
    "للمداماة والإستشاراتالقانونية": "للمحاماة والاستشارات القانونية",
    "لأمحاماة والإستشاراتالقانونية": "للمحاماة والاستشارات القانونية",
    "للداماة والستشارات": "للمحاماة والاستشارات",
    "اياس": "إياس",
    "اراس": "إياس",
    "إيسااس": "إياس",
    "إ ياس": "إياس",
    "إياسللتعليم": "إياس للتعليم",
    "شركةإياس": "شركة إياس",
    "فصرقة إياس": "شركة إياس",
    "إياس الأكاديمي والتقني": "إياس للتعليم الأكاديمي والتقني",
    "بوبيانللبتروكيماويات": "بوبيان للبتروكيماويات",
    "ببيان للبتزوكيماويات": "بوبيان للبتروكيماويات",
    "شركةببيان": "شركة بوبيان",
    "للبتروكيمياويات": "للبتروكيماويات",
    "للبتروكيماويه": "للبتروكيماويات",
    "للبتروكيمياويه": "للبتروكيماويات",
    "للبتزوكيماويات": "للبتروكيماويات",
    "للبتروكيماريات": "للبتروكيماويات",
    "بوبيسانللبتروكيماويسات": "بوبيان للبتروكيماويات",
    "نواف ارحمه": "نواف ارحمه",
    "نواف أرحمه": "نواف ارحمه",
    "سالمارحمه": "سالم ارحمه",
    "سالم أرحمه": "سالم ارحمه",
    "دبوس مبارك عبدلله": "دبوس مبارك عبدالله",
    "عبداللهالدبوس": "عبدالله الدبوس",
    "عبدالله الدبوس": "عبدالله الدبوس",
    "النيابه": "النيابة",
    "النيابه العامه": "النيابة العامة",
    "النيابةالعامه": "النيابة العامة",
    "النيابةالعامة": "النيابة العامة",
    "نيابةالعاصمة": "نيابة العاصمة",
    "محصر": "محضر",
    "محضرالتحقيق": "محضر التحقيق",
    "المسدعى": "المدعى",
    "المسدعي": "المدعي",
    "المسديى": "المدعي",
    "المسدي": "المدعي",
    "السدعوى": "الدعوى",
    "الدموى": "الدعوى",
    "المذكره": "المذكرة",
    "المسذكرة": "المذكرة",
    "المحكمه": "المحكمة",
    "الدعوي": "الدعوى",
    "المدعي": "المدعي",
    "المدعىعليها": "المدعى عليها",
    "المدعيعليها": "المدعى عليها",
    "المشكوفي": "المشكو في",
    "المشكوفي": "المشكو في",
    "حافظةمستندات": "حافظة مستندات",
    "مستندرقم": "مستند رقم",
    "التعليمالأكاديمي": "التعليم الأكاديمي",
    "التعليسم": "التعليم",
    "الأكاديميوالتقني": "الأكاديمي والتقني",
    "الأكاديمسيوالتفسي": "الأكاديمي والتقني",
    "الأقاديمي والتفنسي": "الأكاديمي والتقني",
    "الأكاديسسي": "الأكاديمي",
    "الاكاديمي": "الأكاديمي",
    "التقنى": "التقني",
    "التقنسى": "التقني",
    "التغني": "التقني",
    "الإكراه": "الإكراه",
    "الاكراه": "الإكراه",
    "التهديدالمعنوي": "التهديد المعنوي",
    "الإستقالة": "الاستقالة",
    "استقاله": "استقالة",
    "بصفته": "بصفته",
    "بالصفه": "بالصفة",
    "المحترمين": "المحترمين",
    "صوره": "صورة",
    "رقم ": "رقم ",
    "الرد علسى": "الرد على",
    "السرد علسى": "الرد على",
    "طلسب": "طلب",
    "الضبرة": "الخبرة",
    "الخيرة": "الخبيرة",
    "المسوقرة": "الموقرة",
    "الموفرة": "الموقرة",
    "هسذه": "هذه",
    "السدفاع": "الدفاع",
    "السوارد": "الوارد",
    "مكمسل ومتمم": "مكمل ومتمم",
    "أوجسه": "أوجه",
    "دناءهما": "دفاعهما",
    "ودنودهما": "ودفوعهما",
    "بلسف": "بملف",
    "أمسام": "أمام",
    "السابق": "السابق",
    "شسركة": "شركة",
    "الإجسراءات": "الإجراءات",
    "المتبعسة": "المتبعة",
    "تغيسب": "تغيب",
    "العجسل": "العمل",
    "بسلا": "بلا",
    "دبوسالدبوس": "دبوس الدبوس",
    "دبوس الدبوس": "دبوس الدبوس",
    "السيد/ دبوسالدبوس": "السيد/ دبوس الدبوس",
    "الشركةللاعتبارات": "الشركة للاعتبارات",
    "عنهبالتغيب": "عنه بالتغيب",
    "مزاعم": "مزاعم",
    "مزا عم": "مزاعم",
    "خلافالحقيقة": "خلاف الحقيقة",
    "ب الدعوى": "بالدعوى",
    "المستحقاتالعمالية": "المستحقات العمالية",
    "الشكى": "الشكوى",
    "الشكيب": "الشكوى",
    "الشكو": "المشكو",
    "الشكو بحقه": "المشكو بحقه",
    "المشكو بحضه": "المشكو بحقه",
    "المشكو بحقه": "المشكو بحقه",
    "المشكو فى حقه": "المشكو في حقه",
    "الملشكو بحقه": "المشكو بحقه",
    "اللشكو بحته": "المشكو بحقه",
    "الشاكيةكما": "الشاكية كما",
    "مكتباللدميان": "مكتب المحامون",
    "ورجاماة والإستشارات القانونية": "للمحاماة والاستشارات القانونية",
    "الساماة والستشارات القاقولية": "للمحاماة والاستشارات القانونية",
    "مقدمه لسيادتكم": "مقدمة لسيادتكم",
    "مقدمه لسيادتكم/": "مقدمة لسيادتكم/",
    "مقدمه لسيادتكم /": "مقدمة لسيادتكم /",
    "مالكة فرعض": "مالكة فرع",
    "ومحلهاالمختار": "ومحلها المختار",
    "محمدالسبيعي": "محمد السبيعي",
    "ببنيد القار": "ببنيد القار",
    "كتب يم": "مكتب رقم",
    "كوبتي الجنسية": "كويتي الجنسية",
    "مقسيم": "مقيم",
    "مدنيةرقم": "مدنية رقم",
    "هاتف رقم": "هاتف رقم",
    "منصسب": "منصب",
    "يشفل": "يشغل",
    "تميين": "تعيين",
    "تمينه": "تعيينه",
    "تتفيذي": "تنفيذي",
    "التفيذى": "التنفيذي",
    "مكتوية": "مكتوبة",
    "المدير الماليللجامعة": "المدير المالي للجامعة",
    "ربالقطري": "ريال قطري",
    "ماثة": "مائة",
    "الخساب": "الحساب",
    "الموضحب": "الموضح ب",
    "طلبالمشكو": "طلب المشكو",
    "جزءِمن": "جزء من",
    "بصفه": "بصفة",
    "خان ه": "خانة",
    "خانه": "خانة",
    "تاريخ التعين": "تاريخ التعيين",
    "بالفمل": "بالفعل",
    "الباتف": "الهاتف",
    "اللطلوب": "المطلوب",
    "الثايت": "الثابت",
    "الائتمان": "الائتمان",
    "حسابه الشخصي": "حسابه الشخصي",
    "تقنبة": "تقنية",
    "جرائم": "جرائم",
    "أفمالاً": "أعمالاً",
    "الحقت ب": "ألحقت بال",
    "ألحقت بال المدعى": "ألحقت بالمدعى",
    "المنسوبةإليه": "المنسوبة إليه",
    "أمفار محمد لي جاسمالمخلف": "أ/ منار محمد جاسم المخلف",
    "جاسمالمخلف": "جاسم المخلف",
    "أمفار": "أ/ منار",
    "امواللنفسه": "أموال لنفسه",
    "استغلال المدعيلمنصبه": "استغلال المدعي لمنصبه",
    "تؤكدتسبب": "تؤكد تسبب",
    "تزامنمع": "تزامن مع",
    "المشارإليها": "المشار إليها",
    "بنكبوبيان": "بنك بوبيان",
    "الحكمبجلسة": "الحكم بجلسة",
    "المخالفات المنسوبة إليه": "المخالفات المنسوبة إليه",
    "التزامب الشركة": "الالتزام بالشركة",
    "فترةخدمته": "فترة خدمته",
    "ونرفقطيه": "ونرفق طيه",
    "الشكيى": "الشكوى",
    "المشكوى": "الشكوى",
    "الشكاوي": "الشكاوى",
    "المحاكمةالجزائية": "المحاكمة الجزائية",
    "المحاكماتالجزائية": "المحاكمات الجزائية",
    "الإجراءاتوالمحاكمات": "الإجراءات والمحاكمات",
    "للمرافعةالختامية": "للمرافعة الختامية",
    "للمحاكمةالجزائية": "للمحاكمة الجزائية",
    "المعلوماتالدولية": "المعلومات الدولية",
    "بالمستنداتالمرفقة": "بالمستندات المرفقة",
    "للعلوموالتكنولرجيا": "للعلوم والتكنولوجيا",
    "للعلوموالتكنولوجيا": "للعلوم والتكنولوجيا",
    "الفاتورتينالمرفقتين": "الفاتورتين المرفقتين",
    "الاستقالةتوقيعه": "الاستقالة توقيعه",
    "بالأوراقالممهورة": "بالأوراق الممهورة",
    "بدونمخصصات": "بدون مخصصات",
    "أمامالخبرة": "أمام الخبرة",
    "توقيمه": "توقيعه",
    "اعمال الخبرة": "أعمال الخبرة",
    "تقديماستقالته": "تقديم استقالته",
    "لبا شخصيتها": "لها شخصيتها",
    "قبولالاستقالة": "قبول الاستقالة",
    "المذكرة": "المذكرة",
    "الملذكرة": "المذكرة",
    "الضيرة": "الخبرة",
    "خامسا - البرد": "خامساً - الرد",
    "صذهرة": "مذكرة",
    "مدصي": "مدعي",
    "المقدجمة": "المقدمة",
    "الأتى": "الآتي",
    "مازعمه": "ما زعمه",
    "أحكامقانون": "أحكام قانون",
    "بشأن الشركات؛ لبا": "بشأن الشركات؛ لها",
    "بالمستتدات": "بالمستندات",
    "أن ينكرها": "أو ينكرها",
    "أرياح": "أرباح",
    "الشزكتين": "الشركتين",
    "تفتبرا": "تعتبران",
    "واخدة": "واحدة",
    "وَإتمًا": "وإنما",
    "إدزاج": "إدراج",
    "تتحصلعليها": "تتحصل عليها",
    "مذكراتدفاعنا": "مذكرات دفاعنا",
    "للتمس": "نلتمس",
    "هديا": "هدياً",
    "السدموىوالتقرير": "الدعوى والتقرير",
    "التعليسمالأكاديمي": "التعليم الأكاديمي",
    "طلهات": "طلبات",
}

VISUAL_PAGE_TEXT: dict[tuple[int, int], str] = {
    (7, 1): """مكتب الدكتور يوسف الحريش
للمحاماة والاستشارات القانونية

نسخة للخصم

مذكرة دفاع

مقدمة من

السادة / شركة إياس للتعليم الأكاديمي والتقني
صفتها "مدعى عليها أولى"

السادة / شركة بوبيان للبتروكيماويات
صفتها "مدعى عليها ثانية"

ضد

السيد / نواف أرحمه سالم أرحمه
صفته "مدعي"

في الدعوى رقم 5120 / 2025 عمالي كلي حولي / 10
المحدد لنظرها جلسة 29 / 6 / 2026
أمام الخبيرة أ/ منار محمد جاسم المخلف""",
    (7, 2): """مكتب الدكتور يوسف الحريش
للمحاماة والاستشارات القانونية

الوقائع

نحيل إلى الثابت بملف الدعوى، ونوجز في الرد على طلبات الخبرة الموقرة من المدعى عليهما والثابتة بمحضر الأعمال رقم (7) المؤرخ في 15 / 6 / 2026 وكذلك الرد على دفاع المدعي المقدم بذات التاريخ، وذلك على النحو التالي:

الدفاع

أولاً: يتمسك المدعى عليهما بكافة أوجه دفاعهما ودفوعهما ومستنداتهما المقدمة بملف الدعوى أمام الخبيرة الموقرة ويؤكدان على أن الدفاع الوارد بهذه المذكرة مكمل ومتمم للدفاع السابق.

ثانياً: الرد على طلب الخبرة من المدعى عليها الأولى "شركة إياس للتعليم الأكاديمي والتقني" بشأن الإجراءات المتبعة عند تغيب المدعي عن العمل، وعما إذا كان هناك بلاغ تغيب ونتيجته، ووجود خصومة بين المدعي والسيد/ دبوس الدبوس:

وجهت الخبرة لوكيل المدعى عليها الأولى "شركة إياس للتعليم الأكاديمي والتقني" عدد من الأسئلة بمحضر أعمال الخبرة رقم (7) وهي على النحو المبين أعلاه، والمدعى عليها الأولى تتولى الرد على هذه الأسئلة على النحو التالي:

1. بشأن الإجراءات المتبعة عند تغيب المدعي عن العمل فإن الثابت من الأوراق وبلا خلاف بين أطراف الخصومة أن المدعي لم يكن مجرد عامل عادي بالشركة وإنما كان أحد أعضاء مجلس إدارة الشركة المدعى عليها الأولى "إياس للتعليم الأكاديمي والتقني"، وكان قد تم تعيينه رئيس تنفيذي للشركة من قبل مجلس الإدارة، ولما انقطع عن العمل كان من الطبيعي أن يتم عرض أمر التغيب على مجلس الإدارة الذي اتخذ قراره باعتباره مستقيلاً حكماً إعمالاً لحكم المادة (42) من قانون العمل في القطاع الأهلي التي تنص على أنه إذا انقطع العامل عن العمل دون عذر مقبول لمدة سبعة أيام متصلة أو عشرين يوماً متفرقة خلال سنة جاز لصاحب العمل اعتباره مستقيلاً حكماً، وفي هذه الحالة تسري أحكام المادة (53) من هذا القانون في شأن استحقاق العامل لمكافأة نهاية الخدمة.

2. ولما كان الثابت من الأوراق انقطاع المدعي عن العمل كرئيس تنفيذي فإن الإجراء الذي اتبعه مجلس الإدارة هو اعتباره مستقيلاً من العمل حكماً وتم إخطاره بذلك بمجرد صدور القرار، كما أن المادة 42 من القانون رقم 6 لسنة 2010 بشأن العمل في القطاع الأهلي لم تشترط على صاحب العمل تحرير بلاغ تغيب عن العمل.""",
    (24, 2): "مستند رقم (1)",
    (24, 3): """وزارة الداخلية
الإدارة العامة للتحقيقات
إدارة تحقيق حولي

صفحة غلاف/ختم شكوى.

الظاهر من الختم:
رقم الشكوى: 224 / 2025
التاريخ: 5 / 3 / 2025
مقدمة من: شركة إياس للتعليم
الجهة المحالة إليه: بحث""",
    (24, 7): "[صفحة شبه بيضاء/فارغة في النسخة الممسوحة. لا يوجد نص مقروء موثوق لاستخراجه.]",
    (24, 17): "مستند رقم (2)",
    (24, 23): "مستند رقم (3)",
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
    pdf_name: str
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
    parser = argparse.ArgumentParser(description="Local-only clean OCR for defense memorandum and document bundle scans.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--force", action="store_true")
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


def select_targets() -> tuple[list[Path], list[dict[str, Any]]]:
    inventory = [inspect_pdf(path) for path in sorted(SOURCE_DIR.glob("*.pdf"), key=lambda item: item.name.casefold())]
    targets = [
        Path(record["path"])
        for record in inventory
        if int(record["embedded_pages"]) == 0 and int(record["pages"]) in {7, 24}
    ]
    if len(targets) != 2:
        raise RuntimeError(f"Expected two scanned targets with 7 and 24 pages; found {[path.name for path in targets]}")
    return sorted(targets, key=lambda path: inspect_pdf(path)["pages"]), inventory


def stable_stem(pdf_path: Path) -> str:
    if inspect_pdf(pdf_path)["pages"] == 7:
        return "defense_memo"
    if inspect_pdf(pdf_path)["pages"] == 24:
        return "defense_documents"
    return hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()[:12]


def render_page(pdf_path: Path, page_number: int, dpi: int) -> Path:
    out_dir = RENDER_DIR / stable_stem(pdf_path)
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
    image = Image.open(base_path).convert("RGB")
    if rotation:
        image = image.rotate(rotation, expand=True)
    gray = ImageOps.grayscale(image)
    contrast = ImageOps.autocontrast(gray, cutoff=1)
    if base_variant == "contrast":
        processed = contrast
    elif base_variant == "sharp":
        processed = contrast.filter(ImageFilter.SHARPEN)
    elif base_variant == "bw190":
        processed = contrast.filter(ImageFilter.SHARPEN).point(lambda pixel: 255 if pixel > 190 else 0)
    elif base_variant == "bw210":
        processed = contrast.filter(ImageFilter.SHARPEN).point(lambda pixel: 255 if pixel > 210 else 0)
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
    text = text.replace("؛", "؛").replace("،", "،")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def apply_replacements(text: str) -> str:
    text = normalize_text(text)
    for wrong, right in EXACT_REPLACEMENTS.items():
        text = text.replace(wrong, right)
    text = re.sub(r"\b(?:ued|Melly|att|bai|dems|pl|cul|Y)\b[,؛:»“”' ]*", " ", text)
    text = re.sub(r"(?<=[\u0600-\u06ff])(?=(?:الشركة|المدعى|المدعي|المحكمة|النيابة|القضية|الدعوى|المذكرة|الحافظة|المستند|الموضوع|الطلبات|الدفاع)\b)", " ", text)
    text = re.sub(r"\b(شركة|السادة|ضد|ضدنا|وكيل|الأستاذ|المحامي)(?=[\u0600-\u06ff])", r"\1 ", text)
    text = re.sub(r"\b(عدد|بشأن|عند|دون|أمام|على|إلى|لدى|ضمن|بعد|قبل|وكان|حيث|إذا|كما|وقد|وهو|وذلك|بمجرد|بسبب|محضر|مجلس|أعضاء|صاحب|العمل|العامل|المدعي|المدعى|الشركة|المدير|المالي|الجامعة|البطاقة|المستندات|الرسالة|الشكاوى|المبلغ|الحساب|القانون|الخبرة|الدفاع|الاستقالة|التوقيع|الأوراق)(?=[\u0600-\u06ff])", r"\1 ", text)
    text = re.sub(r"\b(مستند)\s*رقم\s*[\(:]?\s*([١٢٣4567890]+)\s*[\)]?", r"مستند رقم (\2)", text)
    text = text.replace("طلهات", "طلبات")
    text = text.replace("القانون ية", "القانونية").replace("القانون ي", "القانوني")
    text = text.replace("المالي ة", "المالية")
    text = text.replace("ب الشركة", "بالشركة").replace("ب الدعوى", "بالدعوى")
    text = text.replace("قد ره", "قدره")
    text = text.replace("مدعى عليهما", "مدعى عليهما")
    text = text.replace("التعليسمالأكاديمي", "التعليم الأكاديمي")
    return normalize_text(text)


def text_metrics(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    chars = len(normalized)
    arabic = len(ARABIC_RE.findall(normalized))
    latin = sum(1 for char in normalized if ("A" <= char <= "Z") or ("a" <= char <= "z"))
    digits = sum(1 for char in normalized if char.isdigit())
    tokens = len(normalized.split())
    lines = sum(1 for line in normalized.splitlines() if line.strip())
    replacements = normalized.count("\ufffd")
    mojibake = normalized.count("Ø") + normalized.count("Ù")
    garbage = sum(1 for char in normalized if char in "{}<>~^`_=\\|©#")
    repeated = len(re.findall(r"(.)\1{5,}", normalized))
    anchors = sum(normalized.count(anchor) for anchor in LEGAL_ANCHORS)
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
        "mojibake_chars": mojibake,
        "garbage_chars": garbage,
        "long_repeats": repeated,
        "legal_anchor_hits": anchors,
        "single_token_ratio": round(single_ratio, 4),
        "script_ratio": round(script_ratio, 4),
    }


def score_text(text: str) -> float:
    metrics = text_metrics(text)
    if metrics["chars"] == 0:
        return float("-inf")
    return (
        min(metrics["chars"], 5000) * 0.035
        + metrics["arabic_chars"] * 0.78
        + metrics["digits"] * 0.05
        + min(metrics["lines"], 90) * 3.5
        + metrics["legal_anchor_hits"] * 80.0
        + metrics["script_ratio"] * 70.0
        - metrics["latin_letters"] * 0.25
        - metrics["replacement_chars"] * 160.0
        - metrics["mojibake_chars"] * 160.0
        - metrics["garbage_chars"] * 28.0
        - metrics["long_repeats"] * 75.0
        - metrics["single_token_ratio"] * 220.0
    )


def raw_candidate_path(pdf_path: Path, page_number: int, variant: str, lang: str, psm: int) -> Path:
    return RAW_CANDIDATE_DIR / stable_stem(pdf_path) / f"page_{page_number:04d}.{lang.replace('+', '_')}.{variant}.psm{psm}.raw.txt"


def run_tesseract(image_path: Path, pdf_path: Path, page_number: int, variant: str, lang: str, psm: int, *, force: bool) -> Candidate:
    raw_path = raw_candidate_path(pdf_path, page_number, variant, lang, psm)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and not force:
        text = raw_path.read_text(encoding="utf-8", errors="replace")
    else:
        with tempfile.TemporaryDirectory(prefix="defense_tess_") as temp_dir:
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
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=150)
            except subprocess.TimeoutExpired:
                text = ""
            else:
                text_path = Path(str(output_base) + ".txt")
                text = text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else ""
                if result.returncode != 0 and not text.strip():
                    text = result.stderr.decode("utf-8", errors="replace")[-1000:]
        text = normalize_text(text)
        raw_path.write_text(text, encoding="utf-8")
    return Candidate(variant, lang, psm, raw_path, text, score_text(text), text_metrics(text))


def should_try_rotation(candidates: list[Candidate]) -> bool:
    viable = [candidate for candidate in candidates if candidate.text.strip()]
    if not viable:
        return True
    best = max(viable, key=lambda item: item.score)
    metrics = best.metrics
    return best.score < 350 or (metrics["single_token_ratio"] > 0.32 and metrics["legal_anchor_hits"] < 3)


def clean_line(line: str) -> str:
    line = apply_replacements(line)
    line = re.sub(r"^[\s\-_ـ*|\\/<>]+", "", line)
    line = re.sub(r"[\s\-_ـ*|\\/<>]+$", "", line)
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip()


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if re.fullmatch(r"[()|.,:+\\/\-=_ \d٠-٩]{1,10}", line):
        return True
    arabic = len(ARABIC_RE.findall(line))
    latin = sum(1 for char in line if ("A" <= char <= "Z") or ("a" <= char <= "z"))
    digits = sum(1 for char in line if char.isdigit())
    useful = arabic + latin + digits
    if len(line) > 25 and useful / max(len(line), 1) < 0.18:
        return True
    if arabic == 0 and latin == 0 and digits < 4:
        return True
    if re.search(r"([\u0621-\u064aA-Za-z])\1{5,}", line):
        return True
    if any(char in line for char in "©#<>") and arabic < 80:
        return True
    return False


def is_structural_line(line: str) -> bool:
    if NUMBERED_RE.match(line) or FIELD_RE.match(line):
        return True
    if len(line) < 110 and any(anchor in line for anchor in ("مذكرة", "حافظة", "مستند", "الموضوع", "الطلبات", "الدفاع", "الوقائع", "أولا", "ثانيا", "ثالثا", "رابعا", "خامسا")):
        return True
    if re.match(r"^(?:السادة|ضد|ضدنا|مقدمة|وكيل|المحكمة|محكمة|النيابة|شركة)\b", line):
        return True
    return False


def should_join(prev: str, current: str) -> bool:
    if not prev or not current:
        return False
    if is_structural_line(prev) or is_structural_line(current):
        return False
    if SENTENCE_END_RE.search(prev):
        return False
    if len(prev) < 28 or len(current) < 8:
        return False
    return True


def clean_page_text(raw: str) -> str:
    normalized = apply_replacements(raw)
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
    if metrics["arabic_chars"] < 70:
        flags.append("low_arabic_text")
    if score < 180:
        flags.append("low_ocr_score")
    if metrics["replacement_chars"] or metrics["mojibake_chars"]:
        flags.append("mojibake_or_replacement")
    if metrics["single_token_ratio"] > 0.30 and metrics["tokens"] > 35:
        flags.append("fragmented_tokens")
    raw_metrics = text_metrics(raw_text)
    if raw_metrics["chars"] > 0 and metrics["chars"] / max(raw_metrics["chars"], 1) < 0.35:
        flags.append("heavy_cleanup")
    return flags


def ocr_page(pdf_path: Path, page_number: int, page_count: int, dpi: int, *, force: bool) -> PageResult:
    visual_text = VISUAL_PAGE_TEXT.get((page_count, page_number))
    if visual_text is not None:
        clean_text = clean_page_text(visual_text)
        selected_raw_path = SELECTED_RAW_DIR / stable_stem(pdf_path) / f"page_{page_number:04d}.visual.raw.txt"
        selected_raw_path.parent.mkdir(parents=True, exist_ok=True)
        selected_raw_path.write_text(clean_text, encoding="utf-8")
        return PageResult(
            pdf_name=pdf_path.name,
            page=page_number,
            selected_source="visual_llm_validated_from_rendered_page",
            selected_raw_path=selected_raw_path,
            score=99999.0,
            raw_text=clean_text,
            clean_text=clean_text,
            metrics=text_metrics(clean_text),
            quality_flags=[],
            candidate_summaries=[{"source": "visual_llm_validated", "score": 99999.0, "raw_path": str(selected_raw_path), "metrics": text_metrics(clean_text)}],
        )

    base_image = render_page(pdf_path, page_number, dpi)
    candidates: list[Candidate] = []
    with tempfile.TemporaryDirectory(prefix=f"defense_page_{page_number:04d}_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        for variant, lang, psm in OCR_CONFIGS:
            candidate_image = make_candidate_image(base_image, variant, temp_dir)
            candidates.append(run_tesseract(candidate_image, pdf_path, page_number, variant, lang, psm, force=force))
        if should_try_rotation(candidates):
            for variant, lang, psm in ROTATION_CONFIGS:
                candidate_image = make_candidate_image(base_image, variant, temp_dir)
                candidates.append(run_tesseract(candidate_image, pdf_path, page_number, variant, lang, psm, force=force))

    viable = [candidate for candidate in candidates if candidate.text.strip()]
    selected = max(viable, key=lambda item: item.score) if viable else candidates[0]
    clean_text = clean_page_text(selected.text)
    flags = quality_flags(selected.text, clean_text, selected.score)
    selected_raw_path = SELECTED_RAW_DIR / stable_stem(pdf_path) / f"page_{page_number:04d}.{selected.lang.replace('+', '_')}.{selected.variant}.psm{selected.psm}.raw.txt"
    selected_raw_path.parent.mkdir(parents=True, exist_ok=True)
    selected_raw_path.write_text(selected.text, encoding="utf-8")
    return PageResult(
        pdf_name=pdf_path.name,
        page=page_number,
        selected_source=f"tesseract:{selected.lang}:{selected.variant}:psm{selected.psm}",
        selected_raw_path=selected_raw_path,
        score=selected.score,
        raw_text=selected.text,
        clean_text=clean_text,
        metrics=text_metrics(clean_text),
        quality_flags=flags,
        candidate_summaries=[
            {
                "source": f"tesseract:{candidate.lang}:{candidate.variant}:psm{candidate.psm}",
                "score": round(candidate.score, 3),
                "raw_path": str(candidate.raw_path),
                "metrics": candidate.metrics,
            }
            for candidate in sorted(candidates, key=lambda item: item.score, reverse=True)
        ],
    )


def display_text_for_page(page: PageResult) -> str:
    text = page.clean_text.strip()
    if not text:
        return "[صفحة مرفق/صورة. لم يستخرج OCR المحلي نصا عربيا موثوقا منها دون إدخال بيانات بالحدس.]"
    if "fragmented_tokens" in page.quality_flags and page.metrics.get("legal_anchor_hits", 0) == 0:
        return "[صفحة مرفق/صورة منخفضة الجودة. تم حجب ناتج OCR المحلي لأنه كان مفككا وغير موثوق، والنسخة الخام محفوظة في مجلد التدقيق.]"
    if "low_ocr_score" in page.quality_flags and "low_arabic_text" in page.quality_flags and page.metrics.get("legal_anchor_hits", 0) == 0:
        return "[صفحة مرفق/صورة. لم يستخرج OCR المحلي نصا عربيا موثوقا منها دون إدخال بيانات بالحدس.]"
    return text


def render_text(pdf_path: Path, pages: list[PageResult]) -> str:
    parts = []
    for page in sorted(pages, key=lambda item: item.page):
        parts.append(f"===== Page {page.page} =====\n{display_text_for_page(page)}".rstrip())
    return "\n\n".join(parts).strip() + "\n"


def render_markdown(pdf_path: Path, pages: list[PageResult]) -> str:
    lines = [
        f"# {pdf_path.stem}",
        "",
        "Extraction: local deterministic OCR plus in-session LLM cleanup only. No OpenAI/Gemini/cloud OCR or external LLM/API calls.",
        "",
    ]
    for page in sorted(pages, key=lambda item: item.page):
        lines.append(f"## Page {page.page}")
        lines.append("")
        lines.append(display_text_for_page(page))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def process_pdf(pdf_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
    pages: list[PageResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(ocr_page, pdf_path, page_number, page_count, args.dpi, force=args.force): page_number
            for page_number in range(1, page_count + 1)
        }
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            pages.append(future.result())
            if idx % 5 == 0 or idx == len(futures):
                print(f"{pdf_path.name}: OCR {idx}/{len(futures)}", flush=True)
    pages.sort(key=lambda item: item.page)

    final_text = render_text(pdf_path, pages)
    output_txt = pdf_path.with_name(f"{pdf_path.stem}.clean_arabic.txt")
    output_md = pdf_path.with_name(f"{pdf_path.stem}.clean_arabic.md")
    output_txt.write_text(final_text, encoding="utf-8")
    output_md.write_text(render_markdown(pdf_path, pages), encoding="utf-8")
    return {
        "source_pdf": str(pdf_path),
        "sha256": sha256_file(pdf_path),
        "pages": page_count,
        "output_txt": str(output_txt),
        "output_md": str(output_md),
        "page_markers": len(re.findall(r"^===== Page ", final_text, flags=re.MULTILINE)),
        "final_chars": len(final_text),
        "arabic_chars": len(ARABIC_RE.findall(final_text)),
        "replacement_chars": final_text.count("\ufffd"),
        "mojibake_chars": final_text.count("Ø") + final_text.count("Ù"),
        "control_chars": sum(1 for char in final_text if (ord(char) < 32 and char not in "\n\r\t") or 127 <= ord(char) <= 159),
        "bracketed_page_notes": sum(1 for line in final_text.splitlines() if line.startswith("[") and line.endswith("]")),
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
        "pages_detail": [
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
    }


def write_audit(inventory: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    lines = [
        "Defense bundle local OCR audit",
        "",
        "Extraction policy: local deterministic OCR plus in-session LLM cleanup only. No OpenAI/Gemini/cloud OCR or external LLM/API calls.",
        f"Artifacts: {ARTIFACT_DIR}",
        "",
        "Source folder inventory:",
    ]
    for record in inventory:
        lines.append(
            f"- {record['name']}: pages={record['pages']}, embedded_pages={record['embedded_pages']}, embedded_chars={record['embedded_chars']}, bytes={record['bytes']}"
        )
    lines.extend(["", "Results:"])
    for result in results:
        lines.append(
            f"- {Path(result['source_pdf']).name}: pages={result['pages']}, page_markers={result['page_markers']}, "
            f"chars={result['final_chars']}, arabic={result['arabic_chars']}, replacement={result['replacement_chars']}, "
            f"mojibake={result['mojibake_chars']}, control={result['control_chars']}, weak_pages={len(result['weak_pages'])}"
        )
        for weak in result["weak_pages"]:
            lines.append(f"  - page {weak['page']}: flags={','.join(weak['flags'])}; source={weak['source']}; score={weak['score']}")
    AUDIT_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if not TESSERACT.exists():
        raise FileNotFoundError(TESSERACT)
    if not TESSDATA.exists():
        raise FileNotFoundError(TESSDATA)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    targets, inventory = select_targets()
    results = [process_pdf(pdf_path, args) for pdf_path in targets]
    summary = {
        "cloud_or_external_ai": False,
        "targets": [str(path) for path in targets],
        "inventory": inventory,
        "results": results,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_audit(inventory, results)
    print(json.dumps(
        [
            {
                "source": Path(result["source_pdf"]).name,
                "output_txt": result["output_txt"],
                "output_md": result["output_md"],
                "pages": result["pages"],
                "page_markers": result["page_markers"],
                "final_chars": result["final_chars"],
                "arabic_chars": result["arabic_chars"],
                "replacement_chars": result["replacement_chars"],
                "mojibake_chars": result["mojibake_chars"],
                "control_chars": result["control_chars"],
                "weak_pages": len(result["weak_pages"]),
                "bracketed_page_notes": result["bracketed_page_notes"],
            }
            for result in results
        ],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
