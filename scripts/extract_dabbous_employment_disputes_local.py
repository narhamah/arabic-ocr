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

import fitz
from PIL import Image, ImageFilter, ImageOps


FOLDER = Path(r"C:\Users\narha\Dropbox\Dabbous Allegations\04 - Employment Disputes")
TARGET_NAMES = [
    "SKM_36726042300310.pdf",
    "SKM_36726042300311.pdf",
    "SKM_36726060201340.pdf",
    "SKM_36726060201330.pdf",
]
ARTIFACT_DIR = FOLDER / "employment_disputes_ocr_artifacts"
RENDER_DIR = ARTIFACT_DIR / "rendered_pages_z3"
RENDER_300_DIR = ARTIFACT_DIR / "rendered_pages_dpi300"
PREPROCESSED_DIR = ARTIFACT_DIR / "preprocessed_pages_dpi300"
RAW_DIR = ARTIFACT_DIR / "raw_tesseract_pages"
CANDIDATE_RAW_DIR = ARTIFACT_DIR / "raw_tesseract_candidates_v2"
SUMMARY_PATH = ARTIFACT_DIR / "employment_disputes_local_ocr_summary.json"

TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA = Path(r"C:\Users\narha\arabic-ocr\.tessdata")

ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
BIDI_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
SENTENCE_END_RE = re.compile(r"[.؟!؛:]$")
DIGIT_TRANS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

OCR_CANDIDATES: tuple[tuple[str, int], ...] = (
    ("z3_raw", 4),
    ("z3_raw", 6),
    ("dpi300_contrast", 4),
    ("dpi300_contrast", 6),
    ("dpi300_bw200", 4),
)

ENGLISH_OCR_CANDIDATES: tuple[tuple[str, int], ...] = (
    ("z3_raw", 4),
    ("z3_raw", 6),
    ("dpi300_contrast", 6),
    ("dpi300_bw200", 6),
)

ENGLISH_ANCHORS = (
    "Final Settlement",
    "Bank Payment Voucher",
    "Boubyan Petrochemical",
    "Joining date",
    "Last working",
    "Working days",
    "Basic Salary",
    "Total Package",
    "Leave Salary",
    "Indemnity",
    "Employee Signature",
    "Approved by",
    "Delivered",
    "Outlook",
    "Your message has been delivered",
    "Subject",
    "Date",
    "Amount",
    "Pay to the order",
    "Cheque",
    "K.D",
)

FORCE_ARABIC_PAGES = {
    "SKM_36726042300311.pdf": {31},
}

SEPARATOR_PAGE_TEXT = {
    "SKM_36726042300311.pdf": {
        4: "مستند رقم (١)",
        6: "مستند رقم (٢)",
        8: "مستند رقم (٣)",
        11: "مستند رقم (٤)",
        13: "مستند رقم (٥)",
        15: "مستند رقم (٦)",
        21: "مستند رقم (٧)",
        24: "مستند رقم (٨)",
        27: "مستند رقم (٩)",
        30: "مستند رقم (١٠)",
    },
    "SKM_36726060201340.pdf": {
        2: "مستند رقم (١)",
        13: "مستند رقم (٢)",
        20: "مستند رقم (٣)",
    },
}

MANUAL_PAGE_TEXT = {
    "SKM_36726042300311.pdf": {
        9: """Final Settlement

ID: 1022
Name: Nawaf Arhamah
Status: Resignation

Joining date: 27/04/2015
Last working day: 30/06/2024

Working days: 3353
Working years: 9.19
Unpaid Days: 0
Actual Working days: 3353
Actual Working years: 9.19
Less Than 5 years: 1825 days / 5.00 years
More Than 5 years: 1528 days / 4.19 years

Package:
Basic Salary: 9,000
Other Allowances: -
Total Package: 9,000

Settlement factors:
Leave Salary: 54.50 days / 18,865.38
Notice Period: 0.00 days / -
Salary: 0.00 days / -
Bonus: 0.00 days / 32,500.00
Indemnity on Basic: 184.00 days / 63,692.31
Accounting Deduction: -
Salary Deduction: -
Total: 115,057.69

Signature fields:
Employee Signature
H.R Dep
Approved by""",
        10: """Bank Payment Voucher
Boubyan Petrochemical Company K.S.C

Date: 15/07/2024
Number: Y24-0002926/
Check No: 4017
Ref. No: 0055
Drawn Bank: ABK - Over Draft - 069850 201
Paid To: Mr. Nawaf Arhamah Salem Arhamah
Details: Final Settlement - Nawaf Arhamah

Received From Boubyan Petrochemical Company K.S.C the sum of K.D: 115,057.690
Being as under mentioned

A/C No: 202-00106
Account Description: Accounts Payable and Accruals / Accrued leaves
Amount: 18,865.380

A/C No: 203-00220
Account Description: Other credit balances / Provision for Bonus
Amount: 32,500.000

A/C No: 202-00029
Account Description: Non Current Liabilities / Provision for Indemnity
Amount: 63,692.310

User: RAMY
Registered: 15/07/2024
Total: 115,057.690

Signature fields:
General Manager
Financial Manager
Accountant
Recipient Information
Name
ID Type/Number
Signature""",
        14: """HEAD OFFICE BRANCH
AL AHLI BANK OF KUWAIT K.S.C.P.

Pay to the order of: Mr. Nawaf Arhamah Salim Arhamah
Date: 15/07/2024
Currency: Kuwaiti Dinar
Amount in words: One Hundred Fifteen Thousand Fifty Seven and Fils 690 / 1000 Only.
Amount: KD 115,057.690
Drawer: BOUBYAN PETROCHEMICALS CO.

Cheque Number: 000004017
Sort Code: 000400
Account Number: 069850201
Signature present.""",
        31: """Outlook
إخطار

From: Huda Al Tamimi <haltamami@boubyan.com>
Date: Wed 8/20/2025 10:10 AM
To: Nawaf Arhamah <narhamah@boubyan.com>
Attachment: 1 attachment (147 KB)
Attachment filename: إخطار بانقطاع عن العمل.pdf

السيد / نواف أرحمه أرحمه المحترم
تحية طيبة وبعد،،

يرجى الاطلاع على الإخطار المرفق طيه والإحاطة به.

وتفضلوا بقبول فائق التحية.

هدى التميمي
أمين سر مجلس الإدارة
شركة إياس للتعليم الأكاديمي والتقني""",
        32: """Outlook
Delivered: إخطار

From: Microsoft Outlook <MicrosoftExchange329e71ec88ae4615bbc36ab6ce41109e@boubyan.com>
Date: Wed 8/20/2025 10:11 AM
To: Nawaf Arhamah <narhamah@boubyan.com>
Attachment: 1 attachment (163 KB)
Attachment filename: إخطار؛

Your message has been delivered to the following recipients:
Nawaf Arhamah (narhamah@boubyan.com)

Subject: إخطار""",
    },
}

EXACT_CORRECTIONS = {
    "اشركسة": "شركة",
    "شرقسة": "شركة",
    "شركسة": "شركة",
    "بوبيسان": "بوبيان",
    "بوبيبان": "بوبيان",
    "للبتروتيماويسات": "للبتروكيماويات",
    "للبتروكيماويسات": "للبتروكيماويات",
    "للتصليم": "للتعليم",
    "الأقتاكا": "التقني",
    "اللقتوا": "التقني",
    "التقنر": "التقني",
    "اياس": "إياس",
    "إياس": "إياس",
    "أرحمه": "أرحمه",
    "نواف أرحمه سالم أرحمه": "نواف أرحمه سالم أرحمه",
    "عضوأً": "عضواً",
    "عضوأ": "عضواً",
    "عضواً مستقلاً": "عضواً مستقلاً",
    "المراقب مالي": "المراقب المالي",
    "أمين السر": "أمين السر",
    "ماير": "مايو",
    "بصصائر": "بصائر",
    "بصسائر": "بصائر",
    "المرائمات": "المرافعات",
    "المرفمات": "المرافعات",
    "المسادة": "المادة",
    "قانون المرقمات": "قانون المرافعات",
    "قانون المرائمات": "قانون المرافعات",
    "طيقا": "طبقا",
    "مفرر": "مقرر",
    "تعفد": "تعقد",
    "بالاني": "بالآتي",
    "بنسدب": "بندب",
    "حبرا": "خبراء",
    "برام وزارة": "خبراء وزارة",
    "وزارة العسدل": "وزارة العدل",
    "تنسدب": "تندب",
    "أحسد": "أحد",
    "خبرانهسا": "خبرائها",
    "خبرائيسا": "خبرائها",
    "المختمسين": "المختصين",
    "تكسون": "تكون",
    "مهمتسه": "مهمته",
    "ملسف": "ملف",
    "السدهوى": "الدعوى",
    "مابه": "ما به",
    "مسا هسى": "ما عسى",
    "يقدسه": "يقدمه",
    "أطسراف": "أطراف",
    "أمراف": "أطراف",
    "التنامي": "التداعي",
    "وألسك": "وذلك",
    "لبيسان": "لبيان",
    "طبيمسة": "طبيعة",
    "العلاقسة": "العلاقة",
    "وبيسان": "وبيان",
    "التمساملات": "التعاملات",
    "التعساملات": "التعاملات",
    "والتماقسدات": "والتعاقدات",
    "التجاريسة": "التجارية",
    "فيا بينهم": "فيما بينهم",
    "رمسا فيتها": "وما فيها",
    "وقيمتبا": "وقيمتها",
    "ذلسك": "ذلك",
    "هما إذكسان": "عما إذا كان",
    "هنساك": "هناك",
    "نمسة مديوئية": "ثمة مديونية",
    "مديوئية": "مديونية",
    "لصسالع": "لصالح",
    "العالسة": "الحالة",
    "لاولى": "الأولى",
    "قيمتيسا": "قيمتها",
    "مسببها": "سببها",
    "وكسذلك": "وكذلك",
    "العساب": "الحساب",
    "التسداعي": "التداعي",
    "تماملان": "تعاملات",
    "ينهم": "بينهم",
    "الشسيكات": "الشيكات",
    "البنكيية": "البنكية",
    "رقمسي": "رقمي",
    "المسسادرين": "الصادرين",
    "الطالبسة": "الطالبة",
    "المعلسن": "المعلن",
    "إلييسا": "إليها",
    "الثانيسة": "الثانية",
    "البسالغ": "المبالغ",
    "الاونة": "المدونة",
    "مسبيها": "سببها",
    "أضساً": "أيضاً",
    "تكليسف": "تكليف",
    "السسيد": "السيد",
    "الشسيير": "الخبير",
    "لغسبير": "الخبير",
    "بالإنتقسال": "بالانتقال",
    "مسري": "مقري",
    "إلسيهم": "إليهم",
    "علس": "على",
    "كافسة": "كافة",
    "يوسفالحربش": "يوسف الحربش",
    "يوسفالحريش": "يوسف الحربش",
    "الإستشاراتالقانونية": "الإستشارات القانونية",
    "مقدمةمن": "مقدمة من",
    "مذقرة ينفاع": "مذكرة بدفاع",
    "مذصرة بدفاع": "مذكرة بدفاع",
    "صذدصرةبدفاع": "مذكرة بدفاع",
    "اشركة": "شركة",
    "اللمرئسةإيسا اس": "شركة إياس",
    "إيسا اس": "إياس",
    "إيساس": "إياس",
    "إباس": "إياس",
    "اياس": "إياس",
    "التعليسم": "التعليم",
    "التعلسيم": "التعليم",
    "للتتليسم": "للتعليم",
    "الأكاديمسى": "الأكاديمي",
    "الأكاديمسي": "الأكاديمي",
    "الأكاديصي": "الأكاديمي",
    "الأكاديسي": "الأكاديمي",
    "الأكتاديسي": "الأكاديمي",
    "الاكاديمايئ": "الأكاديمي",
    "الاكاديميا": "الأكاديمي",
    "الاكاديمي": "الأكاديمي",
    "والتقننسي": "والتقني",
    "والتقننسى": "والتقني",
    "والتقنسي": "والتقني",
    "والتقنسى": "والتقني",
    "والتفمسي": "والتقني",
    "و التقنايا": "والتقني",
    "التقنايا": "التقني",
    "النقنايا": "التقني",
    "البتر وكيماويات": "البتروكيماويات",
    "البتروتيماويات": "البتروكيماويات",
    "البتروحيماويات": "البتروكيماويات",
    "البتروشيماويات": "البتروكيماويات",
    "البتروئيماؤويسات": "البتروكيماويات",
    "البتروضماويسات": "البتروكيماويات",
    "للبتروحيماويات": "للبتروكيماويات",
    "للبتروشيماويات": "للبتروكيماويات",
    "للبتروئيماؤويسات": "للبتروكيماويات",
    "للبتروكيماويسات": "للبتروكيماويات",
    "بوؤبيسان": "بوبيان",
    "بؤبيان": "بوبيان",
    "مدمسى": "مدعى",
    "مدسى": "مدعى",
    "مدميى": "مدعى",
    "مدصي": "مدعي",
    "الخدصي": "المدعي",
    "المسدعى عليهما": "المدعى عليهما",
    "السدعى عليهما": "المدعى عليهما",
    "المدصي عليهما": "المدعى عليهما",
    "المدصى عنيهما": "المدعى عليهما",
    "المدعى عنيهما": "المدعى عليهما",
    "المدعى عنيخما": "المدعى عليهما",
    "المدصى عليها": "المدعى عليها",
    "المسدعى": "المدعي",
    "السدعى": "المدعي",
    "السدعي": "المدعي",
    "المدصي": "المدعي",
    "مدعسى": "مدعى",
    "مدمى": "مدعى",
    "مدمي": "مدعي",
    "صنتها": "صفتها",
    "صندها": "صفتها",
    "صفدها": "صفتها",
    "صلفته": "صفته",
    "عليشا": "عليها",
    "عليشماأولى": "عليها أولى",
    "عليماأولى": "عليها أولى",
    "عليما أولى": "عليها أولى",
    "أولسسى": "أولى",
    "أواسسى": "أولى",
    "أواسى": "أولى",
    "علياسماتائيسة": "عليهما ثانية",
    "عليماقائيسة": "عليهما ثانية",
    "عليما ثائيسة": "عليهما ثانية",
    "عليهساثائيسة": "عليها ثانية",
    "عليهاثانية": "عليها ثانية",
    "أرحصه": "أرحمه",
    "أرهمه": "أرحمه",
    "ارحمه": "أرحمه",
    "سائم": "سالم",
    "خدمتته": "خدمته",
    "خدمتسه": "خدمته",
    "لسدى": "لدى",
    "شسركة": "شركة",
    "شركةبوبيان": "شركة بوبيان",
    "شركةاياس": "شركة إياس",
    "شركةإياس": "شركة إياس",
    "ذسترة": "فترة",
    "سترةعمل": "فترة عمل",
    "ذترةعمل": "فترة عمل",
    "فترةعمل": "فترة عمل",
    "نسترة": "فترة",
    "ماليةمستقلة": "مالية مستقلة",
    "مجلسالادارة": "مجلس الإدارة",
    "الادارة": "الإدارة",
    "اعضاء": "أعضاء",
    "بانه": "بأنه",
    "يبين جليا": "يبين جلياً",
    "بصسائر": "بصائر",
    "بشسائر": "بصائر",
    "بضائر": "بصائر",
    "بضسائر": "بصائر",
    "بمسائر": "بصائر",
    "الدستوريةالعليا": "الدستورية العليا",
    "المستتدان": "المستندات",
    "المستتدات": "المستندات",
    "مستتدات": "مستندات",
    "لامستظبار": "لاستظهار",
    "لاسستغظبار": "لاستظهار",
    "ستيار": "استظهار",
    "وجسه الحس": "وجه الحق",
    "وجسه التق": "وجه الحق",
    "رجه التق": "وجه الحق",
    "الساهوى": "الدعوى",
    "السدهوى": "الدعوى",
    "السدعوى": "الدعوى",
    "السلاعوى": "الدعوى",
    "التدهون": "الدعوى",
    "الوقتوف": "الوقوف",
    "مليما": "عليها",
    "علسمنا": "على ما",
    "ملسيمننا": "على ما",
    "مديولية": "مديونية",
    "لصسالع": "لصالح",
    "لمسالع": "لصالح",
    "لالع": "لصالح",
    "أيسا": "أي",
    "المعلسن": "المعلن",
    "السيهم": "إليهم",
    "إلسيهم": "إليهم",
    "ممساعمسن": "سماع من",
    "سمساع مسض": "سماع من",
    "سسامسن": "سماع من",
    "شسهود": "شهود",
    "هود أطراف": "شهود أطراف",
    "التسداهي": "التداعي",
    "التسداعي": "التداعي",
    "التداهي": "التداعي",
    "إنلسزم": "إن لزم",
    "إن لزه": "إن لزم",
    "الأأسر": "الأمر",
    "الأسر": "الأمر",
    "ذلسكبيدا": "ذلك تمهيداً",
    "ذلسكقهيدا": "ذلك تمهيداً",
    "ذلسكتهيدا": "ذلك تمهيداً",
    "وذلكبيدا": "وذلك تمهيداً",
    "تقريسر الغسبراء": "تقرير الخبراء",
    "تقرير الغسراء": "تقرير الخبراء",
    "تقرير الخسبراء": "تقرير الخبراء",
    "السزام": "إلزام",
    "إلسزام": "إلزام",
    "المصسروفات": "المصروفات",
    "المصحاريف": "المصاريف",
    "ومقاتل": "ومقابل",
    "ومتاتل": "ومقابل",
    "الأكاديميوالتقني": "الأكاديمي والتقني",
    "الأكاديميوالتقنى": "الأكاديمي والتقني",
    "الأكاديمىوالتقنى": "الأكاديمي والتقني",
    "بوبيانالبتروكيماويات": "بوبيان للبتروكيماويات",
    "بوبيانللبتروكيماويات": "بوبيان للبتروكيماويات",
    "للبتروحيماويسات": "للبتروكيماويات",
    "للبتروكيماويسات": "للبتروكيماويات",
    "شرئسة": "شركة",
    "شرئحة": "شركة",
    "الدورالسادس": "الدور السادس",
    "جاسمالمخلف": "جاسم المخلف",
    "طيبةوبعد": "طيبة وبعد",
    "فائقالتحية": "فائق التحية",
    "هدىالتميمى": "هدى التميمي",
    "هدىالتميمي": "هدى التميمي",
    "مجلسالإدارة": "مجلس الإدارة",
    "التغليم": "التعليم",
    "الأكاديفي": "الأكاديمي",
    "البيسمسسسسسسسسس. سسسسسن": "البيان",
}

REGEX_CORRECTIONS = [
    (re.compile(r"\bفى\b"), "في"),
    (re.compile(r"\s+([،؛:؟,.])"), r"\1"),
    (re.compile(r"([،؛:؟,.])(?=[\u0621-\u064a])"), r"\1 "),
    (re.compile(r"\s*/\s*"), " / "),
    (re.compile(r"\s{2,}"), " "),
]


@dataclass
class PageOcr:
    pdf: str
    page: int
    source: str
    raw_text: str
    rendered_image: str
    raw_path: str
    score: float


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = BIDI_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\u0640", "")
    text = text.translate(str.maketrans({"\u06A9": "\u0643", "\u06CC": "\u064A"}))
    text = text.translate(DIGIT_TRANS)
    # Drop Arabic short vowels; they are mostly OCR noise in these scans.
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    for bad, good in EXACT_CORRECTIONS.items():
        text = text.replace(bad, good)
    for pattern, replacement in REGEX_CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text.strip()


def clean_line(line: str) -> str:
    line = normalize_text(line)
    line = re.sub(r"^[\s\-_ـ*|\\/<>.]+", "", line)
    line = re.sub(r"[\s\-_ـ*|\\/<>.]+$", "", line)
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip()


def strip_repeated_footer_noise(line: str) -> str:
    if not line:
        return line

    footer_markers = (
        "الكويت - شرق",
        "الكويت -شرق",
        "الكويت - شرق --",
        "الكويت - شر ق",
    )
    for marker in footer_markers:
        if line.startswith(marker):
            return ""
        idx = line.find(" " + marker)
        if idx > 0:
            return line[:idx].strip()

    noisy_markers = ("بريد إلكتروني", "بريد إلقكتروني", "تلفون:", "فاكس:")
    if line.startswith(("تلفون", "فاكس", "بريد إلكتروني", "بريد إلقكتروني")):
        return ""
    if any(marker in line for marker in noisy_markers) and len(line) > 80:
        earliest = min([line.find(marker) for marker in noisy_markers if marker in line])
        prefix = line[:earliest].strip(" -")
        return prefix if len(prefix) >= 20 else ""

    return line


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if re.search(r"([\u0621-\u064aA-Za-z])\1{5,}", line):
        return True
    if re.fullmatch(r"[٠-٩ ()/\\|.,:+-]{1,8}", line):
        return True
    arabic = len(ARABIC_RE.findall(line))
    digit_count = sum(ch.isdigit() for ch in line)
    if arabic <= 4 and digit_count >= 8:
        return True
    if arabic == 0 and len(line) < 35:
        return True
    if arabic < 2 and len(line) > 25:
        return True
    if len(line) <= 80 and all(ord(ch) < 128 for ch in line):
        return True
    if len(line) > 40 and arabic / max(len(line), 1) < 0.08:
        return True
    return False


def is_heading_or_list(line: str) -> bool:
    if len(line) < 80 and any(key in line for key in ("مذكرة", "محضر", "جدول الأعمال", "أسباب", "طلبات", "الدفاع", "الموضوع")):
        return True
    if re.match(r"^[٠-٩]+[\-.)،:]\s*", line):
        return True
    if re.match(r"^[أ-ي][\-.)،:]\s+", line):
        return True
    return False


def should_join(prev: str, current: str) -> bool:
    if not prev or not current:
        return False
    if is_heading_or_list(current) or is_heading_or_list(prev):
        return False
    if SENTENCE_END_RE.search(prev):
        return False
    if len(prev) < 25:
        return False
    return True


def clean_page_text(raw: str) -> str:
    lines: list[str] = []
    for original in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = clean_line(original)
        line = strip_repeated_footer_noise(line)
        if is_noise_line(line):
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if lines and should_join(lines[-1], line):
            lines[-1] = normalize_text(f"{lines[-1]} {line}")
        else:
            lines.append(line)
    while lines and lines[-1] == "":
        lines.pop()
    # Collapse repeated blanks but keep natural paragraph spacing.
    collapsed: list[str] = []
    for line in lines:
        if line == "" and (not collapsed or collapsed[-1] == ""):
            continue
        collapsed.append(line)
    return "\n".join(collapsed).strip()


def clean_english_page_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw or "")
    text = BIDI_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\ufeff", "").replace("\u00a0", " ")
    lines: list[str] = []
    for original in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t]{2,}", " ", original.strip())
        line = re.sub(r"^[\s\-_*|\\/<>.]+", "", line)
        line = re.sub(r"[\s\-_*|\\/<>.]+$", "", line)
        if not line:
            continue
        alnum = sum(ch.isalnum() for ch in line)
        if alnum == 0:
            continue
        if len(line) > 30 and alnum / max(len(line), 1) < 0.15:
            continue
        if re.fullmatch(r"[()|.,:+\\/\-=_ ]{1,20}", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def score_text(text: str) -> float:
    normalized = normalize_text(text)
    arabic = len(ARABIC_RE.findall(normalized))
    latin = sum(1 for c in normalized if ("A" <= c <= "Z") or ("a" <= c <= "z"))
    replacements = normalized.count("\ufffd")
    garbage = len(re.findall(r"[<>_~{}\\^`]", normalized))
    legal_terms = sum(normalized.count(term) for term in ("محكمة", "الدعوى", "شركة", "العامل", "مجلس", "المادة", "الخبير"))
    return arabic + legal_terms * 20 - latin * 1.5 - replacements * 100 - garbage * 5


def score_english_text(text: str) -> float:
    normalized = unicodedata.normalize("NFKC", text or "")
    if not normalized.strip():
        return float("-inf")
    latin = sum(1 for c in normalized if ("A" <= c <= "Z") or ("a" <= c <= "z"))
    digits = sum(1 for c in normalized if c.isdigit())
    arabic = len(ARABIC_RE.findall(normalized))
    lines = sum(1 for line in normalized.splitlines() if line.strip())
    anchors = sum(normalized.lower().count(anchor.lower()) for anchor in ENGLISH_ANCHORS)
    table_terms = len(re.findall(r"\b(ID|Name|Date|Amount|Total|Salary|Bonus|Signature|Subject|From|To)\b", normalized, flags=re.I))
    garbage = len(re.findall(r"[{}<>\\^`~]", normalized))
    repeated = len(re.findall(r"(.)\1{5,}", normalized))
    useful_ratio = (latin + digits) / max(len(normalized), 1)

    # Keep this conservative: English wins only on pages with clear English form/email anchors.
    if anchors < 2 and table_terms < 4:
        return float("-inf")
    return (
        latin * 1.4
        + digits * 0.45
        + anchors * 120
        + table_terms * 35
        + min(lines, 80) * 4
        + useful_ratio * 150
        - arabic * 2
        - garbage * 20
        - repeated * 40
    )


def should_try_english(raw_text: str, raw_score: float) -> bool:
    normalized = normalize_text(raw_text)
    arabic = len(ARABIC_RE.findall(normalized))
    latin = sum(1 for c in normalized if ("A" <= c <= "Z") or ("a" <= c <= "z"))
    return raw_score < 700 or (arabic < 450 and latin > 20)


def render_page(pdf_path: Path, page_number: int) -> Path:
    out_dir = RENDER_DIR / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"page_{page_number:04d}.png"
    if img_path.exists():
        return img_path
    with fitz.open(pdf_path) as doc:
        pix = doc[page_number - 1].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
        pix.save(img_path)
    return img_path


def render_page_dpi300(pdf_path: Path, page_number: int) -> Path:
    out_dir = RENDER_300_DIR / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"page_{page_number:04d}.png"
    if img_path.exists():
        return img_path
    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False)
        pix.save(img_path)
    return img_path


def preprocess_image(source_path: Path, pdf_path: Path, page_number: int, variant: str) -> Path:
    out_dir = PREPROCESSED_DIR / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"page_{page_number:04d}.{variant}.png"
    if out_path.exists():
        return out_path

    image = Image.open(source_path).convert("RGB")
    gray = ImageOps.grayscale(image)
    contrast = ImageOps.autocontrast(gray, cutoff=1)
    if variant == "contrast":
        processed = contrast
    elif variant == "bw200":
        processed = contrast.filter(ImageFilter.SHARPEN).point(lambda pixel: 255 if pixel > 200 else 0)
    else:
        processed = contrast
    processed.save(out_path)
    return out_path


def candidate_image_path(pdf_path: Path, page_number: int, variant: str) -> Path:
    if variant == "z3_raw":
        return render_page(pdf_path, page_number)
    if variant.startswith("dpi300_"):
        base = render_page_dpi300(pdf_path, page_number)
        return preprocess_image(base, pdf_path, page_number, variant.removeprefix("dpi300_"))
    raise ValueError(f"Unknown OCR image variant: {variant}")


def run_tesseract(img_path: Path, lang: str, psm: int, raw_path: Path) -> str:
    if raw_path.exists():
        return raw_path.read_text(encoding="utf-8", errors="replace")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(TESSERACT),
        str(img_path),
        str(Path(tempfile.gettempdir()) / f"dabbous_tess_{raw_path.stem}"),
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
        with tempfile.TemporaryDirectory(prefix="dabbous_tess_") as temp_dir:
            output_base = Path(temp_dir) / "ocr"
            cmd[2] = str(output_base)
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=120)
            text_path = Path(str(output_base) + ".txt")
            text = text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else ""
            if result.returncode != 0 and not text.strip():
                text = result.stderr[-1000:].decode("utf-8", errors="replace") if isinstance(result.stderr, bytes) else str(result.stderr)[-1000:]
    except subprocess.TimeoutExpired:
        text = ""

    raw_path.write_text(text, encoding="utf-8")
    return text


def ocr_page(pdf_path: Path, page_number: int) -> PageOcr:
    manual_text = MANUAL_PAGE_TEXT.get(pdf_path.name, {}).get(page_number)
    if manual_text:
        img_path = render_page(pdf_path, page_number)
        raw_dir = RAW_DIR / pdf_path.stem
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"page_{page_number:04d}.visual_manual_transcription.raw.txt"
        raw_path.write_text(manual_text, encoding="utf-8")
        return PageOcr(
            pdf_path.name,
            page_number,
            "visual_manual_transcription_from_scan",
            manual_text,
            str(img_path),
            str(raw_path),
            9999.0,
        )

    separator_text = SEPARATOR_PAGE_TEXT.get(pdf_path.name, {}).get(page_number)
    if separator_text:
        img_path = render_page(pdf_path, page_number)
        raw_dir = RAW_DIR / pdf_path.stem
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"page_{page_number:04d}.visual_separator_label.raw.txt"
        raw_path.write_text(separator_text, encoding="utf-8")
        return PageOcr(
            pdf_path.name,
            page_number,
            "visual_separator_label_from_scan",
            separator_text,
            str(img_path),
            str(raw_path),
            9999.0,
        )

    variants = []
    for image_variant, psm in OCR_CANDIDATES:
        img_path = candidate_image_path(pdf_path, page_number, image_variant)
        candidate_raw_dir = CANDIDATE_RAW_DIR / pdf_path.stem
        raw_candidate_path = candidate_raw_dir / f"page_{page_number:04d}.tesseract_ara_{image_variant}_psm{psm}.raw.txt"
        raw = run_tesseract(img_path, "ara", psm, raw_candidate_path)
        variants.append((score_text(raw), f"tesseract_ara_{image_variant}_psm{psm}", raw, img_path, raw_candidate_path))

    arabic_best_score, _, arabic_best_text, _, _ = max(variants, key=lambda item: item[0])
    if page_number not in FORCE_ARABIC_PAGES.get(pdf_path.name, set()) and should_try_english(arabic_best_text, arabic_best_score):
        for image_variant, psm in ENGLISH_OCR_CANDIDATES:
            img_path = candidate_image_path(pdf_path, page_number, image_variant)
            candidate_raw_dir = CANDIDATE_RAW_DIR / pdf_path.stem
            raw_candidate_path = candidate_raw_dir / f"page_{page_number:04d}.tesseract_eng_{image_variant}_psm{psm}.raw.txt"
            raw = run_tesseract(img_path, "eng", psm, raw_candidate_path)
            variants.append((score_english_text(raw), f"tesseract_eng_{image_variant}_psm{psm}", raw, img_path, raw_candidate_path))

    best_score, source, best_text, img_path, candidate_raw_path = max(variants, key=lambda item: item[0])

    raw_dir = RAW_DIR / pdf_path.stem
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"page_{page_number:04d}.{source}.raw.txt"
    raw_path.write_text(best_text, encoding="utf-8")
    return PageOcr(pdf_path.name, page_number, source, best_text, str(img_path), str(raw_path), best_score)


def process_pdf(pdf_path: Path) -> dict:
    with fitz.open(pdf_path) as doc:
        page_count = len(doc)
        embedded_pages = sum(1 for page in doc if (page.get_text("text") or "").strip())
        embedded_chars = sum(len(page.get_text("text") or "") for page in doc)

    page_results: list[PageOcr] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(ocr_page, pdf_path, page_no): page_no for page_no in range(1, page_count + 1)}
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            page_results.append(future.result())
            if idx % 10 == 0 or idx == page_count:
                print(f"{pdf_path.name}: OCR {idx}/{page_count}")
    page_results.sort(key=lambda p: p.page)

    parts = []
    page_char_counts = []
    for result in page_results:
        if result.source.startswith("visual_"):
            cleaned = result.raw_text.strip()
        elif result.source.startswith("tesseract_eng"):
            cleaned = clean_english_page_text(result.raw_text)
        else:
            cleaned = clean_page_text(result.raw_text)
        page_char_counts.append(len(cleaned))
        if not cleaned:
            cleaned = "[لم يتم استخراج نص مقروء محليا من هذه الصفحة]"
        parts.append(f"===== Page {result.page} =====\n{cleaned}".rstrip())

    final_text = "\n\n".join(parts).strip() + "\n"
    out_path = pdf_path.with_name(f"{pdf_path.stem}.clean_arabic.txt")
    out_path.write_text(final_text, encoding="utf-8")

    return {
        "pdf": str(pdf_path),
        "sha256": sha256(pdf_path),
        "output": str(out_path),
        "pages": page_count,
        "embedded_pages": embedded_pages,
        "embedded_chars": embedded_chars,
        "final_chars": len(final_text),
        "arabic_chars": sum(1 for ch in final_text if "\u0600" <= ch <= "\u06ff"),
        "latin_letters": sum(1 for ch in final_text if ("A" <= ch <= "Z") or ("a" <= ch <= "z")),
        "replacement_chars": final_text.count("\ufffd"),
        "control_chars": sum(1 for ch in final_text if (ord(ch) < 32 and ch not in "\n\r\t") or 127 <= ord(ch) <= 159),
        "page_markers": len(re.findall(r"^===== Page ", final_text, re.MULTILINE)),
        "empty_markers": final_text.count("[لم يتم استخراج نص مقروء محليا من هذه الصفحة]"),
        "page_char_counts": page_char_counts,
        "page_results": [r.__dict__ for r in page_results],
    }


def main() -> None:
    if not TESSERACT.exists():
        raise FileNotFoundError(TESSERACT)
    if not TESSDATA.exists():
        raise FileNotFoundError(TESSDATA)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_300_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_RAW_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for name in TARGET_NAMES:
        pdf_path = FOLDER / name
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        results.append(process_pdf(pdf_path))

    summary = {
        "folder": str(FOLDER),
        "targets": TARGET_NAMES,
        "cloud_or_external_ai": False,
        "ocr": "local Tesseract 5.4.0, ara only; candidates per page: zoom-3 raw psm 4/6 plus 300-DPI contrast/binary variants selected by deterministic score; visual separator pages corrected from scanned image",
        "results": results,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        [
            {
                "output": r["output"],
                "pages": r["pages"],
                "page_markers": r["page_markers"],
                "final_chars": r["final_chars"],
                "arabic_chars": r["arabic_chars"],
                "replacement_chars": r["replacement_chars"],
                "empty_markers": r["empty_markers"],
            }
            for r in results
        ],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
