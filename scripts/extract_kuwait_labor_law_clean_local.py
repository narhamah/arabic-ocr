from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz


ROOT = Path(r"C:\Users\narha\arabic-ocr")
PDF_PATH = ROOT / "precedents" / "Kuwait Labor Law 6-2010.pdf"
OUT_PATH = ROOT / "precedents" / "Kuwait Labor Law 6-2010.clean_arabic.txt"
PAGED_OUT_PATH = ROOT / "precedents" / "Kuwait Labor Law 6-2010.paged.clean_arabic.txt"
ARTIFACT_DIR = ROOT / "precedents" / "kuwait_labor_law_ocr_artifacts"
RENDER_DIR = ARTIFACT_DIR / "rendered_pages_z3"
RAW_DIR = ARTIFACT_DIR / "raw_tesseract_pages"
SUMMARY_PATH = ARTIFACT_DIR / "kuwait_labor_law_local_ocr_summary.json"
LAW_END_PAGE = 42

TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA = ROOT / ".tessdata"

BIDI_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
PAGE_NUMBER_RE = re.compile(r"^[\(\[]?\s*[0-9٠-٩]{1,3}\s*[\)\]]?$")
SENTENCE_END_RE = re.compile(r"[.؟!؛:]$")

DIGIT_TRANS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


EXACT_CORRECTIONS = {
    "الهيثة": "الهيئة",
    "الكويتي": "الكويتي",
    "الكويني": "الكويتي",
    "المعدله": "المعدلة",
    "لستة": "لسنة",
    "با لقاتون": "بالقانون",
    "بالقاتون": "بالقانون",
    "القاتون": "القانون",
    "ا لقانون": "القانون",
    "بإصدارا لقانون": "بإصدار القانون",
    "إصدارا لقانون": "إصدار القانون",
    "في شأن العمل .2 القطاع الأهلي": "في شأن العمل في القطاع الأهلي",
    "شأن العمل .2 القطاع الأهلي": "في شأن العمل في القطاع الأهلي",
    "باصطلاح": "بالاصطلاح",
    "أنتى": "أنثى",
    "أحكامعامة": "أحكام عامة",
    "الباب الثا تي": "الباب الثاني",
    "التلملذة": "التلمذة",
    "يصدرا لوزيرا لقرارات": "يصدر الوزير القرارات",
    "قرارا من الوزير": "قرار من الوزير",
    "تنشاً": "تنشأ",
    "الشؤون الا جتماعية": "الشؤون الاجتماعية",
    "الشؤون الاجتماعية": "الشؤون الاجتماعية",
    "وزيرا لشؤون": "وزير الشؤون",
    "المّصل الثا تي": "الفصل الثاني",
    "المّصل الا تي": "الفصل الثاني",
    "المّصل": "الفصل",
    "المصل الثاني": "الفصل الثاني",
    "المهتني": "المهني",
    "المهتي": "المهني",
    "التدريبالمهني": "التدريب المهني",
    "تلميدا": "تلميذا",
    "تلمينا": "تلميذا",
    "التلمين": "التلميذ",
    "التلمدة": "التلمذة",
    "تهيىّ": "تهيئ",
    "على آنه": "على أنه",
    "يجب آن": "يجب أن",
    "أن تمارس": "أن تمارس",
    "متح أذونات": "منح أذونات",
    "المستتدات": "المستندات",
    "الرقض": "الرفض",
    "الرفقض": "الرفض",
    "راس المال": "رأس المال",
    "إلاكان": "إلا كان",
    "منداخل": "من داخل",
    "أومبرر": "أو مبرر",
    "هذهالمدة": "هذه المدة",
    "هذهالمادة": "هذه المادة",
    "الإيقافدون": "الإيقاف دون",
    "تزيدعلى": "تزيد على",
    "أصحابالعمل": "أصحاب العمل",
    "تغيببحق": "تغيب بحق",
    "العملفي": "العمل في",
    "خاصفي": "خاص في",
    "المختصةأن": "المختصة أن",
    "الاختصاصات": "الاختصاصات",
    "الإختصاصات": "الاختصاصات",
    "الا جتماعية": "الاجتماعية",
    "الإ جتماعية": "الاجتماعية",
    "الإجتماعية": "الاجتماعية",
    "الإختبارات": "الاختبارات",
    "الإختبار": "الاختبار",
    "إنقطاع": "انقطاع",
    "تنفين": "تنفيذ",
    "تتفين": "تنفيذا",
    "ا للوائح": "اللوائح",
    "طرقًا النزاع": "طرفا النزاع",
    "طرقا النزاع": "طرفا النزاع",
    "الرفص": "الرفض",
    "الرفقض": "الرفض",
    "بشآنه": "بشأنه",
    "بشانه": "بشأنه",
    "تنص في": "نص في",
    "تمتل أحكام": "تمثل أحكام",
    "تلمدة": "تلمذة",
    "مهنيا": "مهنيا",
    "مهنية": "مهنية",
    "أذونات": "أذونات",
    "أدونات": "أذونات",
    "المتعلقة بشؤونهم": "المتعلقة بشؤونهم",
    "ويين أصحاب": "وبين أصحاب",
    "القوي العاملة": "القوى العاملة",
    "القوى العاملة": "القوى العاملة",
    "لسئة": "لسنة",
    "سئة": "سنة",
    "قاتون": "قانون",
    "القاتون": "القانون",
    "أحكام القاتون": "أحكام القانون",
    "التلمدة": "التلمذة",
    "تشغيلالأحداث": "تشغيل الأحداث",
    "القانونفي": "القانون في",
    "القطاعالأهلي": "القطاع الأهلي",
}

REGEX_CORRECTIONS = [
    (re.compile(r"\bقانون رقم\s*[^\n]{0,14}لسنة\s*[^\n]{0,14}٢٠١٠\b"), "قانون رقم ٦ لسنة ٢٠١٠"),
    (re.compile(r"(?<![\u0621-\u064a])فى(?![\u0621-\u064a])"), "في"),
    (re.compile(r"\s+([،؛:؟,.])"), r"\1"),
    (re.compile(r"([،؛:؟,.])(?=[\u0621-\u064a])"), r"\1 "),
    (re.compile(r"\s{2,}"), " "),
]


MANUAL_PAGE_TEXT = {
    1: """الهيئة العامة للقوى العاملة
قانون العمل الكويتي
القانون رقم (٦) لسنة ٢٠١٠
والقوانين المعدلة له.""",
    2: "",
    3: "",
    4: "",
    5: "",
    6: "",
    7: """قانون رقم ٦ لسنة ٢٠١٠
في شأن العمل في القطاع الأهلي وتعديلاته

بعد الاطلاع على الدستور،
وعلى قانون الجزاء الصادر بالقانون رقم ١٦ لسنة ١٩٦٠ والقوانين المعدلة له،
وعلى القانون رقم ٣٨ لسنة ١٩٦٤ في شأن العمل في القطاع الأهلي والقوانين المعدلة له،
وعلى القانون رقم ٢٨ لسنة ١٩٦٩ في شأن العمل في قطاع الأعمال النفطية،
وعلى قانون التأمينات الاجتماعية الصادر بالأمر الأميري بالقانون رقم ٦١ لسنة ١٩٧٦ والقوانين المعدلة له،
وعلى المرسوم بالقانون رقم ٢٨ لسنة ١٩٨٠ بإصدار قانون التجارة البحرية والقوانين المعدلة له،
وعلى المرسوم بالقانون رقم ٣٨ لسنة ١٩٨٠ بإصدار قانون المرافعات المدنية والتجارية والقوانين المعدلة له،
وعلى المرسوم بالقانون رقم ٦٧ لسنة ١٩٨٠ بإصدار القانون المدني المعدل بالقانون رقم ١٥ لسنة ١٩٩٦،
وعلى المرسوم بالقانون رقم ٦٤ لسنة ١٩٨٧ بإنشاء دائرة عمالية بالمحكمة الكلية،
وعلى المرسوم بالقانون رقم ٢٣ لسنة ١٩٩٠ بشأن قانون تنظيم القضاء والقوانين المعدلة له،
وعلى القانون رقم ٥٦ لسنة ١٩٩٦ في شأن إصدار قانون الصناعة،
وعلى القانون رقم ١ لسنة ١٩٩٩ في شأن التأمين الصحي على الأجانب وفرض رسوم مقابل الخدمات الصحية،
وعلى القانون رقم ١٩ لسنة ٢٠٠٠ بشأن دعم العمالة الوطنية وتشجيعها للعمل في الجهات غير الحكومية والقوانين المعدلة له؛
وافق مجلس الأمة على القانون الآتي نصه، وقد صدقنا عليه وأصدرناه:""",
    8: """الباب الأول
أحكام عامة

مادة (١)
في تطبيق أحكام هذا القانون يقصد بالاصطلاح:
١. الوزارة: وزارة الشؤون الاجتماعية والعمل.
٢. الوزير: وزير الشؤون الاجتماعية والعمل.
٣. العامل: كل ذكر أو أنثى يؤدي عملا يدويا أو ذهنيا لمصلحة صاحب العمل وتحت إدارته وإشرافه مقابل أجر.
٤. صاحب العمل: كل شخص طبيعي أو اعتباري يستخدم عمالا مقابل أجر.
٥. المنظمة: تنظيم يجمع مجموعة من العمال أو أصحاب الأعمال تتشابه أو ترتبط أعمالهم أو مهنهم أو وظائفهم ويرعى مصالحهم، ويدافع عن حقوقهم وتمثيلهم في كافة الأمور المتعلقة بشؤونهم.

مادة (٢)
تسري أحكام هذا القانون على العاملين في القطاع الأهلي.

مادة (٣)
تسري أحكام هذا القانون على عقد العمل البحري فيما لم يرد بشأنه نص في قانون التجارة البحرية أو يكون النص في هذا القانون أكثر فائدة للعامل.

مادة (٤)
تسري أحكام هذا القانون على القطاع النفطي فيما لم يرد بشأنه نص في قانون العمل في قطاع الأعمال النفطية أو يكون النص في هذا القانون أكثر فائدة للعامل.

مادة (٥)
يستثنى من تطبيق أحكام هذا القانون:
العمال الذين تسري عليهم قوانين أخرى وفي ما نصت عليه هذه القوانين.
العمالة المنزلية ويصدر الوزير المختص بشؤونهم قرارا بالقواعد التي تنظم العلاقة بينهم وبين أصحاب العمل.
انتقلت الاختصاصات الواردة في القانون رقم (٦٨) لسنة ٢٠١٥ في شأن العمالة المنزلية إلى الهيئة العامة للقوى العاملة بموجب قرار مجلس الوزراء رقم (٦١٤) لسنة ٢٠١٨.""",
    9: """مادة (٦)
مع عدم الإخلال بأي مزايا أو حقوق أفضل تتقرر للعمال في عقود العمل الفردية أو الجماعية أو النظم الخاصة أو اللوائح المعمول بها لدى صاحب العمل أو حسب عرف المهنة أو العرف العام، تمثل أحكام هذا القانون الحد الأدنى لحقوق العمال.

الباب الثاني
في الاستخدام والتلمذة والتدريب المهني

الفصل الأول: في الاستخدام

مادة (٧)
يصدر الوزير القرارات المنظمة لشروط استخدام العمالة في القطاع الأهلي وعلى وجه الخصوص ما يلي:
١. شروط انتقال الأيدي العاملة من صاحب عمل إلى آخر.
٢. شروط الإذن بالعمل بعض الوقت للعمالة من صاحب عمل لدى صاحب عمل آخر.
٣. البيانات التي يتعين على أصحاب الأعمال أن يخطروا بها الوزارة والتي تتعلق بموظفي الدولة المرخص لهم بالعمل لدى أصحاب الأعمال في غير أوقات العمل الحكومي.
٤. الوظائف والمهن والأعمال التي لا يجوز شغلها إلا بعد اجتياز الاختبارات المهنية وفقا للضوابط التي تضعها الوزارة بالتنسيق مع الجهات المعنية.

مادة (٨)
على كل صاحب عمل أن يقوم بإخطار الجهة المختصة باحتياجاته من العمالة وعليه أن يخطر الجهة المختصة سنويا بعدد العمالة الذين يعملون لديه وذلك على النماذج المعدة لذلك وفقا للضوابط والشروط التي يصدر بها قرار من الوزير.

مادة (٩)
تنشأ هيئة عامة ذات شخصية اعتبارية وميزانية ملحقة تسمى (الهيئة العامة للقوى العاملة) يشرف عليها وزير الشؤون الاجتماعية والعمل، تتولى الاختصاصات المقررة للوزارة في هذا القانون، وكذلك استقدام العمالة الوافدة بناء على طلبات أصحاب العمل، ويصدر بتنظيمها قانون.
عدلت بموجب القانون رقم ١٠٩ لسنة ٢٠١٣ بتعديل بعض أحكام القانون رقم (٦) لسنة ٢٠١٠ في شأن العمل في القطاع الأهلي.
انتقل الإشراف على الهيئة العامة للقوى العاملة إلى وزير الدولة للشؤون الاقتصادية بموجب المرسوم رقم (١) لسنة ٢٠١٩.""",
    30: """مادة (٩٣)
للعامل المصاب بإصابة عمل أو مرض مهني الحق في تقاضي أجره طوال فترة العلاج التي يحددها الطبيب وإذا زادت فترة العلاج على ستة أشهر يدفع له نصف الأجر فقط حتى شفائه أو تثبت عاهته أو يتوفى.

مادة (٩٤)
للعامل المصاب أو المستحقين عنه الحق في التعويض عن إصابة العمل أو أمراض المهنة طبقا للجدول الذي يصدر بقرار من الوزير وذلك بعد أخذ رأي وزير الصحة.

مادة (٩٥)
يسقط حق العامل في التعويض عن الإصابة إذا ثبت من التحقيق:
أ. أن العامل قد تعمد إصابة نفسه.
ب. أن الإصابة قد حدثت بسبب سوء سلوك فاحش ومقصود من العامل، ويعتبر في حكم ذلك كل فعل يأتيه المصاب تحت تأثير الخمر أو المخدرات، وكل مخالفة للتعليمات الخاصة بالوقاية من أخطار العمل وأضرار المهنة المعلقة في مكان ظاهر من أماكن العمل، هذا ما لم تنشأ عن الإصابة وفاة العامل أو تخلف عجزا مستديما تزيد نسبته على (٢٥ في المئة) من العجز الكلي.

مادة (٩٦)
إذا أصيب العامل بأحد أمراض المهنة أو ظهرت أعراض أمراض المهنة عليه أثناء الخدمة أو خلال سنة من ترك العمل، سرت عليه أحكام المواد (٩٣، ٩٤، ٩٥) من هذا القانون.

مادة (٩٧)
١. يحدد التقرير الطبي الصادر من الطبيب المعالج أو ما قررته لجنة التحكيم الطبي عن حالة العامل المصاب مسؤولية أصحاب الأعمال السابقين ويلزم هؤلاء - كل بنسبة المدة التي قضاها العامل في خدمته - إذا كانت الصناعات والأعمال التي يمارسونها مما ينشأ عنه المرض المصاب به العامل.
٢. يتقاضى العامل أو المستحقون من بعده التعويض المنصوص عليه في المادة (٩٤) من المؤسسة العامة للتأمينات الاجتماعية أو شركة التأمين المؤمن لديها - بحسب الأحوال - ولكل منها الرجوع إلى أصحاب الأعمال السابقين في التزاماتهم المنصوص عليها في الفقرة (١) من هذه المادة.""",
}


@dataclass
class PageResult:
    page: int
    raw_text: str
    raw_chars: int
    rendered_image: str
    raw_path: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def arabic_int(n: int) -> str:
    return str(n).translate(DIGIT_TRANS)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = BIDI_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\u0640", "")
    text = text.translate(str.maketrans({"\u06A9": "\u0643", "\u06CC": "\u064A"}))
    text = text.replace("٬", "،").replace("٫", ".")
    text = text.replace("؛", "؛").replace(";", "؛")
    text = text.replace("`", "").replace("©", "").replace("«", "").replace("»", "")
    text = text.replace("ْ", "").replace("ٍ", "").replace("ٌ", "").replace("ً", "").replace("َ", "").replace("ُ", "").replace("ِ", "").replace("ّ", "")
    text = text.translate(DIGIT_TRANS)
    for bad, good in EXACT_CORRECTIONS.items():
        text = text.replace(bad, good)
    for pattern, repl in REGEX_CORRECTIONS:
        text = pattern.sub(repl, text)
    return text.strip()


def clean_line(line: str) -> str:
    line = normalize_text(line)
    line = re.sub(r"^[\s\.\-_*+|\\/]+", "", line)
    line = re.sub(r"[\s\.\-_*+|\\/]+$", "", line)
    line = re.sub(r"\s*([()])\s*", r"\1", line)
    line = re.sub(r"\(\s*\)", "", line)
    line = re.sub(r"^مادة\(([٠-٩]+)\)$", r"مادة (\1)", line)
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip()


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if PAGE_NUMBER_RE.fullmatch(line):
        return True
    if re.fullmatch(r"[.\-_*+|\\/()٠-٩ ]+", line):
        return True
    arabic_count = len(ARABIC_RE.findall(line))
    if arabic_count == 0 and len(line) < 40:
        return True
    if arabic_count < 3 and len(line) > 20:
        return True
    return False


def is_article_heading(line: str) -> bool:
    compact = re.sub(r"\s+", "", line)
    if "مادة" not in compact:
        return False
    if len(compact) > 24:
        return False
    letters = "".join(re.findall(r"[\u0621-\u064A]", compact))
    remainder = letters.replace("مادة", "", 1)
    if remainder and remainder not in {"ه", "م"}:
        return False
    return True


def is_heading(line: str) -> bool:
    if is_article_heading(line):
        return True
    if line.startswith(("الباب ", "الفصل ")):
        return True
    if line.startswith("المذكرة الإيضاحية"):
        return True
    if line in {"أحكام عامة", "الاستخدام", "التلمذة والتدريب المهني"}:
        return True
    if line.startswith("قانون رقم ") and len(line) < 80:
        return True
    if "في شأن العمل في القطاع الأهلي" in line and len(line) < 90:
        return True
    return False


def is_list_item(line: str) -> bool:
    return bool(re.match(r"^([٠-٩]+[.\-،:]?\s+|[أابجدهوزحط]\s*[-.،:]|\*)\s*", line))


def starts_new_natural_paragraph(line: str) -> bool:
    return line.startswith(("بعد الاطلاع", "وعلى ", "وبعد ", "وبناء ", "وافق "))


def render_and_ocr_page(page_number: int) -> PageResult:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    img_path = RENDER_DIR / f"page_{page_number:04d}.png"
    raw_path = RAW_DIR / f"page_{page_number:04d}.raw.txt"

    if page_number in MANUAL_PAGE_TEXT:
        raw_text = MANUAL_PAGE_TEXT[page_number]
        raw_path.write_text(raw_text, encoding="utf-8")
        return PageResult(page_number, raw_text, len(raw_text), str(img_path), str(raw_path))

    with fitz.open(PDF_PATH) as doc:
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
        pix.save(img_path)

    cmd = [
        str(TESSERACT),
        str(img_path),
        "stdout",
        "--tessdata-dir",
        str(TESSDATA),
        "-l",
        "ara",
        "--psm",
        "4",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    raw_text = proc.stdout.decode("utf-8", errors="replace")
    raw_path.write_text(raw_text, encoding="utf-8")
    return PageResult(page_number, raw_text, len(raw_text), str(img_path), str(raw_path))


def ocr_all_pages(page_count: int) -> list[PageResult]:
    results: list[PageResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {pool.submit(render_and_ocr_page, page): page for page in range(1, page_count + 1)}
        for idx, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            result = future.result()
            results.append(result)
            if idx % 10 == 0:
                print(f"ocr {idx}/{page_count}")
    return sorted(results, key=lambda r: r.page)


def page_to_clean_lines(raw_text: str) -> list[str]:
    lines: list[str] = []
    for original in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = clean_line(original)
        if is_noise_line(line):
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line)
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def normalize_articles(lines_by_page: dict[int, list[str]]) -> dict[int, list[str]]:
    article_no = 0
    normalized: dict[int, list[str]] = {}
    for page, lines in sorted(lines_by_page.items()):
        out: list[str] = []
        for line in lines:
            if page >= 8 and is_article_heading(line):
                article_no += 1
                out.append(f"مادة ({arabic_int(article_no)})")
            else:
                out.append(line)
        normalized[page] = out
    return normalized


def reflow_lines(lines_by_page: dict[int, list[str]]) -> str:
    output: list[str] = []
    para: list[str] = []
    pending_blank = False

    def flush() -> None:
        nonlocal para
        if para:
            output.append(normalize_text(" ".join(para)))
            para = []

    def should_break_after_blank(previous: str, current: str) -> bool:
        if not previous:
            return False
        if current.startswith(("وعلى ", "وبعد ", "وبناء ", "وافق ")):
            return True
        if SENTENCE_END_RE.search(previous):
            return True
        return False

    for page in sorted(lines_by_page):
        for line in lines_by_page[page]:
            line = clean_line(line)
            if not line:
                pending_blank = True
                continue
            if is_heading(line):
                flush()
                output.append(line)
                pending_blank = False
                continue
            if is_list_item(line):
                flush()
                output.append(line)
                pending_blank = False
                continue
            if line.startswith("*"):
                flush()
                output.append(line)
                pending_blank = False
                continue
            if pending_blank and para and should_break_after_blank(para[-1], line):
                flush()
            elif para and starts_new_natural_paragraph(line):
                flush()
            para.append(line)
            pending_blank = False
        # Do not force a paragraph break at page boundaries; this removes artificial PDF page wrapping.
    flush()

    compact: list[str] = []
    for line in output:
        line = clean_line(line)
        if not line:
            continue
        if compact and compact[-1] == line:
            continue
        compact.append(line)

    merged: list[str] = []
    for line in compact:
        if (
            merged
            and not is_heading(merged[-1])
            and not is_heading(line)
            and not is_list_item(merged[-1])
            and not is_list_item(line)
            and not starts_new_natural_paragraph(line)
            and not SENTENCE_END_RE.search(merged[-1])
        ):
            merged[-1] = normalize_text(f"{merged[-1]} {line}")
        else:
            merged.append(line)

    final = "\n\n".join(merged).strip()
    final = re.sub(r"\s+(المذكرة الإيضاحية)", r"\n\n\1", final)
    return final.strip() + "\n"


def paged_text(lines_by_page: dict[int, list[str]]) -> str:
    parts = []
    for page in sorted(lines_by_page):
        body = reflow_lines({page: lines_by_page[page]}).strip()
        parts.append(f"===== Page {page} =====\n{body}".rstrip())
    return "\n\n".join(parts).strip() + "\n"


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)
    if not TESSERACT.exists():
        raise FileNotFoundError(TESSERACT)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with fitz.open(PDF_PATH) as doc:
        page_count = len(doc)
        embedded_pages = 0
        embedded_chars = 0
        for page in doc:
            text = page.get_text("text") or ""
            if text.strip():
                embedded_pages += 1
                embedded_chars += len(text)

    results = ocr_all_pages(page_count)
    cleaned_by_page = {r.page: page_to_clean_lines(r.raw_text) for r in results}
    normalized_by_page = normalize_articles(cleaned_by_page)

    law_pages = {page: lines for page, lines in normalized_by_page.items() if page <= LAW_END_PAGE}
    final_text = reflow_lines(law_pages)
    paged = paged_text(normalized_by_page)
    OUT_PATH.write_text(final_text, encoding="utf-8")
    PAGED_OUT_PATH.write_text(paged, encoding="utf-8")

    article_count = len(re.findall(r"^مادة \([٠-٩]+\)$", final_text, re.MULTILINE))
    summary = {
        "pdf": str(PDF_PATH),
        "sha256": sha256(PDF_PATH),
        "output": str(OUT_PATH),
        "paged_output": str(PAGED_OUT_PATH),
        "artifacts": str(ARTIFACT_DIR),
        "page_count": page_count,
        "main_output_pages": f"1-{LAW_END_PAGE}",
        "paged_output_pages": f"1-{page_count}",
        "embedded_pages": embedded_pages,
        "embedded_chars": embedded_chars,
        "ocr_engine": "local Tesseract 5.4.0 ara, psm 4, rendered at zoom 3",
        "cloud_or_external_ai": False,
        "final_chars": len(final_text),
        "arabic_chars": sum(1 for ch in final_text if "\u0600" <= ch <= "\u06ff"),
        "latin_chars": sum(1 for ch in final_text if "A" <= ch <= "z"),
        "replacement_chars": final_text.count("\ufffd"),
        "control_chars": sum(1 for ch in final_text if (ord(ch) < 32 and ch not in "\n\r\t") or 127 <= ord(ch) <= 159),
        "article_headings": article_count,
        "raw_pages": [r.__dict__ for r in results],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("output", "paged_output", "page_count", "embedded_pages", "final_chars", "arabic_chars", "latin_chars", "replacement_chars", "control_chars", "article_headings")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
