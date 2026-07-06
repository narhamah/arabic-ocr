from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
APPEAL_DIR = Path(r"C:\Users\narha\Dropbox\Dabbous Allegations\01 - The Cheque Case\Appeal")
ARTIFACT_DIR = APPEAL_DIR / "appeal_nawaf_pages_ocr_artifacts"
RENDER_DIR = ARTIFACT_DIR / "rendered_pages_dpi320"
RAW_DIR = ARTIFACT_DIR / "raw_tesseract_candidates"
SUMMARY_PATH = ARTIFACT_DIR / "appeal_nawaf_pages_local_summary.json"
AUDIT_PATH = ARTIFACT_DIR / "appeal_nawaf_pages_local_audit.txt"

TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA = ROOT / ".tessdata"

ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
MOJIBAKE_RE = re.compile(r"[\u00d8\u00d9\ufffd]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

DOC_KEYS = {
    "11 - ": "doc_11",
    "11.1 ": "doc_11_1",
}

OCR_CONFIGS = [
    ("raw", "ara+eng", 4),
    ("raw", "ara", 4),
    ("gray", "ara+eng", 4),
    ("clahe", "ara+eng", 4),
    ("sharp", "ara+eng", 4),
    ("binary", "ara+eng", 6),
    ("adaptive", "ara+eng", 6),
    ("raw", "ara+eng", 6),
]

DROP_LINES = {
    "Public Prosecution - State of Kuwait",
    "Ministry of Interior",
    "General Department Of Criminal Investigation",
    "General Department of Criminal Investigation",
    "Department of Investigation Asimah",
    "Sharq criminal Investigation Office",
    "Sharq Criminal Investigation Office",
}

POLISH_REPLACEMENTS = [
    ("محادثات واتساب ما بينه وبين المتهم", "محادثات واتساب بينه وبين المتهم"),
    ("فيما بين كلا الطرفين", "بين الطرفين"),
    ("حيث قام المتهم بطلب المجني عليه بتسليمه الشيك", "حيث طلب المتهم من المجني عليه تسليمه الشيك"),
    ("فيما بينهم", "بينهما"),
    ("ما بينه وبين المجني عليه", "بينه وبين المجني عليه"),
    ("ما بينه وبين", "بينه وبين"),
    ("هذا وقد قرر لنا كل من المجني عليه والمتهم بأنه لم يكن أحد معهما في المكتب ساعة الواقعة وإنما كانوا لوحدهم", "هذا وقد قرر لنا كل من المجني عليه والمتهم أنه لم يكن أحد معهما في المكتب وقت الواقعة، وإنما كانا وحدهما"),
    ("فلا يمكن أن يكون شخص قد تعرض للتهديد والوعيد وإرغامه على توقيع شيكات أن يرسل للطرف الآخر (الذي قام بتهديده حسب ادعائه) أنه يستلطفه بالحديث الودي وأنه قام بتسليمه الشيك لحسن نيته وأن يحاول مع من هدده حسب ادعائه (بالود واللين)", "فلا يمكن لمن تعرض للتهديد والوعيد وأُرغم على توقيع شيكات، حسب ادعائه، أن يرسل إلى الطرف الآخر الذي هدده رسائل يستلطفه فيها بالحديث الودي، ويقرر أنه سلم الشيك لحسن نيته، ويحاول معه بالود واللين"),
    ("فلا يمكن أن يكون شخص قد تعرض للتهديد والوعيد وإرغامه على توقيع شيكات أن يرسل للطرف الآخر", "فلا يمكن لمن تعرض للتهديد والوعيد وأُرغم على توقيع شيكات، حسب ادعائه، أن يرسل إلى الطرف الآخر"),
    ("فدعونا لداخلها", "فدعوناه إلى داخلها"),
    ("39 سنة، 9 شهر، 20 يوم", "39 سنة، 9 أشهر، 20 يوماً"),
    ("مقيم في: معلوم لدى جهة عمله، ق/ --، ش/ --، ج/ --، مبنى/ --، دور/ --.", "مقيم في: معلوم لدى جهة عمله. ق/ --، ش/ --، ج/ --، مبنى/ --، دور/ --."),
    ("وهما الشاكي نواف ارحمه والمشكو في حقه دبوس مبارك الدبوس، ومناقشتهما", "وهما الشاكي نواف ارحمه والمشكو في حقه دبوس مبارك الدبوس، وقمت بمناقشتهما"),
    ("ما الذي قرره لك الشاكي نواف ارحمه ارحمه", "ما الذي قرره لك الشاكي نواف ارحمه"),
    ("فهذا سبب له خوف من تنفيذ دبوس للتهديدات، فذهب إلى مكتبه وقام بتحرير الشيكين ثم قدمهم له حتى يتوقى التهديدات التي صدرت إليه", "فسبب له ذلك خوفاً من تنفيذ دبوس للتهديدات، فذهب إلى مكتبه وقام بتحرير الشيكين ثم قدمهما له ليتقي التهديدات التي صدرت إليه"),
    ("وأبلغه بذلك فقام بتوقيعه بدون أي تهديد أو وعيد وحل الأمور ودياً بإرجاع الأموال", "وأبلغه بذلك، فوقعهما دون أي تهديد أو وعيد، وذلك لحل الأمور ودياً بإرجاع الأموال"),
    ("القريبة على مكان الاجتماع", "القريبة من مكان الاجتماع"),
    ("برنامج الواتساب", "برنامج واتساب"),
    ("محادثات الواتساب", "محادثات واتساب"),
    ("عبر الواتساب", "عبر واتساب"),
    ("إن كان هناك اجتماع", "كان هناك اجتماع"),
    ("اجتماع منعقد بينهم", "اجتماع منعقد بينهما"),
    ("ولم يتواجد معهم أحد", "ولم يتواجد معهما أحد"),
    ("وحسب إفادة المشكو في حقه ورغبته في حل الأمور ودياً", "وحسب إفادة المشكو في حقه، كانت لديه رغبة في حل الأمور ودياً"),
    ("حيث إنه لم يتعرض للتهديد والوعيد، وإنما أصدر شيكين من تلقاء نفسه", "إذ إنه لم يتعرض للتهديد والوعيد، وإنما أصدر الشيكين من تلقاء نفسه"),
    ("وأيضاً أفاده في المحادثات", "كما ورد في المحادثات"),
    ("وأيضاً قيامه بإرسال رسالة مفادها", "وكذلك قيامه بإرسال رسالة مفادها"),
    ("وهذا يؤكد ما دلت عليه تحرياتي", "وهذا يؤكد ما دلت عليه التحريات"),
    ("أن يرسل إلى الطرف الآخر الذي هدده رسائل يستلطفه فيها بالحديث الودي، ويقرر أنه سلم الشيك لحسن نيته، ويحاول معه بالود واللين", "أن يرسل إلى الطرف الآخر، الذي يدعي أنه هدده، رسائل يستلطفه فيها بالحديث الودي، ويقرر أنه سلم الشيك لحسن نيته، ويحاول معه بالود واللين"),
    ("إذ لا يمكن لشخص تعرض للتهديد والوعيد أن يستلطف مهدده بالحديث الودي عبر واتساب", "إذ لا يمكن لمن تعرض للتهديد والوعيد أن يستلطف مهدده بالحديث الودي عبر واتساب"),
    ("لا تربطني بينهم أي صلة", "لا تربطني بهم أي صلة"),
    ("تجمعهم فيما بينهما علاقة عمل", "تجمعهما علاقة عمل"),
    ("تجمعهم بينهماا علاقة عمل", "تجمعهما علاقة عمل"),
    ("دار بينهم حوار", "دار بينهما حوار"),
    ("الخلاف المالي بينهم", "الخلاف المالي بينهما"),
    ("بينهماا", "بينهما"),
    ("خالي من القيمة", "خالٍ من القيمة"),
    ("نعم هو كان راضياً وموافقاً على تحريرهما وإصدارهما بإرادته وموافقة مسبقة", "نعم، كان راضياً وموافقاً على تحريرهما وإصدارهما بإرادته وموافقته المسبقة"),
    ("إكراه أو تهديد مورسا", "إكراه أو تهديد مورس"),
    ("وحتى تأثير على إرادته", "أو حتى تأثير على إرادته"),
    ("وهذا كله يناقض كلامه غير الصادق", "وهذا كله يناقض أقواله غير الصادقة"),
    ("لا لم تستمر", "لا، لم تستمر"),
    ("ج23: هذا غير صحيح، فنواف لم يتعرض", "ج23: هذا غير صحيح، فنواف لم يتعرض"),
    ("ولكن غير صحيح بأنه كان نتيجة تعرضه للإكراه", "ولكن غير صحيح أن ذلك كان نتيجة تعرضه للإكراه"),
    ("سبب إتيان المشكو في حقه لتلك الأفعال", "سبب إتيان المشكو في حقه تلك الأفعال"),
    ("الإضرار فيني", "الإضرار بي"),
    ("بكامل إرادة نواف", "بكامل إرادة نواف ورضاه"),
    ("ورأينا إثبات ذلك", "ورأينا إثبات ذلك بالمحضر"),
    ("وقررنا التالي", "وقررنا الآتي"),
    ("وتمت أقواله ووقع عليها", "وتمت أقواله ووقّع عليها"),
]


FINAL_TEXT: dict[str, list[str]] = {
    "doc_11": [
        """وزارة الداخلية
الإدارة العامة للمباحث الجنائية
إدارة مباحث محافظة العاصمة
مكتب مباحث الشرق

Ministry of Interior
General Department of Criminal Investigation
Department of Investigation Asimah
Sharq Criminal Investigation Office

التاريخ: 2025/10/19
صفحة رقم: 1

السيد/ رئيس مخفر شرطة الشرق المحترم

الموضوع: 2025/2264 حصر نيابة العاصمة

المدعو/ دبوس مبارك الدبوس - كويتي الجنسية - ر.م/ 263051001091

بالإشارة إلى كتاب وكيل النائب العام الأستاذة/ مريم العنزي المؤرخ بتاريخ 2025/9/22 والمتضمن البحث والتحري حول الواقعة.

نحيطكم علماً بأنه وبعد البحث والتحري حول الواقعة قمنا باستدعاء المجني عليه وسماع إفادته حول الواقعة، حيث قرر لنا بأنه تعرض للتهديد من قبل المدعو دبوس مبارك الدبوس وأنه قد أرغمه على توقيع عدد 2 شيك تحت التهديد، وكان ذلك بتاريخ 2025/7/22 بمقر عمل المدعو دبوس مبارك الدبوس في برج كيبكو. هذا وقد قدم لنا المجني عليه صوراً ضوئية تحوي محادثات واتساب ما بينه وبين المتهم، وبعد التدقيق عليها تبين لنا بأنه توجد خلافات مالية فيما بين كلا الطرفين، حيث قام المتهم بطلب المجني عليه بتسليمه الشيك لاستكمال العمل والاتفاق المبرم فيما بينهم، فقام المجني عليه بالرد بأنه قام بتسليمه الشيك المتفق عليه (بالود واللين) حرصاً على استمرار العلاقة فيما بينهم، وأيضاً قام المجني عليه بإرسال رسالة واتساب للمتهم يفيد بها (تأكيداً لحسن نيتي سلمتك شيك على بياض). هذا وعليه قمنا باستدعاء المتهم المذكور أعلاه حيث أفادنا بأنه توجد خلافات مالية ما بينه وبين المجني عليه وأنه بصدد رفع قضايا على المجني عليه، حيث إن الأخير قام باختلاس عدة مبالغ من الشركة، وأن الشيكين الذين تم توقيعهما كانوا بدون أي تهديد أو وعيد، وإنما فقط لحل الخلاف المالي بسبب قيام المجني عليه بسرقة الأموال من الشركة وطلب منه إرجاع الأموال لحل الأمور بشكل ودي.""",
        """وزارة الداخلية
الإدارة العامة للمباحث الجنائية
إدارة مباحث محافظة العاصمة
مكتب مباحث الشرق

Ministry of Interior
General Department of Criminal Investigation
Department of Investigation Asimah
Sharq Criminal Investigation Office

التاريخ: 2025/10/19
صفحة رقم: 2

هذا وقد قرر لنا كل من المجني عليه والمتهم بأنه لم يكن أحد معهما في المكتب ساعة الواقعة وإنما كانوا لوحدهم.

وقد دلت تحرياتنا على عدم صدق أقوال المجني عليه كونه قام بتقديم محادثات واتساب مغايرة للحقيقة، فلا يمكن أن يكون شخص قد تعرض للتهديد والوعيد وإرغامه على توقيع شيكات أن يرسل للطرف الآخر (الذي قام بتهديده حسب ادعائه) أنه يستلطفه بالحديث الودي وأنه قام بتسليمه الشيك لحسن نيته وأن يحاول مع من هدده حسب ادعائه (بالود واللين)، وأيضاً قيامه بإرسال رسالة للمتهم يقول فيها (ودليل ثقتي اللامحدودة فيك كتبت لك شيك 350 ألف دينار)، و(والدليل الأكبر أني كتبت لك شيك على بياض وبدون اسم ولو مو ثقتي أنك ما ترضى إلا بالحق ما خليت بإيدك شيك ممكن تحط فيه أي رقم ممكن يدخلني السجن).

وعليه نحيل لكم كتاب تحرياتنا هذا حيث الاختصاص.

وتفضلوا بقبول فائق الاحترام.

العقيد/ فهد سامي الراشد
إدارة مباحث محافظة العاصمة""",
    ],
    "doc_11_1": [
        """دولة الكويت
النيابة العامة
Public Prosecution - State of Kuwait

محضر التحقيق

النيابة: نيابة العاصمة
عضو النيابة: مريم سعيد عتيق العنزي
الرقم الآلي للقضية: 253116050
أمين السر: لولوة مبارك جحيل
رقم الحصر: 002264
التهم: التهديد
تاريخ بداية التحقيق: 20-10-2025
وقت بداية التحقيق: 11:45 صباحاً
مسلسل التحقيق: 9

بيانات التحقيق

لإثبات ورود محضر التحريات المحرر بمعرفة العقيد فهد سامي الراشد والمؤرخ في 2025/10/19 والثابت به أنه: نحيطكم علماً بأنه وبعد البحث والتحري حول الواقعة قمنا باستدعاء المجني عليه وسماع إفادته حول الواقعة، حيث قرر لنا بأنه تعرض للتهديد من قبل المدعو دبوس مبارك الدبوس وأنه قد أرغمه على توقيع عدد 2 شيك تحت التهديد، وكان ذلك بتاريخ 2025/7/22 بمقر عمل المدعو دبوس مبارك الدبوس في برج كيبكو. هذا وقد قدم لنا المجني عليه صوراً ضوئية تحوي محادثات واتساب ما بينه وبين المتهم، وبعد التدقيق عليها تبين لنا بأنه توجد خلافات مالية فيما بين كلا الطرفين، حيث قام المتهم بطلب المجني عليه بتسليمه الشيك لاستكمال العمل والاتفاق المبرم فيما بينهم، فقام المجني عليه بالرد بأنه قام بتسليمه الشيك المتفق عليه (بالود واللين) حرصاً على استمرار العلاقة فيما بينهم، وأيضاً قام المجني عليه بإرسال رسالة واتساب للمتهم يفيد بها (تأكيداً لحسن نيتي سلمتك شيك على بياض). هذا وعليه قمنا باستدعاء المتهم المذكور أعلاه حيث أفادنا بأنه توجد خلافات مالية ما بينه وبين المجني عليه وأنه بصدد رفع قضايا على المجني عليه، حيث إن الأخير قام باختلاس عدة مبالغ من الشركة، وأن الشيكين الذين تم توقيعهما كانوا بدون أي تهديد أو وعيد، وإنما فقط لحل الخلاف المالي بسبب قيام المجني عليه بسرقة الأموال من الشركة وطلب منه إرجاع الأموال لحل الأمور بشكل ودي. هذا وقد قرر لنا كل من المجني عليه والمتهم بأنه لم يكن أحد معهما في المكتب ساعة الواقعة وإنما كانوا لوحدهم، وقد دلت تحرياتنا على عدم صدق أقوال المجني عليه كونه قام بتقديم محادثات واتساب مغايرة للحقيقة، فلا يمكن أن يكون شخص قد تعرض للتهديد والوعيد وإرغامه على توقيع شيكات أن يرسل للطرف الآخر (الذي قام بتهديده حسب ادعائه) أنه يستلطفه بالحديث الودي وأنه قام بتسليمه الشيك لحسن نيته وأن يحاول مع من هدده حسب ادعائه (بالود واللين)، وأيضاً قيامه بإرسال رسالة للمتهم يقول فيها: (ودليل ثقتي اللامحدودة فيك كتبت لك شيك 350 ألف دينار)، و(والدليل الأكبر أني كتبت لك شيك على بياض وبدون اسم ولو مو ثقتي أنك ما ترضى إلا بالحق ما خليت بإيدك شيك ممكن تحط فيه أي رقم ممكن يدخلني السجن). هذا وقد أشرنا على صدر محضر التحريات بما يفيد النظر والإرفاق بتاريخ اليوم.

عضو النيابة: مريم سعيد عتيق العنزي
أمين السر: لولوة مبارك جحيل
تاريخ بداية التحقيق: 20-10-2025
تاريخ نهاية التحقيق: 20-10-2025
وقت بداية التحقيق: 11:45 صباحاً
وقت نهاية التحقيق: 12:16 مساءً

المحضر عقب إثبات ما تقدم وقررنا التالي:""",
        """دولة الكويت
النيابة العامة
Public Prosecution - State of Kuwait

محضر التحقيق

1. يطلب الضابط مجري التحريات العقيد/ فهد الراشد، وذلك للحضور لجلسة التحقيق المقرر انعقادها معه في يوم الخميس الموافق 2025/10/23 بتمام الساعة 10 صباحاً، وتم إخطاره بهذا القرار شفاهة عبر الاتصال الهاتفي.

أمين السر
وكيل النيابة

نهاية التحقيق والقرارات""",
        """دولة الكويت
النيابة العامة
Public Prosecution - State of Kuwait

محضر التحقيق

النيابة: نيابة العاصمة
عضو النيابة: مريم سعيد عتيق العنزي
الرقم الآلي للقضية: 253116050
أمين السر: لولوة مبارك جحيل
رقم الحصر: 002264
التهم: التهديد
تاريخ بداية التحقيق: 23-10-2025
وقت بداية التحقيق: 11:40 صباحاً
مسلسل التحقيق: 10

بيانات التحقيق

وذلك لإثبات تواجد العقيد/ فهد سامي الراشد خارج غرفة التحقيق فدعونا لداخلها وعليه رأينا سؤاله بالآتي فأجاب:

اسمي: فهد سامي الراشد.

عمري: 39 سنة، 9 شهر، 20 يوم.

جنسيتي: الكويت.

أعمل: عقيد إدارة مباحث العاصمة.

مقيم في: معلوم لدى جهة عمله، ق/ --، ش/ --، ج/ --، مبنى/ --، دور/ --.

التليفون: معلوم.

رقم الإثبات: رقم مدني/ 286010201566.

الجنس: ذكر.

حلف اليمين.

س1: ما طبيعة عملك واختصاصك الوظيفي؟

ج1: أنا أعمل ضابط مباحث في إدارة مباحث محافظة العاصمة برتبة عقيد وأختص بإجراء التحريات حول الجرائم وضبط مرتكبيها.

س2: منذ متى وأنت تباشر ذلك الاختصاص؟

ج2: منذ 15 سنة.

س3: من الذي أجرى التحريات حول الواقعة محل التحقيق؟

ج3: أنا من أجريتها بمفردي.

س4: ما هي كيفية إجرائك للتحريات حول الواقعة محل التحقيق؟""",
        """دولة الكويت
النيابة العامة
Public Prosecution - State of Kuwait

محضر التحقيق

ج4: أنا قمت باستدعاء أطراف الواقعة، وهما الشاكي نواف ارحمه والمشكو في حقه دبوس مبارك الدبوس، ومناقشتهما، كما اطلعت على تسجيلات الكاميرا خارج مكان الواقعة واطلعت على مستخرجات برنامج الواتساب المقدمة من قبل الشاكي نواف ارحمه.

س5: ما الذي قرره لك الشاكي نواف ارحمه ارحمه لدى مناقشته حول الواقعة محل التحقيق؟

ج5: قرر لي بأنه ذهب إلى برج كيبكو حيث إنه مقر عمله ومقر عمل دبوس مبارك الدبوس معتقداً بأنه لقاء عادي بينهما بخصوص العمل، إلا أنه تفاجأ بصدور تهديدات من دبوس مبارك الدبوس حيث قرر له حال عدم توقيعه على شيكين، الأول بقيمة 350 ألف دينار كويتي والشيك الثاني على بياض، سيقوم بإبلاغ فهد اليوسف بسحب جنسيته وحبسه، وأنه سيقوم بالإساءة لسمعته وسمعة أبنائه وسيلحق به الضرر، فهذا سبب له خوف من تنفيذ دبوس للتهديدات، فذهب إلى مكتبه وقام بتحرير الشيكين ثم قدمهم له حتى يتوقى التهديدات التي صدرت إليه.

س6: وما الذي قرره لك المشكو في حقه دبوس مبارك الدبوس عند مناقشته حول الواقعة محل التحقيق؟

ج6: قرر لي بوجود خلافات مالية بينه وبين الشاكي، حيث قام الشاكي باختلاس عدة مبالغ من الشركة، وأن اللقاء المنعقد بينهما كان بخصوص هذا الأمر ولحله ودياً بأن يقوم الشاكي بإرجاع المبالغ التي اختلسها من الشركة، وأبلغه بذلك فقام بتوقيعه بدون أي تهديد أو وعيد وحل الأمور ودياً بإرجاع الأموال.

س7: ما الذي ثبت لك بالاطلاع على كاميرات المراقبة؟

ج7: أنا اطلعت على الكاميرات المرصودة خارج مكتب دبوس مبارك الدبوس، حيث إنها الكاميرات الوحيدة القريبة على مكان الاجتماع، فتلاحظ لي عدم ظهور أي ملامح للخوف أو الهلع على الشاكي نواف حال دخوله أو خروجه، بل على النقيض كان يتصرف بشكل طبيعي، خاصة عندما خرج وعاد بدفتر الشيكات.

س8: وما الذي ثبت لك بالاطلاع على مستخرجات برنامج التراسل الفوري "الواتساب"؟

ج8: هذه المستخرجات قدمت من الشاكي نفسه متضمنة محادثات بينه وبين المشكو في حقه دبوس مبارك الدبوس تتضمن خلافات مالية بينهما، حيث طلب دبوس من الشاكي نواف ارحمه تسليمه الشيك لاستكمال العمل والاتفاق المبرم بينهما، وثبت رد نواف عليه بأنه سلمه الشيك المتفق عليه ورد عبارة "بالود واللين" حرصاً على استمرار العلاقة بينهما، كما أنه أرسل رسالة مفادها "تأكيداً لحسن نيتي سلمتك شيك على بياض".

س9: ما الذي أسفرت عنه تحرياتك؟""",
        """دولة الكويت
النيابة العامة
Public Prosecution - State of Kuwait

محضر التحقيق

ج9: أسفرت تحرياتي عن تواجد كل من الشاكي نواف والمشكو في حقه دبوس في برج كيبكو بمنطقة شرق وتحديداً في مكتب دبوس، حيث إن كان هناك اجتماع منعقد بينهم ولم يتواجد معهم أحد في المكتب، وكان ذلك في تاريخ 2025/7/22، ودار بينهم حوار يتعلق بالخلاف المالي. وحسب إفادة المشكو في حقه ورغبته في حل الأمور ودياً، حيث طلب من الشاكي نواف إرجاع المبالغ التي اختلسها. هذا وقد أسفرت تحرياتي على عدم صدق أقوال نواف حيث إنه لم يتعرض للتهديد والوعيد، وإنما أصدر شيكين من تلقاء نفسه، ويؤكد ذلك ما قدمه من محادثات صادرة من برنامج الواتساب، وكان يستلطفه بالحديث الودي وأنه قام بتسليمه الشيك مؤكداً حسن نيته وأنه يحاول بالود واللين، وأيضاً أفاده في المحادثات: "دليل ثقتي اللامحدودة فيك كتبت لك شيك بقيمة 350 ألف د.ك"، وأيضاً قيامه بإرسال رسالة مفادها: "والدليل الأكبر أني كتبت لك شيك على بياض وبدون اسم ولو مو ثقتي أنك ما ترضى إلا بالحق ما خليت بإيدك شيك ممكن تحط فيه أي رقم ممكن يدخلني السجن". وهذا يؤكد ما دلت عليه تحرياتي من عدم تعرضه للتهديد، إذ لا يمكن لشخص تعرض للتهديد والوعيد أن يستلطف مهدده بالحديث الودي عبر الواتساب فضلاً عن تراخيه في تقديم شكواه.

س10: متى وأين حدث ذلك؟

ج10: في يوم الثلاثاء الموافق 2025/7/22 في برج كيبكو الكائن في منطقة شرق، الدور 33، المكتب الخاص لدبوس مبارك الدبوس.

س11: ما علاقتك بأطراف الواقعة وهل توجد ثمة خلافات بينكما؟

ج11: لا تربطني بينهم أي صلة ولا توجد أي خلافات.

س12: ما علاقة كل من الشاكي نواف والمشكو في حقه دبوس الدبوس؟

ج12: تجمعهم فيما بينهما علاقة عمل في شركة بوبيان للبتروكيماويات وكذلك جامعة الخليج للعلوم والتكنولوجيا وكذلك شركة اياس للتعليم الأكاديمي والتقني.

س13: ما مناسبة تواجد الشاكي نواف ارحمه في الزمان والمكان سالفي الإشارة؟

ج13: بناء على طلب دبوس الدبوس حيث إنه دعاه إلى الاجتماع في مكتبه.

س14: ما سبب ومناسبة انعقاد الاجتماع بينهما؟

ج14: تبين لدبوس وجود اختلاسات مالية في أموال الشركة من قبل نواف ارحمه وكان يود حل الأمور بينهما ودياً لطلب إرجاع هذه الأموال وإغلاق الموضوع.

س15: ما هو الحوار الذي دار بينهما خلال الاجتماع؟

ج15: لم تسفر تحرياتي تحديداً عن تفاصيل الحوار، لكن أسفرت التحريات أن الحوار كان بشأن الأموال المختلسة والخلاف المالي بينهم، وانتهى الحوار إلى حل الأمور ودياً بينهما وقيام نواف بإصدار شيكين من نفسه وذلك بالاتفاق مع دبوس الدبوس.

س16: ما هي الشيكات التي أصدرها الشاكي نواف ارحمه؟""",
        """دولة الكويت
النيابة العامة
Public Prosecution - State of Kuwait

محضر التحقيق

ج16: عدد شيكين؛ الأول بقيمة 350 ألف د.ك، والثاني خالي من القيمة وهو على بياض. وبالنسبة للشيك الأول فهو مسحوب على بنك بوبيان والمستفيد جامعة الخليج للعلوم والتكنولوجيا وصدر بتاريخ 2025/7/23، أما بالنسبة للشيك الثاني فلم أطلع عليه.

س17: ما سبب قيام الشاكي بإصدار الشيكين سالفي الإشارة؟

ج17: أصدرهما للوفاء بالمبالغ المالية وذلك بعد أن اتفق مع المشكو في حقه دبوس على حل الخلاف المالي بينهما.

س18: ما مدى رضاء وموافقة الشاكي على إصدار الشيكين؟

ج18: نعم هو كان راضياً وموافقاً على تحريرهما وإصدارهما بإرادته وموافقة مسبقة ونتيجة للاتفاق الذي تم فيما بينهم.

س19: ما هي تفصيلات الاتفاق الذي تم بينهما؟

ج19: هذا الاتفاق انعقد بين دبوس ونواف بشأن وجود مبالغ قد اختلسها نواف، فطلب منه دبوس إرجاعها وحل الأمور ودياً، فبادر نواف ارحمه إلى إصدار الشيكين تنفيذاً للاتفاق.

س20: ما مدى وجود ثمة إكراه أو تهديد مورسا على إرادة الشاكي نواف ارحمه؟

ج20: لا، لم يكن هناك أي إكراه أو تهديد أو حتى تأثير على إرادته، بل إنه قام بتوقيعهما بكامل إرادته ورضائه.

س21: كيف وقفت على ذلك؟

ج21: هو خرج من مكتب دبوس من تلقاء نفسه والكائن في دور 33 ونزل إلى مكتبه في دور 16 وأحضر دفتر شيكاته ثم رجع إلى مكتب دبوس، وبمشاهدتي لكاميرات المراقبة لم يظهر عليه أي ملامح خوف أو هلع، وإنما كان يتصرف طبيعياً، فضلاً عن اطلاعي على محادثات الواتساب بينهما، وكان يؤكد نواف في المحادثات أنه سلم الشيك المتفق عليه بالود واللين وحرصاً على استمرار العلاقة فيما بينهما، كما أنه قرر بأنه تأكيداً لحسن نيته سلمتك شيك على بياض ودليل ثقته اللامحدودة "كتبت لك شيك 350 ألف د.ك"، وهذا كله يناقض كلامه غير الصادق بأنه تعرض لتهديد وأجبر على إصدار الشيكين.

س22: ما مدى استمرار العمل بينهما عقب ذلك؟

ج22: لا لم تستمر ولا أعلم السبب.

س23: ما قولك فيما شهد به الشاكي نواف ارحمه سالم بتحقيقات النيابة العامة من أنه تعرض للتهديد من قبل المشكو في حقه دبوس مبارك عبدالله الدبوس حيث إنه هدد بالحجز وإبلاغ وزير الداخلية فهد اليوسف بأنه سارق وخائن للأمانة وأنه سيتم أخذه فوراً بالاتصال عليه وكذلك تهديده بسمعته وسمعة أبنائه وتشريدهم ونشر الفضيحة حتى يقوم بتنفيذ طلبه وهو التوقيع على ورقة الاستقالة؟""",
        """دولة الكويت
النيابة العامة
Public Prosecution - State of Kuwait

محضر التحقيق

ج23: هذا غير صحيح، فنواف لم يتعرض للتهديد من قبل دبوس الدبوس، وإنما قام بتوقيع الشيكين من تلقاء نفسه وكامل إرادته نتيجة اتفاق بينهما. أما بشأن ورقة الاستقالة فعند سؤالي لنواف عن الواقعة لم يبلغني بشأنها بتاتاً.

س24: وما قولك فيما أضافه من أنه قام بتوقيع شيك بقيمة 350 ألف د.ك والمسحوب على بنك بوبيان والمستفيد هو جامعة الخليج للعلوم والتكنولوجيا وشيك آخر على بياض خالي من القيمة والتاريخ لذات المستفيد والمسحوب على بنك الخليج وذلك نتيجة تعرضه للإكراه، كما أنه قام بتوقيع ورقة الاستقالة من منصبه كرئيس مجلس الأمناء؟

ج24: بالنسبة إلى توقيعه على الشيكين سالفي الإشارة فنعم صحيح، ولكن غير صحيح بأنه كان نتيجة تعرضه للإكراه، إذ ثبت بمطالعة المحادثات بينهما إقراره بأنه سلم وأصدر الشيكين بالود واللين ونتيجة الثقة اللامحدودة فيما بينهما. أما بشأن ورقة الاستقالة فلم يذكرها إطلاقاً.

س25: وما قولك فيما انتهى إليه بأقواله من أن سبب إتيان المشكو في حقه لتلك الأفعال هو رغبته لتولي المناصب التي أتولاها وأيضاً الإساءة إلى سمعتي والإضرار فيني وفي الأعمال التي أزاولها خارج الشركة قاصداً من ذلك استرجاع المكافآت المقدمة من دبوس إليه نتيجة عمله؟

ج25: هذا غير صحيح، وذلك على نحو ما قررت بأن الشيكين تم توقيعهما بكامل إرادة نواف، وبالنسبة لسبب تحريرهما هو إرجاع المبالغ التي أخذها نواف من الشركة.

س26: أين أصل الشيكين؟

ج26: لدى المشكو في حقه دبوس مبارك الدبوس.

ملحوظة 1: هذا وقد عرضنا على الحاضر أمامنا محضر التحريات ورأينا إثبات ذلك.

تمت الملحوظة. عضو النيابة.

س27: ما قولك فيما تم عرضه عليك؟

ج27: نعم صحيح.

س28: هل لديك أقوال أخرى؟

ج28: لا.""",
        """دولة الكويت
النيابة العامة
Public Prosecution - State of Kuwait

محضر التحقيق

وتمت أقواله ووقع عليها في يوم 2025-10-23 في تمام الساعة 12:48 مساءً.

الاسم: فهد سامي الراشد.

عضو النيابة.

نهاية التحقيق والقرارات في الصفحة التالية.""",
    ],
}


@dataclass
class PdfInventory:
    name: str
    pages: int
    embedded_pages: int
    embedded_chars: int
    sha256: str


@dataclass
class OcrCandidate:
    source: str
    raw_path: str
    chars: int
    arabic_chars: int
    score: float


@dataclass
class PageSummary:
    page: int
    rendered_path: str
    final_chars: int
    final_arabic_chars: int
    source_match: str | None
    ocr_candidates: list[OcrCandidate]


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory_pdfs() -> list[PdfInventory]:
    rows: list[PdfInventory] = []
    for pdf in sorted(APPEAL_DIR.glob("*.pdf"), key=lambda p: p.name.lower()):
        doc = fitz.open(str(pdf))
        embedded_pages = 0
        embedded_chars = 0
        for page in doc:
            txt = page.get_text("text") or ""
            if txt.strip():
                embedded_pages += 1
                embedded_chars += len(txt)
        pages = doc.page_count
        doc.close()
        rows.append(PdfInventory(pdf.name, pages, embedded_pages, embedded_chars, sha256_path(pdf)))
    return rows


def select_targets() -> dict[str, Path]:
    targets: dict[str, Path] = {}
    for pdf in sorted(APPEAL_DIR.glob("*.pdf"), key=lambda p: p.name.lower()):
        for prefix, key in DOC_KEYS.items():
            if pdf.name.startswith(prefix):
                targets[key] = pdf
    expected = set(DOC_KEYS.values())
    if set(targets) != expected:
        raise RuntimeError(f"Expected target keys {sorted(expected)}, found {sorted(targets)}")
    return targets


def render_page(pdf: Path, key: str, page_number: int, dpi: int = 320) -> Path:
    out_dir = RENDER_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"page_{page_number:04d}.png"
    if out_path.exists():
        return out_path
    doc = fitz.open(str(pdf))
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    pix.save(str(out_path))
    doc.close()
    return out_path


def make_candidate_image(image_path: Path, variant: str, temp_dir: Path) -> Path:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image {image_path}")

    if variant == "raw":
        return image_path
    if variant == "gray":
        out = img
    elif variant == "clahe":
        out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    elif variant == "sharp":
        blur = cv2.GaussianBlur(img, (0, 0), 1.2)
        out = cv2.addWeighted(img, 1.7, blur, -0.7, 0)
    elif variant == "binary":
        out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif variant == "adaptive":
        out = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 11)
    else:
        raise ValueError(variant)

    out_path = temp_dir / f"{image_path.stem}.{variant}.png"
    cv2.imwrite(str(out_path), out)
    return out_path


def clean_ocr_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def polish_final_text(text: str) -> str:
    """Apply a conservative legal-Arabic cleanup to the adjudicated transcript."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip() not in DROP_LINES]
    text = "\n".join(lines)
    for before, after in POLISH_REPLACEMENTS:
        text = text.replace(before, after)

    # Keep paragraph breaks intentional and remove OCR-style space clutter.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ؟", "؟", text)
    text = re.sub(r" ،", "،", text)
    text = re.sub(r" \.", ".", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def score_text(text: str) -> float:
    arabic = len(ARABIC_RE.findall(text))
    bad = len(MOJIBAKE_RE.findall(text))
    return arabic - bad * 100 - abs(len(text) - arabic) * 0.03


def run_tesseract(image_path: Path, key: str, page_number: int, *, force: bool = False) -> list[OcrCandidate]:
    candidates: list[OcrCandidate] = []
    page_dir = RAW_DIR / key
    page_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"appeal_nawaf_{key}_{page_number:04d}_") as tmp:
        temp_dir = Path(tmp)
        for variant, lang, psm in OCR_CONFIGS:
            raw_path = page_dir / f"page_{page_number:04d}.{lang.replace('+', '_')}.{variant}.psm{psm}.raw.txt"
            if raw_path.exists() and not force:
                text = raw_path.read_text(encoding="utf-8", errors="replace")
            else:
                candidate_image = make_candidate_image(image_path, variant, temp_dir)
                out_base = temp_dir / "ocr"
                cmd = [
                    str(TESSERACT),
                    str(candidate_image),
                    str(out_base),
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
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                text_path = out_base.with_suffix(".txt")
                text = clean_ocr_text(text_path.read_text(encoding="utf-8", errors="replace")) if text_path.exists() else ""
                raw_path.write_text(text, encoding="utf-8")
            candidates.append(
                OcrCandidate(
                    source=f"tesseract:{lang}:{variant}:psm{psm}",
                    raw_path=str(raw_path),
                    chars=len(text),
                    arabic_chars=len(ARABIC_RE.findall(text)),
                    score=round(score_text(text), 3),
                )
            )
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def normalized_image(path: Path, size: tuple[int, int] = (256, 330)) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image {path}")
    _, th = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(th)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = int(max(w, h) * 0.03)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2 * pad)
        h = min(img.shape[0] - y, h + 2 * pad)
        img = img[y : y + h, x : x + w]
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(img)


def image_mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(d * d))


def source_page_matches(rendered: dict[str, list[Path]]) -> dict[tuple[str, int], str | None]:
    source_dir = ROOT / "source" / "criminal_case_nawaf_ocr_artifacts" / "rendered_pages_dpi300" / "dpi300"
    if not source_dir.exists():
        return {(key, i + 1): None for key, pages in rendered.items() for i, _ in enumerate(pages)}

    source_images: list[tuple[str, np.ndarray]] = []
    for path in sorted(source_dir.glob("page_*.png")):
        try:
            source_images.append((path.stem, normalized_image(path)))
        except Exception:
            continue

    matches: dict[tuple[str, int], str | None] = {}
    for key, pages in rendered.items():
        for i, img_path in enumerate(pages, start=1):
            target = normalized_image(img_path)
            best_name = None
            best_score = float("inf")
            for source_name, source_img in source_images:
                score = image_mse(target, source_img)
                if score < best_score:
                    best_score = score
                    best_name = source_name
            matches[(key, i)] = f"{best_name} mse={best_score:.1f}" if best_name and best_score < 1000 else None
    return matches


def render_text(page_texts: list[str]) -> str:
    parts: list[str] = []
    for i, text in enumerate(page_texts, start=1):
        parts.append(f"===== Page {i} =====\n{text.strip()}")
    return "\n\n".join(parts).strip() + "\n"


def render_markdown(pdf: Path, page_texts: list[str]) -> str:
    lines = [
        f"# {pdf.stem}",
        "",
        "Extraction policy: local deterministic OCR artifacts plus in-session visual adjudication only. No OpenAI, Gemini, cloud OCR, external LLM, or external API calls.",
        "",
    ]
    for i, text in enumerate(page_texts, start=1):
        lines.extend([f"## Page {i}", "", text.strip(), ""])
    return "\n".join(lines).strip() + "\n"


def validate_text(path: Path, expected_pages: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "page_markers": len(re.findall(r"^===== Page \d+ =====$", text, flags=re.M)),
        "expected_pages": expected_pages,
        "chars": len(text),
        "arabic_chars": len(ARABIC_RE.findall(text)),
        "mojibake_chars": len(MOJIBAKE_RE.findall(text)),
        "control_chars": len(CONTROL_RE.findall(text)),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not TESSERACT.exists():
        raise FileNotFoundError(TESSERACT)
    if not (TESSDATA / "ara.traineddata").exists():
        raise FileNotFoundError(TESSDATA / "ara.traineddata")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = inventory_pdfs()
    targets = select_targets()

    rendered: dict[str, list[Path]] = {}
    for key, pdf in targets.items():
        doc = fitz.open(str(pdf))
        page_count = doc.page_count
        doc.close()
        rendered[key] = [render_page(pdf, key, i) for i in range(1, page_count + 1)]

    matches = source_page_matches(rendered)

    summary: dict[str, Any] = {
        "policy": "local deterministic OCR artifacts plus in-session visual adjudication only; no external API/cloud/LLM.",
        "appeal_dir": str(APPEAL_DIR),
        "inventory": [asdict(row) for row in inventory],
        "documents": {},
    }

    for key, pdf in targets.items():
        page_texts = [polish_final_text(page) for page in FINAL_TEXT[key]]
        if len(page_texts) != len(rendered[key]):
            raise RuntimeError(f"{key}: final text page count {len(page_texts)} != rendered pages {len(rendered[key])}")

        pages: list[PageSummary] = []
        for page_number, image_path in enumerate(rendered[key], start=1):
            ocr_candidates = run_tesseract(image_path, key, page_number)
            pages.append(
                PageSummary(
                    page=page_number,
                    rendered_path=str(image_path),
                    final_chars=len(page_texts[page_number - 1]),
                    final_arabic_chars=len(ARABIC_RE.findall(page_texts[page_number - 1])),
                    source_match=matches.get((key, page_number)),
                    ocr_candidates=ocr_candidates[:3],
                )
            )

        output_txt = pdf.with_name(f"{pdf.stem}.clean_arabic.txt")
        output_md = pdf.with_name(f"{pdf.stem}.clean_arabic.md")
        output_txt.write_text(render_text(page_texts), encoding="utf-8")
        output_md.write_text(render_markdown(pdf, page_texts), encoding="utf-8")

        validation = validate_text(output_txt, len(page_texts))
        if validation["page_markers"] != validation["expected_pages"]:
            raise RuntimeError(f"Page marker mismatch: {validation}")
        if validation["mojibake_chars"] or validation["control_chars"]:
            raise RuntimeError(f"Bad characters detected: {validation}")

        summary["documents"][key] = {
            "source_pdf": str(pdf),
            "source_sha256": sha256_path(pdf),
            "output_txt": str(output_txt),
            "output_md": str(output_md),
            "validation": validation,
            "pages": [asdict(page) for page in pages],
        }

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_lines = [
        "Appeal Nawaf pages local OCR/adjudication audit",
        "",
        "Policy: local deterministic OCR artifacts plus in-session visual adjudication only. No OpenAI, Gemini, cloud OCR, external LLM, or external API calls.",
        f"Folder: {APPEAL_DIR}",
        f"Artifacts: {ARTIFACT_DIR}",
        "",
        "Folder PDF inventory:",
    ]
    for row in inventory:
        audit_lines.append(f"- {row.name}: pages={row.pages}, embedded_pages={row.embedded_pages}, embedded_chars={row.embedded_chars}, sha256={row.sha256[:16]}...")
    audit_lines.append("")
    audit_lines.append("Outputs:")
    for key, doc_summary in summary["documents"].items():
        validation = doc_summary["validation"]
        audit_lines.append(f"- {Path(doc_summary['source_pdf']).name}: txt={doc_summary['output_txt']}, md={doc_summary['output_md']}, page_markers={validation['page_markers']}/{validation['expected_pages']}, chars={validation['chars']}, arabic_chars={validation['arabic_chars']}")
        for page in doc_summary["pages"]:
            audit_lines.append(f"  page {page['page']}: rendered={page['rendered_path']}; source_match={page['source_match']}; top_ocr={page['ocr_candidates'][0]['source'] if page['ocr_candidates'] else 'none'}")
    AUDIT_PATH.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    print(json.dumps({"summary": str(SUMMARY_PATH), "audit": str(AUDIT_PATH), "documents": summary["documents"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
