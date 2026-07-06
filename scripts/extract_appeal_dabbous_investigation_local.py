from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
APPEAL_DIR = Path(r"C:\Users\narha\Dropbox\Dabbous Allegations\01 - The Cheque Case\Appeal")
ARTIFACT_DIR = APPEAL_DIR / "appeal_dabbous_investigation_ocr_artifacts"
RENDER_DIR = ARTIFACT_DIR / "rendered_pages_dpi320"
PREPROCESS_DIR = ARTIFACT_DIR / "preprocessed_tesseract_inputs"
RAW_OCR_DIR = ARTIFACT_DIR / "raw_tesseract_candidates"
HISTORICAL_DIR = ARTIFACT_DIR / "historical_local_artifact_candidates"
SUMMARY_PATH = ARTIFACT_DIR / "appeal_dabbous_investigation_local_summary.json"
AUDIT_PATH = ARTIFACT_DIR / "appeal_dabbous_investigation_local_audit.txt"

TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA = ROOT / ".tessdata"
HISTORICAL_ARTIFACT_DIR = Path(r"C:\Users\narha\Dropbox\Dabbous Allegations\Fraud Case\mahdr_dabbous_ocr_artifacts")

ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
MOJIBAKE_RE = re.compile(r"[\u00d8\u00d9\ufffd]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

OCR_CONFIGS = [
    ("raw", "ara+eng", 4),
    ("gray", "ara+eng", 4),
    ("clahe", "ara+eng", 4),
    ("sharp", "ara+eng", 4),
    ("binary", "ara+eng", 6),
    ("adaptive", "ara+eng", 6),
    ("raw", "ara", 4),
    ("raw", "ara+eng", 6),
]


@dataclass
class PdfInventory:
    name: str
    pages: int
    embedded_pages: int
    embedded_chars: int
    size: int


@dataclass
class OcrCandidate:
    page: int
    variant: str
    lang: str
    psm: int
    success: bool
    text_chars: int
    arabic_chars: int
    mojibake_chars: int
    score: float
    output: str
    error: str = ""


FINAL_PAGES: list[list[str]] = [
    [
        "دولة الكويت. النيابة العامة. محضر التحقيق.",
        "النيابة: نيابة العاصمة. عضو النيابة: مريم سعيد عتيق العنزي. الرقم الآلي للقضية: 253116050. أمين السر: لؤلؤة مبارك جحيل. رقم الحصر: 002264. التهم: التهديد. تاريخ بداية التحقيق: 26-10-2025. وقت بداية التحقيق: 10:09 صباحاً. مسلسل التحقيق: 11.",
        "بيانات التحقيق: وذلك لإثبات تواجد المشكو في حقه/ دبوس مبارك الدبوس خارج غرفة التحقيق، فدعوناه إلى داخلها، وعليه رأينا سؤاله بالآتي فأجاب:",
        "اسمي: دبوس مبارك الدبوس. عمري: 62 سنة، 5 أشهر، 15 يوماً. جنسيتي: الكويت. أعمل: أعمال حرة، رئيس مجلس إدارة شركة بوبيان للبتروكيماويات. مقيم في: م/ الفحيحيل، ق/ 10، ش/ 5، ج/--، مبنى/15، دور/--. التليفون: 99744445. رقم الإثبات/الرقم المدني: 263051001091. الجنس: ذكر.",
        "سئل على سبيل الاستدلال.",
        "س1: ما هي صلتك بالشاكي نواف ارحمه؟",
        "ج1: تجمعني بنواف ارحمه علاقة عمل منذ أكثر من 10 سنوات.",
        "س2: ما مدى وجود خلافات سابقة بينكما؟",
        "ج2: لا توجد خلافات سابقة بيننا أبداً.",
        "س3: ما هي طبيعة الأعمال التي تجمعكما؟",
    ],
    [
        "ج3: كان نواف يتولى العمل كرئيس تنفيذي في شركة بوبيان للبتروكيماويات، وكان يختص بالشؤون المالية والاستراتيجية للشركة ومسؤوليات أخرى، إلى أن استقال قبل قرابة سنة، واستمر بالعمل معي كاستشاري، وكُلّف بأعمال منها جامعة الخليج للعلوم والتكنولوجيا وشركة إياس للتعليم الأكاديمي والتقني.",
        "س4: ما مدى استمرار تلك الأعمال بينكما؟",
        "ج4: لا، لم تستمر الأعمال بيننا فيما عدا شركة واحدة هو ما زال المدير التنفيذي فيها.",
        "س5: وما سبب عدم استمرار الأعمال فيما بينكما؟",
        "ج5: حيث تبين لنا فقدان الأموال الموجودة تحت يده بحكم أعماله، لذلك اتفقنا على عدم الاستمرار في العمل.",
        "س6: ما هي صلة الشاكي نواف ارحمه في جامعة الخليج للعلوم والتكنولوجيا؟",
        "ج6: هو كان رئيس مجلس أمناء جامعة الخليج للعلوم والتكنولوجيا.",
        "س7: ما مدى التقائك بالشاكي في يوم الثلاثاء الموافق 22/7/2025؟",
        "ج7: نعم، التقينا أنا ونواف في يوم الثلاثاء في مكتبي الكائن في برج كيبكو، الدور 33، بمنطقة شرق.",
        "س8: ما سبب ومناسبة اللقاء الذي جمعكما؟",
        "ج8: تم رفع تقرير من شؤون المحاسبة في شركة بوبيان للبتروكيماويات عن وجود مبالغ مفقودة كانت تحت يد نواف ارحمه في جامعة الخليج للعلوم والتكنولوجيا، فتواصلت معه ودعوته إلى مكتبي.",
        "س9: ما هي كيفية إبلاغه بالاجتماع؟",
        "ج9: كان عندنا موعد مسبق للالتقاء لأن الاجتماعات تنعقد بيننا بشكل دوري، فأكدت عليه الحضور.",
        "س10: ما مدى علمه المسبق بفحوى الاجتماع وسببه؟",
        "ج10: لا، لم يكن على علم بموضوع الاجتماع لأن أساساً الاجتماع كان مقرراً تحديده من قبل ولمناقشة أمور أخرى، وعندما رفع لي تقرير شؤون المحاسبة أنا تفاجأت أصلاً بما تضمنه من مبالغ مفقودة، فقررت أن أتكلم مع نواف بخصوص موضوع الاختلاسات في الاجتماع المحدد مسبقاً.",
        "س11: أين مكان ذلك الاجتماع وزمانه تحديداً؟",
        "ج11: كان في مكتبي في برج كيبكو بالدور 33، وهذا هو المكان الذي تنعقد به الاجتماعات عادة، وكان حوالي الساعة 12 ظهراً تقريباً أثناء ساعات العمل.",
        "س12: هل تواجد آخرون برفقتكما؟",
    ],
    [
        "ج12: لا، كنا في المكتب لوحدنا لمناقشة الموضوع.",
        "س13: ما هي المدة الزمنية التي استغرقها الاجتماع المنعقد بينكم؟",
        "ج13: لم يتجاوز الساعة تقريباً، وتخللها خروج ودخول نواف.",
        "س14: ما الحوار الذي دار بينكما؟",
        "ج14: عندما دخل نواف إلى مكتبي رحبت فيه وقلت له حياك، وقدمت له الورقة التي رفعت لي من شؤون المحاسبة والمبين فيها جزء من قيمة الأموال المفقودة، فتفاجأ من الورقة وتغير لونه فعرف موضوع الاجتماع، فبادر وقال لي: «أنا بين إيدينك أرجوك الموضوع يكون بيننا»، وأنا هني رديت عليه: «أنا ما أبي منك شيء، أنا أبي فلوسي وأحتاج أموال الشركة»، وأوضحت له بأن الذي تبين لي يقيناً وجود مبلغ بقيمة 350 ألف د.ك مفقود، وتوجد مبالغ أخرى لم تنته إجراءات المحاسبة في تحديد قيمتها، وأبلغته أن لازم يحل الموضوع ويرجع فلوس الجامعة، فاقترح علي تقديم شيكين؛ الأول بقيمة 350 ألف د.ك يصرف فوراً، والثاني على بياض، وذلك لحل الأمر ودياً وإجراء التسوية بيننا، وهذا كله كان بطلب منه حتى لا أقدم شكوى ضده، فخرج من المكتب ونزل إلى مكتبه وأحضر دفتر الشيكات ووقع عليهما وقدمهما لي.",
        "س15: ما هو سبب إصدار الشيكين على نحو ما قررت؟",
        "ج15: لأنه تبين لنا بناء على تقرير شؤون المحاسبة وجود اختلاسات ومبالغ مفقودة من الجامعة، وأيضاً من الشركات الأخرى التي تولى العمل فيها في السابق، ولتسوية الأمور بيننا وحلها ودياً وعدم الدخول في الشكاوى القضائية، وكان ذلك بطلب منه بعد أن عرضت عليه التقارير المقدمة من شؤون المحاسبة.",
        "س16: ما هي عدد الشيكات التي أصدرها الشاكي نواف ارحمه وكم قيمتها؟",
        "ج16: عدد شيكين؛ الأول بقيمة 350 ألف د.ك، والثاني على بياض.",
        "س17: من هو المستفيد من الشيكين؟",
        "ج17: بالنسبة إلى الشيك الأول والمحدد به قيمة 350 ألف د.ك، المستفيد هو جامعة الخليج للعلوم والتكنولوجيا، حيث تبين أن قيمة الأموال المفقودة من الجامعة هي 350 ألف د.ك كحد أدنى. وبالنسبة إلى الشيك المحرر على بياض، كان خالياً من القيمة واسم المستفيد.",
        "س18: على أي البنوك سحب الشيكين؟",
        "ج18: على بنك بوبيان.",
        "س19: ما ردة فعله حيال إصداره للشيكين للتسوية على نحو ما قررت؟",
    ],
    [
        "ج19: أنا لما أبلغته بموضوع الاختلاسات والأموال المفقودة طلبت منه إرجاع المبالغ وإغلاق الموضوع كأن لم يكن، وبينت له أن الحد الأدنى من الأموال المفقودة هي 350 ألف د.ك وتوجد مبالغ غير واضحة، فهو من عرض علي تسليم شيكين؛ الأول بقيمة 350 ألف د.ك، والثاني يكون على بياض، بعدما أنا طلبت منه تسديد المبالغ، فكان موافقاً وراضياً على السداد بهذه الطريقة حتى لا تقدم ضده شكوى، وكان كله بطلب منه.",
        "س20: من أين أحضر دفتر الشيكات؟",
        "ج20: من مكتبه الكائن في الدور 16.",
        "س21: أين قام بتوقيع الشيكين؟",
        "ج21: لا أعلم، لكنه عاد إلى مكتبي بعد أن طلب مغادرته لإحضار دفتر الشيكات، وعند دخوله قام بتسليمي الشيكين مكتوبين.",
        "س22: بماذا تفسر قبوله تحرير وإصدار الشيكين وتسليمهما لك؟",
        "ج22: لأنه عندما واجهته بتقارير شؤون المحاسبة أقر لي بالخطأ، وطلب مني أني أسامحه، ومن باب التأكيد عرض السداد عن طريق الشيكين، لأن كان همي إرجاع المبالغ وأبلغته بذلك.",
        "س23: ما قولك فيما قرر الشاكي نواف ارحمه بتحقيقات النيابة العامة من أنك تواصلت معه هاتفياً وطلبت منه الحضور إلى مكتبك لمناقشة الخطة الأكاديمية للسنة الدراسية القادمة لجامعة الخليج للعلوم والتكنولوجيا، إذ إنه يتولى إدارة الجامعة، كما أنه الرئيس التنفيذي لشركة إياس للتعليم الأكاديمي، فعقد الاجتماع في مكتبك في برج كيبكو الكائن في منطقة شرق؟",
        "ج23: فيما يتعلق باتصالي عليه هاتفياً وطلب الحضور، فنعم أنا اتصلت وقلت له حياك أنا موجود، لكن لم يتضمن اتصالي عليه أي تحديد لموضوع الاجتماع، لكن وعلى نحو ما قررت هذا الاجتماع محدد مسبقاً وتنعقد الاجتماعات بيننا بشكل دوري، وكانت لمناقشة الخطط الأكاديمية والأمور المتصلة في أعمالنا.",
        "س24: وما قولك فيما أضاف من أن الحوار الذي دار بينكما أنك قررت له: «هذا أهم اجتماع في حياتك، أنت ارتكبت جرائم خيانة أمانة وسرقة لأموال جامعة الخليج ومبالغ وايد كبيرة الحين أنا شايفها»، فرد عليك: «يا بوعبدالله عطني مهلة أوضح لك»، فقررت له: «لا تجادل، الأمور كلها واضحة بالنسبة لي، أنا صار لي أشهر أدور وراك وهذا اللي لقيته وفي وايد أشياء ثانية، فالأحسن لك تدفع الحين لأني كلمت وزير الداخلية فهد اليوسف وناطرك، وهذا الحل الأمثل حتى تحافظ على سمعتك وسمعة أسرتك وما تشرد عيالك، ولا ترى راح يضيع مستقبلك، أنا الحين ما قلت لأحد والموضوع أبي أحله بهدوء، وادفع عشان ما تنتشر السالفة وما نقول للناس إن نواف حرامي، وإلا راح تتحجز وتدخل الحجز»، فقرر لك: «بوعبدالله حاضر، وإذا في خطأ مني ممكن أصلحه، وإذا في غلط سامحني لكن أنا ما سرقت ولا خنت الأمانة»، فأجبته: «هذا كلام مأخوذ خيره، فادفع عشان ما آخذ إجراءات أكثر قسوة؟».",
    ],
    [
        "ج24: هذا الكلام في جزء كبير منه غير صحيح، وعلى حسب ما تسعفني الذاكرة في استرجاع الحوار الذي دار بيننا، أنا كنت أبين له بالأوراق الاختلاسات التي ارتكبها والأموال التي حولها لحسابات تخصه شخصياً، وكنت أطلب منه إرجاع هذه الأموال بأي طريقة يراها مناسبة، وإلا كما قررت له فإني سأمارس حقي في الاتجاه للقضاء، وهو أقر لي بالخطأ الذي ارتكبه وعرض علي السداد عن طريق شيكين، وفعلاً خرج من المكتب وأحضر الشيكين لي، وبعدها تبين لي وجود خطأ وأبلغته وقام بإعادة صياغة إحداهما من نفسه.",
        "س25: وما قولك فيما أضاف من أنه بعد ذلك الحوار غادر المكتب وأحضر دفتر الشيكات، وكان ذلك خوفاً من تهديداتك، إذ إنك هددته بالحجز وإبلاغ وزير الداخلية فهد اليوسف بأنه سارق وخائن للأمانة، وأنه سيتم أخذه فوراً بالاتصال عليه، وكذلك تهديده بسمعته وسمعة أبنائه وتشريدهم ونشر الفضيحة، وهذا كله حتى يقوم بتنفيذ طلباتك، وهي التوقيع على شيكين وعلى ورقة الاستقالة؟",
        "ج25: هذا كله غير صحيح، وهو لما خرج من مكتبي أحضر الشيكين موقعين وخالصين، وأنا لم أهدده، فقط بينت له أن لازم يرجع أموال الشركة وجامعة الخليج، وإلا سأمارس حقي في اللجوء للجهات المختصة. أما ما قرره فلم يصدر مني، وهو من بادر في عرض السداد عن طريق تحرير الشيكين. أما بشأن ورقة الاستقالة، فهو من قام بتوقيع أوراق استقالته من نفسه في شركة إياس وجامعة الخليج وهناك شركات أخرى، وبحكم كوني رئيساً قمت باستبداله لأني لا أثق فيه.",
        "س26: ما قولك فيما أضاف أيضاً في تحقيقات النيابة العامة من أنه قام بتوقيع شيك بقيمة 350 ألف د.ك مسحوب على بنك بوبيان والمستفيد جامعة الخليج للعلوم والتكنولوجيا، وشيك على بياض خالٍ من القيمة والتاريخ، للمستفيد أيضاً جامعة الخليج والمسحوب على بنك بوبيان، وفي اليوم التالي تم تهديده بتوقيع ورقة الاستقالة من جامعة الخليج، وكان ذلك تحت تأثير الإكراه المتمثل بالتهديد بالحجز لدى الشرطة وإبلاغ وزير الداخلية بأنه سارق وخائن للأمانة، وأن بمجرد الاتصال على الوزير سيتم أخذه فوراً، مما أثرت عليه تلك التهديدات وجعلته يمتثل لأوامرك إذ تبين له إمكانية تنفيذ تلك التهديدات؟",
        "ج26: هذا غير صحيح، هو خرج من مكتبي بمحض إرادته وأحضر الشيكين موقعين وخالصين، وأريد أن أنوه بأن الشيك الثاني كان خالياً من المستفيد ومن القيمة لأنه شيك على بياض، وهو من عرض ذلك، وكذلك هو وقع إقراراً منه بالمبالغ وتعهد في السداد، وهو ما يثبت ذلك، وأن صورة الشيك الضوئية الملتقطة من الشيك موضح فيها بأن توقيعه كان دون إكراه وإقرار منه بالمبالغ وتعهد في السداد. كما أني قمت بالاتصال عليه بعد خروجه من مكتبي وتسلمه للشيكين وذهابه إلى مكتبه، وأبلغته بوجود خطأ في أحد الشيكين، فقام بتصحيح الخطأ وأحضر لي الشيك بنفسه، وأن طريقة السداد عن طريق الشيكين هو اقتراح اتفقنا عليه بعد أن ناقشنا مسألة التسوية.",
        "ملحوظة 1: هذا وقد قدم الحاضر أمامنا صورة ضوئية لشيك ومدون تحتها إقرار، وتبين لنا بأنه الأصل، فأخذنا صورة عنها وأرجعنا الأصل للحاضر أمامنا، وأشرنا على الصورة بما يفيد النظر والإرفاق بتاريخ اليوم، وأشرنا عليها #1.",
    ],
    [
        "تمت الملحوظة. عضو النيابة.",
        "س27: ما مضمون ما تم تقديمه؟",
        "ج27: هذه صورة أخذتها عن الشيك الذي قام بتسليمه نواف والمتضمن المبلغ الذي اختلسه من أموال الجامعة، وثبت بها إقرار منه بأن المبلغ تم صرفه له من جامعة الخليج عن طريق الخطأ، وقام برده، وتعهد بإقرار برد كافة المبالغ بعد حصرها، وأن هذا الإقرار منه بدون طلب وبناء على مسؤوليته الشخصية وبدون أي إكراه، وحمل الإقرار توقيعه.",
        "ملحوظة 2: هذا وقد قدم الحاضر أمامنا عدد 6 أوراق تتضمن محادثات عبر برنامج التراسل الفوري واتساب، قمنا بإعادة ترقيمهم تصاعدياً من العدد 1 إلى 6، وأشرنا على الصفحة الأولى بما يفيد النظر والإرفاق بتاريخ اليوم.",
        "تمت الملحوظة. عضو النيابة.",
        "س28: ما مضمون ما تم تقديمه؟",
    ],
    [
        "ج28: هذه محادثات بيني وبين نواف ارحمه بعد قيامه بتسليم شيكين لي بعدة أيام، وتضمنت المحادثة إقراراً منه بقيامه بتسليم الشيكات بشكل ودي ودون إكراه، حيث ثبت في المحادثة الأولى بالنسبة للمؤشر عليها برقم 1 إرساله رسالة لي يقول فيها: «دليل ثقتي اللامحدودة فيك كتبت لك شيك 350 ألف د.ك بدون نقاش ولا جدال»، وأنه كتب لي شيكاً على بياض، وأيضاً تضمنت الرسالة إرساله عبارة: «متى ما يكون عندك تصور كامل بكل المطالبات والمبالغ اللي تشوف أنها انصرفت بالخطأ أنا حاضر أناقشك فيهم، وحق الجامعة وبوبيان ما يضيع». وتضمنت الرسالة، وتحديداً في ورقة رقم 2، أنه يكن لي المودة والاحترام، وهذا يناقض ما قرره بأني أكرهته، فشلون أكون مكرهاً ومهدداً على نحو ما قرر ويرسل لي هذه الرسالة. وتضمنت المحادثات الأخرى إقرارات منه بأن هذه المبالغ نتيجة صرفها له بسبب خطأ، وأنه ما راح يتأخر على سدادها، كما أنه أرسل رسالة يقول فيها: «إن في يوم من الأيام إذا كانت المبالغ أكبر، الشيك على بياض ما زال عندك»، وهذا يؤكد كلامي بأن تسليمه للشيك كان بمحض إرادته. والأهم من ذلك الصفحة الأخيرة للمحادثات التي قدمتها للنيابة، رسالته لي والتي قرر فيها: «أنا حاولت معاك بالود واللين وعطيتك شيك بقيمة 350 ألف د.ك، أكثر وأزيد من أي رقم عليه خلاف بينا، وكل هذا من حرصي على استمرار العلاقة الودية بيننا، وتأكيداً لحسن نيتي سلمتك شيك على بياض». فشلون أنا أكرهته وأرسل لي هذه الرسائل.",
        "س29: ما هو رقم الهاتف الذي أرسل إليك عبر تلك الرسائل؟",
        "ج29: رقم هاتفه الشخصي وهو 99602063.",
        "ملحوظة 3: هذا وقد قمنا بالاطلاع على المحادثات التي قدمت الصور الضوئية عنها من خلال هاتف الحاضر أمامنا بعد أن طلبنا منه تمكيننا من ذلك لمطابقتها، فتبين لنا مطابقة ما تم تقديمه للمحادثات، وأنها أرسلت من رقم هاتف مسجل باسم Nawaf Arhamah، مرسلة من رقم الهاتف 99602063، ورأينا إثبات ذلك، وقمنا بإرجاع الهاتف للحاضر أمامنا.",
        "تمت الملحوظة. عضو النيابة.",
        "س30: متى تمت استقالة الشاكي نواف من منصبه في مجلس الأمناء في جامعة الخليج للعلوم والتكنولوجيا؟",
        "ج30: بتاريخ 31/7/2025.",
        "س31: ما سبب استقالته من الجامعة سالفة الذكر؟",
        "ج31: هو من نفسه استقال، وأعتقد بسبب حرجه من الاختلاسات التي ارتكبها.",
    ],
    [
        "س32: ما قولك فيما قرره سالف الذكر في تحقيقات النيابة العامة من أنه التقى بك مرة أخرى في يوم الأربعاء الموافق 23/7/2025 وطلبت منه توقيع الاستقالة؟",
        "ج32: أنا ما طلبت منه توقيع أي استقالة، لكنه هو استقال من جامعة الخليج بمحض إرادته وكذلك من عدة أماكن أخرى، وأنا استبدلته في شركة إياس للتعليم الأكاديمي.",
        "س33: ما قولك فيما جاء في أقواله من أن سبب إتيانك لتلك الأفعال هو رغبتك بتولي المناصب التي يتولاها والإساءة إلى سمعته والإضرار به وفي الأعمال التي يزاولها خارج مظلة الشركة، قاصداً استرجاع المكافآت التي استحقها نتيجة أعماله عن طريق التهديد؟",
        "ج33: هذا غير صحيح، وهذه المناصب أنا مكنته من العمل فيها وأنا رئيسها وهو المرؤوس، والأموال التي اختلسها لا علاقة لها بالمكافآت لأن الأموال المفقودة تتجاوز حدود المكافآت.",
        "س34: بماذا تعلل اتهامه لك بارتكاب تلك الأفعال؟",
        "ج34: هذه ادعاءاته كلها كاذبة وغير صحيحة، وهو يريد التشهير فيني وبسمعتي كتاجر معروف وبالأعمال التي أزاولها، وأعتقد أنه يريد التملص من دفع الشيكات وابتزازي بشكوى مسبقة قبل أن أتقدم بشكوى ضده بشأن اختلاساته، وهذا بلاغ كاذب، وأنا بصدد تقديم عدة شكاوى منفصلة ضد نواف بشأن اختلاساته.",
        "س35: ما علاقتك بالضابط مجري التحريات فهد سامي الراشد وهل توجد خلافات سابقة بينكم؟",
        "ج35: لا أعرفه ولا توجد خلافات سابقة بيننا، والتقيت فيه بمناسبة سؤاله لي عن القضية.",
        "س36: ما قولك فيما جاء بأقوال مجري التحريات سالف الذكر وتحديداً في إجابة الأسئلة من س17 وحتى س22 «تلوناها عليه»؟",
        "ج36: نعم، صحيح ما جاء في أقوال الضابط التي تليت علي.",
        "ملحوظة 4: هذا وقد عرضنا على الحاضر أمامنا محضر التحريات المحرر بمعرفة سالف الذكر.",
        "تمت الملحوظة. عضو النيابة.",
        "س37: ما قولك فيما تم عرضه عليك؟",
        "ج37: نعم، صحيح ما جاء في التحريات على لساني، وأيضاً ما تضمنته المحادثات التي جاءت في المحضر.",
    ],
    [
        "س38: هل لديك أقوال أخرى؟",
        "ج38: لا.",
        "وتمت أقواله ووقع عليها في يوم 26-10-2025 في تمام الساعة 12:06 مساءً.",
        "الإسم: دبوس مبارك الدبوس [توقيع].",
        "عضو النيابة: مريم سعيد عتيق العنزي [توقيع].",
        "نهاية التحقيق والقرارات في الصفحة التالية.",
    ],
]


def clean_name_for_output(path: Path) -> tuple[Path, Path]:
    stem = path.stem
    return (
        path.with_name(f"{stem}.clean_arabic.txt"),
        path.with_name(f"{stem}.clean_arabic.md"),
    )


def inventory_pdfs(folder: Path) -> list[PdfInventory]:
    rows: list[PdfInventory] = []
    for pdf in sorted(folder.glob("*.pdf")):
        doc = fitz.open(pdf)
        embedded_pages = 0
        embedded_chars = 0
        for page in doc:
            text = page.get_text() or ""
            if text.strip():
                embedded_pages += 1
            embedded_chars += len(text)
        rows.append(
            PdfInventory(
                name=pdf.name,
                pages=doc.page_count,
                embedded_pages=embedded_pages,
                embedded_chars=embedded_chars,
                size=pdf.stat().st_size,
            )
        )
        doc.close()
    return rows


def select_target(folder: Path, inventory: list[PdfInventory]) -> Path:
    candidates = [
        row
        for row in inventory
        if row.name.startswith("09 - ") and row.pages == 9 and row.embedded_pages == 0 and row.embedded_chars < 100
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one 9-page scan starting with '09 - '; found {len(candidates)}")
    return folder / candidates[0].name


def render_pdf(pdf: Path, out_dir: Path, dpi: int = 320) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    doc = fitz.open(pdf)
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    for index, page in enumerate(doc, start=1):
        out_path = out_dir / f"page_{index:04d}.png"
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(out_path)
        rendered.append(out_path)
    doc.close()
    return rendered


def preprocess_image(image_path: Path, variant: str) -> Path:
    PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREPROCESS_DIR / f"{image_path.stem}__{variant}.png"
    if variant == "raw":
        shutil.copyfile(image_path, out_path)
        return out_path

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    if variant == "gray":
        processed = image
    elif variant == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(image)
    elif variant == "sharp":
        blur = cv2.GaussianBlur(image, (0, 0), 1.0)
        processed = cv2.addWeighted(image, 1.5, blur, -0.5, 0)
    elif variant == "binary":
        _, processed = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif variant == "adaptive":
        processed = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)
    else:
        raise ValueError(f"Unknown preprocessing variant: {variant}")

    cv2.imwrite(str(out_path), processed)
    return out_path


def score_text(text: str) -> float:
    arabic = len(ARABIC_RE.findall(text))
    mojibake = len(MOJIBAKE_RE.findall(text))
    controls = len(CONTROL_RE.findall(text))
    return arabic * 2.0 + len(text) * 0.15 - mojibake * 15.0 - controls * 10.0


def run_tesseract(page_num: int, image_path: Path, variant: str, lang: str, psm: int) -> OcrCandidate:
    RAW_OCR_DIR.mkdir(parents=True, exist_ok=True)
    safe_lang = lang.replace("+", "_")
    out_base = RAW_OCR_DIR / f"page_{page_num:04d}__{variant}__{safe_lang}__psm{psm}"
    text_path = out_base.with_suffix(".txt")
    try:
        input_image = preprocess_image(image_path, variant)
        cmd = [
            str(TESSERACT),
            str(input_image),
            str(out_base),
            "--tessdata-dir",
            str(TESSDATA),
            "-l",
            lang,
            "--psm",
            str(psm),
            "--oem",
            "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        text = text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else ""
        return OcrCandidate(
            page=page_num,
            variant=variant,
            lang=lang,
            psm=psm,
            success=result.returncode == 0,
            text_chars=len(text),
            arabic_chars=len(ARABIC_RE.findall(text)),
            mojibake_chars=len(MOJIBAKE_RE.findall(text)),
            score=score_text(text),
            output=str(text_path),
            error=(result.stderr or "").strip(),
        )
    except Exception as exc:
        return OcrCandidate(
            page=page_num,
            variant=variant,
            lang=lang,
            psm=psm,
            success=False,
            text_chars=0,
            arabic_chars=0,
            mojibake_chars=0,
            score=-999999.0,
            output=str(text_path),
            error=str(exc),
        )


def run_local_ocr(rendered_pages: list[Path]) -> dict[int, list[OcrCandidate]]:
    results: dict[int, list[OcrCandidate]] = {i: [] for i in range(1, len(rendered_pages) + 1)}
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {}
        for page_num, image_path in enumerate(rendered_pages, start=1):
            for variant, lang, psm in OCR_CONFIGS:
                future = executor.submit(run_tesseract, page_num, image_path, variant, lang, psm)
                future_map[future] = page_num
        for future in as_completed(future_map):
            candidate = future.result()
            results[candidate.page].append(candidate)

    for page_num, candidates in results.items():
        candidates.sort(key=lambda item: item.score, reverse=True)
        page_json = RAW_OCR_DIR / f"page_{page_num:04d}_tesseract_candidates.json"
        page_json.write_text(
            json.dumps([asdict(candidate) for candidate in candidates], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return results


def copy_historical_candidates() -> list[str]:
    copied: list[str] = []
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORICAL_ARTIFACT_DIR.exists():
        return copied
    for source in sorted(HISTORICAL_ARTIFACT_DIR.glob("page_*.json")):
        data = json.loads(source.read_text(encoding="utf-8"))
        page = int(data.get("page") or source.stem.split("_")[-1])
        chosen = data.get("chosen_after_full_compare") or data.get("chosen") or {}
        text = chosen.get("text", "") if isinstance(chosen, dict) else ""
        out = HISTORICAL_DIR / f"page_{page:04d}__historical_chosen.txt"
        out.write_text(text, encoding="utf-8")
        copied.append(str(out))
    return copied


def build_final_text() -> str:
    sections: list[str] = []
    for index, paragraphs in enumerate(FINAL_PAGES, start=1):
        printed_page = index + 25
        body = "\n\n".join(paragraph.strip() for paragraph in paragraphs if paragraph.strip())
        sections.append(f"===== Page {index} | الصفحة المطبوعة {printed_page} =====\n\n{body}")
    return "\n\n".join(sections).strip() + "\n"


def build_final_markdown(source_pdf: Path, text_body: str) -> str:
    title = source_pdf.stem
    md_body = re.sub(r"^===== Page (\d+) \| الصفحة المطبوعة (\d+) =====$", r"## Page \1 | الصفحة المطبوعة \2", text_body, flags=re.MULTILINE)
    return (
        f"# {title}\n\n"
        "Clean Arabic extraction prepared from local page renders, local Tesseract OCR candidates, existing local artifact candidates, and manual visual adjudication in-session. No new external OCR, cloud API, or paid LLM call is executed by this script.\n\n"
        f"{md_body}"
    )


def validate_output(text: str, page_count: int) -> dict[str, int | bool]:
    page_markers = len(re.findall(r"^===== Page \d+", text, flags=re.MULTILINE))
    mojibake_chars = len(MOJIBAKE_RE.findall(text))
    control_chars = len(CONTROL_RE.findall(text))
    arabic_chars = len(ARABIC_RE.findall(text))
    return {
        "page_markers": page_markers,
        "expected_pages": page_count,
        "page_marker_match": page_markers == page_count,
        "chars": len(text),
        "arabic_chars": arabic_chars,
        "mojibake_chars": mojibake_chars,
        "control_chars": control_chars,
        "empty": len(text.strip()) == 0,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    if not TESSERACT.exists():
        raise RuntimeError(f"Tesseract binary not found: {TESSERACT}")
    if not (TESSDATA / "ara.traineddata").exists():
        raise RuntimeError(f"Arabic tessdata not found: {TESSDATA / 'ara.traineddata'}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = inventory_pdfs(APPEAL_DIR)
    target_pdf = select_target(APPEAL_DIR, inventory)
    rendered_pages = render_pdf(target_pdf, RENDER_DIR)
    local_ocr_results = run_local_ocr(rendered_pages)
    historical_copied = copy_historical_candidates()

    final_txt_path, final_md_path = clean_name_for_output(target_pdf)
    final_text = build_final_text()
    final_md = build_final_markdown(target_pdf, final_text)
    final_txt_path.write_text(final_text, encoding="utf-8")
    final_md_path.write_text(final_md, encoding="utf-8")

    validation = validate_output(final_text, len(rendered_pages))
    summary = {
        "source_pdf": str(target_pdf),
        "output_txt": str(final_txt_path),
        "output_md": str(final_md_path),
        "artifact_dir": str(ARTIFACT_DIR),
        "inventory": [asdict(row) for row in inventory],
        "selected_target": next(row.name for row in inventory if row.name == target_pdf.name),
        "rendered_pages": [str(path) for path in rendered_pages],
        "local_tesseract_configs": [{"variant": v, "lang": l, "psm": p} for v, l, p in OCR_CONFIGS],
        "local_tesseract_best_by_page": {
            str(page): asdict(candidates[0]) if candidates else None for page, candidates in local_ocr_results.items()
        },
        "historical_local_artifact_candidates_copied": historical_copied,
        "validation": validation,
        "notes": [
            "No new external OCR, cloud API, or paid LLM call was executed.",
            "Final clean text is manually adjudicated from the current rendered page images, fresh local Tesseract OCR candidates, and existing local artifact candidates.",
            "Handwritten signatures are represented only where the surrounding printed form makes the signer identity clear.",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_lines = [
        "Appeal Dabbous investigation local OCR audit",
        f"Source PDF: {target_pdf}",
        f"Pages: {len(rendered_pages)}",
        "Embedded text: 0 pages / 0 chars (pure scan)",
        "No new external OCR, cloud API, or paid LLM call executed.",
        f"TXT output: {final_txt_path}",
        f"MD output: {final_md_path}",
        f"Artifact directory: {ARTIFACT_DIR}",
        f"Validation: {json.dumps(validation, ensure_ascii=False)}",
        "",
        "Best local Tesseract candidate per page:",
    ]
    for page, candidates in local_ocr_results.items():
        if candidates:
            best = candidates[0]
            audit_lines.append(
                f"- Page {page}: {best.variant}, lang={best.lang}, psm={best.psm}, chars={best.text_chars}, arabic={best.arabic_chars}, mojibake={best.mojibake_chars}, score={best.score:.1f}"
            )
    AUDIT_PATH.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    print(json.dumps({"txt": str(final_txt_path), "md": str(final_md_path), "validation": validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
