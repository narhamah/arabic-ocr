from __future__ import annotations

import json
import re
from pathlib import Path

import fitz


FOLDER = Path(r"C:\Users\narha\Dropbox\Dabbous Allegations\Fraud Case")
ARTIFACT_DIR = FOLDER / "mahdr_dabbous_ocr_artifacts"

# Page 7: Gemini Flash dropped material from the WhatsApp quote section.
# Page 1: Gemini Pro is more readable/flowing than the Flash table rendering.
PAGE_ENGINE = {
    1: "gemini_pro",
    2: "gemini_flash",
    3: "gemini_flash",
    4: "gemini_flash",
    5: "gemini_flash",
    6: "gemini_flash",
    7: "gemini_pro",
    8: "gemini_flash",
    9: "gemini_flash",
}


def locate_target_pdf() -> Path:
    matches: list[Path] = []
    for pdf_path in FOLDER.glob("*.pdf"):
        try:
            doc = fitz.open(pdf_path)
            embedded_pages = 0
            for page_index in range(doc.page_count):
                if (doc.load_page(page_index).get_text("text") or "").strip():
                    embedded_pages += 1
            if doc.page_count == 9 and embedded_pages == 0:
                matches.append(pdf_path)
            doc.close()
        except Exception:
            continue
    if len(matches) != 1:
        raise RuntimeError(f"Could not uniquely locate target PDF: {[p.name for p in matches]}")
    return matches[0]


def get_engine_text(page_no: int, engine: str) -> str:
    data = json.loads((ARTIFACT_DIR / f"page_{page_no:02d}.json").read_text(encoding="utf-8"))
    if engine == data.get("chosen", {}).get("engine"):
        return data.get("chosen", {}).get("text") or ""
    for output in data.get("outputs", []):
        if output.get("engine") == engine:
            return output.get("text") or ""
    raise KeyError((page_no, engine))


def normalize_qa_line(line: str) -> str:
    stripped = line.strip()
    match = re.match(r"^(س|ج)\s*(\d+)\s*$", stripped)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    match = re.match(r"^(\d+)\s*(س|ج)\s*$", stripped)
    if match:
        return f"{match.group(2)}{match.group(1)}"
    match = re.match(r"^(س|ج)\s*(\d+)\b(.*)$", stripped)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"
    return stripped


def line_cleanup(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("[Image of the Kuwaiti Ministry of Justice Logo]\n", "")
    text = text.replace("دولة\nالكويت", "دولة الكويت")
    text = text.replace("محضر تحقيق", "محضر التحقيق")
    text = text.replace("?", "؟")
    lines = []
    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]{2,}", " ", raw_line).strip()
        line = re.sub(r"^\[Signature:\s*(.*?)\]$", r"\1", line)
        lines.append(normalize_qa_line(line))
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def global_cleanup(text: str) -> str:
    replacements = {
        "الالرقم الآلي": "الرقم الآلي",
        "النيابة العامه": "النيابة العامة",
        "الرقم الالي": "الرقم الآلي",
        "الالي للقضية": "الآلي للقضية",
        "لولوة مبارك جحيل": "لؤلؤة مبارك جحيل",
        "صباحا": "صباحاً",
        "اسمى": "اسمي",
        "عمرى": "عمري",
        "اعمال حره": "أعمال حرة",
        "أعمال حره": "أعمال حرة",
        "ادارة": "إدارة",
        "المدنى": "المدني",
        "الورقه": "الورقة",
        "لاثبات": "لإثبات",
        "الي داخلها": "إلى داخلها",
        "بالاتي": "بالآتي",
        "شركو بوبيان": "شركة بوبيان",
        "شركة بويان": "شركة بوبيان",
        "الستراتيجيه": "الاستراتيجية",
        "استراتيجبة": "استراتيجية",
        "التكنلوجيا": "التكنولوجيا",
        "الاكاديمي": "الأكاديمي",
        "باعمال": "بأعمال",
        "الاعمال": "الأعمال",
        "الماليه": "المالية",
        "الشركه": "الشركة",
        "للشركه": "للشركة",
        "الجامعه": "الجامعة",
        "المحاسبه": "المحاسبة",
        "شئون": "شؤون",
        "الشئون": "الشؤون",
        "الاموال": "الأموال",
        "الموجوده": "الموجودة",
        "المفقوده": "المفقودة",
        "سابقه": "سابقة",
        "سنه": "سنة",
        "الداخليه": "الداخلية",
        "الاستقاله": "الاستقالة",
        "الاساءه": "الإساءة",
        "المرئوس": "المرؤوس",
        "التخاساته": "اختلاساته",
        "التخسها": "اختلسها",
        "اقوال": "أقوال",
        "اقرار": "إقرار",
        "اكراه": "إكراه",
        "اعادة": "إعادة",
        "امامنا": "أمامنا",
        "اوضحت": "أوضحت",
        "اموال": "أموال",
        "ارحمه": "أرحمه",
        "إرحمه": "أرحمه",
        "الاوراق": "الأوراق",
        "اوراق": "أوراق",
        "اخرى": "أخرى",
        "اخري": "أخرى",
        "آخري": "أخرى",
        "اخر": "آخر",
        "اجراءات": "إجراءات",
        "اقدم": "أقدم",
        "فورا": "فوراً",
        "شخصيا": "شخصياً",
        "وديا": "ودياً",
        "الصفحه": "الصفحة",
        "الاولي": "الأولى",
        "الضوئيه": "الضوئية",
        "مطابقه": "مطابقة",
        "مرسله": "مرسلة",
        "استقالت الشاكي": "استقالة الشاكي",
        "برتكاب": "بارتكاب",
        "شكاوي": "شكاوى",
        "بالنسبه": "بالنسبة",
        "القيمه": "القيمة",
        "بقيمه": "بقيمة",
        "ادني": "أدنى",
        "خاليا": "خالياً",
        "القضائيه": "القضائية",
        "المقدمه": "المقدمة",
        "الاول": "الأول",
        " ال أن": " إلى أن",
        "و الاستراتيجية": "والاستراتيجية",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def page_specific_cleanup(page_no: int, text: str) -> str:
    if page_no == 1:
        text = text.replace("التهم: التهديد", "التهمة: التهديد")
        text = text.replace(
            "رقم الإثبات: رقم المدني/ 263051001091",
            "رقم الإثبات: الرقم المدني/ 263051001091",
        )
        text = text.replace(
            "أعمل: أعمال حرة رئيس مجلس إدارة\nشركة بوبيان للبتروكيماويات",
            "أعمل: أعمال حرة، رئيس مجلس إدارة شركة بوبيان للبتروكيماويات",
        )
        text = text.replace(
            "مقيم في: م/ الفحيحيل, ق/ 10, ش/ 5, ج/ -\n-, مبنى/ 15, دور/ --",
            "مقيم في: م/ الفحيحيل، ق/ 10، ش/ 5، ج/ -، مبنى/ 15، دور/ --",
        )
        text = text.replace("\nمريم العنزي\nصفحة 26", "\nعضو النيابة: مريم العنزي\nصفحة 26")
    elif page_no == 3:
        text = text.replace("\nس12\nلا كنا", "\nج12\nلا كنا")
        text = text.replace("استغرقتها الاجتماع", "استغرقه الاجتماع")
        text = text.replace("قتلته حياك", "قلت له حياك")
        text = text.replace("قتلت له حياك", "قلت له حياك")
        text = text.replace("آلكبين", "المبين")
        text = text.replace("قادماً من الورقة", "فتفاجأ من الورقة")
        text = text.replace(
            'انا ابي فلوسي و احتاج أموال الشركة "',
            'أنا أريد فلوسي وأحتاج أموال الشركة ترجع"',
        )
        text = text.replace(
            'انا مابي\nمنك شي انا ابي فلوسي و احتاج أموال الشركة "',
            'أنا لا أريد\nمنك شيئاً؛ أنا أريد فلوسي وأحتاج أموال الشركة ترجع"',
        )
        text = text.replace("لم تنتهي إجراءات", "لم تنته إجراءات")
        text = text.replace("فقترح", "فاقترح")
        text = text.replace("علي بياض", "على بياض")
    elif page_no == 4:
        text = text.replace("من این", "من أين")
        text = text.replace("این قام", "أين قام")
        text = text.replace("الحد الادنى", "الحد الأدنى")
        text = text.replace("علي بياض", "على بياض")
    elif page_no == 5:
        text = text.replace("\nس24\nهذا الكلام", "\nج24\nهذا الكلام")
        text = text.replace("حسابات شخصياً و", "حسابات تخصه شخصياً و")
        text = text.replace("فإني سؤمارس", "فإني سأمارس")
        text = text.replace("لا سؤمارس", "إلا سأمارس")
        text = text.replace("لم احدده فقط", "لم أهدده، فقط")
        text = text.replace("لم احدده، فقط", "لم أهدده، فقط")
    elif page_no == 6:
        text = text.replace(
            "س27\n\nج27\nما مضمون ما تم تقديمه؟\nهذه صوره",
            "س27\nما مضمون ما تم تقديمه؟\nج27\nهذه صورة",
        )
        text = text.replace(
            "س27\nج27\nما مضمون ما تم تقديمه؟\nهذه صورة",
            "س27\nما مضمون ما تم تقديمه؟\nج27\nهذه صورة",
        )
        text = text.replace("صوره اخذتها", "صورة أخذتها")
        text = text.replace("صورة اخذتها", "صورة أخذتها")
        text = text.replace("ملحوظة2", "ملحوظة 2")
        text = text.replace("الواتساب", "واتساب")
        text = text.replace("باعادة ترقيمهم", "بإعادة ترقيمها")
        text = text.replace("بإعادة ترقيمهم", "بإعادة ترقيمها")
        text = text.replace("بإعادة ترقيمةم", "بإعادة ترقيمها")
        text = text.replace("تصاعديا", "تصاعدياً")
        text = text.replace("والارفاق", "والإرفاق")
        text = text.replace("و الارفاق", "والإرفاق")
    elif page_no == 7:
        text = text.replace("بعد قيامنا بتسليم شيكين لي", "بعد قيامه بتسليم شيكين لي")
        text = text.replace("الا محدوده", "اللامحدودة")
        text = text.replace("كتبتلك", "كتبت لك")
        text = text.replace("و ان كتب لي شيك", "وأنه كتب لي شيك")
        text = text.replace("لصفحه الاخيره", "للصفحة الأخيرة")
        text = text.replace("اكرهته و السري هذا الرسايل", "أكرهته وأرسل لي هذه الرسائل")
        text = text.replace("السرت من رقم", "أرسلت من رقم")
        text = text.replace("بإسم", "باسم")
        text = text.replace("مريم الفزع", "مريم العنزي")
    elif page_no == 8:
        text = text.replace(
            "سالف الذكر و تحديداً في اجابة الاسئله",
            "سالف الذكر وتحديداً في إجابة الأسئلة",
        )
        text = text.replace('" تلوناه عليه "', '"تليت عليه"')
        text = text.replace("التي توليت علي", "التي تليت علي")
    elif page_no == 9:
        text = text.replace(
            "س38\nح38\nهل لديك أقوال أخرى ؟\nلا",
            "س38\nهل لديك أقوال أخرى؟\nج38\nلا",
        )
        text = text.replace("و تمت أقواله ووقع عليها", "وتمت تلاوة أقواله ووقع عليها")
        text = text.replace("الإسم", "الاسم")
    return text


def clean_page(page_no: int, text: str) -> str:
    text = line_cleanup(text)
    text = global_cleanup(text)
    text = line_cleanup(text)
    text = page_specific_cleanup(page_no, text)
    return line_cleanup(text)


def main() -> None:
    pdf_path = locate_target_pdf()
    out_path = FOLDER / f"{pdf_path.stem}.full_ocr_consolidated.clean_arabic.txt"
    audit_path = FOLDER / "mahdr_dabbous_full_ocr_consolidated_audit.txt"

    pages = []
    engine_report = []
    for page_no in range(1, 10):
        engine = PAGE_ENGINE[page_no]
        text = clean_page(page_no, get_engine_text(page_no, engine))
        pages.append(f"===== صفحة {page_no} =====\n{text}")
        engine_report.append(f"صفحة {page_no}: {engine}")

    header = (
        "محضر تحقيق النيابة مع دبوس\n"
        "نص موحد ومنقح من مقارنة مصادر OCR المتاحة.\n"
        "مصادر المقارنة: Tesseract Arabic، OpenAI OCR، Gemini Pro، Gemini Flash.\n"
        "اختيار المصدر لكل صفحة:\n"
        + "\n".join(engine_report)
        + "\n\n"
    )
    out_path.write_text(header + "\n\n".join(pages).strip() + "\n", encoding="utf-8")

    audit = [
        "Full OCR consolidated audit",
        f"Source PDF: {pdf_path}",
        f"Output file: {out_path}",
        "Engines compared: tesseract, openai, gemini_pro, gemini_flash",
        "Paddle engines were not available in the current repo venv.",
        "Selections:",
        *engine_report,
    ]
    audit_path.write_text("\n".join(audit) + "\n", encoding="utf-8")

    final_text = out_path.read_text(encoding="utf-8")
    arabic_chars = sum(1 for char in final_text if "\u0600" <= char <= "\u06FF")
    mojibake_markers = final_text.count("Ø") + final_text.count("Ù") + final_text.count("�")
    print(out_path)
    print(f"chars={len(final_text)} arabic={arabic_chars} mojibake_markers={mojibake_markers}")
    print(audit_path)


if __name__ == "__main__":
    main()
