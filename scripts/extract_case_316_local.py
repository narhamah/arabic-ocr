from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import fitz


FOLDER = Path(r"C:\Users\narha\Dropbox\Dabbous Allegations\New Extorion Case")
ARTIFACT_DIR = FOLDER / "case_316_2025_ocr_artifacts"
SUMMARY_PATH = ARTIFACT_DIR / "case_316_2025_summary.json"
AUDIT_PATH = ARTIFACT_DIR / "case_316_2025_audit.txt"

FINAL_PAGE_TEXT = """وزارة الداخلية
الإدارة العامة للتحقيقات

التاريخ: 2026/3/4
الرقم:

شهادة لمن يهمه الأمر

بالاطلاع على ملف الجنحة رقم 2025/316 جنح بيان تبين أنها تتضمن البلاغ المقدم من / سارة محمد سعد العدواني بصفتها وكيلاً عن جامعة الخليج للعلوم والتكنولوجيا بتاريخ 2025/3/25 وبدائرة اختصاص مخفر شرطة السالمية عن تعرض جامعة الخليج للعلوم والتكنولوجيا للسرقة وخيانة الأمانة من قبل رئيس مجلس أمناء الجامعة السابق المدعو / نواف ارحمه سالم ارحمه الذي قام بسحب المبالغ المبينة القدر من حساب الجامعة عبر البطاقة الائتمانية بعد أن حصل على رقمها السري من المدير المالي للجامعة وقام بسداد قيمة فواتير خاصة بشركته المملوكة له ولحسابه الخاص وذلك على النحو المبين بالتحقيقات.

وبسؤال المدعو / نواف ارحمه سالم ارحمه أنكر ما أسند إليه من اتهام.

ومازالت الجنحة المذكورة رهن التحقيق ولم يصدر فيها تصرف نهائي.

وقد حررت هذه الشهادة وسلمت إلى / نور بدر حسين الصانع بصفتها وكيلاً عن نواف ارحمه سالم ارحمه بناء على طلبها.

الشهادة صالحة لمدة ثلاثة أشهر من تاريخ صدورها.

مدير عام / الإدارة العامة للتحقيقات
اللواء حقوقي / فيصل خالد المكراد

ختم/توقيع ظاهر أسفل الصفحة: وزارة الداخلية - الإدارة العامة للتحقيقات، وختم إدارة تحقيق محافظة حولي."""


def inspect_pdf(path: Path) -> dict[str, Any]:
    with fitz.open(path) as doc:
        embedded_chars = 0
        embedded_pages = 0
        sample_parts: list[str] = []
        for index, page in enumerate(doc):
            text = page.get_text("text") or ""
            embedded_chars += len(text)
            if text.strip():
                embedded_pages += 1
            if index < 2:
                sample_parts.append(text[:240].replace("\n", " "))
        return {
            "path": str(path),
            "name": path.name,
            "bytes": path.stat().st_size,
            "pages": doc.page_count,
            "embedded_pages": embedded_pages,
            "embedded_chars": embedded_chars,
            "sample": " | ".join(sample_parts),
        }


def select_target(inventory: list[dict[str, Any]]) -> Path:
    candidates = [
        Path(record["path"])
        for record in inventory
        if record["pages"] == 1 and record["embedded_pages"] == 1 and 900 <= record["embedded_chars"] <= 1200
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one target PDF; found {[path.name for path in candidates]}")
    return candidates[0]


def text_metrics(text: str) -> dict[str, int]:
    return {
        "chars": len(text),
        "arabic_chars": sum(1 for char in text if "\u0600" <= char <= "\u06ff"),
        "page_markers": len(re.findall(r"^===== Page ", text, flags=re.MULTILINE)),
        "replacement_chars": text.count("\ufffd"),
        "mojibake_chars": text.count("Ø") + text.count("Ù"),
        "control_chars": sum(
            1
            for char in text
            if (ord(char) < 32 and char not in "\n\r\t") or 127 <= ord(char) <= 159
        ),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    inventory = [inspect_pdf(path) for path in sorted(FOLDER.glob("*.pdf"), key=lambda item: item.name.casefold())]
    target = select_target(inventory)

    with fitz.open(target) as doc:
        native_text = doc[0].get_text("text") or ""
    native_path = ARTIFACT_DIR / "native_broken_text_layer.txt"
    native_path.write_text(native_text, encoding="utf-8", errors="replace")

    final_text = f"===== Page 1 =====\n{FINAL_PAGE_TEXT.strip()}\n"
    final_md = "\n".join([
        f"# {target.stem}",
        "",
        "Extraction: local deterministic inspection/OCR artifacts plus in-session visual LLM validation only. No external LLM/cloud OCR/API calls.",
        "",
        "## Page 1",
        "",
        FINAL_PAGE_TEXT.strip(),
        "",
    ])
    txt_path = target.with_name(f"{target.stem}.clean_arabic.txt")
    md_path = target.with_name(f"{target.stem}.clean_arabic.md")
    txt_path.write_text(final_text, encoding="utf-8")
    md_path.write_text(final_md, encoding="utf-8")

    benchmark_path = ARTIFACT_DIR / "case_316_ocr_benchmark.json"
    benchmark = []
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))

    summary = {
        "source_pdf": str(target),
        "output_txt": str(txt_path),
        "output_md": str(md_path),
        "artifacts": str(ARTIFACT_DIR),
        "cloud_or_external_ai": False,
        "source_quality": {
            "embedded_text_layer": "present_but_unusable_broken_font_extraction",
            "native_text_path": str(native_path),
        },
        "extraction_method": "Rendered-page visual validation by this session, with local Tesseract OCR candidates preserved for audit.",
        "metrics": text_metrics(final_text),
        "top_ocr_candidates": benchmark[:5],
        "inventory": inventory,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_lines = [
        "Case 316/2025 local extraction audit",
        "",
        f"Source PDF: {target}",
        f"TXT output: {txt_path}",
        f"Markdown output: {md_path}",
        f"Artifacts: {ARTIFACT_DIR}",
        "",
        "Policy: local deterministic inspection/OCR artifacts plus in-session visual LLM validation only. No external LLM/cloud OCR/API calls.",
        "Native text layer finding: present but unusable; it decodes as broken Latin/font text, so the final transcript uses the rendered page image.",
        "",
        "Validation:",
    ]
    for key, value in summary["metrics"].items():
        audit_lines.append(f"- {key}: {value}")
    audit_lines.extend(["", "Source folder inventory:"])
    for record in inventory:
        audit_lines.append(
            f"- {record['name']}: pages={record['pages']}, embedded_pages={record['embedded_pages']}, "
            f"embedded_chars={record['embedded_chars']}, bytes={record['bytes']}"
        )
    AUDIT_PATH.write_text("\n".join(audit_lines).strip() + "\n", encoding="utf-8")

    print(json.dumps({
        "output_txt": str(txt_path),
        "output_md": str(md_path),
        "audit": str(AUDIT_PATH),
        "summary": str(SUMMARY_PATH),
        **summary["metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
