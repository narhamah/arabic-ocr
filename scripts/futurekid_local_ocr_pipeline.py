from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageFilter, ImageOps

from arabic_ocr.arabic_text import clean_arabic_text, contains_arabic
from arabic_ocr.native_pdf import extract_best_native_page_text


SOURCE_DIR = Path(r"C:\Users\narha\Dropbox\Future Kid Valuation\Disclosures")
OUTPUT_DIR = SOURCE_DIR / "clean_txt"
ARTIFACT_DIR = SOURCE_DIR / "ocr_audit_artifacts" / "local_tesseract"
PAGE_CACHE_DIR = ARTIFACT_DIR / "pages"
SUMMARY_PATH = ARTIFACT_DIR / "local_ocr_summary.json"
TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA_DIR = Path(r"C:\Users\narha\arabic-ocr\.tessdata")

OCR_VARIANTS: tuple[tuple[str, int], ...] = (
    ("contrast", 4),
    ("sharp", 4),
    ("bw200", 6),
    ("bw200", 11),
)

DISCLOSURE_ANCHORS = (
    "الكويت",
    "بورصة",
    "شركة",
    "المستقبل",
    "الأجيال",
    "افصاح",
    "إفصاح",
    "الجمعية",
    "مجلس",
    "الإدارة",
    "البيانات",
    "المالية",
    "الربع",
    "النصف",
    "السنوية",
    "الدخل",
    "المركز",
    "الأرباح",
    "الخسائر",
    "رأس",
    "المال",
    "دينار",
    "فلس",
    "سهم",
    "أسهم",
    "تداول",
    "تسهيلات",
    "ائتمانية",
)


@dataclass(slots=True)
class PdfRecord:
    path: Path
    sha256: str
    page_count: int
    duplicate_of: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local-only deterministic OCR for Future Kid disclosures.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--limit", type=int, default=0, help="Optional number of PDFs to process for testing.")
    parser.add_argument("--force", action="store_true", help="Re-run page OCR even when cached.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_stem(pdf_path: Path) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", pdf_path.stem)
    safe = re.sub(r"\s+", " ", safe).strip(" .")
    return safe or hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()[:12]


def inspect_pdf(path: Path) -> tuple[str, int]:
    file_hash = sha256_file(path)
    doc = fitz.open(path)
    page_count = doc.page_count
    doc.close()
    return file_hash, page_count


def load_records(limit: int = 0) -> list[PdfRecord]:
    records: list[PdfRecord] = []
    seen: dict[str, str] = {}
    for pdf_path in sorted(SOURCE_DIR.glob("*.pdf"), key=lambda p: p.name.casefold()):
        file_hash, page_count = inspect_pdf(pdf_path)
        duplicate_of = seen.get(file_hash)
        if duplicate_of is None:
            seen[file_hash] = pdf_path.name
        records.append(PdfRecord(pdf_path, file_hash, page_count, duplicate_of))
        if limit and len(records) >= limit:
            break
    return records


def render_page(pdf_path: Path, page_index: int, dpi: int) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return image


def preprocess(image: Image.Image, variant: str) -> Image.Image:
    gray = ImageOps.grayscale(image)
    contrast = ImageOps.autocontrast(gray)
    if variant == "contrast":
        return contrast
    if variant == "sharp":
        return contrast.filter(ImageFilter.SHARPEN)
    if variant == "smoothsharp":
        return contrast.filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.SHARPEN)
    if variant == "bw":
        sharp = contrast.filter(ImageFilter.SHARPEN)
        return sharp.point(lambda pixel: 255 if pixel > 185 else 0)
    if variant == "bw200":
        sharp = contrast.filter(ImageFilter.SHARPEN)
        return sharp.point(lambda pixel: 255 if pixel > 200 else 0)
    return image


def run_tesseract(image: Image.Image, *, variant: str, psm: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="futurekid_tess_") as temp_dir:
        temp_path = Path(temp_dir)
        image_path = temp_path / f"{variant}_{psm}.png"
        output_base = temp_path / "ocr"
        image.save(image_path)
        command = [
            str(TESSERACT_EXE),
            str(image_path),
            str(output_base),
            "--tessdata-dir",
            str(TESSDATA_DIR),
            "-l",
            "ara+eng",
            "--oem",
            "1",
            "--psm",
            str(psm),
            "-c",
            "preserve_interword_spaces=1",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "engine": "tesseract",
                "variant": variant,
                "psm": psm,
                "success": False,
                "error": f"timeout: {exc}",
                "text": "",
                "score": float("-inf"),
            }

        text_path = Path(str(output_base) + ".txt")
        text = text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else ""
        text = normalize_ocr_text(text)
        return {
            "engine": "tesseract",
            "variant": variant,
            "psm": psm,
            "success": result.returncode == 0,
            "error": result.stderr.strip()[-500:] if result.stderr.strip() else None,
            "text": text,
            "score": score_text(text),
        }


def extract_native_candidate(pdf_path: Path, page_index: int) -> dict[str, Any]:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    text, metadata = extract_best_native_page_text(pdf_path, doc, page, cmap_cache={})
    doc.close()
    text = normalize_ocr_text(text)
    return {
        "engine": "native_pdf",
        "variant": metadata.get("source"),
        "psm": None,
        "success": bool(text.strip()),
        "error": None,
        "text": text,
        "score": score_text(text),
        "metadata": metadata,
    }


def normalize_ocr_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\u00a0", " ")
    # These Latin-1 characters are mojibake artifacts, not valid disclosure text.
    text = text.replace("Ø", "").replace("Ù", "").replace("�", "")
    text = text.replace("ـ", "")
    text = text.replace("؟", "?")
    text = text.replace("؛", ";")
    text = text.replace("،", ",")
    text = text.replace("كـ", "ك")
    text = text.replace("﷼", "ريال")
    text = text.replace("\u06a9", "ك").replace("\u06cc", "ي")
    if contains_arabic(text):
        text = clean_arabic_text(text)
        text = apply_high_confidence_arabic_fixes(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def apply_high_confidence_arabic_fixes(text: str) -> str:
    """Apply conservative domain-specific OCR fixes without inventing content."""
    replacements = {
        "التاريض": "التاريخ",
        "التارض": "التاريخ",
        "ألتاريخ": "التاريخ",
        "اجتاع": "اجتماع",
        "اجعاع": "اجتماع",
        "اجعاء": "اجتماع",
        "اججاع": "اجتماع",
        "الجعية": "الجمعية",
        "الججعية": "الجمعية",
        "العأدية": "العادية",
        "العاديهة": "العادية",
        "الترفيبية": "الترفيهية",
        "التزفيهية": "الترفيهية",
        "النرفيهية": "الترفيهية",
        "الترفهية": "الترفيهية",
        "الترفيهيةالعقارية": "الترفيهية العقارية",
        "التزفيهيةالعقارية": "الترفيهية العقارية",
        "الترفيبيةالعقارية": "الترفيهية العقارية",
        "العفازية": "العقارية",
        "العفاريه": "العقارية",
        "المستمبل": "المستقبل",
        "المستتيبل": "المستقبل",
        "المستقيل": "المستقبل",
        "بورصهالكويت": "بورصة الكويت",
        "بورصةالكويت": "بورصة الكويت",
        "الكويتالمحترمين": "الكويت المحترمين",
        "الكويتالحترمين": "الكويت المحترمين",
        "الكويتاللحرمين": "الكويت المحترمين",
        "المحرمين": "المحترمين",
        "المححرمين": "المحترمين",
        "تحيجة": "تحية",
        "تجيجة": "تحية",
        "طيبةو": "طيبة وبعد",
        "إنعتاد": "إنعقاد",
        "إنعتااد": "إنعقاد",
        "إنعقاد": "انعقاد",
        "لائتخاب": "لانتخاب",
        "العاديةلانتخاب": "العادية لانتخاب",
        "لفتزة": "لفترة",
        "لشترة": "لفترة",
        "مسنوات": "سنوات",
        "سزوات": "سنوات",
        "وسيم الإنعقاد": "وسيتم الانعقاد",
        "وسيت الإنعقاد": "وسيتم الانعقاد",
        "وسيم\\ لإنعقاد": "وسيتم الانعقاد",
        "يمقر": "بمقر",
        "همع الوزارات": "مجمع الوزارات",
        "ممع الوزارات": "مجمع الوزارات",
        "وز ارة": "وزارة",
        "التجارةوالصناعة": "التجارة والصناعة",
        "النصفصاحا": "النصف صباحاً",
        "الصفصاحا": "النصف صباحاً",
        "صاحا": "صباحاً",
        "المرفقات:": "المرفقات:",
        "مرفق موافقة وزارة التجارةوالصناعة": "مرفق موافقة وزارة التجارة والصناعة",
        "سجل تجارى": "سجل تجاري",
        "سحل تجارى": "سجل تجاري",
        "الاتلكتروني": "الإلكتروني",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    text = re.sub(r"(?<=\S)المحترمين\b", " المحترمين", text)
    text = re.sub(r"(?<=\S)العقارية\b", " العقارية", text)
    text = re.sub(r"(?<=\S)لانتخاب\b", " لانتخاب", text)
    return text


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", text))


def score_text(text: str) -> float:
    if not text.strip():
        return float("-inf")
    chars = len(text)
    arabic_chars = sum(1 for char in text if "\u0600" <= char <= "\u06FF")
    latin_chars = sum(1 for char in text if "A" <= char <= "Z" or "a" <= char <= "z")
    digits = sum(1 for char in text if char.isdigit())
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tokens = token_count(text)
    single_char_tokens = sum(1 for token in text.split() if len(token) == 1)
    single_ratio = single_char_tokens / max(tokens, 1)
    mojibake = text.count("Ø") + text.count("Ù") + text.count("�") + text.count("? ?")
    anchors = sum(text.count(anchor) for anchor in DISCLOSURE_ANCHORS)
    long_repeats = len(re.findall(r"(.)\1{5,}", text))
    special_noise = sum(1 for char in text if char in "[]{}<>~¢©&=_\\|")
    latin_noise_words = len(re.findall(r"\b[a-z]{2,}\b", text))
    script_ratio = (arabic_chars + latin_chars + digits) / max(chars, 1)

    score = 0.0
    score += min(chars, 4000) * 0.04
    score += arabic_chars * 0.55
    score += digits * 0.08
    score += min(len(lines), 60) * 4.0
    score += anchors * 100.0
    score += script_ratio * 90.0
    score -= single_ratio * 450.0
    score -= mojibake * 140.0
    score -= long_repeats * 75.0
    score -= special_noise * 25.0
    score -= latin_noise_words * 8.0
    return score


def page_cache_path(pdf_hash: str, page_no: int) -> Path:
    return PAGE_CACHE_DIR / pdf_hash[:16] / f"page_{page_no:04d}.json"


def process_page(pdf_path: Path, pdf_hash: str, page_index: int, dpi: int, *, force: bool) -> dict[str, Any]:
    cache_path = page_cache_path(pdf_hash, page_index + 1)
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    image = render_page(pdf_path, page_index, dpi=dpi)
    candidates = [extract_native_candidate(pdf_path, page_index)]
    for variant, psm in OCR_VARIANTS:
        processed = preprocess(image, variant)
        candidates.append(run_tesseract(processed, variant=variant, psm=psm))

    viable = [candidate for candidate in candidates if candidate.get("text", "").strip()]
    chosen = max(viable, key=lambda item: float(item.get("score", float("-inf")))) if viable else candidates[0]
    result = {
        "source_pdf": str(pdf_path),
        "page": page_index + 1,
        "dpi": dpi,
        "candidates": candidates,
        "chosen": {
            "engine": chosen.get("engine"),
            "variant": chosen.get("variant"),
            "psm": chosen.get("psm"),
            "success": chosen.get("success"),
            "error": chosen.get("error"),
            "score": chosen.get("score"),
            "text": chosen.get("text", ""),
        },
        "quality": quality_summary(chosen.get("text", "")),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def quality_summary(text: str) -> dict[str, Any]:
    text = text or ""
    return {
        "chars": len(text),
        "arabic_chars": sum(1 for char in text if "\u0600" <= char <= "\u06FF"),
        "digit_chars": sum(1 for char in text if char.isdigit()),
        "tokens": token_count(text),
        "mojibake_markers": text.count("Ø") + text.count("Ù") + text.count("�"),
        "lines": sum(1 for line in text.splitlines() if line.strip()),
    }


def render_document_text(record: PdfRecord, pages: list[dict[str, Any]]) -> str:
    header = [
        f"Source PDF: {record.path.name}",
        f"SHA256: {record.sha256}",
        f"Pages: {record.page_count}",
        "Extraction: local deterministic OCR only. No OpenAI, Gemini, or external LLM/API calls.",
        "Engines compared per page: native/glyph PDF extraction + Tesseract ara+eng variants.",
        "",
    ]
    body: list[str] = []
    for page_result in sorted(pages, key=lambda item: int(item["page"])):
        chosen = page_result["chosen"]
        quality = page_result["quality"]
        body.append(
            "===== Page "
            f"{page_result['page']} "
            f"(source={chosen.get('engine')}:{chosen.get('variant')}/psm={chosen.get('psm')}; "
            f"chars={quality['chars']}; arabic={quality['arabic_chars']}) ====="
        )
        body.append(chosen.get("text", "").strip() or "[NO TEXT EXTRACTED BY LOCAL OCR]")
        body.append("")
    return "\n".join(header + body).strip() + "\n"


def process_record(record: PdfRecord, *, workers: int, dpi: int, force: bool) -> dict[str, Any]:
    output_path = OUTPUT_DIR / f"{stable_stem(record.path)}.clean.txt"
    page_results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_page, record.path, record.sha256, page_index, dpi, force=force)
            for page_index in range(record.page_count)
        ]
        for future in as_completed(futures):
            page_results.append(future.result())

    page_results.sort(key=lambda item: int(item["page"]))
    for page_result in page_results:
        page_result["chosen"]["text"] = normalize_ocr_text(page_result["chosen"].get("text", ""))
        page_result["quality"] = quality_summary(page_result["chosen"]["text"])
    output_path.write_text(render_document_text(record, page_results), encoding="utf-8")

    quality_totals = Counter()
    chosen_sources = Counter()
    weak_pages: list[int] = []
    for page_result in page_results:
        chosen = page_result["chosen"]
        quality = page_result["quality"]
        chosen_sources[str(chosen.get("engine"))] += 1
        quality_totals["chars"] += quality["chars"]
        quality_totals["arabic_chars"] += quality["arabic_chars"]
        quality_totals["mojibake_markers"] += quality["mojibake_markers"]
        if quality["chars"] < 40 or quality["mojibake_markers"] > 0:
            weak_pages.append(int(page_result["page"]))

    return {
        "pdf": str(record.path),
        "output": str(output_path),
        "sha256": record.sha256,
        "page_count": record.page_count,
        "duplicate_of": record.duplicate_of,
        "chosen_sources": dict(chosen_sources),
        "total_chars": int(quality_totals["chars"]),
        "total_arabic_chars": int(quality_totals["arabic_chars"]),
        "total_mojibake_markers": int(quality_totals["mojibake_markers"]),
        "weak_pages": weak_pages,
    }


def main() -> None:
    args = parse_args()
    if not TESSERACT_EXE.exists():
        raise RuntimeError(f"Tesseract not found: {TESSERACT_EXE}")
    if not (TESSDATA_DIR / "ara.traineddata").exists():
        raise RuntimeError(f"Arabic traineddata not found: {TESSDATA_DIR / 'ara.traineddata'}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records(limit=args.limit)
    unique_by_hash: dict[str, PdfRecord] = {}
    for record in records:
        unique_by_hash.setdefault(record.sha256, record)

    results_by_hash: dict[str, dict[str, Any]] = {}
    records_summary: list[dict[str, Any]] = []
    unique_records = list(unique_by_hash.values())
    total_pages = sum(record.page_count for record in unique_records)
    print(
        f"Local OCR start: pdfs={len(records)} unique_pdfs={len(unique_records)} "
        f"unique_pages={total_pages} workers={args.workers} dpi={args.dpi}"
    )
    print("Cloud/API engines are not called by this script.")

    for index, record in enumerate(unique_records, start=1):
        result = process_record(record, workers=args.workers, dpi=args.dpi, force=args.force)
        results_by_hash[record.sha256] = result
        records_summary.append(result)
        print(
            f"[{index}/{len(unique_records)}] {record.path.name} "
            f"pages={record.page_count} chars={result['total_chars']} "
            f"weak={len(result['weak_pages'])}"
        )

    for record in records:
        if record.sha256 in results_by_hash and record.path != Path(results_by_hash[record.sha256]["pdf"]):
            original_result = results_by_hash[record.sha256]
            original_text = Path(original_result["output"]).read_text(encoding="utf-8")
            duplicate_text = re.sub(
                r"^Source PDF: .*$",
                f"Source PDF: {record.path.name}",
                original_text,
                count=1,
                flags=re.MULTILINE,
            )
            duplicate_text = duplicate_text.replace(
                "Extraction: local deterministic OCR only. No OpenAI, Gemini, or external LLM/API calls.",
                "Extraction: local deterministic OCR only. No OpenAI, Gemini, or external LLM/API calls.\n"
                f"Duplicate PDF bytes of: {Path(original_result['pdf']).name}",
                1,
            )
            duplicate_output = OUTPUT_DIR / f"{stable_stem(record.path)}.clean.txt"
            duplicate_output.write_text(duplicate_text, encoding="utf-8")
            duplicate_result = dict(original_result)
            duplicate_result.update(
                {
                    "pdf": str(record.path),
                    "output": str(duplicate_output),
                    "duplicate_of": Path(original_result["pdf"]).name,
                }
            )
            records_summary.append(duplicate_result)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(SOURCE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "artifact_dir": str(ARTIFACT_DIR),
        "mode": "local_only_no_cloud_no_external_llm",
        "tesseract": str(TESSERACT_EXE),
        "tessdata_dir": str(TESSDATA_DIR),
        "ocr_variants": [{"variant": variant, "psm": psm} for variant, psm in OCR_VARIANTS],
        "pdf_count": len(records),
        "unique_pdf_count": len(unique_records),
        "unique_page_count": total_pages,
        "output_count": len(list(OUTPUT_DIR.glob("*.clean.txt"))),
        "total_chars": sum(item["total_chars"] for item in records_summary),
        "total_arabic_chars": sum(item["total_arabic_chars"] for item in records_summary),
        "total_mojibake_markers": sum(item["total_mojibake_markers"] for item in records_summary),
        "weak_documents": [
            item for item in records_summary if item["weak_pages"] or item["total_chars"] < max(80, item["page_count"] * 40)
        ],
        "documents": sorted(records_summary, key=lambda item: Path(item["pdf"]).name.casefold()),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "documents"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
