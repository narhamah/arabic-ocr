from __future__ import annotations

import argparse
import hashlib
import json
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


SOURCE_DIR = Path(r"C:\Users\narha\Dropbox\Future Kid Valuation\Financials")
OUTPUT_DIR = SOURCE_DIR / "clean_txt"
ARTIFACT_DIR = SOURCE_DIR / "ocr_audit_artifacts" / "local_tesseract_eng"
PAGE_CACHE_DIR = ARTIFACT_DIR / "pages"
SUMMARY_PATH = ARTIFACT_DIR / "local_ocr_summary.json"
TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA_DIR = Path(r"C:\Users\narha\arabic-ocr\.tessdata")

# Financial statements are mostly dense English tables. These modes trade off
# paragraph flow, table-column preservation, and sparse headers/footers.
OCR_VARIANTS: tuple[tuple[str, int], ...] = (
    ("contrast", 6),
    ("contrast", 4),
    ("sharp", 6),
    ("bw200", 6),
)

FINANCIAL_ANCHORS = (
    "Future Kid",
    "Entertainment",
    "Real Estate",
    "K.S.C",
    "consolidated",
    "statement",
    "financial",
    "income",
    "profit",
    "loss",
    "assets",
    "liabilities",
    "equity",
    "cash",
    "flows",
    "Kuwaiti",
    "Dinars",
    "KD",
    "unaudited",
    "auditor",
    "subsidiaries",
    "quarter",
    "period",
    "year",
)


@dataclass(slots=True)
class PdfRecord:
    path: Path
    sha256: str
    page_count: int
    duplicate_of: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local-only OCR for Future Kid English financial PDFs.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
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
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
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
    if variant == "bw200":
        sharp = contrast.filter(ImageFilter.SHARPEN)
        return sharp.point(lambda pixel: 255 if pixel > 200 else 0)
    return gray


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\u00a0", " ")
    text = text.replace("Ø", "").replace("Ù", "").replace("�", "")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def native_candidate(pdf_path: Path, page_index: int) -> dict[str, Any]:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    text = normalize_text(page.get_text("text") or "")
    doc.close()
    return {
        "engine": "native_pdf",
        "variant": "pymupdf",
        "psm": None,
        "success": bool(text),
        "error": None,
        "text": text,
        "score": score_text(text),
    }


def run_tesseract(image: Image.Image, *, variant: str, psm: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="futurekid_fin_tess_") as temp_dir:
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
            "eng",
            "--oem",
            "1",
            "--psm",
            str(psm),
            "-c",
            "preserve_interword_spaces=1",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=150)
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
        text = normalize_text(text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else "")
        return {
            "engine": "tesseract",
            "variant": variant,
            "psm": psm,
            "success": result.returncode == 0,
            "error": result.stderr.strip()[-500:] if result.stderr.strip() else None,
            "text": text,
            "score": score_text(text),
        }


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9,()./%:-]*", text))


def score_text(text: str) -> float:
    if not text.strip():
        return float("-inf")
    chars = len(text)
    letters = sum(1 for char in text if char.isalpha())
    digits = sum(1 for char in text if char.isdigit())
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tokens = token_count(text)
    anchors = sum(text.lower().count(anchor.lower()) for anchor in FINANCIAL_ANCHORS)
    currency = len(re.findall(r"\b(?:KD|KWD|Dinars?|fils)\b", text, flags=re.I))
    numeric_groups = len(re.findall(r"\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\b\d{4}\b", text))
    special_noise = sum(1 for char in text if char in "{}[]<>|~^")
    mojibake = text.count("Ø") + text.count("Ù") + text.count("�")
    single_char_tokens = sum(1 for token in text.split() if len(token) == 1)
    single_ratio = single_char_tokens / max(tokens, 1)
    script_ratio = (letters + digits) / max(chars, 1)

    score = 0.0
    score += min(chars, 6000) * 0.05
    score += min(letters, 3500) * 0.08
    score += min(digits, 2000) * 0.18
    score += min(len(lines), 100) * 2.5
    score += anchors * 65.0
    score += currency * 20.0
    score += numeric_groups * 8.0
    score += script_ratio * 100.0
    score -= special_noise * 30.0
    score -= mojibake * 150.0
    score -= single_ratio * 300.0
    return score


def page_cache_path(pdf_hash: str, page_no: int) -> Path:
    return PAGE_CACHE_DIR / pdf_hash[:16] / f"page_{page_no:04d}.json"


def quality_summary(text: str) -> dict[str, int]:
    text = text or ""
    return {
        "chars": len(text),
        "letters": sum(1 for char in text if char.isalpha()),
        "digits": sum(1 for char in text if char.isdigit()),
        "tokens": token_count(text),
        "lines": sum(1 for line in text.splitlines() if line.strip()),
        "mojibake_markers": text.count("Ø") + text.count("Ù") + text.count("�"),
    }


def process_page(pdf_path: Path, pdf_hash: str, page_index: int, dpi: int, *, force: bool) -> dict[str, Any]:
    cache_path = page_cache_path(pdf_hash, page_index + 1)
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    image = render_page(pdf_path, page_index, dpi)
    candidates = [native_candidate(pdf_path, page_index)]
    for variant, psm in OCR_VARIANTS:
        candidates.append(run_tesseract(preprocess(image, variant), variant=variant, psm=psm))

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
            "text": normalize_text(chosen.get("text", "")),
        },
    }
    result["quality"] = quality_summary(result["chosen"]["text"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def render_document(record: PdfRecord, pages: list[dict[str, Any]]) -> str:
    header = [
        f"Source PDF: {record.path.name}",
        f"SHA256: {record.sha256}",
        f"Pages: {record.page_count}",
        "Extraction: local deterministic OCR/text extraction only. No OpenAI, Gemini, cloud OCR, or external LLM/API calls.",
        "Engines compared per page: native PDF text + local Tesseract eng OCR variants.",
        "",
    ]
    body: list[str] = []
    for page in sorted(pages, key=lambda item: int(item["page"])):
        chosen = page["chosen"]
        chosen["text"] = normalize_text(chosen.get("text", ""))
        quality = quality_summary(chosen["text"])
        body.append(
            "===== Page "
            f"{page['page']} "
            f"(source={chosen.get('engine')}:{chosen.get('variant')}/psm={chosen.get('psm')}; "
            f"chars={quality['chars']}; digits={quality['digits']}) ====="
        )
        body.append(chosen["text"] or "[NO TEXT EXTRACTED BY LOCAL OCR]")
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
    output_path.write_text(render_document(record, page_results), encoding="utf-8")

    totals = Counter()
    sources = Counter()
    weak_pages: list[int] = []
    for page in page_results:
        text = normalize_text(page["chosen"].get("text", ""))
        quality = quality_summary(text)
        source_key = f"{page['chosen'].get('engine')}:{page['chosen'].get('variant')}/psm={page['chosen'].get('psm')}"
        sources[source_key] += 1
        totals["chars"] += quality["chars"]
        totals["letters"] += quality["letters"]
        totals["digits"] += quality["digits"]
        totals["mojibake_markers"] += quality["mojibake_markers"]
        if quality["chars"] < 80 or quality["mojibake_markers"] > 0:
            weak_pages.append(int(page["page"]))

    return {
        "pdf": str(record.path),
        "output": str(output_path),
        "sha256": record.sha256,
        "page_count": record.page_count,
        "duplicate_of": record.duplicate_of,
        "chosen_sources": dict(sources),
        "total_chars": int(totals["chars"]),
        "total_letters": int(totals["letters"]),
        "total_digits": int(totals["digits"]),
        "total_mojibake_markers": int(totals["mojibake_markers"]),
        "weak_pages": weak_pages,
    }


def main() -> None:
    args = parse_args()
    if not TESSERACT_EXE.exists():
        raise RuntimeError(f"Tesseract not found: {TESSERACT_EXE}")
    if not (TESSDATA_DIR / "eng.traineddata").exists():
        raise RuntimeError(f"English traineddata not found: {TESSDATA_DIR / 'eng.traineddata'}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records(limit=args.limit)
    unique_by_hash: dict[str, PdfRecord] = {}
    for record in records:
        unique_by_hash.setdefault(record.sha256, record)
    unique_records = list(unique_by_hash.values())
    total_pages = sum(record.page_count for record in unique_records)
    print(
        f"Financials local OCR start: pdfs={len(records)} unique_pdfs={len(unique_records)} "
        f"unique_pages={total_pages} workers={args.workers} dpi={args.dpi}"
    )
    print("Cloud/API engines are not called by this script.")

    results_by_hash: dict[str, dict[str, Any]] = {}
    records_summary: list[dict[str, Any]] = []
    for index, record in enumerate(unique_records, start=1):
        result = process_record(record, workers=args.workers, dpi=args.dpi, force=args.force)
        results_by_hash[record.sha256] = result
        records_summary.append(result)
        print(
            f"[{index}/{len(unique_records)}] {record.path.name} pages={record.page_count} "
            f"chars={result['total_chars']} weak={len(result['weak_pages'])}"
        )

    for record in records:
        original = results_by_hash.get(record.sha256)
        if original and record.path != Path(original["pdf"]):
            original_text = Path(original["output"]).read_text(encoding="utf-8")
            duplicate_text = re.sub(
                r"^Source PDF: .*$",
                f"Source PDF: {record.path.name}",
                original_text,
                count=1,
                flags=re.MULTILINE,
            )
            duplicate_text = duplicate_text.replace(
                "Extraction: local deterministic OCR/text extraction only. No OpenAI, Gemini, cloud OCR, or external LLM/API calls.",
                "Extraction: local deterministic OCR/text extraction only. No OpenAI, Gemini, cloud OCR, or external LLM/API calls.\n"
                f"Duplicate PDF bytes of: {Path(original['pdf']).name}",
                1,
            )
            duplicate_output = OUTPUT_DIR / f"{stable_stem(record.path)}.clean.txt"
            duplicate_output.write_text(duplicate_text, encoding="utf-8")
            duplicate_result = dict(original)
            duplicate_result.update(
                {"pdf": str(record.path), "output": str(duplicate_output), "duplicate_of": Path(original["pdf"]).name}
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
        "language": "eng",
        "ocr_variants": [{"variant": variant, "psm": psm} for variant, psm in OCR_VARIANTS],
        "pdf_count": len(records),
        "unique_pdf_count": len(unique_records),
        "unique_page_count": total_pages,
        "output_count": len(list(OUTPUT_DIR.glob("*.clean.txt"))),
        "total_chars": sum(item["total_chars"] for item in records_summary),
        "total_letters": sum(item["total_letters"] for item in records_summary),
        "total_digits": sum(item["total_digits"] for item in records_summary),
        "total_mojibake_markers": sum(item["total_mojibake_markers"] for item in records_summary),
        "weak_documents": [
            item for item in records_summary if item["weak_pages"] or item["total_chars"] < max(500, item["page_count"] * 120)
        ],
        "documents": sorted(records_summary, key=lambda item: Path(item["pdf"]).name.casefold()),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "documents"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
