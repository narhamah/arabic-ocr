from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz

from arabic_ocr.arabic_text import (
    assess_arabic_extraction_quality,
    clean_arabic_text,
    contains_arabic,
)
from arabic_ocr.engines.registry import available_engine_ids
from arabic_ocr.native_pdf import extract_best_native_page_text


SOURCE_DIR = Path(r"C:\Users\narha\Dropbox\Future Kid Valuation\Disclosures")
OUTPUT_DIR = SOURCE_DIR / "clean_txt"
ARTIFACT_DIR = SOURCE_DIR / "ocr_audit_artifacts"
INVENTORY_PATH = ARTIFACT_DIR / "inventory.json"
SUMMARY_PATH = ARTIFACT_DIR / "summary.json"


@dataclass(slots=True)
class PageExtraction:
    page_no: int
    text: str
    source: str
    quality_status: str
    quality_issues: list[str]
    quality_metrics: dict[str, Any]
    embedded_chars: int


def stable_stem(pdf_path: Path) -> str:
    raw = pdf_path.stem.strip()
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", raw)
    safe = re.sub(r"\s+", " ", safe).strip(" .")
    return safe or hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()[:12]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_for_upload(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\u00a0", " ")
    text = clean_arabic_text(text) if contains_arabic(text) else text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def extract_pdf_native(pdf_path: Path) -> tuple[list[PageExtraction], dict[str, Any]]:
    doc = fitz.open(pdf_path)
    cmap_cache: dict[int, dict[int, str]] = {}
    pages: list[PageExtraction] = []
    embedded_pages = 0
    embedded_chars = 0
    image_xrefs: set[int] = set()
    page_sizes: list[tuple[float, float]] = []

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        page_sizes.append((round(page.rect.width, 2), round(page.rect.height, 2)))
        raw_text = page.get_text("text") or ""
        if raw_text.strip():
            embedded_pages += 1
        embedded_chars += len(raw_text)
        for image in page.get_images(full=True):
            image_xrefs.add(int(image[0]))

        page_text, metadata = extract_best_native_page_text(
            pdf_path,
            doc,
            page,
            cmap_cache=cmap_cache,
        )
        page_text = normalize_for_upload(page_text)
        quality = assess_arabic_extraction_quality(page_text)
        pages.append(
            PageExtraction(
                page_no=page_index + 1,
                text=page_text,
                source=str(metadata.get("source") or "unknown"),
                quality_status=quality.status,
                quality_issues=list(quality.issues),
                quality_metrics=dict(quality.metrics),
                embedded_chars=len(raw_text),
            )
        )

    metadata = {
        "page_count": doc.page_count,
        "embedded_pages": embedded_pages,
        "embedded_chars": embedded_chars,
        "image_count": len(image_xrefs),
        "page_sizes": sorted(set(page_sizes)),
    }
    doc.close()
    return pages, metadata


def needs_ocr(pages: list[PageExtraction], metadata: dict[str, Any]) -> bool:
    if metadata["page_count"] == 0:
        return False
    if metadata["embedded_pages"] == 0:
        return True
    empty_pages = sum(1 for page in pages if not page.text.strip())
    fail_pages = sum(1 for page in pages if page.quality_status == "fail")
    if empty_pages:
        return True
    return fail_pages > 0


def render_final_text(pdf_path: Path, pages: list[PageExtraction], metadata: dict[str, Any], file_hash: str) -> str:
    header = [
        f"Source PDF: {pdf_path.name}",
        f"SHA256: {file_hash}",
        f"Pages: {metadata['page_count']}",
        "Extraction: deterministic native/glyph PDF text extraction; OCR required only when audit flags a bad or empty text layer.",
        "",
    ]
    body: list[str] = []
    for page in pages:
        source_note = f"source={page.source}; quality={page.quality_status}"
        if page.quality_issues:
            source_note += "; issues=" + ",".join(page.quality_issues)
        body.append(f"===== Page {page.page_no} ({source_note}) =====")
        body.append(page.text.strip() if page.text.strip() else "[NO TEXT EXTRACTED]")
        body.append("")
    return "\n".join(header + body).strip() + "\n"


def write_outputs() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(SOURCE_DIR.glob("*.pdf"), key=lambda p: p.name.casefold())
    inventory: list[dict[str, Any]] = []
    hash_to_output: dict[str, Path] = {}

    for index, pdf_path in enumerate(pdf_paths, start=1):
        file_hash = sha256_file(pdf_path)
        pages, metadata = extract_pdf_native(pdf_path)
        quality_counts = Counter(page.quality_status for page in pages)
        source_counts = Counter(page.source for page in pages)
        arabic_chars = sum(int(page.quality_metrics.get("arabic_char_count", 0)) for page in pages)
        text_chars = sum(len(page.text) for page in pages)
        ocr_flag = needs_ocr(pages, metadata)

        output_path = OUTPUT_DIR / f"{stable_stem(pdf_path)}.clean.txt"
        if file_hash in hash_to_output:
            original = hash_to_output[file_hash]
            text = (
                f"Source PDF: {pdf_path.name}\n"
                f"SHA256: {file_hash}\n"
                f"Duplicate of extracted file: {original.name}\n"
                "The PDF bytes are identical, so the extracted text is identical.\n"
            )
            output_path.write_text(text, encoding="utf-8")
            duplicate_of = original.name
        else:
            output_path.write_text(render_final_text(pdf_path, pages, metadata, file_hash), encoding="utf-8")
            hash_to_output[file_hash] = output_path
            duplicate_of = None

        record = {
            "index": index,
            "pdf": str(pdf_path),
            "output": str(output_path),
            "sha256": file_hash,
            "duplicate_of_output": duplicate_of,
            "page_count": metadata["page_count"],
            "embedded_pages": metadata["embedded_pages"],
            "embedded_chars": metadata["embedded_chars"],
            "image_count": metadata["image_count"],
            "text_chars": text_chars,
            "arabic_chars": arabic_chars,
            "quality_counts": dict(quality_counts),
            "source_counts": dict(source_counts),
            "needs_ocr": ocr_flag,
            "empty_pages": [page.page_no for page in pages if not page.text.strip()],
            "fail_pages": [page.page_no for page in pages if page.quality_status == "fail"],
            "warn_pages": [page.page_no for page in pages if page.quality_status == "warn"],
            "status": "needs_ocr" if ocr_flag else "native_ok",
        }
        inventory.append(record)
        print(
            f"[{index}/{len(pdf_paths)}] {pdf_path.name} -> {record['status']} "
            f"pages={metadata['page_count']} chars={text_chars}"
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(SOURCE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "artifact_dir": str(ARTIFACT_DIR),
        "pdf_count": len(pdf_paths),
        "available_cloud_engines": available_engine_ids(),
        "total_pages": sum(item["page_count"] for item in inventory),
        "total_text_chars": sum(item["text_chars"] for item in inventory),
        "status_counts": dict(Counter(item["status"] for item in inventory)),
        "quality_counts": dict(Counter(status for item in inventory for status, n in item["quality_counts"].items() for _ in range(n))),
        "needs_ocr": [item for item in inventory if item["needs_ocr"]],
        "duplicates": [item for item in inventory if item["duplicate_of_output"]],
    }
    INVENTORY_PATH.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = write_outputs()
    print(json.dumps(result, ensure_ascii=False, indent=2))
