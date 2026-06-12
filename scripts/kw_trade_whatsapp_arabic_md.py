from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
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

from arabic_ocr.arabic_text import (
    assess_arabic_extraction_quality,
    clean_arabic_text,
    contains_arabic,
    detect_weird_glyph_char_ratio,
)
from arabic_ocr.native_pdf import extract_best_native_page_text


SOURCE_PARENT = Path(r"C:\Users\narha\kw-trade-agent")
SOURCE_PREFIX = "WhatsApp Chat - "
OUTPUT_DIR_NAME = "clean_md"
ARTIFACT_DIR_NAME = "ocr_audit_artifacts"
RUN_DIR_NAME = "local_native_tesseract_ar"
TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA_DIR = Path(r"C:\Users\narha\arabic-ocr\.tessdata")

OCR_VARIANTS: tuple[tuple[str, int], ...] = (
    ("contrast", 4),
    ("contrast", 6),
    ("sharp", 4),
    ("bw200", 6),
)


@dataclass(slots=True)
class PdfRecord:
    path: Path
    sha256: str
    page_count: int
    embedded_pages: int
    embedded_chars: int
    duplicate_of: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local-only clean Markdown extraction for the Aleqtisadyah WhatsApp PDF export."
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional PDF limit for smoke tests.")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI for Tesseract fallback pages.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--force", action="store_true", help="Recompute page caches and Markdown outputs.")
    parser.add_argument(
        "--ocr-statuses",
        default="fail,warn",
        help="Comma-separated native quality statuses that should trigger Tesseract fallback.",
    )
    return parser.parse_args()


def find_source_dir() -> Path:
    matches = [
        path
        for path in SOURCE_PARENT.iterdir()
        if path.is_dir() and path.name.startswith(SOURCE_PREFIX)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {SOURCE_PREFIX!r} folder under {SOURCE_PARENT}, found {len(matches)}"
        )
    return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_stem(path: Path) -> str:
    match = re.match(r"^(\d{8})", path.name)
    if match:
        return match.group(1)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._-")
    return safe or hashlib.sha1(path.name.encode("utf-8", errors="ignore")).hexdigest()[:12]


def inspect_pdf(path: Path) -> tuple[str, int, int, int]:
    file_hash = sha256_file(path)
    doc = fitz.open(path)
    embedded_pages = 0
    embedded_chars = 0
    for page in doc:
        text = page.get_text("text") or ""
        chars = len(text.strip())
        if chars:
            embedded_pages += 1
            embedded_chars += chars
    page_count = doc.page_count
    doc.close()
    return file_hash, page_count, embedded_pages, embedded_chars


def load_records(source_dir: Path, limit: int = 0) -> list[PdfRecord]:
    records: list[PdfRecord] = []
    seen: dict[str, str] = {}
    pdfs = sorted(source_dir.glob("*.pdf"), key=lambda path: path.name.casefold())
    if limit:
        pdfs = pdfs[:limit]
    for pdf_path in pdfs:
        file_hash, page_count, embedded_pages, embedded_chars = inspect_pdf(pdf_path)
        duplicate_of = seen.get(file_hash)
        if duplicate_of is None:
            seen[file_hash] = pdf_path.name
        records.append(
            PdfRecord(
                path=pdf_path,
                sha256=file_hash,
                page_count=page_count,
                embedded_pages=embedded_pages,
                embedded_chars=embedded_chars,
                duplicate_of=duplicate_of,
            )
        )
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
    if variant == "bw200":
        sharp = contrast.filter(ImageFilter.SHARPEN)
        return sharp.point(lambda pixel: 255 if pixel > 200 else 0)
    return contrast


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text)
    text = text.replace("\ufeff", "").replace("\u00a0", " ")
    text = text.replace("\ufffd", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    if contains_arabic(text):
        text = clean_arabic_text(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def quality_for_text(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    assessment = assess_arabic_extraction_quality(normalized)
    chars = len(normalized)
    arabic_chars = sum(1 for char in normalized if "\u0600" <= char <= "\u06ff")
    latin_ext_noise = sum(1 for char in normalized if "\u0100" <= char <= "\u024f")
    replacement_chars = normalized.count("\ufffd")
    weird_ratio = detect_weird_glyph_char_ratio(normalized)
    return {
        "status": assessment.status,
        "issues": assessment.issues,
        "metrics": assessment.metrics,
        "chars": chars,
        "arabic_chars": arabic_chars,
        "digit_chars": sum(1 for char in normalized if char.isdigit()),
        "lines": sum(1 for line in normalized.splitlines() if line.strip()),
        "latin_extended_noise": latin_ext_noise,
        "replacement_chars": replacement_chars,
        "weird_glyph_ratio": weird_ratio,
    }


def score_text(text: str) -> float:
    normalized = normalize_text(text)
    quality = quality_for_text(normalized)
    status_bonus = {
        "pass": 2500.0,
        "warn": 800.0,
        "not_arabic": -200.0,
        "fail": -900.0,
    }.get(str(quality["status"]), 0.0)
    chars = quality["chars"]
    arabic_chars = quality["arabic_chars"]
    digits = quality["digit_chars"]
    lines = quality["lines"]
    noise = quality["latin_extended_noise"] + quality["replacement_chars"] * 20
    weird_ratio = float(quality["weird_glyph_ratio"])
    tokens = len(re.findall(r"[\u0600-\u06ffA-Za-z0-9]+", normalized))
    useful_ratio = (arabic_chars + digits) / max(chars, 1)
    return (
        status_bonus
        + min(chars, 6000) * 0.025
        + arabic_chars * 0.16
        + digits * 0.05
        + min(lines, 80) * 3.0
        + min(tokens, 1000) * 0.2
        + useful_ratio * 300.0
        - noise * 65.0
        - weird_ratio * 2500.0
    )


def run_tesseract(image: Image.Image, *, variant: str, psm: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="kw_trade_tess_") as temp_dir:
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
                "quality": quality_for_text(""),
            }

        text_path = Path(str(output_base) + ".txt")
        text = text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else ""
        text = normalize_text(text)
        return {
            "engine": "tesseract",
            "variant": variant,
            "psm": psm,
            "success": result.returncode == 0,
            "error": result.stderr.strip()[-500:] if result.stderr.strip() else None,
            "text": text,
            "score": score_text(text),
            "quality": quality_for_text(text),
        }


def page_cache_path(page_cache_dir: Path, pdf_hash: str, page_no: int) -> Path:
    return page_cache_dir / pdf_hash[:16] / f"page_{page_no:04d}.json"


def process_page(
    pdf_path: Path,
    pdf_hash: str,
    page_index: int,
    page_cache_dir: Path,
    dpi: int,
    ocr_statuses: set[str],
    *,
    force: bool,
) -> dict[str, Any]:
    cache_path = page_cache_path(page_cache_dir, pdf_hash, page_index + 1)
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    try:
        text, metadata = extract_best_native_page_text(pdf_path, doc, page, cmap_cache={})
    except Exception as exc:
        # Some custom-font newspaper PDFs can break glyph column detection.
        # Keep the run deterministic: fall back to PyMuPDF text, then route
        # through local Tesseract if quality checks mark it weak.
        text = page.get_text("text") or ""
        metadata = {
            "source": "pymupdf_after_glyph_error",
            "quality_status": "fail",
            "quality_issues": [f"native_glyph_exception:{type(exc).__name__}"],
            "quality_metrics": {},
            "candidate_sources": {"pymupdf": bool(text.strip()), "glyph": False},
            "error": repr(exc),
        }
    doc.close()
    text = normalize_text(text)
    native_candidate = {
        "engine": "native_pdf",
        "variant": metadata.get("source"),
        "psm": None,
        "success": bool(text.strip()),
        "error": None,
        "text": text,
        "score": score_text(text),
        "quality": quality_for_text(text),
        "metadata": metadata,
    }
    candidates = [native_candidate]

    native_status = str(native_candidate["quality"]["status"])
    native_quality = native_candidate["quality"]
    native_needs_ocr = (
        native_status in ocr_statuses
        or int(native_quality.get("chars") or 0) < 60
        or int(native_quality.get("replacement_chars") or 0) > 20
        or int(native_quality.get("latin_extended_noise") or 0) > 50
        or float(native_quality.get("weird_glyph_ratio") or 0.0) > 0.10
    )
    if native_needs_ocr:
        image = render_page(pdf_path, page_index, dpi=dpi)
        for variant, psm in OCR_VARIANTS:
            candidates.append(run_tesseract(preprocess(image, variant), variant=variant, psm=psm))

    viable = [candidate for candidate in candidates if candidate.get("text", "").strip()]
    chosen = max(viable, key=lambda item: float(item.get("score", float("-inf")))) if viable else native_candidate
    result = {
        "source_pdf": str(pdf_path),
        "page": page_index + 1,
        "dpi": dpi,
        "native_status": native_status,
        "candidates": candidates,
        "chosen": {
            "engine": chosen.get("engine"),
            "variant": chosen.get("variant"),
            "psm": chosen.get("psm"),
            "success": chosen.get("success"),
            "error": chosen.get("error"),
            "score": chosen.get("score"),
            "quality": chosen.get("quality"),
            "text": chosen.get("text", ""),
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def markdown_escape_inline(text: str) -> str:
    return text.replace("`", "'")


def first_title_lines(page_results: list[dict[str, Any]], max_lines: int = 3) -> list[str]:
    for page_result in page_results:
        text = page_result["chosen"].get("text", "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines[:max_lines]
    return []


def render_markdown(record: PdfRecord, page_results: list[dict[str, Any]]) -> str:
    title = output_stem(record.path)
    title_lines = first_title_lines(page_results)
    lines: list[str] = [
        f"# {title}",
        "",
        "## Document Metadata",
        "",
        f"- Source PDF: `{markdown_escape_inline(record.path.name)}`",
        f"- SHA256: `{record.sha256}`",
        f"- Pages: {record.page_count}",
        f"- Embedded text pages: {record.embedded_pages}",
        f"- Embedded text characters: {record.embedded_chars}",
        "- Extraction mode: local native/glyph text extraction with local Tesseract fallback on weak pages",
        "- External services: none; no OpenAI, Gemini, cloud OCR, or external LLM/API calls",
    ]
    if record.duplicate_of:
        lines.append(f"- Duplicate of: `{markdown_escape_inline(record.duplicate_of)}`")
    if title_lines:
        lines.append("- First readable lines:")
        for line in title_lines:
            lines.append(f"  - {line}")
    lines.extend(["", "---", ""])

    for page_result in page_results:
        chosen = page_result["chosen"]
        quality = chosen.get("quality") or quality_for_text(chosen.get("text", ""))
        psm = chosen.get("psm")
        source = f"{chosen.get('engine')}:{chosen.get('variant')}"
        if psm is not None:
            source += f"/psm={psm}"
        lines.extend(
            [
                f"## Page {page_result['page']}",
                "",
                (
                    f"_Source: {source}; native_quality={page_result.get('native_status')}; "
                    f"chosen_quality={quality.get('status')}; chars={quality.get('chars')}; "
                    f"arabic={quality.get('arabic_chars')}_"
                ),
                "",
                chosen.get("text", "").strip() or "[NO TEXT EXTRACTED BY LOCAL PIPELINE]",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def process_record(
    record: PdfRecord,
    output_dir: Path,
    page_cache_dir: Path,
    dpi: int,
    workers: int,
    ocr_statuses: set[str],
    *,
    force: bool,
) -> dict[str, Any]:
    md_path = output_dir / f"{output_stem(record.path)}.clean.md"

    page_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                process_page,
                record.path,
                record.sha256,
                page_index,
                page_cache_dir,
                dpi,
                ocr_statuses,
                force=force,
            )
            for page_index in range(record.page_count)
        ]
        for future in as_completed(futures):
            page_results.append(future.result())
    page_results.sort(key=lambda item: int(item["page"]))
    for page_result in page_results:
        chosen = page_result["chosen"]
        text = normalize_text(chosen.get("text", ""))
        chosen["text"] = text
        chosen["quality"] = quality_for_text(text)
    md_path.write_text(render_markdown(record, page_results), encoding="utf-8", newline="\n")

    chosen_sources = Counter()
    native_statuses = Counter()
    chosen_statuses = Counter()
    weak_pages: list[int] = []
    ocr_pages: list[int] = []
    total_chars = 0
    total_arabic_chars = 0
    total_noise = 0
    for page_result in page_results:
        chosen = page_result["chosen"]
        quality = chosen.get("quality") or quality_for_text(chosen.get("text", ""))
        source = f"{chosen.get('engine')}:{chosen.get('variant')}"
        if chosen.get("psm") is not None:
            source += f"/psm={chosen.get('psm')}"
        chosen_sources[source] += 1
        native_statuses[str(page_result.get("native_status"))] += 1
        chosen_statuses[str(quality.get("status"))] += 1
        total_chars += int(quality.get("chars") or 0)
        total_arabic_chars += int(quality.get("arabic_chars") or 0)
        total_noise += int(quality.get("latin_extended_noise") or 0) + int(quality.get("replacement_chars") or 0)
        if chosen.get("engine") == "tesseract":
            ocr_pages.append(int(page_result["page"]))
        if quality.get("status") in {"fail", "warn"} or int(quality.get("chars") or 0) < 60:
            weak_pages.append(int(page_result["page"]))

    return {
        "pdf": str(record.path),
        "output": str(md_path),
        "sha256": record.sha256,
        "page_count": record.page_count,
        "embedded_pages": record.embedded_pages,
        "embedded_chars": record.embedded_chars,
        "duplicate_of": record.duplicate_of,
        "chosen_sources": dict(chosen_sources),
        "native_statuses": dict(native_statuses),
        "chosen_statuses": dict(chosen_statuses),
        "total_chars": total_chars,
        "total_arabic_chars": total_arabic_chars,
        "total_noise_markers": total_noise,
        "weak_pages": weak_pages,
        "ocr_pages": ocr_pages,
    }


def write_index(output_dir: Path, source_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Aleqtisadyah WhatsApp PDF Markdown Index",
        "",
        f"Generated: {summary['created_at']}",
        "",
        "Mode: local-only native/glyph extraction with local Tesseract fallback. No cloud/API/LLM calls.",
        "",
        f"- Source folder: `{source_dir}`",
        f"- Markdown output folder: `{output_dir}`",
        f"- PDFs processed: {summary['pdf_count']}",
        f"- Unique PDFs: {summary['unique_pdf_count']}",
        f"- Pages covered: {summary['total_pages']}",
        f"- Total extracted characters: {summary['total_chars']}",
        f"- Documents with weak pages after fallback: {len(summary['weak_documents'])}",
        "",
        "## Files",
        "",
    ]
    for doc in summary["documents"]:
        output_name = Path(doc["output"]).name
        weak = doc.get("weak_pages") or []
        ocr = doc.get("ocr_pages") or []
        note_parts = [
            f"pages: {doc['page_count']}",
            f"chars: {doc.get('total_chars', 0)}",
        ]
        if ocr:
            note_parts.append(f"ocr fallback pages: {len(ocr)}")
        if weak:
            note_parts.append(f"weak pages: {', '.join(map(str, weak[:20]))}{'...' if len(weak) > 20 else ''}")
        lines.append(f"- [{output_name}]({output_name}) - {'; '.join(note_parts)}")
    (output_dir / "00_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_audit(output_dir: Path, source_dir: Path, artifact_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Local Extraction Audit - Aleqtisadyah WhatsApp PDFs",
        "",
        "## Method",
        "",
        "- Enumerated the WhatsApp export folder by parent directory to avoid hard-coded Arabic path failures.",
        "- Inventoried every PDF for page count and embedded text coverage using PyMuPDF.",
        "- Extracted each page with the repo native/glyph Arabic extractor.",
        "- Ran local Tesseract `ara+eng` only when native quality was `fail` or `warn`.",
        "- Compared candidates using deterministic Arabic quality/scoring heuristics.",
        "- Wrote one Markdown file per PDF, preserving page boundaries and source metadata.",
        "- Did not call OpenAI, Gemini, cloud OCR, external LLMs, or external APIs.",
        "",
        "## Summary",
        "",
        f"- Source folder: `{source_dir}`",
        f"- Artifact folder: `{artifact_dir}`",
        f"- PDFs processed: {summary['pdf_count']}",
        f"- Unique PDFs: {summary['unique_pdf_count']}",
        f"- Duplicate files by SHA256: {summary['duplicate_files']}",
        f"- Pages covered: {summary['total_pages']}",
        f"- Total extracted characters: {summary['total_chars']}",
        f"- Total extracted Arabic characters: {summary['total_arabic_chars']}",
        f"- Documents with weak pages after fallback: {len(summary['weak_documents'])}",
        "",
        "## Weak Documents",
        "",
    ]
    if summary["weak_documents"]:
        for doc in summary["weak_documents"]:
            weak = ", ".join(map(str, doc.get("weak_pages", [])[:40]))
            if len(doc.get("weak_pages", [])) > 40:
                weak += "..."
            lines.append(f"- {Path(doc['pdf']).name}: {weak}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Limitation",
            "",
            "This is a local deterministic extraction. It avoids invented text and cloud cost, but pages with broken custom-font mappings or newspaper-style dense layouts can still contain character-level extraction/OCR errors. Weak-page flags mark the places that need manual review if litigation-grade precision is required.",
        ]
    )
    (output_dir / "00_LOCAL_EXTRACTION_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if not TESSERACT_EXE.exists():
        raise RuntimeError(f"Tesseract not found: {TESSERACT_EXE}")
    if not (TESSDATA_DIR / "ara.traineddata").exists():
        raise RuntimeError(f"Arabic traineddata not found: {TESSDATA_DIR / 'ara.traineddata'}")

    source_dir = find_source_dir()
    output_dir = source_dir / OUTPUT_DIR_NAME
    artifact_dir = source_dir / ARTIFACT_DIR_NAME / RUN_DIR_NAME
    page_cache_dir = artifact_dir / "pages"
    summary_path = artifact_dir / "local_extraction_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    page_cache_dir.mkdir(parents=True, exist_ok=True)

    ocr_statuses = {status.strip() for status in args.ocr_statuses.split(",") if status.strip()}
    records = load_records(source_dir, limit=args.limit)
    sha_counts = Counter(record.sha256 for record in records)
    total_pages = sum(record.page_count for record in records)
    print(
        f"Local extraction start: pdfs={len(records)} pages={total_pages} "
        f"dpi={args.dpi} ocr_statuses={sorted(ocr_statuses)}",
        flush=True,
    )
    print("Cloud/API engines are not called by this script.", flush=True)

    documents: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        result = process_record(
            record,
            output_dir,
            page_cache_dir,
            args.dpi,
            args.workers,
            ocr_statuses,
            force=args.force,
        )
        documents.append(result)
        print(
            f"[{index}/{len(records)}] {output_stem(record.path)} "
            f"pages={record.page_count} chars={result.get('total_chars', 0)} "
            f"ocr_pages={len(result.get('ocr_pages', []))} weak={len(result.get('weak_pages', []))}",
            flush=True,
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "artifact_dir": str(artifact_dir),
        "mode": "local_only_native_glyph_plus_tesseract_fallback_no_cloud_no_external_llm",
        "tesseract": str(TESSERACT_EXE),
        "tessdata_dir": str(TESSDATA_DIR),
        "language": "ara+eng",
        "ocr_statuses": sorted(ocr_statuses),
        "ocr_variants": [{"variant": variant, "psm": psm} for variant, psm in OCR_VARIANTS],
        "pdf_count": len(records),
        "unique_pdf_count": len(sha_counts),
        "duplicate_files": sum(count - 1 for count in sha_counts.values() if count > 1),
        "total_pages": total_pages,
        "output_count": len(documents),
        "total_chars": sum(int(doc.get("total_chars", 0)) for doc in documents),
        "total_arabic_chars": sum(int(doc.get("total_arabic_chars", 0)) for doc in documents),
        "total_noise_markers": sum(int(doc.get("total_noise_markers", 0)) for doc in documents),
        "weak_documents": [doc for doc in documents if doc.get("weak_pages")],
        "documents": documents,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_index(output_dir, source_dir, summary)
    write_audit(output_dir, source_dir, artifact_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
