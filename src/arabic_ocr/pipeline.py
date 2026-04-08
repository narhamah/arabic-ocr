"""Pipeline orchestrator for native Arabic extraction plus OCR verification."""

from __future__ import annotations

import logging

from PIL import Image

from arabic_ocr.arabic_pdf import extract_native_pdf
from arabic_ocr.layout import detect_regions
from arabic_ocr.normalizer import normalize
from arabic_ocr.ocr_engines import run_dual_ocr
from arabic_ocr.preprocessor import preprocess
from arabic_ocr.renderer import render_pdf
from arabic_ocr.verifier import verify_and_merge, verify_native_vs_ocr

logger = logging.getLogger(__name__)


def process_pdf(
    pdf_path: str,
    config: dict | None = None,
) -> list[dict]:
    """Process a PDF through native extraction and OCR verification."""
    config = config or {}
    dpi = config.get("dpi", 400)
    pages = config.get("pages", None)
    strip_diacritics = config.get("strip_diacritics", True)
    verify_with_ocr = config.get("verify_with_ocr", True)

    native_results = extract_native_pdf(pdf_path, pages=pages)
    native_by_page = {page.page: page for page in native_results}
    page_images = render_pdf(pdf_path, dpi=dpi, pages=pages)

    results = []
    for page_idx, original_image in enumerate(page_images):
        actual_page = pages[page_idx] if pages else page_idx
        native_page = native_by_page.get(actual_page)
        if native_page and native_page.has_text_layer:
            page_result = _process_native_page(
                original_image=original_image,
                page_num=actual_page,
                native_text=native_page.text,
                strip_diacritics=strip_diacritics,
                verify_with_ocr=verify_with_ocr,
            )
        else:
            page_result = _process_ocr_page(
                original_image=original_image,
                page_num=actual_page,
                strip_diacritics=strip_diacritics,
            )
        results.append(page_result)

    return results


def _process_native_page(
    *,
    original_image: Image.Image,
    page_num: int,
    native_text: str,
    strip_diacritics: bool,
    verify_with_ocr: bool,
) -> dict:
    normalized_native = native_text.strip()
    if not verify_with_ocr:
        return {
            "page": page_num,
            "text": normalized_native,
            "confidence": 0.95,
            "regions": 1,
            "method": "native_pdf",
        }

    ocr_page = _process_ocr_page(
        original_image=original_image,
        page_num=page_num,
        strip_diacritics=strip_diacritics,
    )
    adjudicated = verify_native_vs_ocr(
        native_text=normalized_native,
        ocr_text=ocr_page["text"],
        image=original_image,
    )
    return {
        "page": page_num,
        "text": adjudicated["text"],
        "confidence": adjudicated["confidence"],
        "regions": 1,
        "method": adjudicated["source"],
        "ocr_confidence": ocr_page["confidence"],
    }


def _process_ocr_page(
    *,
    original_image: Image.Image,
    page_num: int,
    strip_diacritics: bool,
) -> dict:
    preprocess(original_image)
    regions = detect_regions(original_image)

    region_texts = []
    total_confidence = 0.0
    num_successful = 0

    for region in regions:
        try:
            region_text, confidence = _process_region(
                original_crop=region["crop"],
                strip_diacritics=strip_diacritics,
            )
            region_texts.append(region_text)
            total_confidence += confidence
            num_successful += 1
        except Exception as exc:
            logger.warning("Region processing failed: %s", exc)

    final_text = "\n\n".join(text for text in region_texts if text)
    avg_confidence = total_confidence / max(num_successful, 1)

    return {
        "page": page_num,
        "text": final_text,
        "confidence": avg_confidence,
        "regions": len(regions),
        "method": "ocr",
    }


def _process_region(
    *,
    original_crop: Image.Image,
    strip_diacritics: bool,
) -> tuple[str, float]:
    ocr_results = run_dual_ocr(original_crop)

    primary_text = ocr_results["primary"].get("text", "")
    secondary_text = ocr_results["secondary"].get("text", "")

    primary_normalized = normalize(primary_text, strip_diacritics=strip_diacritics)
    secondary_normalized = normalize(secondary_text, strip_diacritics=strip_diacritics)

    if primary_normalized and secondary_normalized:
        len_ratio = len(primary_normalized) / max(len(secondary_normalized), 1)
        if len_ratio > 1.3 or len_ratio < 0.7:
            logger.warning("Large length discrepancy between OCR outputs, using shorter")
            if len(primary_normalized) <= len(secondary_normalized):
                return primary_normalized, 0.5
            return secondary_normalized, 0.5

    if not primary_normalized and secondary_normalized:
        return secondary_normalized, 0.5
    if primary_normalized and not secondary_normalized:
        return primary_normalized, 0.5
    if not primary_normalized and not secondary_normalized:
        return "", 0.0

    verified = verify_and_merge(primary_normalized, secondary_normalized, original_crop)
    return verified["text"], verified["confidence"]
