"""Pipeline orchestrator: render -> preprocess -> layout -> OCR -> normalize -> verify.

Processes scanned Arabic PDFs through all stages, producing final text output.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from arabic_ocr.renderer import render_pdf
from arabic_ocr.preprocessor import preprocess
from arabic_ocr.layout import detect_regions
from arabic_ocr.ocr_engines import run_dual_ocr
from arabic_ocr.normalizer import normalize
from arabic_ocr.verifier import verify_and_merge

logger = logging.getLogger(__name__)


def process_pdf(
    pdf_path: str,
    config: dict | None = None,
) -> list[dict]:
    """Process a scanned Arabic PDF through the full OCR pipeline.

    Args:
        pdf_path: Path to the PDF file.
        config: Optional configuration dict with keys:
            - dpi (int): Render resolution, default 400
            - pages (list[int]): Specific pages to process (0-indexed)
            - strip_diacritics (bool): Strip diacritics in normalization, default True

    Returns:
        List of PageResult dicts, each with keys:
            - page (int): 0-indexed page number
            - text (str): Final extracted text
            - confidence (float): Overall confidence score
            - regions (int): Number of regions detected
    """
    config = config or {}
    dpi = config.get("dpi", 400)
    pages = config.get("pages", None)
    strip_diacritics = config.get("strip_diacritics", True)

    # Stage 0: Render PDF to images
    page_images = render_pdf(pdf_path, dpi=dpi, pages=pages)

    results = []
    for page_idx, original_image in enumerate(page_images):
        actual_page = pages[page_idx] if pages else page_idx
        page_result = _process_page(
            original_image=original_image,
            page_num=actual_page,
            strip_diacritics=strip_diacritics,
        )
        results.append(page_result)

    return results


def _process_page(
    *,
    original_image: Image.Image,
    page_num: int,
    strip_diacritics: bool,
) -> dict:
    """Process a single page through the pipeline.

    Args:
        original_image: Raw rendered page image.
        page_num: Page number for output.
        strip_diacritics: Whether to strip diacritics.

    Returns:
        PageResult dict.
    """
    # Stage 1: Preprocess (for layout detection, not for VLMs)
    preprocessed = preprocess(original_image)

    # Stage 2: Layout detection
    regions = detect_regions(original_image)

    # Process each region
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
        except Exception as e:
            logger.warning(f"Region processing failed: {e}")
            continue

    # Combine region texts
    final_text = "\n\n".join(t for t in region_texts if t)
    avg_confidence = total_confidence / max(num_successful, 1)

    return {
        "page": page_num,
        "text": final_text,
        "confidence": avg_confidence,
        "regions": len(regions),
    }


def _process_region(
    *,
    original_crop: Image.Image,
    strip_diacritics: bool,
) -> tuple[str, float]:
    """Process a single region through OCR -> normalize -> verify.

    Args:
        original_crop: Original (not preprocessed) cropped region image.
        strip_diacritics: Whether to strip diacritics.

    Returns:
        Tuple of (text, confidence).
    """
    # Stage 3: Dual VLM OCR on ORIGINAL image (not preprocessed)
    ocr_results = run_dual_ocr(original_crop)

    primary_text = ocr_results["primary"].get("text", "")
    secondary_text = ocr_results["secondary"].get("text", "")

    # Stage 4: Normalize BEFORE comparison
    primary_normalized = normalize(primary_text, strip_diacritics=strip_diacritics)
    secondary_normalized = normalize(secondary_text, strip_diacritics=strip_diacritics)

    # Sanity check: if one output is >30% longer/shorter, it likely hallucinated
    if primary_normalized and secondary_normalized:
        len_ratio = len(primary_normalized) / max(len(secondary_normalized), 1)
        if len_ratio > 1.3 or len_ratio < 0.7:
            # Use the shorter one as it's less likely to be hallucinated
            logger.warning("Large length discrepancy between OCR outputs, using shorter")
            if len(primary_normalized) <= len(secondary_normalized):
                return primary_normalized, 0.5
            return secondary_normalized, 0.5

    # Handle cases where one engine failed
    if not primary_normalized and secondary_normalized:
        return secondary_normalized, 0.5
    if primary_normalized and not secondary_normalized:
        return primary_normalized, 0.5
    if not primary_normalized and not secondary_normalized:
        return "", 0.0

    # Stage 5: Cross-verify
    verified = verify_and_merge(primary_normalized, secondary_normalized, original_crop)

    return verified["text"], verified["confidence"]
