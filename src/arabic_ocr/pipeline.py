"""Pipeline orchestration for scanned Arabic PDFs."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from arabic_ocr.engines.base import EngineOutput, OCREngine
from arabic_ocr.engines.registry import get_primary_secondary
from arabic_ocr.exporters import legacy_page_results
from arabic_ocr.field_refinement import refine_high_value_lines
from arabic_ocr.layout import detect_regions
from arabic_ocr.models import (
    BoundingBox,
    OCRBlock,
    OCRDocument,
    OCRLine,
    OCRPage,
    OCRSpan,
    PageAsset,
    PipelineConfig,
)
from arabic_ocr.native_pdf import choose_best_page_text, should_compare_native_ocr
from arabic_ocr.normalizer import normalize
from arabic_ocr.preparation import reflow_block_text
from arabic_ocr.preprocessor import preprocess
from arabic_ocr.quality import (
    extract_high_value_fields,
    is_low_quality_text,
    looks_like_hallucinated_ocr_text,
    ocr_quality_score,
    strip_ocr_boilerplate_lines,
    token_stats,
)
from arabic_ocr.renderer import inspect_pdf
from arabic_ocr.verifier import verify_and_merge

logger = logging.getLogger(__name__)


def process_pdf(pdf_path: str, config: dict | None = None) -> list[dict]:
    """Legacy page-oriented OCR API."""
    config_obj = PipelineConfig.from_mapping(config)
    document = process_pdf_document(pdf_path, config=config_obj)
    return legacy_page_results(document, profile=config_obj.profile)


def process_pdf_document(
    pdf_path: str,
    config: dict | PipelineConfig | None = None,
) -> OCRDocument:
    """Process a PDF into the canonical structured document model."""
    config_obj = config if isinstance(config, PipelineConfig) else PipelineConfig.from_mapping(config)
    assets = inspect_pdf(
        pdf_path,
        dpi=config_obj.dpi,
        pages=config_obj.pages,
        render_images=True,
    )

    document = OCRDocument(
        source_pdf=str(pdf_path),
        dpi=config_obj.dpi,
        metadata={
            "engine_mode": config_obj.engine,
            "profile": config_obj.profile,
            "strip_diacritics": config_obj.strip_diacritics,
        },
    )

    primary_engine, secondary_engine = get_primary_secondary(config_obj.engine)
    _emit_progress(
        config_obj,
        "document_start",
        source_pdf=str(pdf_path),
        page_total=len(assets),
        engine_mode=config_obj.engine,
    )
    for index, asset in enumerate(assets, start=1):
        _emit_progress(
            config_obj,
            "page_start",
            page_number=asset.page_number,
            page_index=index,
            page_total=len(assets),
            has_embedded_text=asset.has_embedded_text,
        )
        if _use_embedded_text(asset, config_obj):
            page = _resolve_page_from_native_and_ocr(
                asset=asset,
                config=config_obj,
                primary_engine=primary_engine,
                secondary_engine=secondary_engine,
                page_index=index,
                page_total=len(assets),
            )
        else:
            page = _process_page(
                asset=asset,
                config=config_obj,
                primary_engine=primary_engine,
                secondary_engine=secondary_engine,
                page_index=index,
                page_total=len(assets),
            )
        document.pages.append(page)
        _write_debug_assets(asset, config_obj)
        _emit_progress(
            config_obj,
            "page_complete",
            page_number=asset.page_number,
            page_index=index,
            page_total=len(assets),
            block_count=len(page.blocks),
            source_kind=page.source_kind,
        )

    _emit_progress(config_obj, "document_complete", page_total=len(document.pages))
    return document


def _use_embedded_text(asset: PageAsset, config: PipelineConfig) -> bool:
    """Prefer embedded/native text when the page already contains useful text."""
    return config.use_embedded_text and bool(asset.embedded_text.strip())


def _resolve_page_from_native_and_ocr(
    *,
    asset: PageAsset,
    config: PipelineConfig,
    primary_engine: OCREngine,
    secondary_engine: OCREngine | None,
    page_index: int,
    page_total: int,
) -> OCRPage:
    """Choose the better page between native text extraction and OCR."""
    native_page = _page_from_embedded_text(asset)
    if not config.compare_native_ocr or not should_compare_native_ocr(asset.embedded_text, asset.metadata):
        _emit_progress(
            config,
            "page_embedded_text",
            page_number=asset.page_number,
            page_index=page_index,
            page_total=page_total,
        )
        return native_page

    ocr_page = _process_page(
        asset=asset,
        config=config,
        primary_engine=primary_engine,
        secondary_engine=secondary_engine,
        page_index=page_index,
        page_total=page_total,
    )
    _, comparison = choose_best_page_text(native_page.text, ocr_page.text)

    comparison_metadata = {
        **comparison,
        "native_source": asset.metadata.get("source"),
        "native_quality_status": asset.metadata.get("quality_status"),
    }
    if comparison["winner"] == "ocr" and ocr_page.text.strip():
        ocr_page.metadata["native_ocr_comparison"] = comparison_metadata
        return ocr_page

    native_page.metadata["native_ocr_comparison"] = comparison_metadata
    _emit_progress(
        config,
        "page_embedded_text",
        page_number=asset.page_number,
        page_index=page_index,
        page_total=page_total,
    )
    return native_page


def _page_from_embedded_text(asset: PageAsset) -> OCRPage:
    """Build a page directly from embedded/native PDF text."""
    text = asset.embedded_text.strip()
    source_engine = str(asset.metadata.get("source") or "embedded_text")
    lines = [
        OCRLine(
            spans=[OCRSpan(text=line, confidence=1.0, engine_id=source_engine)],
            confidence=1.0,
            engine_id=source_engine,
        )
        for line in text.splitlines()
        if line.strip()
    ]
    if not lines:
        lines = [
            OCRLine(
                spans=[OCRSpan(text=text, confidence=1.0, engine_id=source_engine)],
                confidence=1.0,
                engine_id=source_engine,
            )
        ]

    page_width, page_height = _page_dimensions(asset)
    return OCRPage(
        page_number=asset.page_number,
        width=page_width,
        height=page_height,
        source_kind="embedded_text",
        blocks=[
            OCRBlock(
                block_type="embedded_text",
                bbox=BoundingBox(0, 0, page_width, page_height),
                reading_order=0,
                lines=lines,
                confidence=1.0,
                engine_id=source_engine,
                metadata={
                    "token_stats": token_stats(text),
                    "native_extraction": dict(asset.metadata),
                },
            )
        ],
        metadata={"native_extraction": dict(asset.metadata)},
    )


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
    asset: PageAsset,
    config: PipelineConfig,
    primary_engine: OCREngine,
    secondary_engine: OCREngine | None,
    page_index: int,
    page_total: int,
) -> OCRPage:
    """Process a scanned page through preprocess -> layout -> OCR."""
    raw_image = asset.images.raw
    if raw_image is None:
        raise RuntimeError("Scanned page processing requires a rendered image")

    _emit_progress(
        config,
        "page_scan_start",
        page_number=asset.page_number,
        page_index=page_index,
        page_total=page_total,
    )
    enhanced_image = preprocess(raw_image)
    asset.images.enhanced = enhanced_image
    _emit_progress(
        config,
        "page_preprocessed",
        page_number=asset.page_number,
        page_index=page_index,
        page_total=page_total,
    )
    regions = detect_regions(enhanced_image, source_image=raw_image)
    regions = _filter_regions(regions, page_size=raw_image.size)
    _emit_progress(
        config,
        "page_layout_detected",
        page_number=asset.page_number,
        page_index=page_index,
        page_total=page_total,
        region_count=len(regions),
    )

    blocks: list[OCRBlock] = []
    for reading_order, region in enumerate(regions):
        _emit_progress(
            config,
            "block_start",
            page_number=asset.page_number,
            page_index=page_index,
            page_total=page_total,
            block_index=reading_order + 1,
            block_total=len(regions),
            block_type=region.get("label", "text"),
        )
        try:
            block = _region_to_block(
                region=region,
                reading_order=reading_order,
                config=config,
                primary_engine=primary_engine,
                secondary_engine=secondary_engine,
                page_number=asset.page_number,
                page_index=page_index,
                page_total=page_total,
            )
        except Exception as exc:  # pragma: no cover - defensive in production
            logger.warning("Region processing failed: %s", exc)
            continue
        if not block.lines or not block.text.strip():
            continue
        blocks.append(block)
        _emit_progress(
            config,
            "block_complete",
            page_number=asset.page_number,
            page_index=page_index,
            page_total=page_total,
            block_index=reading_order + 1,
            block_total=len(regions),
            block_type=block.block_type,
        )

    page_width, page_height = raw_image.size
    return OCRPage(
        page_number=asset.page_number,
        width=page_width,
        height=page_height,
        source_kind="scan",
        blocks=blocks,
        metadata={"embedded_text_present": bool(asset.embedded_text.strip())},
    )


def _process_page(
    *,
    asset: PageAsset,
    config: PipelineConfig,
    primary_engine: OCREngine,
    secondary_engine: OCREngine | None,
    page_index: int,
    page_total: int,
) -> OCRPage:
    """Compatibility wrapper for the canonical scanned-page processor."""
    return _process_ocr_page(
        asset=asset,
        config=config,
        primary_engine=primary_engine,
        secondary_engine=secondary_engine,
        page_index=page_index,
        page_total=page_total,
    )


def _region_to_block(
    *,
    region: dict,
    reading_order: int,
    config: PipelineConfig,
    primary_engine: OCREngine,
    secondary_engine: OCREngine | None,
    page_number: int | None = None,
    page_index: int | None = None,
    page_total: int | None = None,
) -> OCRBlock:
    """Recognize a single region and turn it into a structured block."""
    raw_crop = region["crop"]
    merged = _process_region_details(
        original_crop=raw_crop,
        strip_diacritics=config.strip_diacritics,
        engine_mode=config.engine,
        verify_disagreements=config.verify_disagreements,
        primary_engine=primary_engine,
        secondary_engine=secondary_engine,
    )
    block_text, confidence, metadata = merged
    block_bbox = BoundingBox.from_tuple(region["bbox"])
    lines = _text_to_lines(
        block_text,
        confidence,
        metadata.get("final_engine_id"),
        block_bbox=block_bbox,
        region_image=raw_crop,
    )
    block = OCRBlock(
        block_type=region.get("label", "text"),
        bbox=block_bbox,
        reading_order=reading_order,
        lines=lines,
        confidence=confidence,
        engine_id=metadata.get("final_engine_id"),
        metadata={
            "token_stats": token_stats(block_text),
            "fields": extract_high_value_fields(block_text),
            **metadata,
        },
    )
    if config.refine_fields:
        block = refine_high_value_lines(
            block,
            region_crop=raw_crop,
            progress_callback=config.progress_callback,
            page_number=page_number,
            block_index=reading_order,
            page_index=page_index,
            page_total=page_total,
        )
        block.metadata["token_stats"] = token_stats(block.text)
        block.metadata["fields"] = extract_high_value_fields(block.text)
    return block


def _process_region(
    *,
    original_crop: Image.Image,
    strip_diacritics: bool,
) -> tuple[str, float]:
    """Backwards-compatible helper for processing a single region."""
    text, confidence, _ = _process_region_details(
        original_crop=original_crop,
        strip_diacritics=strip_diacritics,
        engine_mode="gemini",
        verify_disagreements=True,
    )
    return text, confidence


def _process_region_details(
    *,
    original_crop: Image.Image,
    strip_diacritics: bool,
    engine_mode: str = "gemini",
    verify_disagreements: bool = True,
    primary_engine: OCREngine | None = None,
    secondary_engine: OCREngine | None = None,
) -> tuple[str, float, dict]:
    """Process a single region through OCR -> merge -> verification."""
    if primary_engine is None:
        primary_engine, secondary_engine = get_primary_secondary(engine_mode)

    primary_output = primary_engine.recognize(original_crop)
    secondary_output = None
    if secondary_engine is not None and (
        engine_mode == "gemini"
        or not primary_output.success
        or is_low_quality_text(primary_output.text)
    ):
        secondary_output = secondary_engine.recognize(original_crop)

    merged_text, confidence, metadata = _merge_outputs(
        primary_output=primary_output,
        secondary_output=secondary_output,
        image=original_crop,
        strip_diacritics=strip_diacritics,
        verify_disagreements=verify_disagreements,
    )
    merged_text, confidence, metadata = _maybe_rescue_low_quality_region(
        text=merged_text,
        confidence=confidence,
        metadata=metadata,
        original_crop=original_crop,
        strip_diacritics=strip_diacritics,
        primary_engine=primary_engine,
        secondary_engine=secondary_engine,
        primary_output=primary_output,
        secondary_output=secondary_output,
    )
    return merged_text, confidence, metadata


def _merge_outputs(
    *,
    primary_output: EngineOutput,
    secondary_output: EngineOutput | None,
    image: Image.Image,
    strip_diacritics: bool,
    verify_disagreements: bool,
) -> tuple[str, float, dict]:
    """Merge primary and secondary OCR results."""
    primary_text = primary_output.text or ""
    secondary_text = secondary_output.text if secondary_output else ""
    primary_text = _sanitize_ocr_text(primary_text)
    secondary_text = _sanitize_ocr_text(secondary_text)
    primary_normalized = normalize(primary_text, strip_diacritics=strip_diacritics)
    secondary_normalized = normalize(secondary_text, strip_diacritics=strip_diacritics)

    metadata = {
        "primary_success": primary_output.success,
        "secondary_success": secondary_output.success if secondary_output else None,
        "primary_error": primary_output.error,
        "secondary_error": secondary_output.error if secondary_output else None,
        "primary_engine_id": primary_output.metadata.get("engine_id"),
        "secondary_engine_id": secondary_output.metadata.get("engine_id") if secondary_output else None,
        "primary_quality_score": ocr_quality_score(primary_normalized),
        "secondary_quality_score": ocr_quality_score(secondary_normalized),
    }

    if primary_normalized and secondary_normalized and verify_disagreements:
        len_ratio = len(primary_normalized) / max(len(secondary_normalized), 1)
        if len_ratio > 1.3 or len_ratio < 0.7:
            if metadata["primary_quality_score"] > metadata["secondary_quality_score"] + 0.2:
                metadata["final_engine_id"] = primary_output.metadata.get("engine_id")
                return primary_normalized, 0.5, metadata
            if metadata["secondary_quality_score"] > metadata["primary_quality_score"] + 0.2:
                metadata["final_engine_id"] = secondary_output.metadata.get("engine_id") if secondary_output else None
                return secondary_normalized, 0.5, metadata
            if len(primary_normalized) <= len(secondary_normalized):
                metadata["final_engine_id"] = primary_output.metadata.get("engine_id")
                return primary_normalized, 0.5, metadata
            metadata["final_engine_id"] = secondary_output.metadata.get("engine_id") if secondary_output else None
            return secondary_normalized, 0.5, metadata

    if not primary_normalized and secondary_normalized:
        metadata["final_engine_id"] = secondary_output.metadata.get("engine_id") if secondary_output else None
        secondary_normalized = _maybe_reflow_block_text(secondary_normalized, metadata)
        return secondary_normalized, 0.5, metadata
    if primary_normalized and not secondary_normalized:
        metadata["final_engine_id"] = primary_output.metadata.get("engine_id")
        primary_normalized = _maybe_reflow_block_text(primary_normalized, metadata)
        return primary_normalized, 0.5, metadata
    if not primary_normalized and not secondary_normalized:
        metadata["final_engine_id"] = primary_output.metadata.get("engine_id")
        return "", 0.0, metadata

    if verify_disagreements and secondary_normalized:
        verified = verify_and_merge(primary_normalized, secondary_normalized, image)
        metadata["final_engine_id"] = "merged"
        metadata["disagreements"] = verified["disagreements"]
        metadata["total_words"] = verified["total_words"]
        merged_text = _maybe_reflow_block_text(verified["text"], metadata)
        return merged_text, verified["confidence"], metadata

    metadata["final_engine_id"] = primary_output.metadata.get("engine_id")
    primary_normalized = _maybe_reflow_block_text(primary_normalized, metadata)
    return primary_normalized, primary_output.confidence or 0.5, metadata


def _text_to_lines(
    text: str,
    confidence: float,
    engine_id: str | None,
    *,
    block_bbox: BoundingBox | None = None,
    region_image: Image.Image | None = None,
) -> list[OCRLine]:
    """Split merged block text into line objects."""
    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not stripped_lines and text:
        stripped_lines = [text.strip()]
    if not stripped_lines:
        return []

    lines: list[OCRLine] = []
    line_boxes = _estimate_line_bboxes(
        block_bbox,
        len(stripped_lines),
        region_image=region_image,
    )
    for index, line in enumerate(stripped_lines):
        line_bbox = line_boxes[index] if index < len(line_boxes) else None
        lines.append(OCRLine(
            spans=[OCRSpan(text=line, confidence=confidence, engine_id=engine_id, bbox=line_bbox)],
            confidence=confidence,
            bbox=line_bbox,
            engine_id=engine_id,
        ))
    return lines


def _page_dimensions(asset: PageAsset) -> tuple[int, int]:
    """Return page dimensions, preferring rendered pixel dimensions when available."""
    if asset.images.raw is not None:
        return asset.images.raw.size
    return asset.width, asset.height


def _write_debug_assets(asset: PageAsset, config: PipelineConfig) -> None:
    """Persist optional page-level debug imagery."""
    if config.save_debug_dir is None:
        return

    debug_dir = Path(config.save_debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = f"page_{asset.page_number + 1:04d}"
    if asset.images.raw is not None:
        asset.images.raw.save(debug_dir / f"{stem}_raw.png")
    if asset.images.enhanced is not None:
        asset.images.enhanced.save(debug_dir / f"{stem}_enhanced.png")


def _sanitize_ocr_text(text: str) -> str:
    """Drop obvious non-OCR boilerplate responses from model outputs."""
    cleaned = strip_ocr_boilerplate_lines(text)
    if looks_like_hallucinated_ocr_text(cleaned):
        return ""
    return cleaned


def _maybe_reflow_block_text(text: str, metadata: dict) -> str:
    """Restore line breaks for long single-line OpenAI OCR output."""
    stats = token_stats(text)
    if not text or "\n" in text or stats["token_count"] < 80:
        return text
    if metadata.get("final_engine_id") != "openai":
        return text

    try:
        reflowed = reflow_block_text(text)
    except Exception:
        return text
    return reflowed or text


def _maybe_rescue_low_quality_region(
    *,
    text: str,
    confidence: float,
    metadata: dict,
    original_crop: Image.Image,
    strip_diacritics: bool,
    primary_engine: OCREngine,
    secondary_engine: OCREngine | None,
    primary_output: EngineOutput,
    secondary_output: EngineOutput | None,
) -> tuple[str, float, dict]:
    """Retry weak OCR blocks with higher-resolution and rotated variants."""
    trigger = _should_run_ocr_rescue(text, image=original_crop)
    if trigger is None:
        return text, confidence, metadata

    primary_score = metadata.get("primary_quality_score", -1.0)
    secondary_score = metadata.get("secondary_quality_score", -1.0)
    rescue_engine = primary_engine
    if secondary_engine is not None and secondary_output is not None and secondary_score > primary_score:
        rescue_engine = secondary_engine

    best_text = text
    best_confidence = confidence
    best_score = ocr_quality_score(text)
    best_variant = None
    best_variant_metadata: dict[str, object] = {}

    chunked_candidate = _run_chunked_ocr_rescue(
        image=original_crop,
        text=text,
        strip_diacritics=strip_diacritics,
        trigger=trigger,
        rescue_engine=rescue_engine,
    )
    if chunked_candidate is not None:
        candidate_text, candidate_confidence, candidate_variant, candidate_metadata = chunked_candidate
        candidate_score = ocr_quality_score(candidate_text)
        if _candidate_improves_ocr_result(
            candidate_text=candidate_text,
            candidate_score=candidate_score,
            best_text=best_text,
            best_score=best_score,
        ):
            best_text = candidate_text
            best_confidence = max(best_confidence, candidate_confidence)
            best_score = candidate_score
            best_variant = candidate_variant
            best_variant_metadata = candidate_metadata

    for variant_name, variant_image in _build_rescue_variants(
        original_crop,
        include_right_angle_rotations=not bool(text.strip()),
    ):
        try:
            output = rescue_engine.recognize(variant_image)
        except Exception:
            continue

        candidate_text = _normalize_engine_output_text(
            output.text,
            strip_diacritics=strip_diacritics,
            engine_id=rescue_engine.id,
        )
        if not candidate_text:
            continue

        candidate_score = ocr_quality_score(candidate_text)
        if not _candidate_improves_ocr_result(
            candidate_text=candidate_text,
            candidate_score=candidate_score,
            best_text=best_text,
            best_score=best_score,
        ):
            continue

        best_text = candidate_text
        best_confidence = max(best_confidence, output.confidence or 0.5)
        best_score = candidate_score
        best_variant = variant_name
        best_variant_metadata = {}

    if best_variant is None:
        return text, confidence, metadata

    rescued_metadata = dict(metadata)
    rescued_metadata["final_engine_id"] = rescue_engine.id
    rescued_metadata["ocr_rescue"] = {
        "trigger": trigger,
        "variant": best_variant,
        "engine_id": rescue_engine.id,
        "initial_quality_score": round(ocr_quality_score(text), 4),
        "rescued_quality_score": round(best_score, 4),
    }
    rescued_metadata["ocr_rescue"].update(best_variant_metadata)
    return best_text, best_confidence, rescued_metadata


def _should_run_ocr_rescue(text: str, *, image: Image.Image) -> str | None:
    """Decide whether a region deserves extra OCR passes."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    if is_low_quality_text(stripped):
        return "low_quality"

    width, height = image.size
    area = width * height
    token_count = int(token_stats(stripped)["token_count"])
    sparse_large_crop_threshold = max(40, area // 140_000)
    if area >= 6_000_000 and token_count < sparse_large_crop_threshold:
        return "sparse_large_crop"
    if area >= 1_000_000 and token_count < 24:
        return "under_read_large_crop"
    return None


def _run_chunked_ocr_rescue(
    *,
    image: Image.Image,
    text: str,
    strip_diacritics: bool,
    trigger: str,
    rescue_engine: OCREngine,
) -> tuple[str, float, str, dict[str, object]] | None:
    """Retry oversized sparse regions as overlapping horizontal OCR chunks."""
    if not _should_try_chunked_ocr(image=image, text=text, trigger=trigger):
        return None

    chunk_specs = _build_chunked_ocr_crops(image)
    if len(chunk_specs) < 2:
        return None

    chunk_texts: list[str] = []
    confidences: list[float] = []
    for _, chunk_image in chunk_specs:
        try:
            output = rescue_engine.recognize(chunk_image)
        except Exception:
            continue

        chunk_text = _normalize_engine_output_text(
            output.text,
            strip_diacritics=strip_diacritics,
            engine_id=rescue_engine.id,
        )
        if not chunk_text:
            continue

        chunk_texts.append(chunk_text)
        confidences.append(output.confidence or 0.5)

    if not chunk_texts:
        return None

    stitched_text = _stitch_chunked_text(chunk_texts)
    if not stitched_text:
        return None

    average_confidence = sum(confidences) / len(confidences)
    return (
        stitched_text,
        average_confidence,
        "chunked_horizontal",
        {
            "chunk_count": len(chunk_specs),
            "successful_chunk_count": len(chunk_texts),
        },
    )


def _should_try_chunked_ocr(*, image: Image.Image, text: str, trigger: str) -> bool:
    """Limit chunked OCR retries to large crops that appear under-read."""
    width, height = image.size
    area = width * height
    stats = token_stats(text)
    token_count = int(stats["token_count"])
    line_count = sum(1 for line in text.splitlines() if line.strip())

    if height < 1800 and area < 5_000_000:
        return False
    if trigger in {"sparse_large_crop", "under_read_large_crop"}:
        return True
    if trigger == "empty" and height >= 2200:
        return True
    if trigger == "low_quality" and area >= 5_000_000 and token_count < 120:
        return True
    if area >= 8_000_000 and line_count < 8:
        return True
    return False


def _build_chunked_ocr_crops(
    image: Image.Image,
    *,
    target_height: int = 1600,
    overlap: int = 180,
    boundary_search: int = 220,
) -> list[tuple[str, Image.Image]]:
    """Split tall crops into overlapping horizontal OCR chunks."""
    width, height = image.size
    if height <= target_height + overlap:
        return []

    row_ink = (np.asarray(image.convert("L")) < 235).sum(axis=1)
    min_chunk_height = max(target_height // 2, 900)
    chunks: list[tuple[str, Image.Image]] = []
    start = 0
    chunk_index = 1

    while start < height:
        proposed_end = min(height, start + target_height)
        if proposed_end >= height:
            end = height
        else:
            search_start = max(start + min_chunk_height, proposed_end - boundary_search)
            search_end = min(height - 1, proposed_end + boundary_search)
            if search_end <= search_start:
                end = proposed_end
            else:
                cut_offset = int(np.argmin(row_ink[search_start:search_end + 1]))
                end = max(search_start + cut_offset, start + min_chunk_height)

        if end <= start:
            end = min(height, start + target_height)

        chunks.append((f"chunk_{chunk_index}", image.crop((0, start, width, end))))
        if end >= height:
            break

        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = min(height, start + min_chunk_height)
        start = next_start
        chunk_index += 1

    return chunks if len(chunks) > 1 else []


def _stitch_chunked_text(chunk_texts: list[str]) -> str:
    """Join chunk OCR results while dropping duplicated overlap lines."""
    merged_lines: list[str] = []
    for chunk_text in chunk_texts:
        lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
        if not lines:
            continue
        if not merged_lines:
            merged_lines.extend(lines)
            continue

        overlap = _find_chunk_line_overlap(merged_lines, lines)
        merged_lines.extend(lines[overlap:])

    return "\n".join(merged_lines).strip()


def _find_chunk_line_overlap(existing_lines: list[str], new_lines: list[str], *, max_overlap: int = 6) -> int:
    """Find exact normalized line overlap between adjacent OCR chunks."""
    limit = min(len(existing_lines), len(new_lines), max_overlap)
    if limit <= 0:
        return 0

    existing_signatures = [_line_signature(line) for line in existing_lines]
    new_signatures = [_line_signature(line) for line in new_lines]
    for overlap in range(limit, 0, -1):
        if existing_signatures[-overlap:] == new_signatures[:overlap]:
            return overlap
    return 0


def _line_signature(text: str) -> str:
    """Normalize a line enough to compare chunk overlap boundaries."""
    normalized = normalize(text, strip_diacritics=True)
    return " ".join(normalized.split())


def _candidate_improves_ocr_result(
    *,
    candidate_text: str,
    candidate_score: float,
    best_text: str,
    best_score: float,
) -> bool:
    """Accept either a clear quality gain or a modest gain with meaningfully more text."""
    if candidate_score > best_score + 0.2:
        return True

    candidate_tokens = int(token_stats(candidate_text)["token_count"])
    best_tokens = int(token_stats(best_text)["token_count"])
    if candidate_score > best_score + 0.05 and candidate_tokens >= int(best_tokens * 1.15):
        return True
    return best_score < 0.0 and candidate_score > best_score


def _build_rescue_variants(
    image: Image.Image,
    *,
    include_right_angle_rotations: bool,
) -> list[tuple[str, Image.Image]]:
    """Build transformed OCR retries for weak regions."""
    enhanced = preprocess(image)
    variants: list[tuple[str, Image.Image]] = [
        ("upscaled", _upscale_for_ocr(image)),
        ("enhanced_upscaled", _upscale_for_ocr(enhanced)),
        ("rot180_upscaled", _upscale_for_ocr(image.rotate(180, expand=True))),
        ("rot180_enhanced_upscaled", _upscale_for_ocr(enhanced.rotate(180, expand=True))),
    ]
    if include_right_angle_rotations:
        variants.extend([
            ("rot90_upscaled", _upscale_for_ocr(image.rotate(90, expand=True))),
            ("rot270_upscaled", _upscale_for_ocr(image.rotate(270, expand=True))),
        ])
    return variants


def _normalize_engine_output_text(
    text: str,
    *,
    strip_diacritics: bool,
    engine_id: str | None,
) -> str:
    """Apply the same cleanup path used for baseline OCR output."""
    cleaned = _sanitize_ocr_text(text or "")
    if "\n" in cleaned or "\r" in cleaned:
        normalized_lines = [
            normalize(line, strip_diacritics=strip_diacritics)
            for line in cleaned.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        ]
        normalized = "\n".join(line for line in normalized_lines if line)
    else:
        normalized = normalize(cleaned, strip_diacritics=strip_diacritics)
    if not normalized:
        return ""
    return _maybe_reflow_block_text(normalized, {"final_engine_id": engine_id})


def _upscale_for_ocr(image: Image.Image, *, max_side: int = 3072) -> Image.Image:
    """Upscale small OCR crops to improve recognition of dense print."""
    longest_side = max(image.size)
    if longest_side >= max_side:
        return image.copy()

    scale = min(2.0, max_side / max(longest_side, 1))
    if scale <= 1.05:
        return image.copy()

    resampling_module = getattr(Image, "Resampling", Image)
    return image.resize(
        (max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale))),
        resample=resampling_module.LANCZOS,
    )


def _filter_regions(regions: list[dict], *, page_size: tuple[int, int]) -> list[dict]:
    """Drop tiny heuristic fragments that tend to produce OCR hallucinations."""
    if len(regions) <= 1:
        return regions

    page_width, page_height = page_size
    page_area = max(page_width * page_height, 1)
    filtered: list[dict] = []
    for region in regions:
        if region.get("label") != "heuristic_block":
            filtered.append(region)
            continue

        x1, y1, x2, y2 = region["bbox"]
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        area_ratio = (width * height) / page_area
        width_ratio = width / page_width
        height_ratio = height / page_height
        if area_ratio < 0.008 and (width_ratio < 0.35 or height_ratio < 0.04):
            continue
        filtered.append(region)

    return filtered or regions


def _emit_progress(config: PipelineConfig, event: str, **payload: object) -> None:
    """Emit a progress event if the caller configured a callback."""
    if config.progress_callback is None:
        return
    config.progress_callback(event, payload)


def _estimate_line_bboxes(
    block_bbox: BoundingBox | None,
    line_count: int,
    *,
    region_image: Image.Image | None = None,
) -> list[BoundingBox]:
    """Estimate line boxes using the block image, falling back to equal slices."""
    if block_bbox is None or line_count <= 0:
        return []

    if region_image is not None:
        segments = _detect_line_segments(region_image)
        fitted_segments = _fit_segments_to_count(segments, line_count, image_height=region_image.size[1])
        if fitted_segments:
            return [
                BoundingBox(
                    block_bbox.x1,
                    block_bbox.y1 + y1,
                    block_bbox.x2,
                    block_bbox.y1 + max(y2, y1 + 1),
                )
                for y1, y2 in fitted_segments
            ]

    total_height = max(block_bbox.y2 - block_bbox.y1, 1)
    line_height = total_height / line_count
    boxes: list[BoundingBox] = []
    for index in range(line_count):
        y1 = int(block_bbox.y1 + index * line_height)
        y2 = int(block_bbox.y1 + (index + 1) * line_height)
        boxes.append(BoundingBox(block_bbox.x1, y1, block_bbox.x2, max(y2, y1 + 1)))
    return boxes


def _detect_line_segments(image: Image.Image) -> list[tuple[int, int]]:
    """Detect visual text rows inside a cropped block image."""
    gray = np.asarray(image.convert("L"))
    if gray.size == 0:
        return []

    content_mask = gray < 235
    if not bool(content_mask.any()):
        return []

    height, width = content_mask.shape
    row_threshold = max(6, int(width * 0.008))
    row_mask = content_mask.sum(axis=1) > row_threshold
    return _find_segments(
        row_mask,
        max_gap=max(4, height // 180),
        min_length=max(8, height // 300),
    )


def _find_segments(mask: np.ndarray, *, max_gap: int, min_length: int) -> list[tuple[int, int]]:
    """Group True values into contiguous segments, bridging short gaps."""
    segments: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0

    for index, value in enumerate(mask.tolist()):
        if value:
            if start is None:
                start = index
            gap = 0
            continue

        if start is None:
            continue

        gap += 1
        if gap > max_gap:
            end = index - gap + 1
            if end - start >= min_length:
                segments.append((start, end))
            start = None
            gap = 0

    if start is not None:
        end = len(mask)
        if end - start >= min_length:
            segments.append((start, end))

    return segments


def _fit_segments_to_count(
    segments: list[tuple[int, int]],
    target_count: int,
    *,
    image_height: int,
) -> list[tuple[int, int]]:
    """Merge or split visual segments to match the OCR line count."""
    if target_count <= 0:
        return []
    if not segments:
        return []

    fitted = list(segments)
    while len(fitted) > target_count:
        smallest_gap = None
        smallest_index = None
        for index in range(len(fitted) - 1):
            gap = fitted[index + 1][0] - fitted[index][1]
            if smallest_gap is None or gap < smallest_gap:
                smallest_gap = gap
                smallest_index = index
        if smallest_index is None:
            break
        start = fitted[smallest_index][0]
        end = fitted[smallest_index + 1][1]
        fitted[smallest_index:smallest_index + 2] = [(start, end)]

    while len(fitted) < target_count:
        heights = [end - start for start, end in fitted]
        split_index = max(range(len(fitted)), key=lambda idx: heights[idx])
        start, end = fitted[split_index]
        if end - start < max(12, image_height // 160):
            break
        midpoint = start + (end - start) // 2
        if midpoint <= start or midpoint >= end:
            break
        fitted[split_index:split_index + 1] = [(start, midpoint), (midpoint, end)]

    if len(fitted) != target_count:
        return []
    return fitted
