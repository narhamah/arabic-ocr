"""Stage 2: Layout detection and RTL sorting."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image


def detect_regions(
    image: Image.Image,
    *,
    source_image: Image.Image | None = None,
) -> list[dict]:
    """Detect regions from a layout image and crop from the source image."""
    crop_source = source_image or image
    regions = _try_yolo_detection(image, crop_source=crop_source)

    if not regions:
        regions = _heuristic_detection(image, crop_source=crop_source)
    if not regions:
        regions = _full_page_fallback(crop_source)

    return _sort_rtl_reading_order(regions)


def _try_yolo_detection(
    image: Image.Image,
    *,
    crop_source: Image.Image | None = None,
) -> list[dict]:
    """Try DocLayout-YOLO detection. Returns empty list if unavailable."""
    crop_source = crop_source or image
    model = _load_yolo_model()
    if model is None:
        return []

    try:
        results = model.predict(image, imgsz=1024, conf=0.2)
    except Exception:
        return []

    regions = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            label = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            crop = crop_source.crop((x1, y1, x2, y2))
            regions.append({
                "label": label,
                "bbox": (x1, y1, x2, y2),
                "crop": crop,
                "confidence": conf,
            })
    return regions


@lru_cache(maxsize=1)
def _load_yolo_model():
    """Load and cache the optional DocLayout-YOLO model."""
    try:
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    try:
        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        return YOLOv10(model_path)
    except Exception:
        return None


def _heuristic_detection(
    image: Image.Image,
    *,
    crop_source: Image.Image | None = None,
) -> list[dict]:
    """Split a page into simple row bands and RTL columns using whitespace."""
    crop_source = crop_source or image
    gray = np.asarray(image.convert("L"))
    if gray.size == 0:
        return []

    content_mask = gray < 235
    if not bool(content_mask.any()):
        return []

    height, width = content_mask.shape
    row_threshold = max(8, int(width * 0.01))
    row_mask = content_mask.sum(axis=1) > row_threshold
    row_segments = _find_segments(
        row_mask,
        max_gap=max(12, height // 120),
        min_length=max(24, height // 80),
    )

    regions: list[dict] = []
    for top, bottom in row_segments:
        band_mask = content_mask[top:bottom, :]
        col_threshold = max(6, int(max(bottom - top, 1) * 0.03))
        col_mask = band_mask.sum(axis=0) > col_threshold
        col_segments = _find_segments(
            col_mask,
            max_gap=max(24, width // 80),
            min_length=max(48, width // 12),
        )

        boxes = _boxes_from_segments(
            col_segments=col_segments,
            top=top,
            bottom=bottom,
            page_width=width,
            page_height=height,
        )
        for x1, y1, x2, y2 in boxes:
            crop = crop_source.crop((x1, y1, x2, y2))
            regions.append({
                "label": "heuristic_block",
                "bbox": (x1, y1, x2, y2),
                "crop": crop,
                "confidence": 0.6,
            })

    if len(regions) <= 1:
        return []
    return regions


def _boxes_from_segments(
    *,
    col_segments: list[tuple[int, int]],
    top: int,
    bottom: int,
    page_width: int,
    page_height: int,
) -> list[tuple[int, int, int, int]]:
    if len(col_segments) >= 2:
        return [
            _expand_box(
                x1=left,
                y1=top,
                x2=right,
                y2=bottom,
                page_width=page_width,
                page_height=page_height,
            )
            for left, right in col_segments
        ]

    return [
        _expand_box(
            x1=0,
            y1=top,
            x2=page_width,
            y2=bottom,
            page_width=page_width,
            page_height=page_height,
        )
    ]


def _expand_box(
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    page_width: int,
    page_height: int,
) -> tuple[int, int, int, int]:
    pad_x = max(12, page_width // 120)
    pad_y = max(10, page_height // 180)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(page_width, x2 + pad_x),
        min(page_height, y2 + pad_y),
    )


def _find_segments(mask: np.ndarray, *, max_gap: int, min_length: int) -> list[tuple[int, int]]:
    """Group True values into segments, bridging short gaps."""
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


def _full_page_fallback(image: Image.Image) -> list[dict]:
    """Return a single region covering the full page."""
    width, height = image.size
    return [{
        "label": "full_page",
        "bbox": (0, 0, width, height),
        "crop": image.copy(),
        "confidence": 1.0,
    }]


def _sort_rtl_reading_order(regions: list[dict]) -> list[dict]:
    """Sort regions in Arabic reading order."""
    if len(regions) <= 1:
        return regions

    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        region["_x_center"] = (x1 + x2) / 2
        region["_y_center"] = (y1 + y2) / 2

    sorted_by_x = sorted(regions, key=lambda region: region["_x_center"], reverse=True)
    page_width = max(region["bbox"][2] for region in regions)

    columns: list[list[dict]] = []
    for region in sorted_by_x:
        for column in columns:
            column_x = sum(item["_x_center"] for item in column) / len(column)
            if abs(region["_x_center"] - column_x) < page_width * 0.2:
                column.append(region)
                break
        else:
            columns.append([region])

    columns.sort(
        key=lambda column: sum(item["_x_center"] for item in column) / len(column),
        reverse=True,
    )

    ordered: list[dict] = []
    for column in columns:
        column.sort(key=lambda region: region["_y_center"])
        ordered.extend(column)

    for region in ordered:
        region.pop("_x_center", None)
        region.pop("_y_center", None)

    return ordered
