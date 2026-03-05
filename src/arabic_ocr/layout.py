"""Stage 2: Layout detection and RTL sorting.

Detects text regions in a page image, sorts them in Arabic reading order
(right-to-left columns, top-to-bottom within columns).

Uses DocLayout-YOLO when available, falls back to single full-page region.
"""

from __future__ import annotations

from PIL import Image


def detect_regions(image: Image.Image) -> list[dict]:
    """Detect text regions in an image.

    Args:
        image: PIL Image of a document page.

    Returns:
        List of region dicts, each with keys:
            - label: str (e.g., "text", "title", "full_page")
            - bbox: tuple (x1, y1, x2, y2)
            - crop: PIL.Image.Image of the cropped region
            - confidence: float 0.0-1.0
    """
    regions = _try_yolo_detection(image)

    if not regions:
        regions = _full_page_fallback(image)

    regions = _sort_rtl_reading_order(regions)
    return regions


def _try_yolo_detection(image: Image.Image) -> list[dict]:
    """Try DocLayout-YOLO detection. Returns empty list if unavailable."""
    try:
        from doclayout_yolo import YOLOv10
    except ImportError:
        return []

    try:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        model = YOLOv10(model_path)
        results = model.predict(image, imgsz=1024, conf=0.2)

        regions = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                label = result.names[int(box.cls[0])]
                conf = float(box.conf[0])
                crop = image.crop((x1, y1, x2, y2))
                regions.append({
                    "label": label,
                    "bbox": (x1, y1, x2, y2),
                    "crop": crop,
                    "confidence": conf,
                })

        return regions
    except Exception:
        return []


def _full_page_fallback(image: Image.Image) -> list[dict]:
    """Return single region covering the full page."""
    w, h = image.size
    return [{
        "label": "full_page",
        "bbox": (0, 0, w, h),
        "crop": image.copy(),
        "confidence": 1.0,
    }]


def _sort_rtl_reading_order(regions: list[dict]) -> list[dict]:
    """Sort regions in Arabic reading order.

    Groups regions into columns by x-center proximity,
    sorts columns right-to-left, sorts within column top-to-bottom.
    """
    if len(regions) <= 1:
        return regions

    # Calculate x-center for each region
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        region["_x_center"] = (x1 + x2) / 2
        region["_y_center"] = (y1 + y2) / 2

    # Group into columns using x-center clustering
    sorted_by_x = sorted(regions, key=lambda r: r["_x_center"], reverse=True)

    columns: list[list[dict]] = []
    for region in sorted_by_x:
        placed = False
        for col in columns:
            col_x = sum(r["_x_center"] for r in col) / len(col)
            img_width = max(r["bbox"][2] for r in regions)
            # If x-center is within 20% of image width, same column
            if abs(region["_x_center"] - col_x) < img_width * 0.2:
                col.append(region)
                placed = True
                break
        if not placed:
            columns.append([region])

    # Sort columns right-to-left (by average x-center, descending)
    columns.sort(key=lambda col: sum(r["_x_center"] for r in col) / len(col), reverse=True)

    # Sort within each column top-to-bottom
    result = []
    for col in columns:
        col.sort(key=lambda r: r["_y_center"])
        result.extend(col)

    # Clean up temp keys
    for region in result:
        region.pop("_x_center", None)
        region.pop("_y_center", None)

    return result
