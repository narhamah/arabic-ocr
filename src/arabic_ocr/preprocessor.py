"""Stage 1: Image preprocessing for Arabic OCR.

Pipeline: grayscale -> CLAHE -> denoise -> deskew.
Returns grayscale images; some OCR engines work best from raw imagery while
classical layout/text detectors benefit from a lightly enhanced view.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:  # pragma: no cover - depends on optional extra
    import cv2
except ImportError:  # pragma: no cover - exercised indirectly in test env
    cv2 = None


def preprocess(image: Image.Image) -> Image.Image:
    """Preprocess an image for OCR/layout detection."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    if image.mode != "L":
        gray = np.array(image.convert("L"))
    else:
        gray = np.array(image)

    gray = _apply_clahe(gray)
    gray = _denoise(gray)
    gray = _deskew(gray)

    return Image.fromarray(gray, mode="L")


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply local contrast enhancement."""
    if cv2 is None:
        pil_image = Image.fromarray(gray, mode="L")
        enhanced = ImageOps.autocontrast(pil_image, cutoff=1)
        return np.array(enhanced, dtype=np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _denoise(gray: np.ndarray) -> np.ndarray:
    """Denoise while preserving small Arabic dots."""
    if cv2 is None:
        pil_image = Image.fromarray(gray, mode="L")
        denoised = pil_image.filter(ImageFilter.MedianFilter(size=3))
        return np.array(denoised, dtype=np.uint8)

    return cv2.fastNlMeansDenoising(
        gray,
        h=8,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Detect and correct skew angle when OpenCV is available."""
    if cv2 is None:
        return gray

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lines = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        return gray

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        return gray

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return gray

    height, width = gray.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(
        gray,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
