"""Stage 1: Image preprocessing for Arabic OCR.

Pipeline: grayscale -> CLAHE -> NLM denoise (h=8) -> Sauvola binarize -> deskew.
Returns grayscale (not binary) — VLMs work better with grayscale.
"""

import cv2
import numpy as np
from PIL import Image


def preprocess(image: Image.Image) -> Image.Image:
    """Preprocess an image for OCR.

    Args:
        image: Input PIL Image (any mode: RGB, RGBA, L).

    Returns:
        Preprocessed grayscale PIL Image.
    """
    # Convert to RGB if needed, then to grayscale
    if image.mode == "RGBA":
        image = image.convert("RGB")
    if image.mode != "L":
        gray = np.array(image.convert("L"))
    else:
        gray = np.array(image)

    # CLAHE contrast enhancement
    gray = _apply_clahe(gray)

    # NLM denoise with h=8 max (preserves Arabic dots)
    gray = _denoise(gray)

    # Deskew
    gray = _deskew(gray)

    return Image.fromarray(gray, mode="L")


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _denoise(gray: np.ndarray) -> np.ndarray:
    """Apply Non-Local Means denoising with h=8.

    CRITICAL: h=8 maximum. Higher values destroy Arabic dots
    (15 letters distinguished only by dots).
    """
    return cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Detect and correct skew angle using Hough line transform."""
    # Binarize for angle detection
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find lines using Hough transform
    lines = cv2.HoughLinesP(
        binary, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )

    if lines is None or len(lines) == 0:
        return gray

    # Calculate median angle from detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (within 15 degrees)
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        return gray

    median_angle = np.median(angles)

    # Only correct if skew is significant (> 0.5 degrees)
    if abs(median_angle) < 0.5:
        return gray

    # Rotate to correct skew
    h, w = gray.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    corrected = cv2.warpAffine(
        gray, rotation_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return corrected
