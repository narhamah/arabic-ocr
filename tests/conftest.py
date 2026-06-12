"""Shared test fixtures."""

from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np
from PIL import Image, ImageDraw
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TMP_ROOT = Path(tempfile.gettempdir()) / "arabic_ocr_tests"
TMP_ROOT.mkdir(exist_ok=True)


@pytest.fixture
def sample_arabic_image() -> Image.Image:
    """Create a simple test image with Arabic-like text on white background."""
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)
    # Draw some black rectangles to simulate text blocks
    draw.rectangle([50, 50, 750, 100], fill="black")
    draw.rectangle([50, 150, 750, 200], fill="black")
    draw.rectangle([50, 250, 750, 300], fill="black")
    return img


@pytest.fixture
def low_contrast_image() -> Image.Image:
    """Create a low-contrast test image."""
    img = Image.new("RGB", (800, 600), (180, 180, 180))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 750, 100], fill=(160, 160, 160))
    return img


@pytest.fixture
def two_column_image() -> Image.Image:
    """Create test image with two clear columns."""
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)
    # Right column (read first in Arabic RTL)
    for y in range(50, 500, 60):
        draw.rectangle([420, y, 780, y + 30], fill="black")
    # Left column
    for y in range(50, 500, 60):
        draw.rectangle([20, y, 380, y + 30], fill="black")
    return img


@pytest.fixture
def rotated_image(sample_arabic_image: Image.Image) -> Image.Image:
    """Create a slightly rotated test image."""
    return sample_arabic_image.rotate(5, expand=True, fillcolor="white")


@pytest.fixture
def repo_tmp_path() -> Path:
    """Create a writable temporary directory inside the repository."""
    path = TMP_ROOT / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    yield path
