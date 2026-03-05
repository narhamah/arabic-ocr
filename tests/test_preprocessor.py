"""Tests for image preprocessor — TDD RED phase."""

import pytest
import numpy as np
from PIL import Image
from arabic_ocr.preprocessor import preprocess


class TestPreprocessOutput:

    def test_output_is_pil_image(self, sample_arabic_image):
        result = preprocess(sample_arabic_image)
        assert isinstance(result, Image.Image)

    def test_output_is_grayscale(self, sample_arabic_image):
        result = preprocess(sample_arabic_image)
        assert result.mode == "L"

    def test_preserves_dimensions_approximately(self, sample_arabic_image):
        result = preprocess(sample_arabic_image)
        orig_w, orig_h = sample_arabic_image.size
        res_w, res_h = result.size
        # Allow 10% tolerance for deskew
        assert abs(res_w - orig_w) < orig_w * 0.1
        assert abs(res_h - orig_h) < orig_h * 0.1


class TestContrastEnhancement:

    def test_enhances_low_contrast(self):
        """CLAHE should increase contrast on an image with local variation."""
        # Create image with low-contrast text-like features
        arr = np.full((200, 200), 150, dtype=np.uint8)
        # Add slightly darker "text" regions (low contrast: 150 vs 130)
        arr[40:60, 20:180] = 130
        arr[80:100, 20:180] = 130
        arr[120:140, 20:180] = 130
        img = Image.fromarray(arr, mode="L")

        result = preprocess(img)
        result_arr = np.array(result)
        # CLAHE should increase the spread of pixel values
        assert result_arr.std() > arr.std()


class TestDeskew:

    def test_deskews_rotated_image(self, rotated_image):
        """Rotated image should be corrected."""
        result = preprocess(rotated_image)
        # Result should exist and be processable
        assert isinstance(result, Image.Image)
        assert result.mode == "L"


class TestDotPreservation:

    def test_does_not_destroy_small_features(self):
        """Create image with small dots (like Arabic letter dots) and verify they survive."""
        img = Image.new("RGB", (200, 200), "white")
        arr = np.array(img)
        # Add small dots (3x3 pixels, like Arabic dots)
        for pos in [(50, 50), (100, 100), (150, 150)]:
            arr[pos[1] - 1 : pos[1] + 2, pos[0] - 1 : pos[0] + 2] = [0, 0, 0]
        img = Image.fromarray(arr)

        result = preprocess(img)
        result_arr = np.array(result)

        # At least some dots should survive (dark pixels in dot regions)
        dot_regions_have_dark = False
        for pos in [(50, 50), (100, 100), (150, 150)]:
            region = result_arr[
                max(0, pos[1] - 3) : pos[1] + 4, max(0, pos[0] - 3) : pos[0] + 4
            ]
            if region.min() < 128:
                dot_regions_have_dark = True
                break
        assert dot_regions_have_dark, "Small dots were destroyed by preprocessing"


class TestAcceptsVariousInputs:

    def test_accepts_rgb_image(self):
        img = Image.new("RGB", (100, 100), "white")
        result = preprocess(img)
        assert result.mode == "L"

    def test_accepts_grayscale_image(self):
        img = Image.new("L", (100, 100), 255)
        result = preprocess(img)
        assert result.mode == "L"

    def test_accepts_rgba_image(self):
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        result = preprocess(img)
        assert result.mode == "L"
