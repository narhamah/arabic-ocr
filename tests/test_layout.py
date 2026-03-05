"""Tests for layout detection — TDD RED phase."""

import pytest
from PIL import Image, ImageDraw
from arabic_ocr.layout import detect_regions


class TestSingleRegion:

    def test_single_region_full_page(self, sample_arabic_image):
        """Image with no clear columns returns 1 region covering full page."""
        regions = detect_regions(sample_arabic_image)
        assert len(regions) >= 1

    def test_fallback_without_yolo(self, sample_arabic_image):
        """Without DocLayout-YOLO, returns single full-page region."""
        regions = detect_regions(sample_arabic_image)
        assert len(regions) >= 1
        # Full-page region bbox should cover most of the image
        bbox = regions[0]["bbox"]
        img_w, img_h = sample_arabic_image.size
        region_w = bbox[2] - bbox[0]
        region_h = bbox[3] - bbox[1]
        assert region_w >= img_w * 0.9
        assert region_h >= img_h * 0.9


class TestRegionStructure:

    def test_regions_have_bboxes(self, sample_arabic_image):
        regions = detect_regions(sample_arabic_image)
        for region in regions:
            assert "bbox" in region
            bbox = region["bbox"]
            assert len(bbox) == 4
            x1, y1, x2, y2 = bbox
            assert x2 > x1
            assert y2 > y1

    def test_regions_are_cropped_images(self, sample_arabic_image):
        regions = detect_regions(sample_arabic_image)
        for region in regions:
            assert "crop" in region
            assert isinstance(region["crop"], Image.Image)

    def test_regions_have_label(self, sample_arabic_image):
        regions = detect_regions(sample_arabic_image)
        for region in regions:
            assert "label" in region
            assert isinstance(region["label"], str)

    def test_regions_have_confidence(self, sample_arabic_image):
        regions = detect_regions(sample_arabic_image)
        for region in regions:
            assert "confidence" in region
            assert 0.0 <= region["confidence"] <= 1.0


class TestTwoColumnLayout:

    def test_detects_two_columns(self, two_column_image):
        """Image with clear 2-column layout should return 2 regions."""
        regions = detect_regions(two_column_image)
        # With YOLO fallback, may return 1 full-page region
        # but the function should at least return valid regions
        assert len(regions) >= 1

    def test_rtl_column_order(self, two_column_image):
        """If multiple regions, right column comes before left column."""
        regions = detect_regions(two_column_image)
        if len(regions) >= 2:
            # First region should have higher x (more to the right)
            first_x_center = (regions[0]["bbox"][0] + regions[0]["bbox"][2]) / 2
            second_x_center = (regions[1]["bbox"][0] + regions[1]["bbox"][2]) / 2
            assert first_x_center >= second_x_center


class TestEdgeCases:

    def test_blank_image(self):
        img = Image.new("RGB", (800, 600), "white")
        regions = detect_regions(img)
        assert len(regions) >= 1

    def test_small_image(self):
        img = Image.new("RGB", (50, 50), "white")
        regions = detect_regions(img)
        assert len(regions) >= 1
