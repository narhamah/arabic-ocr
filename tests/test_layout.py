"""Tests for layout detection."""

from __future__ import annotations

from PIL import Image

from arabic_ocr.layout import detect_regions


class TestSingleRegion:

    def test_single_region_returns_valid_regions(self, sample_arabic_image):
        regions = detect_regions(sample_arabic_image)
        assert len(regions) >= 1

    def test_heuristic_split_without_yolo(self, sample_arabic_image):
        regions = detect_regions(sample_arabic_image)
        assert len(regions) >= 3
        assert all(region["label"] in {"heuristic_block", "full_page"} for region in regions)


class TestRegionStructure:

    def test_regions_have_bboxes(self, sample_arabic_image):
        regions = detect_regions(sample_arabic_image)
        for region in regions:
            x1, y1, x2, y2 = region["bbox"]
            assert x2 > x1
            assert y2 > y1

    def test_regions_are_cropped_images(self, sample_arabic_image):
        regions = detect_regions(sample_arabic_image)
        for region in regions:
            assert isinstance(region["crop"], Image.Image)

    def test_regions_have_label(self, sample_arabic_image):
        regions = detect_regions(sample_arabic_image)
        for region in regions:
            assert isinstance(region["label"], str)

    def test_regions_have_confidence(self, sample_arabic_image):
        regions = detect_regions(sample_arabic_image)
        for region in regions:
            assert 0.0 <= region["confidence"] <= 1.0


class TestTwoColumnLayout:

    def test_detects_two_columns(self, two_column_image):
        regions = detect_regions(two_column_image)
        assert len(regions) >= 2

    def test_rtl_column_order(self, two_column_image):
        regions = detect_regions(two_column_image)
        assert len(regions) >= 2
        first_x_center = (regions[0]["bbox"][0] + regions[0]["bbox"][2]) / 2
        second_x_center = (regions[1]["bbox"][0] + regions[1]["bbox"][2]) / 2
        assert first_x_center >= second_x_center


class TestEdgeCases:

    def test_blank_image(self):
        img = Image.new("RGB", (800, 600), "white")
        regions = detect_regions(img)
        assert len(regions) == 1
        assert regions[0]["label"] == "full_page"

    def test_small_image(self):
        img = Image.new("RGB", (50, 50), "white")
        regions = detect_regions(img)
        assert len(regions) >= 1
