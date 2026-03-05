"""Tests for PDF renderer — TDD RED phase."""

import pytest
from pathlib import Path
from PIL import Image
from arabic_ocr.renderer import render_pdf

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestRenderPDF:

    def test_renders_single_page_pdf(self):
        images = render_pdf(FIXTURES_DIR / "single_page.pdf")
        assert len(images) == 1

    def test_renders_multi_page_pdf(self):
        images = render_pdf(FIXTURES_DIR / "multi_page.pdf")
        assert len(images) == 3

    def test_image_is_pil_image(self):
        images = render_pdf(FIXTURES_DIR / "single_page.pdf")
        assert isinstance(images[0], Image.Image)

    def test_image_is_rgb(self):
        images = render_pdf(FIXTURES_DIR / "single_page.pdf")
        assert images[0].mode == "RGB"

    def test_image_dimensions_at_400dpi(self):
        """A4 at 400 DPI should be approximately 3307x4677 pixels."""
        images = render_pdf(FIXTURES_DIR / "single_page.pdf", dpi=400)
        w, h = images[0].size
        # Allow 10% tolerance
        assert 2900 < w < 3700
        assert 4200 < h < 5200

    def test_image_dimensions_at_200dpi(self):
        """Lower DPI should produce smaller images."""
        images_400 = render_pdf(FIXTURES_DIR / "single_page.pdf", dpi=400)
        images_200 = render_pdf(FIXTURES_DIR / "single_page.pdf", dpi=200)
        assert images_200[0].size[0] < images_400[0].size[0]
        assert images_200[0].size[1] < images_400[0].size[1]

    def test_specific_pages(self):
        images = render_pdf(FIXTURES_DIR / "multi_page.pdf", pages=[0, 2])
        assert len(images) == 2

    def test_invalid_pdf_raises(self):
        with pytest.raises(Exception):
            render_pdf(FIXTURES_DIR / "not_a_pdf.txt")

    def test_nonexistent_file_raises(self):
        with pytest.raises(Exception):
            render_pdf(FIXTURES_DIR / "does_not_exist.pdf")
