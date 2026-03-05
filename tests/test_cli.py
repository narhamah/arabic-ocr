"""Tests for CLI interface."""

import pytest
from unittest.mock import patch
from click.testing import CliRunner
from pathlib import Path
from arabic_ocr.cli import main, _parse_pages

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestParsePages:

    def test_single_page(self):
        assert _parse_pages("1") == [0]

    def test_comma_separated(self):
        assert _parse_pages("1,3,7") == [0, 2, 6]

    def test_range(self):
        assert _parse_pages("1-3") == [0, 1, 2]

    def test_mixed(self):
        assert _parse_pages("1,3-5") == [0, 2, 3, 4]


class TestCLI:

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Extract Arabic text" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_nonexistent_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["/nonexistent/file.pdf"])
        assert result.exit_code != 0

    @patch("arabic_ocr.pipeline.process_pdf")
    def test_text_output(self, mock_process):
        mock_process.return_value = [{"page": 0, "text": "Arabic text", "confidence": 1.0, "regions": 1}]
        runner = CliRunner()
        result = runner.invoke(main, [str(FIXTURES_DIR / "single_page.pdf")])
        assert result.exit_code == 0
        assert "Arabic text" in result.output

    @patch("arabic_ocr.pipeline.process_pdf")
    def test_json_output(self, mock_process):
        mock_process.return_value = [{"page": 0, "text": "Arabic text", "confidence": 1.0, "regions": 1}]
        runner = CliRunner()
        result = runner.invoke(main, [str(FIXTURES_DIR / "single_page.pdf"), "-f", "json"])
        assert result.exit_code == 0
        assert '"text"' in result.output

    @patch("arabic_ocr.pipeline.process_pdf")
    def test_output_file(self, mock_process, tmp_path):
        mock_process.return_value = [{"page": 0, "text": "Output text", "confidence": 1.0, "regions": 1}]
        out_file = tmp_path / "output.txt"
        runner = CliRunner()
        result = runner.invoke(main, [str(FIXTURES_DIR / "single_page.pdf"), "-o", str(out_file)])
        assert result.exit_code == 0
        assert out_file.read_text() == "Output text"

    @patch("arabic_ocr.pipeline.process_pdf")
    def test_verbose_flag(self, mock_process):
        mock_process.return_value = [{"page": 0, "text": "", "confidence": 1.0, "regions": 1}]
        runner = CliRunner()
        result = runner.invoke(main, [str(FIXTURES_DIR / "single_page.pdf"), "-v"])
        assert result.exit_code == 0
