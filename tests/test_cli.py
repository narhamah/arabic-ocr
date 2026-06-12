"""Tests for CLI interface."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from arabic_ocr.cli import _parse_pages, main
from arabic_ocr.models import BoundingBox, OCRBlock, OCRDocument, OCRLine, OCRPage, OCRSpan
from arabic_ocr.preparation import PreparedContextBundle

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _sample_document() -> OCRDocument:
    page = OCRPage(
        page_number=0,
        width=800,
        height=1000,
        source_kind="scan",
        blocks=[
            OCRBlock(
                block_type="text",
                bbox=BoundingBox(0, 0, 800, 1000),
                reading_order=0,
                confidence=1.0,
                engine_id="mock",
                lines=[
                    OCRLine(
                        spans=[OCRSpan(text="Arabic text", confidence=1.0, engine_id="mock")],
                        confidence=1.0,
                        engine_id="mock",
                    )
                ],
            )
        ],
    )
    return OCRDocument(source_pdf=str(FIXTURES_DIR / "single_page.pdf"), dpi=400, pages=[page])


class TestParsePages:

    def test_single_page(self):
        assert _parse_pages("1") == [0]

    def test_comma_separated(self):
        assert _parse_pages("1,3,7") == [0, 2, 6]

    def test_range(self):
        assert _parse_pages("1-3") == [0, 1, 2]

    def test_mixed(self):
        assert _parse_pages("1,3-5") == [0, 2, 3, 4]

    def test_invalid_zero_page(self):
        try:
            _parse_pages("0")
        except ValueError as exc:
            assert "positive" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("Expected ValueError for page 0")


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
        assert "0.2.0" in result.output

    def test_nonexistent_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["/nonexistent/file.pdf"])
        assert result.exit_code != 0

    @patch("arabic_ocr.pipeline.process_pdf_document")
    def test_text_output(self, mock_process):
        mock_process.return_value = _sample_document()
        runner = CliRunner()
        result = runner.invoke(main, [str(FIXTURES_DIR / "single_page.pdf")])
        assert result.exit_code == 0
        assert "Arabic text" in result.output
        assert "Page 1" in result.output

    @patch("arabic_ocr.pipeline.process_pdf_document")
    def test_json_output(self, mock_process):
        mock_process.return_value = _sample_document()
        runner = CliRunner()
        result = runner.invoke(main, [str(FIXTURES_DIR / "single_page.pdf"), "-f", "json"])
        assert result.exit_code == 0
        assert '"source_pdf"' in result.output

    @patch("arabic_ocr.pipeline.process_pdf_document")
    def test_output_file(self, mock_process, repo_tmp_path):
        mock_process.return_value = _sample_document()
        out_file = repo_tmp_path / "output.txt"
        runner = CliRunner()
        result = runner.invoke(main, [str(FIXTURES_DIR / "single_page.pdf"), "-o", str(out_file)])
        assert result.exit_code == 0
        assert "Arabic text" in out_file.read_text()

    @patch("arabic_ocr.pipeline.process_pdf_document")
    def test_output_dir(self, mock_process, repo_tmp_path):
        mock_process.return_value = _sample_document()
        output_dir = repo_tmp_path / "bundle"
        runner = CliRunner()
        result = runner.invoke(main, [str(FIXTURES_DIR / "single_page.pdf"), "--output-dir", str(output_dir)])
        assert result.exit_code == 0
        assert (output_dir / "verbatim.md").exists()
        assert (output_dir / "context.md").exists()
        assert (output_dir / "provenance.json").exists()

    @patch("arabic_ocr.cli.prepare_context_bundle")
    @patch("arabic_ocr.pipeline.process_pdf_document")
    def test_output_dir_casefile_writes_extra_artifacts(self, mock_process, mock_prepare_bundle, repo_tmp_path):
        mock_process.return_value = _sample_document()
        mock_prepare_bundle.return_value = PreparedContextBundle(
            context_markdown="# Case File Bundle",
            case_packet_markdown="# Defense Workbench",
            case_index={"bundle_title": "Case bundle", "documents": []},
        )
        output_dir = repo_tmp_path / "bundle_casefile"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(FIXTURES_DIR / "single_page.pdf"),
                "--output-dir",
                str(output_dir),
                "--context-prep",
                "openai_casefile",
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "case_packet.md").exists()
        assert (output_dir / "case_index.json").exists()

    @patch("arabic_ocr.pipeline.process_pdf_document")
    def test_verbose_flag(self, mock_process):
        mock_process.return_value = _sample_document()
        runner = CliRunner()
        result = runner.invoke(main, [str(FIXTURES_DIR / "single_page.pdf"), "-v"])
        assert result.exit_code == 0

    def test_invalid_page_argument(self):
        runner = CliRunner()
        result = runner.invoke(main, [str(FIXTURES_DIR / "single_page.pdf"), "-p", "0"])
        assert result.exit_code != 0
        assert "positive" in result.output
