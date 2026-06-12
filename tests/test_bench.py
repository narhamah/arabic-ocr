"""Tests for benchmark helpers."""

from pathlib import Path
from unittest.mock import patch

from arabic_ocr.bench import character_error_rate, run_benchmark, word_error_rate
from arabic_ocr.models import BoundingBox, OCRBlock, OCRDocument, OCRLine, OCRPage, OCRSpan


def _sample_document() -> OCRDocument:
    return OCRDocument(
        source_pdf="sample.pdf",
        dpi=400,
        pages=[
            OCRPage(
                page_number=0,
                width=800,
                height=1000,
                source_kind="scan",
                blocks=[
                    OCRBlock(
                        block_type="text",
                        bbox=BoundingBox(0, 0, 10, 10),
                        reading_order=0,
                        confidence=1.0,
                        engine_id="mock",
                        lines=[
                            OCRLine(
                                spans=[OCRSpan(text="hello world", confidence=1.0, engine_id="mock")],
                                confidence=1.0,
                                engine_id="mock",
                            )
                        ],
                    )
                ],
            )
        ],
    )


def test_character_error_rate_zero_for_match():
    assert character_error_rate("abc", "abc") == 0.0


def test_word_error_rate_nonzero_for_difference():
    assert word_error_rate("hello world", "hello there") > 0.0


@patch("arabic_ocr.bench.process_pdf_document")
def test_run_benchmark_writes_summary(mock_process, repo_tmp_path):
    input_dir = repo_tmp_path / "pdfs"
    input_dir.mkdir()
    pdf_path = input_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    output_path = repo_tmp_path / "summary.json"

    mock_process.return_value = _sample_document()
    summary = run_benchmark(input_dir, output_path=output_path)

    assert summary["documents"][0]["pdf"].endswith("sample.pdf")
    assert output_path.exists()
