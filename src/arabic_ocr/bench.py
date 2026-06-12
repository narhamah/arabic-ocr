"""Benchmark helpers for OCR experiments."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from arabic_ocr.exporters import document_to_dict, render_text
from arabic_ocr.pipeline import process_pdf_document
from arabic_ocr.quality import token_stats


def run_benchmark(
    input_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    engine: str = "auto",
    profile: str = "llm",
    ground_truth_dir: str | Path | None = None,
) -> dict:
    """Run OCR over a directory of PDFs and capture basic metrics."""
    input_dir = Path(input_dir)
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    results = []
    for pdf_path in pdf_paths:
        start = perf_counter()
        document = process_pdf_document(
            str(pdf_path),
            config={"engine": engine, "profile": profile},
        )
        elapsed = perf_counter() - start
        text = render_text(document, profile=profile)
        metrics = {
            "pdf": str(pdf_path),
            "pages": len(document.pages),
            "seconds": elapsed,
            "token_stats": token_stats(text),
            "document": document_to_dict(document),
        }
        if ground_truth_dir is not None:
            truth_path = Path(ground_truth_dir) / f"{pdf_path.stem}.txt"
            if truth_path.exists():
                truth = truth_path.read_text(encoding="utf-8")
                metrics["cer"] = character_error_rate(truth, text)
                metrics["wer"] = word_error_rate(truth, text)
        results.append(metrics)

    summary = {
        "engine": engine,
        "profile": profile,
        "documents": results,
    }
    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute character error rate."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute word error rate."""
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return _levenshtein(reference_words, hypothesis_words) / len(reference_words)


def _levenshtein(reference, hypothesis) -> int:
    """Generic Levenshtein distance for sequences."""
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            insertions = previous[j] + 1
            deletions = current[j - 1] + 1
            substitutions = previous[j - 1] + (ref_item != hyp_item)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]
