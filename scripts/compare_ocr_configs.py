"""Run bounded OCR comparisons across multiple configs on representative PDFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from arabic_ocr.cli import _parse_pages
from arabic_ocr.exporters import document_to_json, render_markdown, render_text
from arabic_ocr.pipeline import process_pdf_document
from arabic_ocr.quality import (
    extract_high_value_fields,
    looks_like_hallucinated_ocr_text,
    ocr_quality_score,
    token_overlap_ratio,
    token_stats,
)
from arabic_ocr.utils import load_env

_REPO_DEMO_SPECS = [
    ("ملف قضية الشيك الكامل.pdf", "3,24,80"),
    (str(Path("source") / "ملف جنايات نواف ارحمه.pdf"), "3,32,95"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path("out_bench_compare")),
        help="Directory for benchmark outputs",
    )
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        help="OCR config to benchmark; repeatable. Defaults to auto, openai, gemini.",
    )
    parser.add_argument(
        "--spec",
        action="append",
        help="Benchmark spec in the form path|1,5,9 or path|1-3",
    )
    parser.add_argument(
        "--preset",
        choices=["repo_demo"],
        default="repo_demo",
        help="Predefined benchmark document/page set to use when --spec is omitted",
    )
    parser.add_argument(
        "--profile",
        default="llm",
        choices=["faithful", "llm"],
        help="Normalization profile for saved outputs",
    )
    parser.add_argument(
        "--no-field-refine",
        action="store_true",
        help="Disable targeted OpenAI field refinement during benchmarking",
    )
    args = parser.parse_args()

    load_env()

    configs = args.configs or ["auto", "openai", "gemini"]
    specs = _parse_specs(args.spec, preset=args.preset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "configs": configs,
        "profile": args.profile,
        "specs": [{"pdf": pdf, "pages": pages_spec} for pdf, pages_spec in specs],
        "documents": [],
    }

    for pdf_path_str, pages_spec in specs:
        pdf_path = Path(pdf_path_str)
        page_indices = _parse_pages(pages_spec)
        page_numbers = [page + 1 for page in page_indices]
        pdf_output_dir = output_dir / _safe_stem(pdf_path)
        pdf_output_dir.mkdir(parents=True, exist_ok=True)
        pdf_summary = {
            "pdf": str(pdf_path),
            "pages": page_numbers,
            "configs": [],
        }

        rendered_by_config: dict[str, str] = {}
        for config_name in configs:
            run_output_dir = pdf_output_dir / config_name
            run_output_dir.mkdir(parents=True, exist_ok=True)
            start = perf_counter()
            document = process_pdf_document(
                str(pdf_path),
                config={
                    "engine": config_name,
                    "profile": args.profile,
                    "pages": page_indices,
                    "refine_fields": not args.no_field_refine,
                },
            )
            elapsed = perf_counter() - start

            rendered_text = render_text(document, profile=args.profile)
            rendered_markdown = render_markdown(document, profile=args.profile)
            rendered_by_config[config_name] = rendered_text
            (run_output_dir / "verbatim.txt").write_text(rendered_text, encoding="utf-8")
            (run_output_dir / "context.md").write_text(rendered_markdown, encoding="utf-8")
            (run_output_dir / "provenance.json").write_text(document_to_json(document), encoding="utf-8")

            metrics = _document_metrics(document, seconds=elapsed)
            (run_output_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            pdf_summary["configs"].append({
                "name": config_name,
                "metrics": metrics,
            })

        pdf_summary["pairwise_overlap"] = _pairwise_overlap(rendered_by_config)
        (pdf_output_dir / "summary.json").write_text(
            json.dumps(pdf_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["documents"].append(pdf_summary)

    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote OCR comparison outputs to: {output_dir}")


def _parse_specs(raw_specs: list[str] | None, *, preset: str | None) -> list[tuple[str, str]]:
    if raw_specs:
        specs = []
        for item in raw_specs:
            if "|" not in item:
                raise SystemExit(f"Invalid --spec '{item}'. Expected path|pages.")
            path_text, pages_spec = item.split("|", 1)
            specs.append((path_text.strip(), pages_spec.strip()))
        return specs
    if preset == "repo_demo":
        return list(_REPO_DEMO_SPECS)
    raise SystemExit("No benchmark specs were provided.")


def _safe_stem(path: Path) -> str:
    return "".join(char if char.isalnum() else "_" for char in path.stem).strip("_") or "document"


def _document_metrics(document, *, seconds: float) -> dict[str, object]:
    page_metrics = []
    full_text = render_text(document, profile="llm")
    field_totals = {"case_numbers": 0, "dates": 0, "articles": 0}
    hallucinated_line_total = 0
    for page in document.pages:
        fields = extract_high_value_fields(page.text)
        for field_name, values in fields.items():
            field_totals[field_name] += len(values)
        hallucinated_line_total += sum(
            1 for line in page.text.splitlines() if looks_like_hallucinated_ocr_text(line)
        )
        page_metrics.append({
            "page": page.page_number + 1,
            "confidence": page.confidence,
            "quality_score": ocr_quality_score(page.text),
            "token_stats": token_stats(page.text),
            "block_count": len(page.blocks),
            "field_counts": {name: len(values) for name, values in fields.items()},
        })

    avg_page_confidence = mean(page["confidence"] for page in page_metrics) if page_metrics else 0.0
    avg_page_quality = mean(page["quality_score"] for page in page_metrics) if page_metrics else -1.0
    return {
        "seconds": round(seconds, 3),
        "page_count": len(document.pages),
        "avg_page_confidence": round(avg_page_confidence, 4),
        "avg_page_quality_score": round(avg_page_quality, 4),
        "token_stats": token_stats(full_text),
        "field_totals": field_totals,
        "hallucinated_line_total": hallucinated_line_total,
        "page_metrics": page_metrics,
    }


def _pairwise_overlap(rendered_by_config: dict[str, str]) -> list[dict[str, object]]:
    names = sorted(rendered_by_config)
    comparisons: list[dict[str, object]] = []
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            left_text = rendered_by_config[left_name]
            right_text = rendered_by_config[right_name]
            comparisons.append({
                "left": left_name,
                "right": right_name,
                "left_to_right_token_overlap": round(token_overlap_ratio(left_text, right_text), 4),
                "right_to_left_token_overlap": round(token_overlap_ratio(right_text, left_text), 4),
            })
    return comparisons


if __name__ == "__main__":
    main()
