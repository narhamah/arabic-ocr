"""Build local-only cleaned and grouped outputs from OCR provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arabic_ocr.exporters import document_from_json
from arabic_ocr.local_casefile_bundle import (
    build_manual_case_index,
    clean_document,
    render_cleanup_report,
    render_cleaned_markdown,
    render_grouped_context,
    render_manual_case_index_markdown,
    render_ocr_risk_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing provenance.json from the OCR run")
    parser.add_argument("--output-dir", help="Directory for cleaned/grouped outputs; defaults to input dir")
    parser.add_argument("--preset", default="check_case01", help="Manual grouping preset to apply")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance_path = input_dir / "provenance.json"
    if not provenance_path.exists():
        raise SystemExit(f"Missing provenance file: {provenance_path}")

    document = document_from_json(provenance_path.read_text(encoding="utf-8"))
    cleanup = clean_document(document)
    case_index = build_manual_case_index(document, preset=args.preset)

    cleaned_context = render_cleaned_markdown(
        document,
        cleanup.cleaned_llm_pages,
        title="Cleaned OCR Context",
        intro="This file removes obvious scanner watermarks and hallucinated no-text filler lines only. No API or model review was used after OCR.",
    )
    cleaned_verbatim = render_cleaned_markdown(
        document,
        cleanup.cleaned_verbatim_pages,
        title="Cleaned OCR Verbatim",
        intro="This file preserves the locally cleaned page text with lighter normalization. No API or model review was used after OCR.",
    )

    (output_dir / "cleaned_context.md").write_text(cleaned_context, encoding="utf-8")
    (output_dir / "cleaned_verbatim.md").write_text(cleaned_verbatim, encoding="utf-8")
    (output_dir / "cleanup_report.md").write_text(
        render_cleanup_report(document, cleanup, source_dir=input_dir.name),
        encoding="utf-8",
    )
    (output_dir / "manual_case_index.json").write_text(
        json.dumps(case_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "manual_case_index.md").write_text(
        render_manual_case_index_markdown(case_index),
        encoding="utf-8",
    )
    (output_dir / "grouped_context.md").write_text(
        render_grouped_context(document, case_index, cleanup.cleaned_llm_pages),
        encoding="utf-8",
    )
    (output_dir / "ocr_risk_report.md").write_text(
        render_ocr_risk_report(case_index, cleanup),
        encoding="utf-8",
    )

    print(f"Wrote local bundle outputs to: {output_dir}")


if __name__ == "__main__":
    main()
