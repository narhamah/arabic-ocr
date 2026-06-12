"""Generate a cleaned case-file bundle from saved provenance and case index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arabic_ocr.casefile_review import review_casefile_bundle
from arabic_ocr.exporters import document_from_json
from arabic_ocr.utils import load_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing provenance.json and case_index.json")
    parser.add_argument("--output-dir", help="Directory for reviewed outputs; defaults to input dir")
    parser.add_argument("--model", help="OpenAI model to use for review")
    parser.add_argument("--reasoning-effort", help="OpenAI reasoning effort to use for review, for example xhigh")
    args = parser.parse_args()

    load_env()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = input_dir / "provenance.json"
    case_index_path = input_dir / "case_index.json"
    checkpoint_path = output_dir / "review_checkpoint.json"
    if not provenance_path.exists():
        raise SystemExit(f"Missing provenance file: {provenance_path}")
    if not case_index_path.exists():
        raise SystemExit(f"Missing case index file: {case_index_path}")

    document = document_from_json(provenance_path.read_text(encoding="utf-8"))
    case_index = json.loads(case_index_path.read_text(encoding="utf-8"))
    existing_reviews = []
    if checkpoint_path.exists():
        existing_reviews = json.loads(checkpoint_path.read_text(encoding="utf-8")).get("reviewed_documents") or []

    def save_checkpoint(reviewed_documents: list[dict]) -> None:
        checkpoint_path.write_text(
            json.dumps({"reviewed_documents": reviewed_documents}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def progress_callback(event: str, payload: dict) -> None:
        if event == "casefile_review_document":
            segment_total = int(payload.get("segment_total") or 1)
            if segment_total > 1:
                print(
                    f"Reviewing document {payload['document_index']}/{payload['document_total']} "
                    f"segment {payload.get('segment_index', 1)}/{segment_total} "
                    f"(pages {payload['page_start']}-{payload['page_end']})"
                , flush=True)
            else:
                print(
                    f"Reviewing document {payload['document_index']}/{payload['document_total']} "
                    f"(pages {payload['page_start']}-{payload['page_end']})"
                , flush=True)
        elif event == "casefile_review_resume":
            print(
                f"Resuming completed document {payload['document_index']}/{payload['document_total']} "
                f"(pages {payload['page_start']}-{payload['page_end']})",
                flush=True,
            )

    reviewed = review_casefile_bundle(
        document,
        case_index,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        existing_reviews=existing_reviews,
        checkpoint_callback=save_checkpoint,
        progress_callback=progress_callback,
    )

    (output_dir / "reviewed_context.md").write_text(reviewed.reviewed_context_markdown, encoding="utf-8")
    (output_dir / "review_notes.md").write_text(reviewed.review_notes_markdown, encoding="utf-8")
    print(f"Reviewed files written to: {output_dir}")


if __name__ == "__main__":
    main()
