# arabic-ocr

Arabic OCR pipeline for scanned legal and court PDFs.

The repo now uses a structured-document pipeline:

1. inspect PDF pages and prefer embedded text when available
2. render scanned pages with PyMuPDF
3. build raw and enhanced image tracks
4. detect layout regions in RTL reading order
5. run pluggable OCR engines
6. merge or adjudicate disputed spans
7. export faithful text, LLM-ready text, and provenance JSON

## Runtime

- Supported Python: `>=3.10`
- Recommended for serious OCR stacks: Python `3.11`
- Python `3.13` works for the lightweight/tested path in this repo, but optional OCR engines such as Paddle-based stacks may require a separate environment

## Install

Core:

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[cloud]"
pip install -e ".[opencv]"
pip install -e ".[layout]"
pip install -e ".[paddle]"
```

## Environment

Copy `.env.example` to `.env` and set keys only for the cloud paths you use.

```bash
cp .env.example .env
```

Variables:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`

## CLI

Basic text output:

```bash
arabic-ocr document.pdf
```

LLM-ready Markdown:

```bash
arabic-ocr document.pdf --profile llm --format markdown -o context.md
```

OpenAI-assisted OCR plus OpenAI context preparation:

```bash
arabic-ocr document.pdf --engine openai --profile llm --format markdown --context-prep openai
```

Whole-bundle case-file organization for counsel review:

```bash
arabic-ocr document.pdf --engine openai --profile llm --format markdown --context-prep openai_casefile --output-dir out_casefile
```

Structured output bundle:

```bash
arabic-ocr document.pdf --output-dir out --engine auto
```

This writes:

- `out/verbatim.md`
- `out/context.md`
- `out/provenance.json`

When `--context-prep openai_casefile` is used, the output directory also includes:

- `out/case_packet.md`
- `out/case_index.json`

Useful options:

```bash
arabic-ocr document.pdf \
  --engine auto \
  --profile faithful \
  --format json \
  --context-prep rule \
  --save-provenance-json result.json \
  --save-debug-dir debug \
  -p "1-5"
```

Progress is shown on stderr by default so long runs do not look stuck. Use `--no-progress` to suppress it.

Quality notes:

- `--engine auto` now prefers cloud VLM OCR when configured, then falls back to local Paddle adapters
- low-quality regions are retried with higher-resolution and rotated crops before the block is finalized
- oversized sparse regions are retried as overlapping horizontal OCR chunks before rescue gives up on them

## Python API

Legacy page summaries:

```python
from arabic_ocr.pipeline import process_pdf

pages = process_pdf("document.pdf", config={"engine": "auto", "profile": "llm"})
```

Canonical structured document:

```python
from arabic_ocr.pipeline import process_pdf_document
from arabic_ocr.exporters import render_markdown, document_to_json

document = process_pdf_document("document.pdf", config={"engine": "auto"})
context_markdown = render_markdown(document, profile="llm")
provenance_json = document_to_json(document)
```

## Architecture

Main modules:

- `renderer.py`: PDF inspection and rendering
- `preprocessor.py`: optional OpenCV-backed enhancement with safe fallback
- `layout.py`: layout detection plus RTL ordering
- `engines/`: pluggable OCR engine adapters
- `pipeline.py`: document orchestration, merge heuristics, chunked large-region fallback, and low-quality OCR rescue passes
- `normalizer.py`: faithful and LLM normalization profiles
- `preparation.py`: optional OpenAI-assisted context preparation
- `preparation.py`: page-level cleanup plus whole-bundle case-file organization
- `verifier.py`: disagreement merge with OpenAI-first tiebreaking and Claude fallback
- `exporters.py`: text, markdown, and JSON outputs
- `bench.py`: benchmark helpers

## Testing

Default tests:

```bash
pytest -q
```

Explicit unit-only run:

```bash
pytest -q -m "not integration"
```

Live API tests are marked `integration` and `slow` and are excluded from the default test run. Run them explicitly:

```bash
pytest -q -m integration tests/test_live_api.py
```

## Notes

- Paddle adapters are wired as optional best-effort integrations so the repo stays installable without heavyweight OCR stacks.
- OpenAI is the preferred cloud path for OCR improvement and context preparation. Gemini remains optional and uses the maintained `google.genai` SDK.
- The canonical source of truth is the structured JSON document model. Text and Markdown outputs are derived from it.
- Debug image export is page-level today and is intended to support future region/span inspection flows.
