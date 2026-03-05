# arabic-ocr

High-accuracy Arabic OCR pipeline for scanned PDFs. Uses dual Gemini VLM models with Claude-based cross-verification for maximum accuracy.

## Architecture

```
Scanned PDF → Render (400 DPI) → Preprocess → Layout Detection →
  Dual VLM OCR (Gemini 2.5 Pro + 2.0 Flash) → Normalize → Cross-Verify (Claude) → Text
```

## Installation

```bash
pip install -e ".[dev]"
```

For layout detection (optional):
```bash
pip install -e ".[layout]"
```

## Configuration

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Required environment variables:
- `GEMINI_API_KEY` — Google Gemini API key (for OCR)
- `ANTHROPIC_API_KEY` — Anthropic API key (for cross-verification tiebreaker)

## Usage

### CLI

```bash
# Basic usage — extract text to stdout
arabic-ocr document.pdf

# Save to file
arabic-ocr document.pdf -o output.txt

# JSON output with metadata
arabic-ocr document.pdf -f json -o result.json

# Process specific pages (1-indexed)
arabic-ocr document.pdf -p "1-5"
arabic-ocr document.pdf -p "1,3,7"

# Custom DPI and preserve diacritics
arabic-ocr document.pdf --dpi 300 --keep-diacritics

# Verbose mode
arabic-ocr document.pdf -v
```

### Python API

```python
from arabic_ocr.pipeline import process_pdf

# Basic usage
results = process_pdf("document.pdf")
for page in results:
    print(f"Page {page['page']}: {page['text'][:100]}...")

# With configuration
results = process_pdf("document.pdf", config={
    "dpi": 300,
    "pages": [0, 1, 2],
    "strip_diacritics": False,
})
```

## Testing

```bash
# Unit tests (no API keys needed)
pytest -m "not integration" -v

# All tests with coverage
pytest --cov=arabic_ocr --cov-report=term-missing
```

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 0 | `renderer.py` | PDF → PIL Images at 400 DPI (PyMuPDF) |
| 1 | `preprocessor.py` | CLAHE + NLM denoise (h=8) + deskew |
| 2 | `layout.py` | Region detection + RTL sorting |
| 3 | `ocr_engines.py` | Dual Gemini OCR (original images) |
| 4 | `normalizer.py` | Unicode/Arabic text normalization |
| 5 | `verifier.py` | Cross-model verification + Claude tiebreaker |
