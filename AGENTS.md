# Agent Operating Notes

## Arabic OCR Must-Do Rules

This repo is used for Arabic legal OCR. Do not take shortcuts.

1. Always inventory the source folder before extraction.
   - List all PDFs in the folder.
   - For each PDF, check page count, embedded text pages, and embedded character count with PyMuPDF.
   - If the user names one file, still check whether similarly named PDFs exist in the same folder so the target is not confused.

2. Never trust a PDF text layer until quality is checked.
   - Some Arabic PDFs have embedded text that is broken old-font glyph mapping.
   - A file with many embedded characters can still be unusable.
   - Use `arabic_ocr.native_pdf.extract_best_native_page_text` and inspect `quality_status`.
   - Spot-check actual output text, not just file size.

3. For scanned PDFs with `embedded_pages=0`, native/glyph extraction cannot recover text.
   - Use OCR. Do not create a Markdown/TXT file from empty embedded text.
   - If local OCR is unavailable, say so before spending cloud calls.
   - If the user asks for maximum accuracy, use available OCR engines and keep raw artifacts.

4. Arabic path handling on Windows is fragile.
   - Avoid hard-coded Arabic path literals inside Python scripts passed through PowerShell stdin.
   - Prefer enumerating the parent folder with `Path.glob("*.pdf")` and selecting by page/text characteristics.
   - If a Unicode filename must be referenced, let PowerShell pass it directly or derive output paths from the actual `Path` object.
   - In repo notes and generated scripts, prefer ASCII-safe artifact folder names.

5. Do not leave OCR/API jobs running after interruption.
   - After a user aborts a long OCR run, inspect `Get-CimInstance Win32_Process -Filter "name = 'python.exe'"`.
   - Stop only the relevant `arabic_ocr.cli` or extraction-script processes.
   - Do not kill unrelated MCP/background Python processes.

6. Control cloud cost explicitly.
   - The repo has cloud OCR engines: `openai`, `gemini_pro`, `gemini_flash`.
   - Cloud OCR is sometimes necessary for scanned Arabic legal documents, but it must not be used accidentally.
   - If the user says no LLM/API/cloud, use deterministic native/glyph and installed local OCR only.
   - If the user says "use everything" or "maximum accuracy", cloud OCR may be used, but keep page-level artifacts and report what was used.
   - For maximum-accuracy scanned Arabic legal OCR, run every currently available engine from `arabic_ocr.engines.registry.available_engine_ids()` plus any working local OCR such as Tesseract. Do not stop at only one cloud model.

7. Validate final files before reporting success.
   - Open the generated TXT/MD and inspect the first page and at least one middle page.
   - Check for empty output, numeric-only output, mojibake, broken glyph text, and repeated scanner garbage.
   - Confirm the final file exists in the requested folder and has plausible size.

## Known Tooling Findings

### Tesseract

- Installed binary:
  `C:\Program Files\Tesseract-OCR\tesseract.exe`
- It was installed by winget but was not on `PATH`.
- Initially it only had `eng` and `osd`; Arabic OCR was impossible until Arabic traineddata was added.
- Arabic traineddata was downloaded locally to:
  `C:\Users\narha\arabic-ocr\.tessdata\ara.traineddata`
- Use Tesseract with:
  `--tessdata-dir C:\Users\narha\arabic-ocr\.tessdata -l ara+eng`
- Tesseract is useful as a local validator/fallback, but on legal investigation scans it may garble low-contrast lines, handwriting, and dense forms. It should not be the only source when the user demands "squeaky clean" output.

### Paddle OCR

- `paddleocr` was not installed in the repo environment during the Dabbous investigation run.
- The repo environment uses Python 3.13, while Paddle OCR support is typically more reliable on older supported Python versions.
- If Paddle is needed, first check `py -0p`, create a separate Python 3.10 environment, and do not install large OCR stacks into the main repo venv blindly.

### Local binaries checked

- `tesseract`: installed but not on `PATH`
- `pdftotext`: missing
- `ocrmypdf`: missing
- `magick`: missing

## Specific Mistakes To Avoid

1. Do not generate "upload-ready" files from embedded text alone when the source is scanned or the extracted text is obviously broken.
2. Do not mark a folder "complete" only because every PDF has a matching `.md` file.
3. Do not skip newly added PDFs after an initial inventory.
4. Do not use OpenAI/Gemini context cleanup when the user asked to avoid LLM cleanup.
5. Do not accidentally spend cloud OCR calls after the user says to stop AI/API work.
6. Do not hard-code Arabic filenames in Python stdin scripts; this caused `????` path failures.
7. Do not leave interrupted OCR jobs running in the background.
8. Do not miss `gemini_flash` when it is available; this happened during the first Dabbous investigation run and had to be corrected afterward.

## Dabbous Investigation File Findings

Target folder:
`C:\Users\narha\Dropbox\Dabbous Allegations\Fraud Case`

Target file:
The 9-page scanned Public Prosecution investigation PDF in the target folder.
Do not hard-code the Arabic filename in Python scripts passed through
PowerShell stdin. Enumerate `*.pdf` and select the file with:
- `page_count == 9`
- `embedded_pages == 0`

Folder also contained:
- `criminal-60-ar.pdf` with 41 pages and embedded text.
- A Boubyan bank statement PDF with 59 pages and embedded text.
- The target investigation PDF with 9 pages and no embedded text.

The target investigation PDF is a pure scan:
- `pages=9`
- `embedded_pages=0`
- `embedded_chars=0`

Result files produced beside the PDF:
- `<target pdf stem>.clean_arabic.txt`
- `<target pdf stem>.clean_arabic.md`
- `<target pdf stem>.full_ocr_consolidated.clean_arabic.txt`
- `mahdr_dabbous_full_ocr_consolidated_audit.txt`
- `mahdr_dabbous_ocr_artifacts\`

Extraction method used:
- Local Tesseract Arabic (`ara+eng`) with multiple preprocessing variants and PSM modes.
- OpenAI OCR.
- Gemini Pro OCR.
- Gemini Flash OCR was added afterward to close the missed-engine gap.
- Page-level scoring/selection with raw outputs saved in JSON artifacts.
- Final consolidated TXT was manually adjudicated from all available sources: page 1 and page 7 from Gemini Pro, pages 2-6 and 8-9 from Gemini Flash, with cleanup corrections applied by `scripts/consolidate_dabbous_ocr.py`.

Important validation finding:
- Tesseract Arabic alone was not clean enough on page 1. It recognized many headers and Q/A lines, but garbled parts of the form.
- For "squeaky clean" legal scans, combine local OCR with cloud OCR and preserve artifacts for review.

## Preferred Output Pattern

For a single requested PDF:

1. Create final TXT in the same folder as the PDF:
   `<pdf stem>.clean_arabic.txt`
2. Optionally create a Markdown sibling:
   `<pdf stem>.clean_arabic.md`
3. Create audit artifacts in an ASCII-named folder:
   `<short_ascii_name>_ocr_artifacts\`
4. Include page markers:
   `===== Page N =====`
5. Keep every source page represented, even if a page has only partial readable text.

## Future Kid Disclosures Findings

Target folder:
`C:\Users\narha\Dropbox\Future Kid Valuation\Disclosures`

User instruction for this run:
- Do not call OpenAI, Gemini, cloud OCR, or any external LLM/API.
- Use deterministic/local extraction and local OCR only.

Observed source quality:
- The folder contains 244 PDF files.
- The files are overwhelmingly scanned/image PDFs; native/glyph PDF text extraction is mostly empty.
- Local OCR is therefore required for usable text.

Local-only outputs produced:
- Final TXT files:
  `C:\Users\narha\Dropbox\Future Kid Valuation\Disclosures\clean_txt\*.clean.txt`
- Output index:
  `C:\Users\narha\Dropbox\Future Kid Valuation\Disclosures\clean_txt\00_INDEX.txt`
- Local OCR audit:
  `C:\Users\narha\Dropbox\Future Kid Valuation\Disclosures\clean_txt\00_LOCAL_OCR_AUDIT.txt`
- Raw page OCR artifacts:
  `C:\Users\narha\Dropbox\Future Kid Valuation\Disclosures\ocr_audit_artifacts\local_tesseract\pages\`
- Summary JSON:
  `C:\Users\narha\Dropbox\Future Kid Valuation\Disclosures\ocr_audit_artifacts\local_tesseract\local_ocr_summary.json`

Pipeline script:
`scripts\futurekid_local_ocr_pipeline.py`

Validation from the completed local-only run:
- 244 PDFs mapped to 244 TXT outputs.
- 239 unique PDFs after byte-level deduplication.
- 969 unique pages OCRed.
- Page marker count matches PDF page count for every output.
- No `[NO TEXT EXTRACTED BY LOCAL OCR]` markers remain.
- Real mojibake markers `U+00D8`, `U+00D9`, and `U+FFFD` are zero.

Important limitation:
- Do not claim these Tesseract-only outputs are 100% character-perfect. They are local deterministic OCR outputs with no invented text, but some low-resolution scans still contain OCR misreads. Character-perfect certification requires manual page review or a better OCR engine; cloud/external LLM was explicitly disallowed for this run.

## Future Kid Financials Findings

Target folder:
`C:\Users\narha\Dropbox\Future Kid Valuation\Financials`

User instruction for this run:
- Do not call OpenAI, Gemini, cloud OCR, or any external LLM/API.
- The financial statements are in English; use deterministic/local extraction and local OCR only.

Observed source quality:
- The folder contains 34 PDF files.
- Total source coverage is 916 pages.
- Most files are scanned/image PDFs with no useful embedded text.
- A few files have partial embedded text, but native text still must be quality-checked and compared against OCR before use.

Local-only outputs produced:
- Final TXT files:
  `C:\Users\narha\Dropbox\Future Kid Valuation\Financials\clean_txt\*.clean.txt`
- Final Markdown files:
  `C:\Users\narha\Dropbox\Future Kid Valuation\Financials\clean_md\*.clean.md`
- Markdown index:
  `C:\Users\narha\Dropbox\Future Kid Valuation\Financials\clean_md\00_INDEX.md`
- Markdown local OCR audit:
  `C:\Users\narha\Dropbox\Future Kid Valuation\Financials\clean_md\00_LOCAL_OCR_AUDIT.md`
- Raw page OCR artifacts:
  `C:\Users\narha\Dropbox\Future Kid Valuation\Financials\ocr_audit_artifacts\local_tesseract_eng\pages\`
- Summary JSON:
  `C:\Users\narha\Dropbox\Future Kid Valuation\Financials\ocr_audit_artifacts\local_tesseract_eng\local_ocr_summary.json`

Pipeline script:
`scripts\futurekid_financials_local_ocr.py`

Validation from the completed local-only run:
- 34 PDFs mapped to 34 TXT outputs and 34 Markdown outputs.
- 916 pages represented.
- Page marker count matches PDF page count for every Markdown output.
- No `[NO TEXT EXTRACTED BY LOCAL OCR]` markers remain.
- Real mojibake markers `U+00D8`, `U+00D9`, and `U+FFFD` are zero.
- 9 documents had weak-page flags from local scoring; these flags are preserved in Markdown metadata and in `00_LOCAL_OCR_AUDIT.md`.

Important limitation:
- Do not claim these Tesseract-only English financial outputs are 100% character-perfect. They are the strongest local-only no-cloud outputs from this repo path, with no invented text, but low-contrast scans, table cells, seals, signatures, and dense footnotes can still contain OCR misreads.

## KW Trade WhatsApp Aleqtisadyah Findings

Target folder:
`C:\Users\narha\kw-trade-agent\WhatsApp Chat - الاقتصادية`

User instruction for this run:
- Extract all Arabic PDFs to clean Markdown at scale.
- Do not use AI verification, OpenAI, Gemini, cloud OCR, external LLMs, or external APIs.
- Use deterministic/local tooling only.

Observed source quality:
- The folder contains 468 PDF files.
- Total source coverage is 11,786 pages.
- There are 465 unique PDFs and 3 duplicate files by SHA256.
- Filenames are mojibake from WhatsApp export; do not trust names for Arabic metadata.
- Most PDFs have embedded text, but some pages contain broken custom-font glyph mappings, blank native text, Latin-extended glyph garbage, or `U+FFFD` replacement-character damage.
- Native/glyph extraction is usually strong, but must be quality-routed page by page.

Local-only outputs produced:
- Final Markdown files:
  `C:\Users\narha\kw-trade-agent\WhatsApp Chat - الاقتصادية\clean_md\*.clean.md`
- Markdown index:
  `C:\Users\narha\kw-trade-agent\WhatsApp Chat - الاقتصادية\clean_md\00_INDEX.md`
- Local extraction audit:
  `C:\Users\narha\kw-trade-agent\WhatsApp Chat - الاقتصادية\clean_md\00_LOCAL_EXTRACTION_AUDIT.md`
- Raw page artifacts:
  `C:\Users\narha\kw-trade-agent\WhatsApp Chat - الاقتصادية\ocr_audit_artifacts\local_native_tesseract_ar\pages\`
- Summary JSON:
  `C:\Users\narha\kw-trade-agent\WhatsApp Chat - الاقتصادية\ocr_audit_artifacts\local_native_tesseract_ar\local_extraction_summary.json`

Pipeline script:
`scripts\kw_trade_whatsapp_arabic_md.py`

Important implementation notes:
- Enumerate `C:\Users\narha\kw-trade-agent` and select the folder by `WhatsApp Chat - ` prefix; do not hard-code the Arabic folder path inside Python stdin scripts.
- Use numeric WhatsApp attachment prefixes such as `00000005` for output Markdown stems because source filenames are mojibake.
- The deterministic route is native/glyph extraction first, then local Tesseract `ara+eng` only for pages with failed/warned native quality, blank/very short text, heavy replacement-character damage, Latin-extended glyph noise, or high weird-glyph ratio.
- `extract_page_text_glyph_level` can raise an `IndexError` in `_detect_column_boundary` on some newspaper PDFs. The script catches native/glyph exceptions, falls back to raw PyMuPDF text, and routes weak pages to Tesseract.
- Windows redirected stdout may default to `cp1252`; configure stdout/stderr as UTF-8 before printing Arabic JSON.

Validation from the completed local-only run:
- 468 PDFs mapped to 468 Markdown outputs.
- 11,786 page headings represented.
- Page marker count matches PDF page count for every Markdown output.
- No `[NO TEXT EXTRACTED BY LOCAL PIPELINE]` markers remain.
- No `U+FFFD` replacement characters remain in the Markdown outputs.
- Summary JSON reports 48,426,219 extracted characters and 33,785,877 Arabic characters.

Important limitation:
- Do not claim the newspaper corpus is character-perfect. It is deterministic, local, upload-ready extraction with no invented text, but dense newspaper layouts, ads, stock tables, logos, and tiny captions may still contain OCR or extraction errors. Weak-page flags are retained for manual review.
