"""CLI entry point for arabic-ocr."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from arabic_ocr.utils import load_env


@click.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--pages", "-p", type=str, help='Page range, e.g. "1-5" or "1,3,7"')
@click.option("--dpi", default=400, help="Render DPI (default: 400)")
@click.option("--keep-diacritics", is_flag=True, help="Preserve diacritical marks")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.version_option(version="0.1.0")
def main(
    pdf_path: str,
    output: str | None,
    pages: str | None,
    dpi: int,
    keep_diacritics: bool,
    output_format: str,
    verbose: bool,
) -> None:
    """Extract Arabic text from scanned PDF with high accuracy."""
    load_env()

    from arabic_ocr.pipeline import process_pdf

    config: dict = {"dpi": dpi, "strip_diacritics": not keep_diacritics}

    if pages:
        config["pages"] = _parse_pages(pages)

    if verbose:
        click.echo(f"Processing: {pdf_path}", err=True)
        click.echo(f"DPI: {dpi}, Keep diacritics: {keep_diacritics}", err=True)

    try:
        results = process_pdf(pdf_path, config=config)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        output_text = json.dumps(results, ensure_ascii=False, indent=2)
    else:
        output_text = "\n\n---\n\n".join(r["text"] for r in results)

    if output:
        Path(output).write_text(output_text, encoding="utf-8")
        if verbose:
            click.echo(f"Output written to: {output}", err=True)
    else:
        click.echo(output_text)


def _parse_pages(pages_str: str) -> list[int]:
    """Parse page range string like '1-5' or '1,3,7' to 0-indexed list."""
    result = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start) - 1, int(end)))
        else:
            result.append(int(part) - 1)
    return result
