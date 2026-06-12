"""CLI entry point for arabic-ocr."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from arabic_ocr.exporters import document_to_json, render_markdown, render_text
from arabic_ocr.preparation import prepare_context_bundle
from arabic_ocr.utils import load_env


@click.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Primary output file path")
@click.option("--output-dir", type=click.Path(), help="Write verbatim/context/json outputs into a directory")
@click.option("--save-provenance-json", type=click.Path(), help="Write the canonical provenance JSON to this path")
@click.option("--save-debug-dir", type=click.Path(), help="Optional debug artifact directory")
@click.option("--pages", "-p", type=str, help='Page range, e.g. "1-5" or "1,3,7"')
@click.option("--dpi", default=400, help="Render DPI (default: 400)")
@click.option(
    "--engine",
    type=click.Choice(["auto", "openai", "paddleocr_vl", "paddle_ppstructure", "gemini"]),
    default="auto",
    show_default=True,
    help="OCR engine strategy",
)
@click.option(
    "--profile",
    type=click.Choice(["faithful", "llm"]),
    default="faithful",
    show_default=True,
    help="Normalization profile for the primary text output",
)
@click.option("--keep-diacritics", is_flag=True, help="Preserve diacritical marks")
@click.option("--field-refine/--no-field-refine", default=True, show_default=True, help="Run targeted OpenAI refinement on high-value lines")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json", "markdown"]),
    default="text",
    help="Primary output format",
)
@click.option(
    "--context-prep",
    type=click.Choice(["rule", "openai", "auto", "openai_casefile"]),
    default="rule",
    show_default=True,
    help="How to prepare the LLM-ready Markdown context output",
)
@click.option("--progress/--no-progress", default=True, show_default=True, help="Show page/stage progress on stderr")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.version_option(version="0.2.0")
def main(
    pdf_path: str,
    output: str | None,
    output_dir: str | None,
    save_provenance_json: str | None,
    save_debug_dir: str | None,
    pages: str | None,
    dpi: int,
    engine: str,
    profile: str,
    keep_diacritics: bool,
    field_refine: bool,
    output_format: str,
    context_prep: str,
    progress: bool,
    verbose: bool,
) -> None:
    """Extract Arabic text from scanned PDF with high accuracy."""
    load_env()
    output_dir_path = Path(output_dir) if output_dir else None

    from arabic_ocr.pipeline import process_pdf_document

    config: dict = {
        "dpi": dpi,
        "strip_diacritics": not keep_diacritics,
        "engine": engine,
        "profile": profile,
        "refine_fields": field_refine,
    }
    if progress:
        config["progress_callback"] = _build_progress_callback(verbose=verbose)
    if save_debug_dir:
        config["save_debug_dir"] = save_debug_dir
    if pages:
        try:
            config["pages"] = _parse_pages(pages)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--pages") from exc

    if verbose:
        click.echo(f"Processing: {pdf_path}", err=True)
        click.echo(
            f"DPI: {dpi}, Engine: {engine}, Profile: {profile}, Keep diacritics: {keep_diacritics}",
            err=True,
        )

    try:
        document = process_pdf_document(pdf_path, config=config)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if output_dir_path:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "verbatim.md").write_text(
            render_markdown(document, profile="faithful"),
            encoding="utf-8",
        )
        (output_dir_path / "provenance.json").write_text(
            document_to_json(document),
            encoding="utf-8",
        )

    prepared_bundle = None
    prepared_context_markdown = render_markdown(document, profile="llm")
    needs_prepared_context = bool(output_dir) or (output_format == "markdown" and profile == "llm")
    if needs_prepared_context:
        prepared_bundle = prepare_context_bundle(
            document,
            strategy=context_prep,
            progress_callback=config.get("progress_callback"),
        )
        prepared_context_markdown = prepared_bundle.context_markdown

    primary_output = _render_output(
        document=document,
        output_format=output_format,
        profile=profile,
        prepared_context_markdown=prepared_context_markdown,
    )

    if output_dir:
        (output_dir_path / "context.md").write_text(
            prepared_context_markdown,
            encoding="utf-8",
        )
        if prepared_bundle and prepared_bundle.case_packet_markdown:
            (output_dir_path / "case_packet.md").write_text(
                prepared_bundle.case_packet_markdown,
                encoding="utf-8",
            )
        if prepared_bundle and prepared_bundle.case_index_json():
            (output_dir_path / "case_index.json").write_text(
                prepared_bundle.case_index_json(),
                encoding="utf-8",
            )
        if verbose:
            click.echo(f"Output directory written to: {output_dir_path}", err=True)

    if save_provenance_json:
        Path(save_provenance_json).write_text(document_to_json(document), encoding="utf-8")
        if verbose:
            click.echo(f"Provenance JSON written to: {save_provenance_json}", err=True)

    if output:
        Path(output).write_text(primary_output, encoding="utf-8")
        if verbose:
            click.echo(f"Primary output written to: {output}", err=True)
    elif not output_dir:
        click.echo(primary_output)


def _render_output(*, document, output_format: str, profile: str, prepared_context_markdown: str) -> str:
    if output_format == "json":
        return document_to_json(document)
    if output_format == "markdown" and profile == "llm":
        return prepared_context_markdown
    if output_format == "markdown":
        return render_markdown(document, profile=profile)
    return render_text(document, profile=profile)


def _parse_pages(pages_str: str) -> list[int]:
    """Parse page range string like '1-5' or '1,3,7' to 0-indexed list."""
    result: list[int] = []
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            raise ValueError("Page list contains an empty segment")
        if "-" in part:
            start, end = part.split("-", 1)
            start_page = int(start)
            end_page = int(end)
            if start_page < 1 or end_page < 1:
                raise ValueError("Pages are 1-indexed and must be positive")
            if end_page < start_page:
                raise ValueError("Page ranges must be ascending")
            result.extend(range(start_page - 1, end_page))
        else:
            page_number = int(part)
            if page_number < 1:
                raise ValueError("Pages are 1-indexed and must be positive")
            result.append(page_number - 1)
    return sorted(dict.fromkeys(result))


def _build_progress_callback(*, verbose: bool):
    def callback(event: str, payload: dict) -> None:
        page_number = payload.get("page_number")
        page_label = f"page {int(page_number) + 1}" if page_number is not None else "document"
        if event == "document_start":
            click.echo(
                f"Starting OCR for {payload['page_total']} page(s) with engine mode '{payload['engine_mode']}'",
                err=True,
            )
        elif event == "page_embedded_text":
            click.echo(f"[{payload['page_index']}/{payload['page_total']}] {page_label}: using embedded text", err=True)
        elif event == "page_scan_start":
            click.echo(f"[{payload['page_index']}/{payload['page_total']}] {page_label}: scan OCR started", err=True)
        elif event == "page_layout_detected":
            click.echo(
                f"[{payload['page_index']}/{payload['page_total']}] {page_label}: {payload['region_count']} region(s) detected",
                err=True,
            )
        elif event == "page_complete":
            click.echo(f"[{payload['page_index']}/{payload['page_total']}] {page_label}: done", err=True)
        elif event == "context_preparation_page":
            click.echo(
                f"Preparing context with OpenAI: page {int(payload['page_number']) + 1} ({payload['page_index']}/{payload['page_total']})",
                err=True,
            )
        elif event == "casefile_preparation_start":
            click.echo(
                f"Preparing case bundle with OpenAI across {payload['page_total']} page(s)",
                err=True,
            )
        elif event == "casefile_preparation_complete":
            click.echo(
                f"Case bundle preparation complete: {payload['document_total']} logical document(s) inferred",
                err=True,
            )
        elif verbose and event == "field_refinement":
            click.echo(
                f"[{payload['page_index']}/{payload['page_total']}] page {int(payload['page_number']) + 1}: refine line {int(payload['line_index']) + 1} ({payload['field_label']})",
                err=True,
            )
        elif event == "document_complete":
            click.echo(f"Completed OCR for {payload['page_total']} page(s)", err=True)
        elif verbose and event == "page_preprocessed":
            click.echo(f"[{payload['page_index']}/{payload['page_total']}] {page_label}: preprocessing complete", err=True)
        elif verbose and event == "block_start":
            click.echo(
                f"[{payload['page_index']}/{payload['page_total']}] {page_label}: region {payload['block_index']}/{payload['block_total']} ({payload['block_type']})",
                err=True,
            )

    return callback


if __name__ == "__main__":
    main()
