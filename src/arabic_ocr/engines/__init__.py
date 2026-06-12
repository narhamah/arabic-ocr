"""OCR engine registry and adapters."""

from arabic_ocr.engines.registry import available_engine_ids, get_engine, get_primary_secondary

__all__ = ["available_engine_ids", "get_engine", "get_primary_secondary"]
