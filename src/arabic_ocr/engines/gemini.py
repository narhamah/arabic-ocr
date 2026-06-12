"""Gemini-based OCR engine adapters."""

from __future__ import annotations

import importlib.util

from PIL import Image

from arabic_ocr.engines.base import EngineOutput, OCREngine
from arabic_ocr.ocr_engines import ocr_gemini_flash, ocr_gemini_pro


class GeminiProEngine(OCREngine):
    id = "gemini_pro"
    name = "Gemini 2.5 Pro"
    local = False
    supports_layout = False
    supports_tables = False
    returns_bboxes = False
    returns_confidence = False

    def is_available(self) -> bool:
        return importlib.util.find_spec("google.genai") is not None

    def recognize(self, image: Image.Image) -> EngineOutput:
        result = ocr_gemini_pro(image)
        confidence = 0.75 if result["success"] and result["text"] else 0.0
        return EngineOutput(
            text=result["text"],
            success=result["success"],
            confidence=confidence,
            error=result["error"],
            metadata={"engine_id": self.id},
        )


class GeminiFlashEngine(OCREngine):
    id = "gemini_flash"
    name = "Gemini 2.5 Flash"
    local = False
    supports_layout = False
    supports_tables = False
    returns_bboxes = False
    returns_confidence = False

    def is_available(self) -> bool:
        return importlib.util.find_spec("google.genai") is not None

    def recognize(self, image: Image.Image) -> EngineOutput:
        result = ocr_gemini_flash(image)
        confidence = 0.65 if result["success"] and result["text"] else 0.0
        return EngineOutput(
            text=result["text"],
            success=result["success"],
            confidence=confidence,
            error=result["error"],
            metadata={"engine_id": self.id},
        )
