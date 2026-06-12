"""OpenAI Responses API OCR engine adapter."""

from __future__ import annotations

import importlib

from PIL import Image

from arabic_ocr.engines.base import EngineOutput, OCREngine
from arabic_ocr.ocr_engines import ocr_openai


class OpenAIVisionEngine(OCREngine):
    id = "openai"
    name = "OpenAI Vision OCR"
    local = False
    supports_layout = False
    supports_tables = False
    returns_bboxes = False
    returns_confidence = False

    def is_available(self) -> bool:
        return importlib.util.find_spec("openai") is not None

    def recognize(self, image: Image.Image) -> EngineOutput:
        result = ocr_openai(image)
        confidence = 0.8 if result["success"] and result["text"] else 0.0
        return EngineOutput(
            text=result["text"],
            success=result["success"],
            confidence=confidence,
            error=result["error"],
            metadata={"engine_id": self.id},
        )
