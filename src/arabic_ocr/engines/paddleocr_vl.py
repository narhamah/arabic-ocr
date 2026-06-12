"""Best-effort adapter for a PaddleOCR-VL style engine."""

from __future__ import annotations

import importlib

import numpy as np
from PIL import Image

from arabic_ocr.engines.base import EngineOutput, OCREngine


class PaddleOCRVLEngine(OCREngine):
    id = "paddleocr_vl"
    name = "PaddleOCR-VL"
    local = True
    supports_layout = True
    supports_tables = True
    returns_bboxes = True
    returns_confidence = False

    def is_available(self) -> bool:
        return importlib.util.find_spec("paddleocr") is not None

    def recognize(self, image: Image.Image) -> EngineOutput:
        if not self.is_available():
            return EngineOutput(
                text="",
                success=False,
                error="Optional dependency 'paddleocr' is not installed",
            )

        module = importlib.import_module("paddleocr")
        ocr_cls = getattr(module, "PaddleOCR", None)
        if ocr_cls is None:
            return EngineOutput(
                text="",
                success=False,
                error="Installed paddleocr package does not expose PaddleOCR",
            )

        try:
            model = ocr_cls(lang="ar")
            result = model.ocr(np.array(image))
        except Exception as exc:  # pragma: no cover - requires optional dependency
            return EngineOutput(text="", success=False, error=str(exc))

        texts: list[str] = []
        if isinstance(result, list):
            for line_group in result:
                if not isinstance(line_group, list):
                    continue
                for item in line_group:
                    if (
                        isinstance(item, (list, tuple))
                        and len(item) >= 2
                        and isinstance(item[1], (list, tuple))
                        and item[1]
                    ):
                        texts.append(str(item[1][0]))

        text = "\n".join(texts).strip()
        return EngineOutput(
            text=text,
            success=bool(text),
            confidence=0.85 if text else 0.0,
            metadata={"engine_id": self.id},
        )
