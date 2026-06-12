"""Best-effort adapter for Paddle PP-Structure style engines."""

from __future__ import annotations

import importlib

import numpy as np
from PIL import Image

from arabic_ocr.engines.base import EngineOutput, OCREngine


class PaddlePPStructureEngine(OCREngine):
    id = "paddle_ppstructure"
    name = "Paddle PP-Structure"
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
        structure_cls = getattr(module, "PPStructure", None) or getattr(module, "PPStructureV3", None)
        if structure_cls is None:
            return EngineOutput(
                text="",
                success=False,
                error="Installed paddleocr package does not expose PPStructure",
            )

        try:
            model = structure_cls(lang="ar")
            result = model(np.array(image))
        except Exception as exc:  # pragma: no cover - requires optional dependency
            return EngineOutput(text="", success=False, error=str(exc))

        texts: list[str] = []
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    if "res" in item and isinstance(item["res"], list):
                        for row in item["res"]:
                            if isinstance(row, dict) and row.get("text"):
                                texts.append(str(row["text"]))
                    elif item.get("text"):
                        texts.append(str(item["text"]))

        text = "\n".join(texts).strip()
        return EngineOutput(
            text=text,
            success=bool(text),
            confidence=0.8 if text else 0.0,
            metadata={"engine_id": self.id},
        )
