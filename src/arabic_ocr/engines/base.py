"""Base interfaces for OCR engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass(slots=True)
class EngineOutput:
    """Normalized output returned by OCR engines."""

    text: str
    success: bool
    confidence: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OCREngine(ABC):
    """Abstract OCR engine contract."""

    id: str
    name: str
    local: bool
    supports_layout: bool
    supports_tables: bool
    returns_bboxes: bool
    returns_confidence: bool

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the engine is usable in the current environment."""

    @abstractmethod
    def recognize(self, image: Image.Image) -> EngineOutput:
        """Recognize text from an image."""
