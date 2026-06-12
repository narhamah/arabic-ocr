"""Typed document models used across the OCR pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(slots=True)
class BoundingBox:
    """Axis-aligned bounding box."""

    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_tuple(cls, bbox: tuple[int, int, int, int]) -> "BoundingBox":
        return cls(*bbox)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BoundingBox | None":
        if not data:
            return None
        return cls(
            x1=int(data["x1"]),
            y1=int(data["y1"]),
            x2=int(data["x2"]),
            y2=int(data["y2"]),
        )

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict[str, int]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


@dataclass(slots=True)
class OCRSpan:
    """Fine-grained recognized text span."""

    text: str
    confidence: float | None = None
    bbox: BoundingBox | None = None
    engine_id: str | None = None
    adjudicated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OCRSpan":
        return cls(
            text=str(data.get("text", "")),
            confidence=data.get("confidence"),
            bbox=BoundingBox.from_dict(data.get("bbox")),
            engine_id=data.get("engine_id"),
            adjudicated=bool(data.get("adjudicated", False)),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "engine_id": self.engine_id,
            "adjudicated": self.adjudicated,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class OCRLine:
    """OCR line made of spans."""

    spans: list[OCRSpan] = field(default_factory=list)
    confidence: float | None = None
    bbox: BoundingBox | None = None
    engine_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(span.text for span in self.spans).strip()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OCRLine":
        return cls(
            spans=[OCRSpan.from_dict(span) for span in data.get("spans") or []],
            confidence=data.get("confidence"),
            bbox=BoundingBox.from_dict(data.get("bbox")),
            engine_id=data.get("engine_id"),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "engine_id": self.engine_id,
            "spans": [span.to_dict() for span in self.spans],
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class OCRBlock:
    """Page block such as text, table, header, or stamp."""

    block_type: str
    bbox: BoundingBox
    reading_order: int
    lines: list[OCRLine] = field(default_factory=list)
    confidence: float | None = None
    engine_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text).strip()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OCRBlock":
        return cls(
            block_type=str(data.get("type", "text")),
            bbox=BoundingBox.from_dict(data.get("bbox")) or BoundingBox(0, 0, 0, 0),
            reading_order=int(data.get("reading_order", 0)),
            lines=[OCRLine.from_dict(line) for line in data.get("lines") or []],
            confidence=data.get("confidence"),
            engine_id=data.get("engine_id"),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.block_type,
            "bbox": self.bbox.to_dict(),
            "reading_order": self.reading_order,
            "confidence": self.confidence,
            "engine_id": self.engine_id,
            "text": self.text,
            "lines": [line.to_dict() for line in self.lines],
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class OCRPage:
    """Structured OCR page."""

    page_number: int
    width: int
    height: int
    source_kind: str
    blocks: list[OCRBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        ordered_blocks = sorted(self.blocks, key=lambda block: block.reading_order)
        return "\n\n".join(block.text for block in ordered_blocks if block.text).strip()

    @property
    def confidence(self) -> float:
        confidences = [block.confidence for block in self.blocks if block.confidence is not None]
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OCRPage":
        return cls(
            page_number=int(data.get("page", 0)),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            source_kind=str(data.get("source_kind", "scan")),
            blocks=[OCRBlock.from_dict(block) for block in data.get("blocks") or []],
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page_number,
            "width": self.width,
            "height": self.height,
            "source_kind": self.source_kind,
            "confidence": self.confidence,
            "text": self.text,
            "blocks": [block.to_dict() for block in self.blocks],
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class OCRDocument:
    """Canonical structured OCR document."""

    source_pdf: str
    dpi: int
    pages: list[OCRPage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OCRDocument":
        return cls(
            source_pdf=str(data.get("source_pdf", "")),
            dpi=int(data.get("dpi", 400)),
            pages=[OCRPage.from_dict(page) for page in data.get("pages") or []],
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_pdf": self.source_pdf,
            "dpi": self.dpi,
            "metadata": dict(self.metadata),
            "pages": [page.to_dict() for page in self.pages],
        }


@dataclass(slots=True)
class PageImages:
    """Two-track page imagery used during OCR."""

    raw: Image.Image | None = None
    enhanced: Image.Image | None = None


@dataclass(slots=True)
class PageAsset:
    """Internal page asset produced during PDF triage."""

    page_number: int
    width: int
    height: int
    embedded_text: str = ""
    images: PageImages = field(default_factory=PageImages)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_embedded_text(self) -> bool:
        return bool(self.embedded_text.strip())


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for OCR processing."""

    dpi: int = 400
    pages: list[int] | None = None
    strip_diacritics: bool = True
    engine: str = "auto"
    profile: str = "faithful"
    use_embedded_text: bool = True
    compare_native_ocr: bool = True
    verify_disagreements: bool = True
    refine_fields: bool = True
    save_debug_dir: Path | None = None
    progress_callback: ProgressCallback | None = None

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None) -> "PipelineConfig":
        if not config:
            return cls()
        return cls(
            dpi=int(config.get("dpi", 400)),
            pages=config.get("pages"),
            strip_diacritics=bool(config.get("strip_diacritics", True)),
            engine=str(config.get("engine", "auto")),
            profile=str(config.get("profile", "faithful")),
            use_embedded_text=bool(config.get("use_embedded_text", True)),
            compare_native_ocr=bool(config.get("compare_native_ocr", True)),
            verify_disagreements=bool(config.get("verify_disagreements", True)),
            refine_fields=bool(config.get("refine_fields", True)),
            save_debug_dir=Path(config["save_debug_dir"]) if config.get("save_debug_dir") else None,
            progress_callback=config.get("progress_callback"),
        )
