"""Engine registry and routing helpers."""

from __future__ import annotations

import os

from arabic_ocr.engines.base import OCREngine
from arabic_ocr.engines.gemini import GeminiFlashEngine, GeminiProEngine
from arabic_ocr.engines.openai_vision import OpenAIVisionEngine
from arabic_ocr.engines.paddle_ppstructure import PaddlePPStructureEngine
from arabic_ocr.engines.paddleocr_vl import PaddleOCRVLEngine

_ENGINE_REGISTRY: dict[str, OCREngine] = {
    "openai": OpenAIVisionEngine(),
    "gemini_pro": GeminiProEngine(),
    "gemini_flash": GeminiFlashEngine(),
    "paddle_ppstructure": PaddlePPStructureEngine(),
    "paddleocr_vl": PaddleOCRVLEngine(),
}


def get_engine(engine_id: str) -> OCREngine:
    """Return an engine by id."""
    return _ENGINE_REGISTRY[engine_id]


def available_engine_ids() -> list[str]:
    """List engines available in the current runtime."""
    return [engine_id for engine_id, engine in _ENGINE_REGISTRY.items() if engine.is_available()]


def get_primary_secondary(mode: str) -> tuple[OCREngine, OCREngine | None]:
    """Resolve engine mode into a primary and optional secondary engine."""
    if mode == "openai":
        return get_engine("openai"), None
    if mode == "paddleocr_vl":
        secondary = get_engine("paddle_ppstructure") if get_engine("paddle_ppstructure").is_available() else _cloud_fallback()
        return get_engine("paddleocr_vl"), secondary
    if mode == "paddle_ppstructure":
        return get_engine("paddle_ppstructure"), _cloud_fallback()
    if mode == "gemini":
        return get_engine("gemini_pro"), get_engine("gemini_flash")

    cloud_primary, cloud_secondary = _best_available_cloud_pair()
    if cloud_primary is not None:
        return cloud_primary, cloud_secondary
    if get_engine("paddleocr_vl").is_available():
        secondary = get_engine("paddle_ppstructure") if get_engine("paddle_ppstructure").is_available() else _cloud_fallback()
        return get_engine("paddleocr_vl"), secondary
    if get_engine("paddle_ppstructure").is_available():
        return get_engine("paddle_ppstructure"), _cloud_fallback()
    return get_engine("gemini_pro"), get_engine("gemini_flash")


def _cloud_fallback() -> OCREngine | None:
    if _has_env("OPENAI_API_KEY") and get_engine("openai").is_available():
        return get_engine("openai")
    if _has_env("GEMINI_API_KEY"):
        if not get_engine("gemini_pro").is_available():
            return None
        return get_engine("gemini_pro")
    return None


def _best_available_cloud_pair() -> tuple[OCREngine | None, OCREngine | None]:
    """Prefer cloud VLMs for best OCR quality when credentials are available."""
    openai_available = _has_env("OPENAI_API_KEY") and get_engine("openai").is_available()
    gemini_available = _has_env("GEMINI_API_KEY") and get_engine("gemini_pro").is_available()

    if openai_available and gemini_available:
        return get_engine("openai"), get_engine("gemini_pro")
    if openai_available:
        return get_engine("openai"), None
    if gemini_available:
        secondary = get_engine("gemini_flash") if get_engine("gemini_flash").is_available() else None
        return get_engine("gemini_pro"), secondary
    return None, None


def _has_env(name: str) -> bool:
    return bool(os.getenv(name))
