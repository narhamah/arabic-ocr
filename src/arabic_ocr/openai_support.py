"""Shared OpenAI Responses API helpers."""

from __future__ import annotations

import base64
import importlib.util
import io
import os
from typing import Any

from PIL import Image


def has_openai_client() -> bool:
    """Return whether the OpenAI SDK is importable."""
    return importlib.util.find_spec("openai") is not None


def has_openai_credentials() -> bool:
    """Return whether an OpenAI API key is configured."""
    return bool(os.getenv("OPENAI_API_KEY"))


def openai_ready() -> bool:
    """Return whether OpenAI can be called in the current runtime."""
    return has_openai_client() and has_openai_credentials()


def image_to_data_url(image: Image.Image, *, image_format: str = "PNG") -> str:
    """Encode a PIL image as a data URL for OpenAI image input."""
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    mime_type = f"image/{image_format.lower()}"
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def create_client():
    """Create an OpenAI client using OPENAI_API_KEY."""
    from openai import OpenAI

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_output_text(response: Any) -> str:
    """Return textual content from a Responses API result."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    pieces: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                pieces.append(str(text))
    return "\n".join(piece for piece in pieces if piece).strip()
