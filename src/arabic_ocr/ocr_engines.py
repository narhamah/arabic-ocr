"""Cloud OCR helpers used by the engine adapters."""

from __future__ import annotations

import os

from PIL import Image

from arabic_ocr.openai_support import create_client, extract_output_text, image_to_data_url, openai_ready

_OCR_PROMPT = """Extract ALL visible text from this scanned Arabic legal document image exactly as written.

Rules:
- Preserve the original text with 100% fidelity - do not paraphrase or correct grammar
- Read columns right-to-left (Arabic reading order)
- Within each column, read top-to-bottom
- Include headings, body text, footnotes, tables, stamps, and captions when legible
- Include any English text as-is
- Preserve line breaks aggressively:
  - each distinct printed line should be a separate output line
  - keep headers, form labels, and body paragraphs on separate lines
  - insert a blank line between clearly separate sections
  - preserve obvious table rows as one line per row
- Do NOT describe the image or add any commentary
- Do NOT add diacritics/tashkeel unless clearly visible
- Output ONLY the extracted text"""


def ocr_gemini_pro(image: Image.Image) -> dict:
    """Run OCR using Gemini 2.5 Pro."""
    return _call_gemini(image=image, model="gemini-2.5-pro", prompt=_OCR_PROMPT)


def ocr_gemini_flash(image: Image.Image) -> dict:
    """Run OCR using Gemini 2.5 Flash."""
    return _call_gemini(image=image, model="gemini-2.5-flash", prompt=_OCR_PROMPT)


def ocr_openai(image: Image.Image) -> dict:
    """Run OCR using an OpenAI multimodal model."""
    model = os.getenv("OPENAI_OCR_MODEL", "gpt-4.1")
    return _call_openai(image=image, model=model, prompt=_OCR_PROMPT)


def run_dual_ocr(image: Image.Image) -> dict:
    """Run both Gemini models and return both results."""
    primary = ocr_gemini_pro(image)
    secondary = ocr_gemini_flash(image)

    if not primary.get("success"):
        fallback = ocr_openai(image)
        if fallback.get("success"):
            primary = fallback

    if not secondary.get("success"):
        fallback = ocr_openai(image)
        if fallback.get("success"):
            secondary = fallback

    if not primary.get("success") and secondary.get("success"):
        primary = secondary
    if primary.get("success") and not secondary.get("success"):
        secondary = primary

    return {"primary": primary, "secondary": secondary}


def _call_gemini(*, image: Image.Image, model: str, prompt: str) -> dict:
    """Call Gemini API with an image and prompt using the google.genai SDK."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"text": "", "success": False, "error": "GEMINI_API_KEY not set"}

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[
                prompt,
                types.Part.from_bytes(data=_image_bytes(image), mime_type="image/png"),
            ],
        )
        text = getattr(response, "text", None) or ""
        return {"text": text, "success": True, "error": None}
    except Exception as exc:
        return {"text": "", "success": False, "error": str(exc)}


def _call_openai(*, image: Image.Image, model: str, prompt: str) -> dict:
    """Call the OpenAI Responses API with an image and OCR prompt."""
    if not openai_ready():
        if not os.getenv("OPENAI_API_KEY"):
            return {"text": "", "success": False, "error": "OPENAI_API_KEY not set"}
        return {"text": "", "success": False, "error": "Optional dependency 'openai' is not installed"}

    try:
        client = create_client()
        response = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_to_data_url(image)},
                ],
            }],
        )
        return {"text": extract_output_text(response), "success": True, "error": None}
    except Exception as exc:
        return {"text": "", "success": False, "error": str(exc)}


def _image_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
