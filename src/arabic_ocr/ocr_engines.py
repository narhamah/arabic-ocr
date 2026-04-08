"""OCR engines for Arabic document images."""

from __future__ import annotations

import os

from PIL import Image

_OCR_PROMPT = """Extract ALL text from this Arabic document image exactly as written.

Rules:
- Preserve the original text with 100% fidelity. Do not paraphrase.
- Read Arabic in normal reading order.
- Preserve numbers, dates, IBAN/account numbers, currencies, and punctuation exactly.
- Include English text exactly as written.
- Preserve paragraph breaks and visible line breaks when meaningful.
- Do NOT add commentary or descriptions.
- Do NOT add diacritics unless clearly visible.
- Output ONLY the extracted text."""


def ocr_gemini_pro(image: Image.Image) -> dict:
    return _call_gemini(image=image, model="gemini-2.5-pro", prompt=_OCR_PROMPT)


def ocr_gemini_flash(image: Image.Image) -> dict:
    return _call_gemini(image=image, model="gemini-2.0-flash", prompt=_OCR_PROMPT)


def ocr_openai(image: Image.Image) -> dict:
    return _call_openai(image=image, model="gpt-4.1", prompt=_OCR_PROMPT)


def run_dual_ocr(image: Image.Image) -> dict:
    """Run multiple OCR engines and keep two best available outputs."""
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
    try:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"text": "", "success": False, "error": "GEMINI_API_KEY not set"}

        genai.configure(api_key=api_key)
        genai_model = genai.GenerativeModel(model)
        response = genai_model.generate_content([prompt, image])
        text = response.text if response.text else ""
        return {"text": text, "success": True, "error": None}
    except Exception as exc:
        return {"text": "", "success": False, "error": str(exc)}


def _call_openai(*, image: Image.Image, model: str, prompt: str) -> dict:
    try:
        from openai import OpenAI
    except ImportError:
        return {"text": "", "success": False, "error": "openai package not installed"}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"text": "", "success": False, "error": "OPENAI_API_KEY not set"}

    try:
        import base64
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_b64}",
                    },
                ],
            }],
        )
        text = response.output_text if getattr(response, "output_text", None) else ""
        return {"text": text, "success": bool(text), "error": None if text else "Empty response"}
    except Exception as exc:
        return {"text": "", "success": False, "error": str(exc)}
