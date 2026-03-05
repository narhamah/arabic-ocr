"""Stage 3: Dual-VLM OCR using Gemini models.

Primary: Gemini 2.5 Pro (highest Arabic accuracy)
Secondary: Gemini 2.0 Flash (fast, cheap, second opinion)

Both receive the ORIGINAL image (not preprocessed) — VLMs do their own
internal preprocessing and do it better than ours.
"""

from __future__ import annotations

import io
import os

from PIL import Image

_OCR_PROMPT = """Extract ALL Arabic text from this scanned newspaper image exactly as written.

Rules:
- Preserve the original text with 100% fidelity — do not paraphrase or correct grammar
- Read columns right-to-left (Arabic reading order)
- Within each column, read top-to-bottom
- Include headlines, body text, and captions
- Include any English text as-is
- Preserve paragraph breaks between distinct sections
- Do NOT describe the image or add any commentary
- Do NOT add diacritics/tashkeel unless clearly visible
- Output ONLY the extracted text"""


def ocr_gemini_pro(image: Image.Image) -> dict:
    """Run OCR using Gemini 2.5 Pro.

    Args:
        image: Original PIL Image (not preprocessed).

    Returns:
        Dict with keys: text (str), success (bool), error (str | None).
    """
    return _call_gemini(image=image, model="gemini-2.5-pro", prompt=_OCR_PROMPT)


def ocr_gemini_flash(image: Image.Image) -> dict:
    """Run OCR using Gemini 2.0 Flash.

    Args:
        image: Original PIL Image (not preprocessed).

    Returns:
        Dict with keys: text (str), success (bool), error (str | None).
    """
    return _call_gemini(image=image, model="gemini-2.0-flash", prompt=_OCR_PROMPT)


def run_dual_ocr(image: Image.Image) -> dict:
    """Run both Gemini models and return both results.

    Args:
        image: Original PIL Image.

    Returns:
        Dict with keys: primary (Gemini Pro result), secondary (Gemini Flash result).
    """
    primary = ocr_gemini_pro(image)
    secondary = ocr_gemini_flash(image)
    return {"primary": primary, "secondary": secondary}


def _call_gemini(*, image: Image.Image, model: str, prompt: str) -> dict:
    """Call Gemini API with an image and prompt.

    Args:
        image: PIL Image to send.
        model: Gemini model name.
        prompt: Text prompt for OCR.

    Returns:
        Dict with keys: text (str), success (bool), error (str | None).
    """
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

    except Exception as e:
        return {"text": "", "success": False, "error": str(e)}
