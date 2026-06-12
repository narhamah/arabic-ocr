"""Cross-model verification with OpenAI-first adjudication and Claude fallback."""

from __future__ import annotations

import base64
import difflib
import io
import os
import re

from PIL import Image

from arabic_ocr.openai_support import create_client, extract_output_text, image_to_data_url, openai_ready


def verify_and_merge(primary: str, secondary: str, image: Image.Image) -> dict:
    """Compare two OCR outputs and resolve disagreements."""
    if not primary and not secondary:
        return {"text": "", "confidence": 1.0, "disagreements": 0, "total_words": 0}

    primary_words = primary.split()
    secondary_words = secondary.split()
    if primary_words == secondary_words:
        return {
            "text": primary,
            "confidence": 1.0,
            "disagreements": 0,
            "total_words": len(primary_words),
        }

    matcher = difflib.SequenceMatcher(None, primary_words, secondary_words)
    agreements = []
    disagreements = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            agreements.extend(primary_words[i1:i2])
        else:
            disagreements.append({
                "primary": " ".join(primary_words[i1:i2]),
                "secondary": " ".join(secondary_words[j1:j2]),
                "position": len(agreements) + len(disagreements),
            })

    total_words = max(len(primary_words), len(secondary_words))
    num_disagreements = len(disagreements)
    if disagreements:
        try:
            resolved_text = _call_tiebreaker(
                primary=primary,
                secondary=secondary,
                disagreements=disagreements,
                image=image,
            )
            if resolved_text:
                confidence = 1.0 - (num_disagreements / max(total_words, 1))
                return {
                    "text": resolved_text,
                    "confidence": max(0.0, confidence),
                    "disagreements": num_disagreements,
                    "total_words": total_words,
                }
        except Exception:
            pass

        confidence = 1.0 - (num_disagreements / max(total_words, 1))
        return {
            "text": primary,
            "confidence": max(0.0, confidence),
            "disagreements": num_disagreements,
            "total_words": total_words,
        }

    return {
        "text": primary,
        "confidence": 1.0,
        "disagreements": 0,
        "total_words": total_words,
    }


def verify_native_vs_ocr(*, native_text: str, ocr_text: str, image: Image.Image) -> dict:
    """Adjudicate native PDF text against OCR for a full page."""
    native_text = native_text or ""
    ocr_text = ocr_text or ""
    digit_agreement = _extract_numbers(native_text) == _extract_numbers(ocr_text)
    similarity = _similarity(native_text, ocr_text)

    if native_text.strip() and (native_text == ocr_text or (similarity >= 0.92 and digit_agreement)):
        return {
            "text": native_text,
            "confidence": 1.0 if native_text == ocr_text else max(0.85, similarity),
            "source": "native_confirmed_by_ocr",
            "digit_agreement": digit_agreement,
            "similarity": similarity,
        }

    if not native_text.strip() and ocr_text.strip():
        return {
            "text": ocr_text,
            "confidence": 0.75,
            "source": "ocr_only",
            "digit_agreement": digit_agreement,
            "similarity": similarity,
        }

    if native_text.strip() and not ocr_text.strip():
        return {
            "text": native_text,
            "confidence": 0.75,
            "source": "native_only",
            "digit_agreement": digit_agreement,
            "similarity": similarity,
        }

    resolved = _call_openai_page_resolver(native_text=native_text, ocr_text=ocr_text, image=image)
    if not resolved:
        resolved = _call_claude_page_resolver(native_text=native_text, ocr_text=ocr_text, image=image)
    if resolved:
        return {
            "text": resolved,
            "confidence": 0.8,
            "source": "page_resolver",
            "digit_agreement": digit_agreement,
            "similarity": similarity,
        }

    native_score = _consensus_score(native_text, [ocr_text])
    ocr_score = _consensus_score(ocr_text, [native_text])
    if ocr_score > native_score and ocr_text.strip():
        return {
            "text": ocr_text,
            "confidence": max(0.4, min(0.75, similarity)),
            "source": "ocr_preferred",
            "digit_agreement": digit_agreement,
            "similarity": similarity,
        }
    return {
        "text": native_text,
        "confidence": max(0.4, min(0.75, similarity)),
        "source": "native_preferred",
        "digit_agreement": digit_agreement,
        "similarity": similarity,
    }


def _call_tiebreaker(
    *,
    primary: str,
    secondary: str,
    disagreements: list[dict],
    image: Image.Image,
) -> str | None:
    """Use OpenAI first, then Claude if available."""
    if openai_ready():
        resolved = _call_openai_tiebreaker(
            primary=primary,
            secondary=secondary,
            disagreements=disagreements,
            image=image,
        )
        if resolved:
            return resolved
    return _call_claude_tiebreaker(
        primary=primary,
        secondary=secondary,
        disagreements=disagreements,
        image=image,
    )


def _call_openai_tiebreaker(
    *,
    primary: str,
    secondary: str,
    disagreements: list[dict],
    image: Image.Image,
) -> str | None:
    """Call OpenAI to resolve OCR disagreements."""
    if not openai_ready():
        return None

    disagreement_lines = [
        f'- A: "{item["primary"]}" vs B: "{item["secondary"]}"'
        for item in disagreements
    ]
    disagreement_text = "\n".join(disagreement_lines)
    prompt = f"""Two OCR systems read this Arabic legal document image differently.

System A read:
{primary}

System B read:
{secondary}

Focus only on the disputed spans below and use the image to choose the correct reading.
Disputes:
{disagreement_text}

Return ONLY the complete corrected text.
Do not paraphrase.
Do not add commentary.
Do not add diacritics unless clearly visible."""

    client = create_client()
    model = os.getenv("OPENAI_TIEBREAKER_MODEL", "gpt-4.1-mini")
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
    resolved_text = extract_output_text(response)
    return resolved_text or None


def _call_claude_tiebreaker(
    *,
    primary: str,
    secondary: str,
    disagreements: list[dict],
    image: Image.Image,
) -> str | None:
    """Call Claude to resolve OCR disagreements."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    import anthropic

    from base64 import b64encode
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = b64encode(buffer.getvalue()).decode("utf-8")

    disagreement_lines = [
        f'  System A: "{item["primary"]}" vs System B: "{item["secondary"]}"'
        for item in disagreements
    ]
    disagreement_text = "\n".join(disagreement_lines)

    prompt = f"""Two OCR systems read this Arabic legal document image differently.

System A read: {primary}
System B read: {secondary}

The systems agree on most text but disagree on these words:
{disagreement_text}

Look at the image and determine the correct reading for each disagreement.
Return ONLY the complete corrected text.
Do not paraphrase. Do not add diacritics. Just pick the correct reading for each disputed word."""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }],
    )

    return message.content[0].text if message.content else None


def _call_claude_page_resolver(
    *,
    native_text: str,
    ocr_text: str,
    image: Image.Image,
) -> str | None:
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    image_b64 = _image_to_base64_png(image)
    prompt = (
        "You are validating a bank-document page.\n"
        "Candidate A is native PDF text extraction.\n"
        "Candidate B is OCR from the rendered page.\n"
        "Compare both against the image and return ONLY the complete correct page text.\n"
        "Preserve Arabic exactly. Preserve all numbers exactly. Do not paraphrase.\n\n"
        f"Candidate A:\n{native_text}\n\nCandidate B:\n{ocr_text}"
    )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return message.content[0].text if message.content else None


def _call_openai_page_resolver(
    *,
    native_text: str,
    ocr_text: str,
    image: Image.Image,
) -> str | None:
    try:
        from openai import OpenAI
    except ImportError:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    image_b64 = _image_to_base64_png(image)
    prompt = (
        "Validate this bank-document page against two candidate transcriptions.\n"
        "Return only the complete correct page text.\n"
        "Keep Arabic exactly as seen. Keep numbers, punctuation, and line breaks accurate.\n"
        "Do not summarize.\n\n"
        f"Candidate A:\n{native_text}\n\nCandidate B:\n{ocr_text}"
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-4.1",
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

    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    return None


def _image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _extract_numbers(text: str) -> list[str]:
    return re.findall(r"[\d.,/:-]+", text)


def _consensus_score(candidate: str, others: list[str]) -> float:
    if not candidate:
        return 0.0

    score = 0.0
    for other in others:
        score += _similarity(candidate, other)
        if _extract_numbers(candidate) == _extract_numbers(other):
            score += 0.25
    return score
