"""Verification and adjudication helpers for OCR and hybrid extraction."""

from __future__ import annotations

import base64
import difflib
import io
import os
import re

from PIL import Image


def verify_and_merge(
    primary: str, secondary: str, image: Image.Image
) -> dict:
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
            resolved_text = _call_claude_tiebreaker(
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


def verify_native_vs_ocr(
    *,
    native_text: str,
    ocr_text: str,
    image: Image.Image,
) -> dict:
    """Adjudicate between native PDF extraction and OCR confirmation."""
    native_clean = native_text.strip()
    ocr_clean = ocr_text.strip()

    if not native_clean and not ocr_clean:
        return {
            "text": "",
            "confidence": 0.0,
            "source": "empty",
            "similarity": 1.0,
            "digit_agreement": True,
        }
    if native_clean and not ocr_clean:
        return {
            "text": native_clean,
            "confidence": 0.75,
            "source": "native_only",
            "similarity": 0.0,
            "digit_agreement": False,
        }
    if ocr_clean and not native_clean:
        return {
            "text": ocr_clean,
            "confidence": 0.6,
            "source": "ocr_only",
            "similarity": 0.0,
            "digit_agreement": False,
        }

    similarity = _similarity(native_clean, ocr_clean)
    digit_agreement = _extract_numbers(native_clean) == _extract_numbers(ocr_clean)
    len_ratio = len(ocr_clean) / max(len(native_clean), 1)

    if similarity >= 0.985 and digit_agreement:
        return {
            "text": native_clean,
            "confidence": 0.99,
            "source": "native_confirmed_by_ocr",
            "similarity": similarity,
            "digit_agreement": True,
        }

    # A real text layer is usually more trustworthy than OCR when the two
    # outputs diverge heavily. Do not let image models rewrite the page unless
    # the OCR is close enough to act as corroboration rather than invention.
    if similarity < 0.75 or len_ratio < 0.85 or len_ratio > 1.15:
        return {
            "text": native_clean,
            "confidence": 0.9,
            "source": "native_preferred_large_mismatch",
            "similarity": similarity,
            "digit_agreement": digit_agreement,
        }

    resolved_candidates = [
        ("native", native_clean),
        ("ocr", ocr_clean),
    ]

    claude_text = _call_claude_page_resolver(
        native_text=native_clean,
        ocr_text=ocr_clean,
        image=image,
    )
    claude_clean = claude_text.strip() if claude_text else ""
    if claude_clean:
        resolved_candidates.append(("claude", claude_clean))

    openai_text = _call_openai_page_resolver(
        native_text=native_clean,
        ocr_text=ocr_clean,
        image=image,
    )
    openai_clean = openai_text.strip() if openai_text else ""
    if openai_clean:
        resolved_candidates.append(("openai", openai_clean))

    if claude_clean and openai_clean and claude_clean == openai_clean:
        return {
            "text": claude_clean,
            "confidence": 0.98,
            "source": "claude",
            "similarity": similarity,
            "digit_agreement": _extract_numbers(claude_clean) == _extract_numbers(ocr_clean),
        }

    best_source, best_text = max(
        resolved_candidates,
        key=lambda item: _consensus_score(item[1], [text for _name, text in resolved_candidates]),
    )
    best_similarity = max(
        _similarity(best_text, candidate)
        for _name, candidate in resolved_candidates
    )

    return {
        "text": best_text,
        "confidence": min(0.98, max(0.55, best_similarity)),
        "source": best_source,
        "similarity": similarity,
        "digit_agreement": digit_agreement,
    }


def _call_claude_tiebreaker(
    *,
    primary: str,
    secondary: str,
    disagreements: list[dict],
    image: Image.Image,
) -> str | None:
    """Call Claude to resolve OCR disagreements."""
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    image_b64 = _image_to_base64_png(image)

    disagreement_lines = []
    for disagreement in disagreements:
        disagreement_lines.append(
            f'  System A: "{disagreement["primary"]}" vs System B: "{disagreement["secondary"]}"'
        )
    disagreement_text = "\n".join(disagreement_lines)

    prompt = f"""Two OCR systems read this Arabic image differently.

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
