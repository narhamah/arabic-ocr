"""Stage 5: Cross-model verification with Claude tiebreaker.

Compares outputs from two OCR engines:
- Where they agree → accept (high confidence)
- Where they disagree → Claude resolves as tiebreaker
"""

from __future__ import annotations

import difflib
import io
import base64
import os

from PIL import Image


def verify_and_merge(
    primary: str, secondary: str, image: Image.Image
) -> dict:
    """Compare two OCR outputs and resolve disagreements.

    Args:
        primary: Text from primary OCR engine (Gemini Pro).
        secondary: Text from secondary OCR engine (Gemini Flash).
        image: Original image for Claude tiebreaker context.

    Returns:
        Dict with keys:
            - text: Final merged text
            - confidence: Overall confidence score (0.0-1.0)
            - disagreements: Number of word disagreements
            - total_words: Total number of words compared
    """
    # Handle empty inputs
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

    # Find disagreements using SequenceMatcher
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

    # If there are disagreements, try Claude tiebreaker
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

        # Fallback to primary if Claude fails
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


def _call_claude_tiebreaker(
    *,
    primary: str,
    secondary: str,
    disagreements: list[dict],
    image: Image.Image,
) -> str | None:
    """Call Claude to resolve OCR disagreements.

    Args:
        primary: Full primary OCR text.
        secondary: Full secondary OCR text.
        disagreements: List of disagreement dicts.
        image: Original image for visual verification.

    Returns:
        Resolved text string, or None on failure.
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    # Convert image to base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Build disagreement list for prompt
    disagreement_lines = []
    for d in disagreements:
        disagreement_lines.append(f"  System A: \"{d['primary']}\" vs System B: \"{d['secondary']}\"")
    disagreement_text = "\n".join(disagreement_lines)

    prompt = f"""Two OCR systems read this Arabic newspaper image differently.

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
