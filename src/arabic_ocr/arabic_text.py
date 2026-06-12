"""Arabic and RTL text detection, cleaning, normalization, and quality checks."""

from dataclasses import dataclass
import re
import unicodedata

# Arabic Unicode ranges
_ARABIC_BLOCK = re.compile(r"[\u0600-\u06FF]")
_ARABIC_SUPPLEMENT = re.compile(r"[\u0750-\u077F]")
_ARABIC_EXTENDED_A = re.compile(r"[\u08A0-\u08FF]")
_ARABIC_PRESENTATION_A = re.compile(r"[\uFB50-\uFDFF]")
_ARABIC_PRESENTATION_B = re.compile(r"[\uFE70-\uFEFF]")

# Arabic diacritics (tashkeel) — fathah, dammah, kasrah, sukun, shadda, etc.
_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]")

# Tatweel (kashida) — decorative elongation character
_TATWEEL = "\u0640"

# Common Arabic letter normalization pairs
_ARABIC_NORMALIZATIONS: dict[str, str] = {
    "\u0622": "\u0627",  # alef with madda -> alef
    "\u0623": "\u0627",  # alef with hamza above -> alef
    "\u0625": "\u0627",  # alef with hamza below -> alef
    "\u0649": "\u064A",  # alef maksura -> yaa
    "\u0629": "\u0647",  # taa marbuta -> haa
}
_ARABIC_SCRIPT_VARIANTS: dict[str, str] = {
    "\u06A9": "\u0643",  # Persian kaf -> Arabic kaf
    "\u06CC": "\u064A",  # Persian yeh -> Arabic yeh
}

# Arabic-specific punctuation normalization
_ARABIC_PUNCTUATION: dict[str, str] = {
    "\u060C": ",",     # Arabic comma
    "\u061B": ";",     # Arabic semicolon
    "\u061F": "?",     # Arabic question mark
    "\u066B": ".",     # Arabic decimal separator
    "\u066C": ",",     # Arabic thousands separator
}


# Pattern to detect if a word is entirely Arabic letters (no spaces/punctuation)
_ARABIC_WORD = re.compile(r"^[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+$")
_LATIN_EXTENDED_GLYPHS = re.compile(r"[\u00C0-\u02AF\u1D00-\u1EFF]")
_ARABIC_REPEATED_RUNS = re.compile(r"([\u0600-\u06FF])\1{2,}")


@dataclass(slots=True)
class ArabicQualityAssessment:
    """Document-level quality assessment for extracted Arabic text."""

    status: str
    issues: list[str]
    metrics: dict[str, float | int]


def contains_arabic(text: str) -> bool:
    """Check if text contains Arabic script characters.

    Args:
        text: Text to check.

    Returns:
        True if Arabic characters are found.
    """
    return bool(
        _ARABIC_BLOCK.search(text)
        or _ARABIC_SUPPLEMENT.search(text)
        or _ARABIC_EXTENDED_A.search(text)
        or _ARABIC_PRESENTATION_A.search(text)
        or _ARABIC_PRESENTATION_B.search(text)
    )


def detect_rtl_ratio(text: str) -> float:
    """Calculate the ratio of RTL characters in the text.

    Args:
        text: Text to analyze.

    Returns:
        Float between 0.0 and 1.0 representing RTL character ratio.
    """
    if not text:
        return 0.0
    rtl_count = 0
    total_alpha = 0
    for ch in text:
        cat = unicodedata.bidirectional(ch)
        if cat in ("R", "AL", "AN"):
            rtl_count += 1
            total_alpha += 1
        elif cat == "L":
            total_alpha += 1
    if total_alpha == 0:
        return 0.0
    return rtl_count / total_alpha


def is_predominantly_rtl(text: str) -> bool:
    """Check if text is predominantly right-to-left.

    Args:
        text: Text to check.

    Returns:
        True if more than 50% of alphabetic characters are RTL.
    """
    return detect_rtl_ratio(text) > 0.5


def detect_suspicious_arabic_char_ratio(text: str) -> float:
    """Estimate how much of the Arabic text uses unlikely code points.

    Some PDFs decode Arabic glyphs into diacritics, supplementary Arabic
    letters, or presentation-form characters instead of normal Arabic prose.
    A high ratio usually indicates a bad font/CMap extraction rather than
    legitimate Arabic text.
    """
    if not text:
        return 0.0

    arabic_chars = 0
    suspicious_chars = 0

    for ch in text:
        cp = ord(ch)

        # Standard Arabic prose characters we expect in extracted legal text.
        is_standard_arabic = (
            0x0621 <= cp <= 0x064A
            or cp == 0x0640
            or cp in (0x060C, 0x061B, 0x061F, 0x066A, 0x066B, 0x066C)
            or 0x0660 <= cp <= 0x0669
        )

        if 0x0600 <= cp <= 0x06FF:
            arabic_chars += 1
            if not is_standard_arabic:
                suspicious_chars += 1
        elif 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
            arabic_chars += 1
            suspicious_chars += 1
        elif 0xFB50 <= cp <= 0xFEFF:
            arabic_chars += 1
            suspicious_chars += 1

    if arabic_chars == 0:
        return 0.0
    return suspicious_chars / arabic_chars


def has_suspicious_arabic_mapping(text: str, threshold: float = 0.03) -> bool:
    """Return True when Arabic text looks like a bad PDF character map."""
    return detect_suspicious_arabic_char_ratio(text) >= threshold


def detect_weird_glyph_char_ratio(text: str) -> float:
    """Estimate how much text is made of likely bad Latin-extended glyph noise."""
    if not text:
        return 0.0

    weird_chars = len(_LATIN_EXTENDED_GLYPHS.findall(text))
    return weird_chars / len(text)


def detect_repeated_arabic_char_run_ratio(text: str) -> float:
    """Estimate how much Arabic text is made of repeated run-length noise.

    Bad PDF text layers sometimes turn prose into tokens with long repeated
    letters such as "القضاااإد" or "تتت".  Those runs are rare in normal legal
    prose, but common in degraded extraction output.
    """
    if not text:
        return 0.0

    arabic_chars = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x0600 <= cp <= 0x06FF
            or 0x0750 <= cp <= 0x077F
            or 0x08A0 <= cp <= 0x08FF
            or 0xFB50 <= cp <= 0xFDFF
            or 0xFE70 <= cp <= 0xFEFF
        ):
            arabic_chars += 1

    if arabic_chars == 0:
        return 0.0

    repeated_chars = 0
    for match in _ARABIC_REPEATED_RUNS.finditer(text):
        repeated_chars += len(match.group(0))

    return repeated_chars / arabic_chars


def assess_arabic_extraction_quality(text: str) -> ArabicQualityAssessment:
    """Assess whether extracted Arabic text is safe to ingest into the corpus."""
    weird_glyph_ratio = detect_weird_glyph_char_ratio(text)
    rtl_ratio = detect_rtl_ratio(text)
    repeated_run_ratio = detect_repeated_arabic_char_run_ratio(text)

    if not text.strip():
        return ArabicQualityAssessment(
            status="not_arabic",
            issues=[],
            metrics={
                "rtl_ratio": rtl_ratio,
                "suspicious_arabic_char_ratio": detect_suspicious_arabic_char_ratio(text),
                "weird_glyph_ratio": weird_glyph_ratio,
                "repeated_arabic_run_ratio": repeated_run_ratio,
                "arabic_char_count": 0,
            },
        )

    if not contains_arabic(text):
        issues = ["latin_extended_glyph_noise"] if weird_glyph_ratio >= 0.01 else []
        status = "fail" if weird_glyph_ratio >= 0.10 else "not_arabic"
        return ArabicQualityAssessment(
            status=status,
            issues=issues,
            metrics={
                "rtl_ratio": rtl_ratio,
                "suspicious_arabic_char_ratio": 0.0,
                "weird_glyph_ratio": weird_glyph_ratio,
                "repeated_arabic_run_ratio": 0.0,
                "arabic_char_count": 0,
            },
        )

    arabic_char_count = sum(1 for ch in text if contains_arabic(ch))
    suspicious_ratio = detect_suspicious_arabic_char_ratio(text)

    issues: list[str] = []
    if suspicious_ratio >= 0.03:
        issues.append("suspicious_arabic_mapping")
    if weird_glyph_ratio >= 0.01:
        issues.append("latin_extended_glyph_noise")
    if repeated_run_ratio >= 0.03:
        issues.append("repeated_arabic_run_noise")
    if rtl_ratio < 0.25:
        issues.append("low_rtl_ratio")

    if (
        weird_glyph_ratio >= 0.10
        or suspicious_ratio >= 0.05
        or repeated_run_ratio >= 0.05
    ):
        status = "fail"
    elif issues:
        status = "warn"
    else:
        status = "pass"

    return ArabicQualityAssessment(
        status=status,
        issues=issues,
        metrics={
            "rtl_ratio": rtl_ratio,
            "suspicious_arabic_char_ratio": suspicious_ratio,
            "weird_glyph_ratio": weird_glyph_ratio,
            "repeated_arabic_run_ratio": repeated_run_ratio,
            "arabic_char_count": arabic_char_count,
        },
    )


def remove_tashkeel(text: str) -> str:
    """Remove Arabic diacritical marks (tashkeel/harakat).

    Removes fathah, dammah, kasrah, sukun, shadda, and other combining marks
    used for vocalization. Useful for search and matching.

    Args:
        text: Arabic text with diacritics.

    Returns:
        Text with diacritics removed.
    """
    return _TASHKEEL.sub("", text)


def remove_tatweel(text: str) -> str:
    """Remove tatweel/kashida characters used for decorative elongation.

    Args:
        text: Arabic text possibly containing tatweel.

    Returns:
        Text with tatweel removed.
    """
    return text.replace(_TATWEEL, "")


def normalize_arabic_letters(text: str) -> str:
    """Normalize Arabic letter variants to their base forms.

    Normalizes alef variants to bare alef, alef maksura to yaa,
    and taa marbuta to haa.

    Args:
        text: Arabic text.

    Returns:
        Text with normalized letter forms.
    """
    for original, normalized in _ARABIC_NORMALIZATIONS.items():
        text = text.replace(original, normalized)
    return text


def normalize_arabic_script_variants(text: str) -> str:
    """Normalize script-equivalent Arabic variants from font fallback paths."""
    for original, normalized in _ARABIC_SCRIPT_VARIANTS.items():
        text = text.replace(original, normalized)
    return text


def fix_rtl_punctuation(text: str) -> str:
    """Fix common punctuation issues in RTL text.

    Normalizes Arabic-specific punctuation to standard equivalents
    and fixes misplaced punctuation common in PDF extraction.

    Args:
        text: Text with potentially misplaced RTL punctuation.

    Returns:
        Text with corrected punctuation.
    """
    for arabic_punct, standard in _ARABIC_PUNCTUATION.items():
        text = text.replace(arabic_punct, standard)
    return text


def normalize_presentation_forms(text: str) -> str:
    """Normalize Arabic Presentation Forms back to standard Arabic characters.

    PDFs often store Arabic using Presentation Forms A (U+FB50-U+FDFF) and
    Presentation Forms B (U+FE70-U+FEFF) instead of standard Arabic block
    characters (U+0600-U+06FF). NFKC normalization converts these back.

    Args:
        text: Text potentially containing Arabic presentation forms.

    Returns:
        Text with presentation forms normalized to standard characters.
    """
    if _ARABIC_PRESENTATION_A.search(text) or _ARABIC_PRESENTATION_B.search(text):
        text = unicodedata.normalize("NFKC", text)
    return text


def fix_bidi_text(text: str) -> str:
    """Apply Unicode BiDi algorithm to fix display order of mixed text.

    Uses python-bidi to reorder mixed LTR/RTL text into logical order.

    Args:
        text: Text with potentially incorrect bidi ordering.

    Returns:
        Text with corrected bidirectional ordering.
    """
    try:
        # python-bidi >= 0.6.0
        from bidi import get_display
    except ImportError:
        try:
            # python-bidi < 0.5.0
            from bidi.algorithm import get_display
        except ImportError:
            return text

    lines = text.split("\n")
    fixed: list[str] = []
    for line in lines:
        if contains_arabic(line):
            fixed.append(get_display(line))
        else:
            fixed.append(line)
    return "\n".join(fixed)


def fix_lam_alef_ligature(text: str) -> str:
    """Fix reversed lam-ligature decomposition from PDF extraction.

    Many Arabic PDF fonts store lam-based ligatures with the second character
    before the lam in the ToUnicode CMap. PyMuPDF extracts the multi-char
    mapping in reversed order, producing garbled text.

    Lam-alef ligatures (most common):
      اإل → الإ    (alef + alef-hamza-below + lam → alef + lam + alef-hamza-below)
      األ → الأ    (alef + alef-hamza-above + lam → alef + lam + alef-hamza-above)
      اآل → الآ    (alef + alef-madda + lam → alef + lam + alef-madda)
      ال → لا      (bare alef + lam → lam + alef, only in specific contexts)

    Lam-meem ligature:
      امل → الم    (alef + meem + lam → alef + lam + meem)

    Args:
        text: Arabic text with potential lam-ligature garbling.

    Returns:
        Text with corrected lam-ligature order.
    """
    # Lam-alef: alef (ا) + hamza-alef variant (أ/إ/آ) + lam (ل)
    # Replace with: alef (ا) + lam (ل) + hamza-alef variant
    text = text.replace("\u0627\u0625\u0644", "\u0627\u0644\u0625")  # اإل → الإ
    text = text.replace("\u0627\u0623\u0644", "\u0627\u0644\u0623")  # األ → الأ
    text = text.replace("\u0627\u0622\u0644", "\u0627\u0644\u0622")  # اآل → الآ

    # Bare lam-alef: اال → الا (alef + alef + lam → alef + lam + alef)
    # This occurs in words like: االتحاد→الاتحاد, اتصاالت→اتصالات,
    # حاالت→حالات. The sequence اال almost never occurs naturally in
    # Arabic — it's virtually always a reversed lam-alef ligature.
    text = text.replace("\u0627\u0627\u0644", "\u0627\u0644\u0627")  # اال → الا

    # Common definitions used across multiple fixes below.
    _NOT_ARABIC = r'(?<![\u0600-\u06FF])'
    _PREFIXES = '\u0648\u0628\u0641'  # و ب ف
    _CONSONANTS = (
        '\u0628\u062A\u062B\u062C\u062D\u062E\u062F\u0630'  # ب ت ث ج ح خ د ذ
        '\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638'  # ر ز س ش ص ض ط ظ
        '\u0639\u063A\u0641\u0642\u0643\u0645\u0646\u0647'  # ع غ ف ق ك م ن ه
        '\u0648\u064A'                                       # و ي
    )
    _CONS_CLASS = '[' + _CONSONANTS + ']'
    _FOLLOWED_BY_ARABIC = r'(?=[\u0600-\u06FF])'

    # === Ba-alef reversal (MUST run before general lam-consonant fixes) ===
    # The preposition ب + definite article ال produces a ba-alef ligature
    # that CMap reverses: با → اب.  Combined with lam-consonant reversal:
    #   ابXYل → بالYX  (e.g., ابحملكمة → بالمحكمة)
    #   ابXل  → بالX   (e.g., ابحلكم → بالحكم)
    #   ابل   → بال    (e.g., ابلوزارة → بالوزارة)
    #   ابأل  → بالأ   (e.g., ابألشعة → بالأشعة)

    # 2-consonant after اب: ابXYل → بالYX
    def _fix_ba_2cons(m: re.Match) -> str:
        c1, c2 = m.group(1), m.group(2)
        return '\u0628\u0627\u0644' + c2 + c1  # بال + reversed pair

    text = re.sub(
        _NOT_ARABIC + r'\u0627\u0628(' + _CONS_CLASS + r')(' + _CONS_CLASS
        + r')\u0644' + _FOLLOWED_BY_ARABIC,
        _fix_ba_2cons,
        text,
    )

    # 1-consonant after اب: ابXل → بالX
    text = re.sub(
        _NOT_ARABIC + r'\u0627\u0628(' + _CONS_CLASS + r')\u0644'
        + _FOLLOWED_BY_ARABIC,
        '\u0628\u0627\u0644\\1',  # بالX
        text,
    )

    # Simple ابل → بال (when followed by Arabic chars)
    text = re.sub(
        _NOT_ARABIC + r'\u0627\u0628\u0644' + _FOLLOWED_BY_ARABIC,
        '\u0628\u0627\u0644',  # بال
        text,
    )
    # With hamza variants: ابأل → بالأ, ابإل → بالإ, ابآل → بالآ
    text = re.sub(
        _NOT_ARABIC + r'\u0627\u0628([\u0623\u0625\u0622])\u0644',
        '\u0628\u0627\u0644\\1',  # بالأ/بالإ/بالآ
        text,
    )

    # === Lam-consonant reversals: اXل → الX at word start ===
    # CMap maps lam-consonant ligatures (لح, لج, لخ, etc.) with characters
    # reversed.  At word start after alef (the definite article), this
    # produces اXل instead of الX.  Require continuation after ل to avoid
    # false positives with short words (أهل has hamza, not bare alef).

    # 2-consonant reversal: اXYل → الYX (e.g., احملكوم → المحكوم)
    def _fix_2cons(m: re.Match) -> str:
        prefix = m.group(1) or ''
        c1, c2 = m.group(2), m.group(3)
        return prefix + '\u0627\u0644' + c2 + c1

    text = re.sub(
        _NOT_ARABIC + r'([' + _PREFIXES + r']?)'
        r'\u0627(' + _CONS_CLASS + r')(' + _CONS_CLASS + r')\u0644'
        + _FOLLOWED_BY_ARABIC,
        _fix_2cons,
        text,
    )

    # 1-consonant reversal: اXل → الX (e.g., احلكم → الحكم)
    text = re.sub(
        _NOT_ARABIC + r'\u0627(' + _CONS_CLASS + r')\u0644'
        + _FOLLOWED_BY_ARABIC,
        '\u0627\u0644\\1',  # الX
        text,
    )

    # 1-consonant with prefix: (prefix)اXل → (prefix)الX
    text = re.sub(
        _NOT_ARABIC + r'([' + _PREFIXES + r'])\u0627(' + _CONS_CLASS
        + r')\u0644' + _FOLLOWED_BY_ARABIC,
        '\\1\u0627\u0644\\2',
        text,
    )

    # === Common name reversals: حم↔مح in names ===
    # Mid-word ligature reversals where مح gets reversed to حم.
    # Only fix known high-frequency patterns to avoid false positives.
    _NAME_FIXES = [
        ('\u062D\u0645\u0645\u062F', '\u0645\u062D\u0645\u062F'),      # حممد → محمد
        ('\u062D\u0645\u0645\u0648\u062F', '\u0645\u062D\u0645\u0648\u062F'),  # حممود → محمود
        ('\u0623\u0645\u062D', '\u0623\u062D\u0645'),                   # أمح → أحم
    ]
    for wrong, right in _NAME_FIXES:
        text = text.replace(wrong, right)

    # رمح → رحم only in name contexts (followed by ن or ة), to avoid
    # false positives with رمح (spear).
    text = re.sub(
        r'\u0631\u0645\u062D([\u0646\u0629])',  # رمح + (ن or ة)
        '\u0631\u062D\u0645\\1',                 # رحم + same
        text,
    )

    # Lam-meem: امل → الم for the reversed definite article ال + meem.
    # Only replace when امل is at the start of a word (not mid-word).
    # Arabic has inseparable prefixes (و ب ف ل) that attach without space,
    # e.g. واملال = و+المال, باملال = ب+المال.
    # We must NOT replace امل mid-word (e.g., عامل، كامل، معامل، تأمل).
    _AML = '\u0627\u0645\u0644'  # امل
    _ALM = '\u0627\u0644\u0645'  # الم

    # Fix bare word-initial: not preceded by Arabic letter
    text = re.sub(_NOT_ARABIC + _AML, _ALM, text)

    # Fix prefixed: not preceded by Arabic letter + single prefix + امل
    text = re.sub(
        _NOT_ARABIC + r'([' + _PREFIXES + r'])' + _AML,
        r'\1' + _ALM,
        text,
    )

    return text



def normalize_arabic_numerals(text: str) -> str:
    """Normalize Eastern Arabic-Indic numerals to Western Arabic numerals.

    Converts ٠١٢٣٤٥٦٧٨٩ to 0123456789.

    Args:
        text: Text possibly containing Eastern Arabic-Indic digits.

    Returns:
        Text with numerals standardized to Western Arabic digits.
    """
    _EASTERN_ARABIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return text.translate(_EASTERN_ARABIC)


def clean_arabic_text(text: str, strip_tashkeel: bool = True,
                      strip_tatweel: bool = True,
                      normalize_letters: bool = False) -> str:
    """Apply Arabic-specific text cleaning.

    Args:
        text: Arabic text to clean.
        strip_tashkeel: Remove diacritical marks.
        strip_tatweel: Remove decorative elongation.
        normalize_letters: Normalize letter variants (lossy — use for search only).
            Normalizes alef variants to bare alef, alef-maksura to ya,
            ta-marbuta to ha, and Eastern Arabic numerals to Western.

    Returns:
        Cleaned Arabic text.
    """
    text = normalize_presentation_forms(text)
    text = normalize_arabic_script_variants(text)
    if strip_tatweel:
        text = remove_tatweel(text)
    if strip_tashkeel:
        text = remove_tashkeel(text)
    # Normalize Arabic punctuation (،→, etc.) BEFORE ligature fixes,
    # because ، is U+060C which falls in [\u0600-\u06FF] and would
    # block the word-boundary lookbehinds in the ligature fixers.
    text = fix_rtl_punctuation(text)
    # Fix ligature reversal AFTER tashkeel/tatweel removal, because
    # diacritical marks (e.g., shadda) can appear before امل and
    # block word-boundary detection. With tashkeel stripped, the
    # lookbehind correctly sees the non-Arabic character boundary.
    text = fix_lam_alef_ligature(text)
    if normalize_letters:
        text = normalize_arabic_letters(text)
        text = normalize_arabic_numerals(text)
    text = re.sub(r"\s+([,.:;!?\u061F])", r"\1", text)
    return text


def _is_arabic_word(word: str) -> bool:
    """Check if a word consists entirely of Arabic characters."""
    return bool(_ARABIC_WORD.match(word))


def _reverse_arabic_word(word: str) -> str:
    """Reverse an Arabic word to fix visual-to-logical order.

    When PDF extractors output Arabic in visual (LTR) order, each word's
    letters are reversed. This restores logical (RTL) order.
    """
    return word[::-1]


def fix_garbled_arabic(text: str) -> str:
    """Fix common Arabic PDF extraction problems.

    Handles:
    1. Presentation Forms normalization (isolated/final/medial/initial forms)
    2. Reversed Arabic words (visual-to-logical reordering)
    3. Broken lam-alef ligatures

    This should be called early in the extraction pipeline, before other
    Arabic cleaning steps.

    Args:
        text: Text potentially containing garbled Arabic from PDF extraction.

    Returns:
        Text with Arabic garbling issues fixed.
    """
    if not text:
        return ""

    if not contains_arabic(text):
        return text

    # Step 1: Normalize presentation forms (isolated/connected forms → standard)
    text = normalize_presentation_forms(text)

    # Step 2: Fix reversed Arabic words
    # PDF extractors sometimes output Arabic letters in visual (LTR) order.
    # We detect this by checking if reversing a word produces more valid
    # Arabic character connections.
    words = text.split()
    fixed_words: list[str] = []
    for word in words:
        if _is_arabic_word(word) and len(word) > 1:
            # Check if the word looks reversed by testing character connectivity
            # Heuristic: in correct Arabic, certain letter combinations are
            # much more common (e.g., lam before alef). If reversing produces
            # more natural combinations, the word was likely reversed.
            reversed_word = _reverse_arabic_word(word)
            if _score_arabic_naturalness(reversed_word) > _score_arabic_naturalness(word):
                fixed_words.append(reversed_word)
            else:
                fixed_words.append(word)
        else:
            fixed_words.append(word)
    text = " ".join(fixed_words)

    return text


def _score_arabic_naturalness(word: str) -> int:
    """Score how natural an Arabic word looks based on common bigrams.

    Higher score = more likely to be correctly ordered Arabic text.
    Uses common Arabic letter pair frequencies as heuristic.
    """
    score = 0
    # Common Arabic bigrams (lam-alef, alef-lam, etc.)
    common_pairs = {
        ("\u0644", "\u0627"),  # lam-alef (لا) — extremely common
        ("\u0627", "\u0644"),  # alef-lam (ال) — definite article
        ("\u0645", "\u0646"),  # mim-nun
        ("\u0639", "\u0644"),  # ain-lam
        ("\u0641", "\u064a"),  # fa-ya
        ("\u0628", "\u0627"),  # ba-alef
        ("\u062a", "\u0627"),  # ta-alef
    }

    for i in range(len(word) - 1):
        pair = (word[i], word[i + 1])
        if pair in common_pairs:
            score += 1
        # Definite article ال at start is very common
        if i == 0 and word[i] == "\u0627" and len(word) > 1 and word[1] == "\u0644":
            score += 3

    return score
