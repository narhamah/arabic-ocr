"""Tests for Arabic text normalizer — written BEFORE implementation (TDD RED phase)."""

import pytest
from arabic_ocr.normalizer import normalize


class TestNFKCNormalization:
    """NFKC converts Arabic Presentation Forms to standard Arabic."""

    def test_presentation_form_b_initial(self):
        # U+FE91 (beh initial form) -> U+0628 (beh)
        result = normalize("\uFE91")
        assert "\u0628" in result

    def test_presentation_form_lam_alef_ligature(self):
        # U+FEFB (lam-alef ligature) -> U+0644 U+0627
        result = normalize("\uFEFB")
        assert "\u0644" in result and "\u0627" in result

    def test_preserves_standard_arabic(self):
        text = "بسم الله الرحمن الرحيم"
        # Already standard, alef normalization will change أ but this text has none
        assert normalize(text, strip_diacritics=True) == "بسم الله الرحمن الرحيم"


class TestBiDiStripping:
    """Remove bidirectional control characters."""

    def test_strips_rtl_mark(self):
        result = normalize("hello\u200Fworld")
        assert "\u200F" not in result

    def test_strips_ltr_mark(self):
        result = normalize("hello\u200Eworld")
        assert "\u200E" not in result

    def test_strips_bidi_embedding(self):
        result = normalize("\u202Atext\u202C")
        assert "\u202A" not in result
        assert "\u202C" not in result
        assert "text" in result

    def test_strips_zero_width_space(self):
        result = normalize("word\u200Bword")
        assert "\u200B" not in result

    def test_strips_bom(self):
        result = normalize("\uFEFFtext")
        assert "\uFEFF" not in result


class TestTatweelStripping:
    """Remove kashida/tatweel elongation character."""

    def test_strips_tatweel(self):
        assert normalize("الـــكـــويـــت") == "الكويت"

    def test_no_effect_without_tatweel(self):
        assert normalize("الكويت") == "الكويت"


class TestAlefNormalization:
    """Normalize Alef variants to bare Alef."""

    def test_alef_hamza_above(self):
        assert normalize("أحمد") == "احمد"

    def test_alef_hamza_below(self):
        assert normalize("إسلام") == "اسلام"

    def test_alef_madda(self):
        assert normalize("آمن") == "امن"

    def test_alef_wasla(self):
        assert normalize("\u0671من") == "امن"


class TestAlefMaksuraNormalization:
    """Normalize Alef Maksura to Yeh."""

    def test_alef_maksura_to_yeh(self):
        assert normalize("على") == "علي"

    def test_final_alef_maksura(self):
        assert normalize("موسى") == "موسي"


class TestDiacriticStripping:
    """Strip harakat/tashkeel diacritics."""

    def test_strips_fatha(self):
        assert normalize("كَتَبَ") == "كتب"

    def test_strips_shadda(self):
        assert normalize("محمَّد") == "محمد"

    def test_strips_kasra(self):
        assert normalize("بِسْمِ") == "بسم"

    def test_preserve_diacritics_option(self):
        assert normalize("كَتَبَ", strip_diacritics=False) == "كَتَبَ"

    def test_strips_sukun(self):
        assert normalize("مِنْ") == "من"


class TestNumeralNormalization:
    """Convert Eastern Arabic numerals to Western."""

    def test_eastern_to_western(self):
        assert normalize("٢٠٢٦") == "2026"

    def test_preserves_western(self):
        assert normalize("2026") == "2026"

    def test_mixed(self):
        assert normalize("العدد ١٨٥٥٩") == "العدد 18559"

    def test_all_eastern_digits(self):
        assert normalize("٠١٢٣٤٥٦٧٨٩") == "0123456789"


class TestWhitespaceCleanup:
    """Collapse multiple spaces, strip edges."""

    def test_collapses_spaces(self):
        assert normalize("كلمة    كلمة") == "كلمة كلمة"

    def test_strips_edges(self):
        assert normalize("  نص  ") == "نص"

    def test_collapses_mixed_whitespace(self):
        assert normalize("كلمة\t\n  كلمة") == "كلمة كلمة"


class TestFullPipeline:
    """End-to-end normalization."""

    def test_real_ocr_output(self):
        """Simulated messy OCR output with multiple issues."""
        raw = "\u202Bأكـــد\u200F الـوزيـر\u0640 أنَّ العدد ١٢٣\u202C"
        result = normalize(raw)
        assert result == "اكد الوزير ان العدد 123"

    def test_empty_string(self):
        assert normalize("") == ""

    def test_only_whitespace(self):
        assert normalize("   \t\n  ") == ""

    def test_only_control_chars(self):
        assert normalize("\u200F\u200E\u202A") == ""
