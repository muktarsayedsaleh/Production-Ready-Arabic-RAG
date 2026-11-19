import string
import unicodedata
from dataclasses import fields, replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.text.arabic import DEFAULT_CONFIG, ArabicNormalizer, NormalizerConfig, normalize

ALL_OFF = NormalizerConfig(**{f.name: False for f in fields(NormalizerConfig)})


def only(**enabled: bool) -> NormalizerConfig:
    """A config with every rule off except the ones named."""
    return replace(ALL_OFF, **enabled)


# --- One table per rule. Each case runs with only that rule on, so a rule can't pass
# --- because a different rule happened to do its job.


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ﻻ", "لا"),  # lam-alef ligature
        ("ﻷ", "لأ"),  # lam-alef with hamza ligature
        ("ﺑﺴﻢ", "بسم"),  # initial/medial/final presentation forms
        ("ﷲ", "الله"),  # Allah ligature
        ("ﷺ", "صلى الله عليه وسلم"),
        ("ﬁle", "file"),  # Latin compatibility ligature
        ("آ", "آ"),  # alef + combining madda composes
        ("أ", "أ"),  # alef + combining hamza above composes
        ("ؤ", "ؤ"),
    ],
)
def test_compatibility_forms(text: str, expected: str) -> None:
    assert normalize(text, only(compatibility_forms=True)) == expected


def test_nfc_is_applied_even_when_compatibility_forms_are_off() -> None:
    config = only()

    assert normalize("آ", config) == "آ"
    assert normalize("ﻻ", config) == "ﻻ"  # presentation form kept without NFKC


@pytest.mark.parametrize(
    "char",
    [
        "​",  # zero-width space
        "‌",  # zero-width non-joiner
        "‍",  # zero-width joiner
        "‎",  # left-to-right mark
        "‏",  # right-to-left mark
        "؜",  # Arabic letter mark
        "‪",  # LRE
        "‫",  # RLE
        "‬",  # PDF
        "‭",  # LRO
        "‮",  # RLO
        "⁦",  # LRI
        "⁧",  # RLI
        "⁨",  # FSI
        "⁩",  # PDI
        "⁠",  # word joiner
        "﻿",  # byte order mark
        "­",  # soft hyphen
        "؀",  # Arabic number sign
        "۝",  # Arabic end of ayah
    ],
)
def test_format_chars_are_removed(char: str) -> None:
    assert normalize(f"مر{char}حبا", only(remove_format_chars=True)) == "مرحبا"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("الْعَرَبِيَّةُ", "العربية"),  # fatha, sukun, kasra, shadda, damma
        ("كتابًا", "كتابا"),  # fathatan
        ("كتابٌ", "كتاب"),  # dammatan
        ("كتابٍ", "كتاب"),  # kasratan
        ("هٰذا", "هذا"),  # superscript alef
        ("الرَّحْمَٰنِ", "الرحمن"),
        ("صلى الله عليه وسلمؐ", "صلى الله عليه وسلم"),  # Quranic honorific mark
        ("قالۖ", "قال"),  # Quranic annotation mark
    ],
)
def test_strip_diacritics(text: str, expected: str) -> None:
    assert normalize(text, only(strip_diacritics=True)) == expected


def test_strip_diacritics_keeps_base_letters_and_hamza_letters() -> None:
    text = "أإآؤئء"

    assert normalize(text, only(strip_diacritics=True)) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [("مـــرحبا", "مرحبا"), ("الســلام", "السلام"), ("ـ", "")],
)
def test_strip_tatweel(text: str, expected: str) -> None:
    assert normalize(text, only(strip_tatweel=True)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("کتاب", "كتاب"), ("علی", "علي"), ("کی", "كي")],
)
def test_fold_persian_letters(text: str, expected: str) -> None:
    assert normalize(text, only(fold_persian_letters=True)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("أحمد", "احمد"),
        ("إسلام", "اسلام"),
        ("آمن", "امن"),
        ("ٱلله", "الله"),  # alef wasla
        ("ٲ", "ا"),  # alef with wavy hamza above
        ("ٳ", "ا"),  # alef with wavy hamza below
    ],
)
def test_fold_alef(text: str, expected: str) -> None:
    assert normalize(text, only(fold_alef=True)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("على", "علي"), ("مستشفى", "مستشفي"), ("مصطفى", "مصطفي")],
)
def test_fold_alef_maqsura(text: str, expected: str) -> None:
    assert normalize(text, only(fold_alef_maqsura=True)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("مدرسة", "مدرسه"), ("جامعة الملك", "جامعه الملك")],
)
def test_fold_teh_marbuta(text: str, expected: str) -> None:
    assert normalize(text, only(fold_teh_marbuta=True)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("مؤسسة", "مءسسة"), ("رئيس", "رءيس"), ("سؤال", "سءال")],
)
def test_fold_hamza_carriers(text: str, expected: str) -> None:
    assert normalize(text, only(fold_hamza_carriers=True)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("٠١٢٣٤٥٦٧٨٩", "0123456789"),  # Arabic-Indic
        ("۰۱۲۳۴۵۶۷۸۹", "0123456789"),  # Extended Arabic-Indic (Persian)
        ("٣٫٥", "3.5"),  # Arabic decimal separator
        ("١٬٠٠٠", "1,000"),  # Arabic thousands separator
        ("INV-٢٠٤١", "INV-2041"),
    ],
)
def test_ascii_digits(text: str, expected: str) -> None:
    assert normalize(text, only(ascii_digits=True)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("نعم، لا", "نعم, لا"),
        ("أولاً؛ ثانياً", "أولاً; ثانياً"),
        ("لماذا؟", "لماذا?"),
        ("٥٠٪", "٥٠%"),
        ("ختام۔", "ختام."),
    ],
)
def test_ascii_punctuation(text: str, expected: str) -> None:
    assert normalize(text, only(ascii_punctuation=True)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a  b\t\tc", "a b c"),
        ("  leading and trailing  ", "leading and trailing"),
        ("line one  \n  line two", "line one\nline two"),
        ("windows\r\nline\rbreaks", "windows\nline\nbreaks"),
        ("page\x0cbreak", "page\nbreak"),
        ("unicode line paragraph", "unicode\nline\nparagraph"),
        ("para one\n\n\n\n\npara two", "para one\n\npara two"),
        ("no break em space", "no break em space"),
        ("   \n\n  ", ""),
    ],
)
def test_normalize_whitespace(text: str, expected: str) -> None:
    assert normalize(text, only(normalize_whitespace=True)) == expected


# --- Defaults


def test_defaults_on_mixed_document_text() -> None:
    text = "فـــاتورة  رقم‏ INV-2041 بتاريخ ٢٠٢٤/٠٣/١٥ إلى السيِّد أحمد علی"

    assert normalize(text) == "فاتورة رقم INV-2041 بتاريخ 2024/03/15 الي السيد احمد علي"


def test_defaults_keep_teh_marbuta_hamza_carriers_and_arabic_punctuation() -> None:
    text = "مدرسة، مؤسسة؟ رئيس؛"

    assert normalize(text) == text


def test_query_and_document_spellings_meet() -> None:
    document = "ذَهَبَ أحمدُ إلى المستشفى يوم ١٥"
    queries = ["ذهب احمد الي المستشفي يوم 15", "ذهب أحمد إلى المستشفى يوم ١٥"]

    assert {normalize(query) for query in queries} == {normalize(document)}


def test_extracted_pdf_text_with_presentation_forms_matches_clean_text() -> None:
    from_pdf = "ﺍﻟﻤﻜﺘﺒﺔ ﺍﻟﻌﺎﻣﺔ"  # what some PDF extractors return

    assert normalize(from_pdf) == normalize("المكتبة العامة")


def test_empty_string() -> None:
    assert normalize("") == ""


def test_all_rules_off_only_applies_nfc() -> None:
    text = "أَحْمَد ـ ٥ ، x  y"

    assert normalize(text, ALL_OFF) == unicodedata.normalize("NFC", text)


def test_normalizer_instance_matches_function() -> None:
    config = replace(DEFAULT_CONFIG, fold_teh_marbuta=True)

    assert ArabicNormalizer(config)("مدرسة") == normalize("مدرسة", config) == "مدرسه"


# --- Config fingerprint


def test_fingerprint_is_stable_for_equal_configs() -> None:
    assert NormalizerConfig().fingerprint() == NormalizerConfig().fingerprint()


def test_fingerprint_changes_when_any_rule_changes() -> None:
    prints = {
        replace(DEFAULT_CONFIG, **{f.name: not getattr(DEFAULT_CONFIG, f.name)}).fingerprint()
        for f in fields(NormalizerConfig)
    }

    assert len(prints) == len(fields(NormalizerConfig))
    assert DEFAULT_CONFIG.fingerprint() not in prints


@pytest.mark.parametrize(
    "text",
    [
        # Found by fuzzing: NFKC turns U+FCF2 into tatweel + two marks; removing the tatweel
        # leaves marks in non-canonical order after U+0610, which a second pass reorders.
        "ؐﳲ",
        # Removing the tatweel lets alef and madda compose into آ, which then needs folding.
        "اـٓ",
    ],
)
def test_removals_that_expose_new_compositions_still_settle(text: str) -> None:
    config = only(compatibility_forms=True, strip_tatweel=True, fold_alef=True)

    once = normalize(text, config)

    assert normalize(once, config) == once


# --- Properties

ARABIC_HEAVY = st.text(
    alphabet=st.one_of(
        st.characters(min_codepoint=0x0600, max_codepoint=0x06FF),  # Arabic
        st.characters(min_codepoint=0xFB50, max_codepoint=0xFDFF),  # presentation forms A
        st.characters(min_codepoint=0xFE70, max_codepoint=0xFEFF),  # presentation forms B
        st.sampled_from(list(" \t\n\r ​‌‏‫ ")),
        st.sampled_from(list(string.ascii_letters + string.digits + "-/.,")),
    ),
    max_size=200,
)

CONFIGS = st.builds(NormalizerConfig, **{f.name: st.booleans() for f in fields(NormalizerConfig)})


@settings(max_examples=500)
@given(text=st.text(max_size=200))
def test_default_normalization_is_idempotent_on_any_text(text: str) -> None:
    once = normalize(text)

    assert normalize(once) == once


@settings(max_examples=500)
@given(text=ARABIC_HEAVY, config=CONFIGS)
def test_every_config_is_idempotent_on_arabic_text(text: str, config: NormalizerConfig) -> None:
    once = normalize(text, config)

    assert normalize(once, config) == once


@settings(max_examples=300)
@given(
    words=st.lists(
        st.text(alphabet=string.ascii_letters + string.digits + string.punctuation, min_size=1),
        max_size=20,
    )
)
def test_ascii_text_with_single_spaces_is_unchanged(words: list[str]) -> None:
    text = " ".join(words)

    assert normalize(text) == text


@settings(max_examples=300)
@given(text=ARABIC_HEAVY)
def test_defaults_leave_no_marks_tatweel_or_format_chars(text: str) -> None:
    result = normalize(text)

    leftovers = {
        char
        for char in result
        if char == "ـ" or unicodedata.category(char) == "Cf" or "ً" <= char <= "ٟ" or char == "ٰ"
    }
    assert not leftovers


@settings(max_examples=300)
@given(text=ARABIC_HEAVY)
def test_defaults_leave_no_foldable_presentation_forms(text: str) -> None:
    result = normalize(text)

    # Presentation-form glyphs with a compatibility mapping must be folded. A few code
    # points in these blocks are standalone symbols (e.g. U+FBB2) with nothing to fold to.
    def foldable(char: str) -> bool:
        in_block = "ﭐ" <= char <= "﷿" or "ﹰ" <= char <= "﻿"
        return in_block and unicodedata.decomposition(char).startswith("<")

    assert not any(foldable(char) for char in result)
